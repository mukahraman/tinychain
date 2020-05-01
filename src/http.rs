use std::collections::{HashMap, HashSet};
use std::convert::{Infallible, TryInto};
use std::sync::Arc;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use crate::error;
use crate::host::Host;
use crate::state::State;
use crate::value::{Op, TCPath, TCResult, TCValue, ValueId};

fn line_numbers(s: &str) -> String {
    s.lines()
        .enumerate()
        .map(|(i, l)| format!("{} {}", i, l))
        .collect::<Vec<String>>()
        .join("\n")
}

pub async fn listen(
    host: Arc<Host>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let make_svc = make_service_fn(|_conn| {
        let host = host.clone();
        async { Ok::<_, Infallible>(service_fn(move |req| handle(host.clone(), req))) }
    });

    let addr = ([127, 0, 0, 1], port).into();
    let server = Server::bind(&addr).serve(make_svc);

    println!("Listening on http://{}", addr);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }

    Ok(())
}

async fn get(host: Arc<Host>, path: &TCPath, params: &HashMap<String, String>) -> TCResult<State> {
    let key = if let Some(key) = params.get("key") {
        serde_json::from_str::<TCValue>(key)
            .map_err(|e| error::bad_request("Unable to parse 'key' param", e))?
    } else {
        TCValue::None
    };

    host.get(host.new_transaction()?, path, key).await
}

async fn route(
    host: Arc<Host>,
    method: Method,
    path: &str,
    params: HashMap<String, String>,
    body: Vec<u8>,
) -> TCResult<Vec<u8>> {
    let path: TCPath = path.try_into()?;

    match method {
        Method::GET => match get(host, &path, &params).await? {
            State::Value(val) => Ok(serde_json::to_string_pretty(&val)?.as_bytes().to_vec()),
            state => Err(error::bad_request(
                "Attempt to GET unserializable state {}",
                state,
            )),
        },
        Method::POST => {
            let capture: HashSet<ValueId> = if let Some(c) = params.get("capture") {
                c.split('/')
                    .map(|s| s.try_into())
                    .collect::<TCResult<HashSet<ValueId>>>()
            } else {
                Ok(HashSet::new())
            }?;
            let values = match serde_json::from_slice::<Vec<(ValueId, TCValue)>>(&body) {
                Ok(graph) => graph,
                Err(cause) => {
                    let body = line_numbers(std::str::from_utf8(&body).unwrap());
                    return Err(error::bad_request(
                        &format!("{}\n\nUnable to parse request", body),
                        cause,
                    ));
                }
            };

            let txn = host.clone().transact(Op::post(None, path, values))?;
            let mut results: HashMap<ValueId, TCValue> = HashMap::new();
            match txn.execute(capture).await {
                Ok(responses) => {
                    for (id, r) in responses {
                        match r {
                            State::Value(val) => {
                                results.insert(id, val.clone());
                            }
                            other => {
                                return Err(error::bad_request(
                                    "Attempt to capture an unserializable value",
                                    other,
                                ));
                            }
                        }
                    }
                }
                Err(cause) => {
                    return Err(cause);
                }
            };

            txn.commit().await;

            serde_json::to_string_pretty(&results)
                .and_then(|s| Ok(s.into_bytes()))
                .or_else(|e| {
                    let msg = "Your request completed successfully but there was an error serializing the response";
                    Err(error::bad_request(msg, e))
                })
        }
        _ => Err(error::not_found(path)),
    }
}

async fn handle(host: Arc<Host>, req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path();

    let params: HashMap<String, String> = uri
        .query()
        .map(|v| {
            url::form_urlencoded::parse(v.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_else(HashMap::new);

    let body = &hyper::body::to_bytes(req.into_body()).await?;

    transform_error(route(host, method, path, params, body.to_vec()).await)
}

fn transform_error(result: TCResult<Vec<u8>>) -> Result<Response<Body>, hyper::Error> {
    match result {
        Ok(contents) => Ok(Response::new(Body::from(contents))),
        Err(cause) => {
            let mut response = Response::new(Body::from(cause.message().to_string()));
            *response.status_mut() = match cause.reason() {
                error::Code::BadRequest => StatusCode::BAD_REQUEST,
                error::Code::Internal => StatusCode::INTERNAL_SERVER_ERROR,
                error::Code::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
                error::Code::NotFound => StatusCode::NOT_FOUND,
                error::Code::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            };
            Ok(response)
        }
    }
}
