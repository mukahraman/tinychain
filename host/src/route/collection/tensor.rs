use afarray::Array;
use futures::{Future, TryFutureExt};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_btree::Node;
use tc_error::*;
use tc_tensor::*;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tcgeneric::{label, PathSegment, TCBoxTryFuture, Tuple};

use crate::collection::{Collection, Tensor};
use crate::fs;
use crate::route::{GetHandler, PostHandler, PutHandler};
use crate::scalar::{Bound, Number, NumberClass, Range, Value};
use crate::state::State;
use crate::txn::Txn;

use super::{Handler, Route};

struct ConstantHandler;

impl<'a> Handler<'a> for ConstantHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<(Vec<u64>, Number)>() {
                    let (shape, value): (Vec<u64>, Number) = key.opt_cast_into().unwrap();
                    constant(&txn, shape, value).await
                } else {
                    Err(TCError::bad_request("invalid tensor schema", key))
                }
            })
        }))
    }
}

struct CreateHandler {
    class: TensorType,
}

impl<'a> Handler<'a> for CreateHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<Schema>() {
                    let Schema { shape, dtype } = key.opt_cast_into().unwrap();

                    match self.class {
                        TensorType::Dense => constant(&txn, shape, dtype.zero()).await,
                        TensorType::Sparse => Err(TCError::not_implemented("create sparse tensor")),
                    }
                } else {
                    Err(TCError::bad_request(
                        "invalid schema for constant tensor",
                        key,
                    ))
                }
            })
        }))
    }
}

struct ExpandHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for ExpandHandler<T>
where
    T: TensorTransform + Send + 'a,
    Tensor: From<T::Expand>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let axis = key.try_cast_into(|v| TCError::bad_request("invalid tensor axis", v))?;

                self.tensor
                    .expand_dims(axis)
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for ExpandHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct RangeHandler;

impl<'a> Handler<'a> for RangeHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<(Vec<u64>, Number, Number)>() {
                    let (shape, start, stop): (Vec<u64>, Number, Number) =
                        key.opt_cast_into().unwrap();

                    let file = create_file(&txn).await?;

                    DenseTensor::range(file, *txn.id(), shape, start, stop)
                        .map_ok(Tensor::from)
                        .map_ok(Collection::from)
                        .map_ok(State::from)
                        .await
                } else {
                    Err(TCError::bad_request("invalid schema for range tensor", key))
                }
            })
        }))
    }
}

struct TransposeHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for TransposeHandler<T>
where
    T: TensorTransform + Send + 'a,
    Tensor: From<T::Transpose>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let transpose = if key.is_none() {
                    self.tensor.transpose(None)
                } else {
                    let permutation = key.try_cast_into(|v| {
                        TCError::bad_request("invalid permutation for transpose", v)
                    })?;

                    self.tensor.transpose(Some(permutation))
                };

                transpose
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for TransposeHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

impl Route for TensorType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateHandler { class: *self }))
        } else if path.len() == 1 && self == &Self::Dense {
            match path[0].as_str() {
                "constant" => Some(Box::new(ConstantHandler)),
                "range" => Some(Box::new(RangeHandler)),
                _ => None,
            }
        } else {
            None
        }
    }
}

struct DualHandler {
    tensor: Tensor,
    op: fn(Tensor, Tensor) -> TCResult<Tensor>,
}

impl DualHandler {
    fn new<T>(tensor: T, op: fn(Tensor, Tensor) -> TCResult<Tensor>) -> Self
    where
        Tensor: From<T>,
    {
        Self {
            tensor: tensor.into(),
            op,
        }
    }
}

impl<'a> Handler<'a> for DualHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let l = self.tensor;
                let r: Tensor = params.require(&label("r").into())?;
                params.expect_empty()?;

                if l.shape() == r.shape() {
                    debug!("tensor dual op with shapes {} {}", l.shape(), r.shape());
                    (self.op)(l, r).map(Collection::from).map(State::from)
                } else {
                    let (l, r) = broadcast(l, r)?;
                    debug!("tensor dual op with shapes {} {}", l.shape(), r.shape());
                    (self.op)(l, r).map(Collection::from).map(State::from)
                }
            })
        }))
    }
}

struct ReduceHandler<'a, T: TensorReduce<fs::Dir>> {
    tensor: &'a T,
    reduce: fn(T, usize) -> TCResult<<T as TensorReduce<fs::Dir>>::Reduce>,
    reduce_all: fn(&'a T, Txn) -> TCBoxTryFuture<'a, Number>,
}

impl<'a, T: TensorReduce<fs::Dir>> ReduceHandler<'a, T> {
    fn new(
        tensor: &'a T,
        reduce: fn(T, usize) -> TCResult<<T as TensorReduce<fs::Dir>>::Reduce>,
        reduce_all: fn(&'a T, Txn) -> TCBoxTryFuture<'a, Number>,
    ) -> Self {
        Self {
            tensor,
            reduce,
            reduce_all,
        }
    }
}

impl<'a, T: TensorReduce<fs::Dir> + Clone + Sync> Handler<'a> for ReduceHandler<'a, T>
where
    Tensor: From<<T as TensorReduce<fs::Dir>>::Reduce>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    (self.reduce_all)(self.tensor, txn.clone())
                        .map_ok(Value::from)
                        .map_ok(State::from)
                        .await
                } else {
                    let axis = key.try_cast_into(|v| TCError::bad_request("invalid axis", v))?;
                    (self.reduce)(self.tensor.clone(), axis)
                        .map(Tensor::from)
                        .map(Collection::from)
                        .map(State::from)
                }
            })
        }))
    }
}

struct TensorHandler<T> {
    tensor: T,
}

impl<'a, T: 'a> Handler<'a> for TensorHandler<T>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorTransform
        + Clone
        + Send
        + Sync,
    <T as TensorTransform>::Slice: TensorAccess + Send,
    Tensor: From<<T as TensorTransform>::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                debug!("GET Tensor: {}", key);
                let bounds = cast_bounds(self.tensor.shape(), key)?;
                self.tensor
                    .slice(bounds)
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            debug!("PUT Tensor: {} <- {}", key, value);
            Box::pin(write(self.tensor, txn, key, value))
        }))
    }
}

impl<T> From<T> for TensorHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct UnaryHandler {
    tensor: Tensor,
    op: fn(&Tensor) -> TCResult<Tensor>,
}

impl UnaryHandler {
    fn new(tensor: Tensor, op: fn(&Tensor) -> TCResult<Tensor>) -> Self {
        Self { tensor, op }
    }
}

impl<'a> Handler<'a> for UnaryHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let tensor = if key.is_none() {
                    self.tensor
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key.into())?;
                    self.tensor.slice(bounds)?
                };

                (self.op)(&tensor).map(Collection::from).map(State::from)
            })
        }))
    }
}

struct UnaryHandlerAsync<F: Send> {
    tensor: Tensor,
    op: fn(Tensor, Txn) -> F,
}

impl<'a, F: Send> UnaryHandlerAsync<F> {
    fn new(tensor: Tensor, op: fn(Tensor, Txn) -> F) -> Self {
        Self { tensor, op }
    }
}

impl<'a, F> Handler<'a> for UnaryHandlerAsync<F>
where
    F: Future<Output = TCResult<bool>> + Send + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let txn = txn.clone();

                if key.is_none() {
                    (self.op)(self.tensor, txn)
                        .map_ok(State::from)
                        .await
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key.into())?;
                    let slice = self.tensor.slice(bounds)?;
                    (self.op)(slice, txn).map_ok(State::from).await
                }
            })
        }))
    }
}

impl<B: DenseAccess<fs::File<Array>, fs::File<Node>, fs::Dir, Txn>> Route
    for DenseTensor<fs::File<Array>, fs::File<Node>, fs::Dir, Txn, B>
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl<A: SparseAccess<fs::File<Array>, fs::File<Node>, fs::Dir, Txn>> Route
    for SparseTensor<fs::File<Array>, fs::File<Node>, fs::Dir, Txn, A>
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl Route for Tensor {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

fn route<'a, T>(tensor: &'a T, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorCompare<Tensor, Compare = Tensor, Dense = Tensor>
        + TensorBoolean<Tensor, Combine = Tensor>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorMath<fs::Dir, Tensor, Combine = Tensor>
        + TensorReduce<fs::Dir, Txn = Txn>
        + TensorTransform
        + TensorUnary<fs::Dir, Txn = Txn>
        + Clone
        + Send
        + Sync,
    Collection: From<T>,
    Tensor: From<T>,
    <T as TensorTransform>::Slice: TensorAccess + Send,
    Tensor: From<<T as TensorReduce<fs::Dir>>::Reduce>,
    Tensor: From<<T as TensorTransform>::Expand>,
    Tensor: From<<T as TensorTransform>::Slice>,
    Tensor: From<<T as TensorTransform>::Transpose>,
{
    if path.is_empty() {
        Some(Box::new(TensorHandler::from(tensor.clone())))
    } else if path.len() == 1 {
        let cloned = tensor.clone();

        match path[0].as_str() {
            // boolean ops
            "and" => Some(Box::new(DualHandler::new(cloned, TensorBoolean::and))),
            "or" => Some(Box::new(DualHandler::new(cloned, TensorBoolean::or))),
            "xor" => Some(Box::new(DualHandler::new(cloned, TensorBoolean::xor))),

            // comparison ops
            "eq" => Some(Box::new(DualHandler::new(cloned, TensorCompare::eq))),
            "gt" => Some(Box::new(DualHandler::new(cloned, TensorCompare::gt))),
            "gte" => Some(Box::new(DualHandler::new(cloned, TensorCompare::gte))),
            "lt" => Some(Box::new(DualHandler::new(cloned, TensorCompare::lt))),
            "lte" => Some(Box::new(DualHandler::new(cloned, TensorCompare::lte))),
            "ne" => Some(Box::new(DualHandler::new(cloned, TensorCompare::ne))),

            // unary ops
            "abs" => Some(Box::new(UnaryHandler::new(cloned.into(), TensorUnary::abs))),
            "all" => Some(Box::new(UnaryHandlerAsync::new(
                cloned.into(),
                TensorUnary::all,
            ))),
            "any" => Some(Box::new(UnaryHandlerAsync::new(
                cloned.into(),
                TensorUnary::any,
            ))),
            "not" => Some(Box::new(UnaryHandler::new(cloned.into(), TensorUnary::not))),

            // basic math
            "add" => Some(Box::new(DualHandler::new(cloned, TensorMath::add))),
            "div" => Some(Box::new(DualHandler::new(cloned, TensorMath::div))),
            "mul" => Some(Box::new(DualHandler::new(cloned, TensorMath::mul))),
            "sub" => Some(Box::new(DualHandler::new(cloned, TensorMath::sub))),

            // reduce ops
            "product" => Some(Box::new(ReduceHandler::new(
                tensor,
                TensorReduce::product,
                TensorReduce::product_all,
            ))),
            "sum" => Some(Box::new(ReduceHandler::new(
                tensor,
                TensorReduce::sum,
                TensorReduce::sum_all,
            ))),

            // transforms
            "expand_dims" => Some(Box::new(ExpandHandler::from(cloned))),
            "transpose" => Some(Box::new(TransposeHandler::from(cloned))),

            _ => None,
        }
    } else {
        None
    }
}

async fn constant<S>(txn: &Txn, shape: S, value: Number) -> TCResult<State>
where
    Shape: From<S>,
{
    let file = create_file(txn).await?;

    DenseTensor::constant(file, *txn.id(), shape, value)
        .map_ok(Tensor::from)
        .map_ok(Collection::from)
        .map_ok(State::from)
        .await
}

async fn write<T>(tensor: T, txn: &Txn, key: Value, value: State) -> TCResult<()>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorTransform
        + Clone,
    <T as TensorTransform>::Slice: TensorAccess + Send,
{
    debug!("write {} to {}", value, key);
    let bounds = cast_bounds(tensor.shape(), key)?;

    match value {
        State::Collection(Collection::Tensor(value)) => tensor.write(txn.clone(), bounds, value).await,
        State::Scalar(scalar) => {
            let value =
                scalar.try_cast_into(|v| TCError::bad_request("invalid tensor element", v))?;

            tensor.write_value(*txn.id(), bounds, value).await
        }
        other => Err(TCError::bad_request(
            "cannot write this value to tensor",
            other,
        )),
    }
}

async fn create_file(txn: &Txn) -> TCResult<fs::File<afarray::Array>> {
    txn.context()
        .create_file_tmp(*txn.id(), TensorType::Dense)
        .await
}

fn cast_bound(dim: u64, bound: Value) -> TCResult<u64> {
    let bound = i64::try_cast_from(bound, |v| TCError::bad_request("invalid bound", v))?;
    if bound.abs() as u64 > dim {
        return Err(TCError::bad_request(
            format!("Index out of bounds for dimension {}", dim),
            bound,
        ));
    }

    if bound < 0 {
        Ok(dim - bound.abs() as u64)
    } else {
        Ok(bound as u64)
    }
}

fn cast_range(dim: u64, range: Range) -> TCResult<AxisBounds> {
    debug!("cast range from {} with dimension {}", range, dim);

    let start = match range.start {
        Bound::Un => 0,
        Bound::In(start) => cast_bound(dim, start)?,
        Bound::Ex(start) => cast_bound(dim, start)? + 1,
    };

    let end = match range.end {
        Bound::Un => dim,
        Bound::In(end) => cast_bound(dim, end)? + 1,
        Bound::Ex(end) => cast_bound(dim, end)?,
    };

    if end > start {
        Ok(AxisBounds::In(start..end))
    } else {
        Err(TCError::bad_request(
            "invalid range",
            Tuple::from(vec![start, end]),
        ))
    }
}

pub fn cast_bounds(shape: &Shape, value: Value) -> TCResult<Bounds> {
    debug!("tensor bounds from {} (shape is {})", value, shape);

    match value {
        Value::None => Ok(Bounds::all(shape)),
        Value::Number(i) => {
            let bound = cast_bound(shape[0], i.into())?;
            Ok(Bounds::from(vec![bound]))
        }
        Value::Tuple(range) if range.matches::<(Bound, Bound)>() => {
            if shape.is_empty() {
                return Err(TCError::bad_request(
                    "empty Tensor has no valid bounds, but found",
                    range,
                ));
            }

            let range = range.opt_cast_into().unwrap();
            Ok(Bounds::from(vec![cast_range(shape[0], range)?]))
        }
        Value::Tuple(bounds) => {
            if bounds.len() > shape.len() {
                return Err(TCError::unsupported(format!(
                    "tensor of shape {} does not support bounds with {} axes",
                    shape,
                    bounds.len()
                )));
            }

            let mut axes = Vec::with_capacity(shape.len());

            for (axis, bound) in bounds.into_inner().into_iter().enumerate() {
                debug!(
                    "bound for axis {} with dimension {} is {}",
                    axis, shape[axis], bound
                );

                let bound = if bound.is_none() {
                    AxisBounds::all(shape[axis])
                } else if bound.matches::<Range>() {
                    let range = Range::opt_cast_from(bound).unwrap();
                    cast_range(shape[axis], range)?
                } else if bound.matches::<Vec<u64>>() {
                    bound.opt_cast_into().map(AxisBounds::Of).unwrap()
                } else if let Value::Number(value) = bound {
                    cast_bound(shape[axis], value.into()).map(AxisBounds::At)?
                } else {
                    return Err(TCError::bad_request(
                        format!("invalid bound for axis {}", axis),
                        bound,
                    ));
                };

                axes.push(bound);
            }

            Ok(Bounds { axes })
        }
        other => Err(TCError::bad_request("invalid tensor bounds", other)),
    }
}
