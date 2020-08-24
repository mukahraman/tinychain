use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::ser::Serialize;

use crate::class::{State, TCResult};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::Value;

pub mod btree;
pub mod graph;
pub mod schema;
pub mod table;
pub mod tensor;

pub type BTree = btree::BTree;
pub type BTreeSlice = btree::BTreeSlice;
pub type Graph = graph::Graph;
pub type Table = table::Table;
pub type Tensor = tensor::Tensor;

#[async_trait]
pub trait Collect: Transact + Send + Sync {
    type Selector: Clone + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

    type Item: Clone
        + DeserializeOwned
        + Serialize
        + TryFrom<Value, Error = error::TCError>
        + Send
        + Sync
        + 'static;

    type Slice: Into<State>;

    async fn get(self: Arc<Self>, txn: Arc<Txn>, selector: Self::Selector)
        -> TCResult<Self::Slice>;

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        value: Self::Item,
    ) -> TCResult<()>;
}

pub enum CollectionType {
    BTree,
    Graph,
    Table,
    Tensor,
}

#[derive(Clone)]
pub enum Collection {
    BTree(btree::BTree),
    Graph(graph::Graph),
    Table(table::Table),
    Tensor(tensor::Tensor),
}

impl Collection {
    pub async fn get(&self, _txn: Arc<Txn>, _selector: Value) -> TCResult<State> {
        Err(error::not_implemented())
    }

    pub async fn put(&self, _txn: &Arc<Txn>, _selector: &Value, _state: State) -> TCResult<Self> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Transact for Collection {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Graph(graph) => graph.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Graph(graph) => graph.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }
}

impl From<btree::BTree> for Collection {
    fn from(b: btree::BTree) -> Collection {
        Self::BTree(b)
    }
}

impl From<graph::Graph> for Collection {
    fn from(g: graph::Graph) -> Collection {
        Self::Graph(g)
    }
}

impl From<table::Table> for Collection {
    fn from(t: table::Table) -> Collection {
        Self::Table(t)
    }
}

impl From<tensor::Tensor> for Collection {
    fn from(t: tensor::Tensor) -> Collection {
        Self::Tensor(t)
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(_) => write!(f, "(B-tree)"),
            Self::Graph(_) => write!(f, "(graph)"),
            Self::Table(_) => write!(f, "(table)"),
            Self::Tensor(_) => write!(f, "(tensor)"),
        }
    }
}
