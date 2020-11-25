use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::iter::{self, FromIterator};
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join_all, try_join_all, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};

use crate::class::{Class, Instance, NativeClass, TCResult, TCStream};
use crate::collection::btree::{self, BTreeFile, BTreeInstance};
use crate::collection::class::*;
use crate::collection::schema::{Column, IndexSchema, Row, TableSchema};
use crate::collection::{Collection, CollectionBase};
use crate::error;
use crate::scalar::{label, Id, Link, PathSegment, Scalar, TCPathBuf, TryCastInto, Value};
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{self, Bounds, ColumnBound};
use super::view::{IndexSlice, MergeSource, Merged, TableSlice};
use super::{Table, TableImpl, TableInstance, TableType};

const PRIMARY_INDEX: &str = "primary";

#[derive(Clone, Eq, PartialEq)]
pub enum TableBaseType {
    Index,
    ReadOnly,
    Table,
}

impl Class for TableBaseType {
    type Instance = TableBase;
}

impl NativeClass for TableBaseType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Ok(TableBaseType::Table)
        } else if suffix.len() == 1 && suffix[0].as_str() == "index" {
            Ok(TableBaseType::Index)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        CollectionType::prefix().append(label("table"))
    }
}

#[async_trait]
impl CollectionClass for TableBaseType {
    type Instance = TableBase;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<TableBase> {
        let schema =
            schema.try_cast_into(|v| error::bad_request("Expected TableSchema but found", v))?;

        TableIndex::create(txn, schema)
            .map_ok(TableBase::from)
            .await
    }
}

impl From<TableBaseType> for CollectionType {
    fn from(tbt: TableBaseType) -> CollectionType {
        CollectionType::Base(CollectionBaseType::Table(tbt))
    }
}

impl From<TableBaseType> for Link {
    fn from(tbt: TableBaseType) -> Link {
        let prefix = TableType::prefix();

        use TableBaseType::*;
        match tbt {
            Index => prefix.append(label("index")).into(),
            ReadOnly => prefix.append(label("ro_index")).into(),
            Table => prefix.into(),
        }
    }
}

impl fmt::Display for TableBaseType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Index => write!(f, "Index"),
            Self::ReadOnly => write!(f, "Index (read-only)"),
            Self::Table => write!(f, "Table"),
        }
    }
}

#[derive(Clone)]
pub enum TableBase {
    Index(TableImpl<Index>),
    ROIndex(TableImpl<ReadOnly>),
    Table(TableImpl<TableIndex>),
}

impl Instance for TableBase {
    type Class = TableBaseType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Index(_) => TableBaseType::Index,
            Self::ROIndex(_) => TableBaseType::ReadOnly,
            Self::Table(_) => TableBaseType::Table,
        }
    }
}

#[async_trait]
impl TableInstance for TableBase {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Index(index) => index.count(txn_id).await,
            Self::ROIndex(index) => index.count(txn_id).await,
            Self::Table(table) => table.count(txn_id).await,
        }
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::Index(index) => index.into_inner().clone().delete(txn_id).await,
            Self::ROIndex(index) => index.into_inner().clone().delete(txn_id).await,
            Self::Table(table) => table.into_inner().clone().delete(txn_id).await,
        }
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.delete_row(txn_id, row).await,
            Self::ROIndex(index) => index.delete_row(txn_id, row).await,
            Self::Table(table) => table.delete_row(txn_id, row).await,
        }
    }

    async fn insert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        match self {
            Self::Index(index) => TableInstance::insert(index.deref(), txn_id, key, values).await,
            Self::ROIndex(index) => TableInstance::insert(index.deref(), txn_id, key, values).await,
            Self::Table(table) => TableInstance::insert(table.deref(), txn_id, key, values).await,
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match self {
            Self::Index(index) => index.key(),
            Self::ROIndex(index) => index.key(),
            Self::Table(table) => table.key(),
        }
    }

    fn values(&'_ self) -> &'_ [Column] {
        match self {
            Self::Index(index) => index.values(),
            Self::ROIndex(index) => index.values(),
            Self::Table(table) => table.values(),
        }
    }

    fn order_by(&self, columns: Vec<Id>, reverse: bool) -> TCResult<Table> {
        match self {
            Self::Index(index) => index.order_by(columns, reverse),
            Self::ROIndex(index) => index.order_by(columns, reverse),
            Self::Table(table) => table.order_by(columns, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        match self {
            Self::Index(index) => index.reversed(),
            Self::ROIndex(index) => index.reversed(),
            Self::Table(table) => table.reversed(),
        }
    }

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Table> {
        match self {
            Self::Index(index) => index.slice(bounds),
            Self::ROIndex(index) => index.slice(bounds),
            Self::Table(table) => table.slice(bounds),
        }
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        match self {
            Self::Index(index) => index.into_inner().stream(txn_id).await,
            Self::ROIndex(index) => index.into_inner().stream(txn_id).await,
            Self::Table(table) => table.into_inner().stream(txn_id).await,
        }
    }

    fn validate_bounds(&self, bounds: &bounds::Bounds) -> TCResult<()> {
        match self {
            Self::Index(index) => index.validate_bounds(bounds),
            Self::ROIndex(index) => index.validate_bounds(bounds),
            Self::Table(table) => table.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match self {
            Self::Index(index) => index.validate_order(order),
            Self::ROIndex(index) => index.validate_order(order),
            Self::Table(table) => table.validate_order(order),
        }
    }

    async fn update(self, txn: Txn, value: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.into_inner().update(txn, value).await,
            Self::ROIndex(index) => index.into_inner().update(txn, value).await,
            Self::Table(table) => table.into_inner().update(txn, value).await,
        }
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        match self {
            Self::Index(index) => index.update_row(txn_id, row, value).await,
            Self::ROIndex(index) => index.update_row(txn_id, row, value).await,
            Self::Table(table) => table.update_row(txn_id, row, value).await,
        }
    }

    async fn upsert(&self, txn_id: &TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        match self {
            Self::Index(index) => TableInstance::upsert(index.deref(), txn_id, key, values).await,
            Self::ROIndex(index) => TableInstance::upsert(index.deref(), txn_id, key, values).await,
            Self::Table(table) => TableInstance::upsert(table.deref(), txn_id, key, values).await,
        }
    }
}

#[async_trait]
impl Transact for TableBase {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Index(index) => index.commit(txn_id).await,
            Self::ROIndex(_) => (), // no-op
            Self::Table(table) => table.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Index(index) => index.rollback(txn_id).await,
            Self::ROIndex(_) => (), // no-op
            Self::Table(table) => table.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Index(index) => index.finalize(txn_id).await,
            Self::ROIndex(_) => (), // no-op
            Self::Table(table) => table.finalize(txn_id).await,
        }
    }
}

impl From<Index> for TableBase {
    fn from(index: Index) -> Self {
        Self::Index(index.into())
    }
}

impl From<ReadOnly> for TableBase {
    fn from(index: ReadOnly) -> Self {
        Self::ROIndex(index.into())
    }
}

impl From<TableIndex> for TableBase {
    fn from(index: TableIndex) -> Self {
        Self::Table(index.into())
    }
}

impl From<TableBase> for Collection {
    fn from(table: TableBase) -> Collection {
        Collection::Base(CollectionBase::Table(table.into()))
    }
}

#[derive(Clone)]
pub struct Index {
    btree: BTreeFile,
    schema: IndexSchema,
}

impl Index {
    pub async fn create(txn: &Txn, schema: IndexSchema) -> TCResult<Index> {
        let btree = BTreeFile::create(txn, schema.clone().into()).await?;
        Ok(Index { btree, schema })
    }

    pub async fn get(&self, txn_id: TxnId, key: Vec<Value>) -> TCResult<Option<Vec<Value>>> {
        let key = self.schema.validate_key(key)?;
        let mut rows = self.btree.stream(txn_id, key.into(), false).await?;
        Ok(rows.next().await)
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        BTreeInstance::is_empty(&self.btree, txn).await
    }

    pub async fn len(&self, txn_id: TxnId) -> TCResult<u64> {
        self.btree.len(txn_id, btree::BTreeRange::default()).await
    }

    pub fn index_slice(&self, bounds: Bounds) -> TCResult<IndexSlice> {
        let bounds = bounds::validate(bounds, &self.schema().columns())?;
        IndexSlice::new(self.btree.clone(), self.schema().clone(), bounds)
    }

    async fn insert(&self, txn_id: &TxnId, row: Row, reject_extra_columns: bool) -> TCResult<()> {
        let key = self.schema().values_from_row(row, reject_extra_columns)?;
        self.btree.insert(txn_id, key).await
    }

    pub fn schema(&'_ self) -> &'_ IndexSchema {
        &self.schema
    }

    pub fn validate_slice_bounds(&self, outer: Bounds, inner: Bounds) -> TCResult<()> {
        let outer = bounds::validate(outer, &self.schema().columns())?;
        let inner = bounds::validate(inner, &self.schema().columns())?;

        let outer = bounds::btree_range(&outer, &self.schema().columns())?;
        let inner = bounds::btree_range(&inner, &self.schema().columns())?;

        if outer.contains(&inner, &self.schema.columns())? {
            Ok(())
        } else {
            Err(error::bad_request(
                "Slice does not contain requested bounds",
                "",
            ))
        }
    }
}

impl Instance for Index {
    type Class = TableBaseType;

    fn class(&self) -> Self::Class {
        Self::Class::Index
    }
}

#[async_trait]
impl TableInstance for Index {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.len(txn_id).await
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        self.btree
            .delete(&txn_id, btree::BTreeRange::default())
            .await
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        let key = self.schema.values_from_row(row, false)?;
        self.btree.delete(txn_id, key.into()).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn order_by(&self, order: Vec<Id>, reverse: bool) -> TCResult<Table> {
        if self.schema.starts_with(&order) {
            if reverse {
                self.reversed()
            } else {
                Ok(self.clone().into())
            }
        } else {
            let order: Vec<String> = order.iter().map(|id| id.to_string()).collect();
            Err(error::bad_request(
                &format!("Index with schema {} does not support order", self.schema),
                order.join(", "),
            ))
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(IndexSlice::all(self.btree.clone(), self.schema.clone(), true).into())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        self.index_slice(bounds).map(|is| is.into())
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.btree
            .stream(txn_id, btree::BTreeRange::default(), false)
            .await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds = bounds::validate(bounds.clone(), &self.schema().columns())?;

        for (column, bound_column) in self.schema.columns()[0..bounds.len()]
            .iter()
            .zip(bounds.keys())
        {
            if column.name() != bound_column {
                return Err(error::bad_request(
                    &format!(
                        "Expected column {} in index range selector but found",
                        column.name()
                    ),
                    bound_column,
                ));
            }
        }

        Ok(())
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        if !self.schema.starts_with(&order) {
            let order: Vec<String> = order.iter().map(|c| c.to_string()).collect();
            Err(error::bad_request(
                &format!("Cannot order index with schema {} by", self.schema),
                order.join(", "),
            ))
        } else {
            Ok(())
        }
    }

    async fn update(self, txn: Txn, row: Row) -> TCResult<()> {
        let key: btree::Key = self.schema().values_from_row(row, false)?;
        self.btree
            .update(txn.id(), btree::BTreeRange::default(), &key)
            .await
    }
}

impl From<Index> for Table {
    fn from(index: Index) -> Table {
        Table::Base(TableBase::from(index).into())
    }
}

#[async_trait]
impl Transact for Index {
    async fn commit(&self, txn_id: &TxnId) {
        self.btree.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.btree.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.btree.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub struct ReadOnly {
    index: IndexSlice,
}

impl ReadOnly {
    pub async fn copy_from(
        source: Table,
        txn: Txn,
        key_columns: Option<Vec<Id>>,
    ) -> TCResult<ReadOnly> {
        let source_schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let (schema, btree) = if let Some(columns) = key_columns {
            let column_names: HashSet<&Id> = columns.iter().collect();
            let schema = source_schema.subset(column_names)?;
            let btree =
                BTreeFile::create(&txn.subcontext_tmp().await?, schema.clone().into()).await?;

            let rows = source.select(columns)?.stream(txn.id().clone()).await?;
            btree.insert_from(txn.id(), rows).await?;
            (schema, btree)
        } else {
            let btree =
                BTreeFile::create(&txn.subcontext_tmp().await?, source_schema.clone().into())
                    .await?;

            let rows = source.stream(txn.id().clone()).await?;
            btree.insert_from(txn.id(), rows).await?;
            (source_schema, btree)
        };

        let index = Index { schema, btree };

        index
            .index_slice(bounds::all())
            .map(|index| ReadOnly { index })
    }

    pub fn into_reversed(self) -> ReadOnly {
        ReadOnly {
            index: self.index.into_reversed(),
        }
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.index.is_empty(txn).await
    }
}

impl Instance for ReadOnly {
    type Class = TableBaseType;

    fn class(&self) -> Self::Class {
        Self::Class::ReadOnly
    }
}

#[async_trait]
impl TableInstance for ReadOnly {
    type Stream = <Index as TableInstance>::Stream;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.index.clone().count(txn_id).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.index.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.index.values()
    }

    fn order_by(&self, order: Vec<Id>, reverse: bool) -> TCResult<Table> {
        self.index.validate_order(&order)?;

        if reverse {
            self.reversed()
        } else {
            Ok(self.clone().into())
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(self.clone().into_reversed().into())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        self.validate_bounds(&bounds)?;
        self.index
            .slice_index(bounds)
            .map(|index| ReadOnly { index }.into())
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.index.clone().stream(txn_id).await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        self.index.validate_bounds(bounds)
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.index.validate_order(order)
    }
}

impl From<ReadOnly> for Table {
    fn from(index: ReadOnly) -> Table {
        Table::Base(TableBase::from(index).into())
    }
}

#[derive(Clone)]
pub struct TableIndex {
    primary: Index,
    auxiliary: BTreeMap<Id, Index>,
}

impl TableIndex {
    pub async fn create(txn: &Txn, schema: TableSchema) -> TCResult<TableIndex> {
        let primary = Index::create(
            &txn.subcontext(PRIMARY_INDEX.parse()?).await?,
            schema.primary().clone(),
        )
        .await?;

        let auxiliary: BTreeMap<Id, Index> =
            try_join_all(schema.indices().iter().map(|(name, column_names)| {
                Self::create_index(txn, schema.primary(), name.clone(), column_names.to_vec())
                    .map_ok(move |index| (name.clone(), index))
            }))
            .await?
            .into_iter()
            .collect();

        Ok(TableIndex { primary, auxiliary })
    }

    async fn create_index(
        txn: &Txn,
        primary: &IndexSchema,
        name: Id,
        key: Vec<Id>,
    ) -> TCResult<Index> {
        if name.as_str() == PRIMARY_INDEX {
            return Err(error::bad_request(
                "This index name is reserved",
                PRIMARY_INDEX,
            ));
        }

        let index_key_set: HashSet<&Id> = key.iter().collect();
        if index_key_set.len() != key.len() {
            return Err(error::bad_request(
                &format!("Duplicate column in index {}", name),
                key.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        let mut columns: HashMap<Id, Column> = primary
            .columns()
            .iter()
            .cloned()
            .map(|c| (c.name().clone(), c))
            .collect();

        let key: Vec<Column> = key
            .iter()
            .map(|c| columns.remove(&c).ok_or_else(|| error::not_found(c)))
            .collect::<TCResult<Vec<Column>>>()?;

        let values: Vec<Column> = primary
            .key()
            .iter()
            .filter(|c| !index_key_set.contains(c.name()))
            .cloned()
            .collect();
        let schema: IndexSchema = (key, values).into();

        let btree =
            btree::BTreeFile::create(&txn.subcontext_tmp().await?, schema.clone().into()).await?;

        Ok(Index { btree, schema })
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.primary.is_empty(txn).await
    }

    pub fn primary(&'_ self) -> &'_ Index {
        &self.primary
    }

    pub fn supporting_index(&self, bounds: &Bounds) -> TCResult<Index> {
        if self.primary.validate_bounds(bounds).is_ok() {
            return Ok(self.primary.clone());
        }

        for index in self.auxiliary.values() {
            if index.validate_bounds(bounds).is_ok() {
                return Ok(index.clone());
            }
        }

        Err(error::bad_request(
            "This table has no index which supports bounds",
            super::bounds::format(bounds),
        ))
    }

    pub async fn get(&self, txn_id: TxnId, key: Vec<Value>) -> TCResult<Option<Vec<Value>>> {
        self.primary.get(txn_id, key).await
    }

    pub async fn get_owned(self, txn_id: TxnId, key: Vec<Value>) -> TCResult<Option<Vec<Value>>> {
        self.get(txn_id, key).await
    }

    pub async fn insert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        if self.get(txn_id.clone(), key.to_vec()).await?.is_some() {
            let key: Vec<String> = key.iter().map(|v| v.to_string()).collect();
            Err(error::bad_request(
                "Tried to insert but this key already exists",
                format!("[{}]", key.join(", ")),
            ))
        } else {
            self.upsert(&txn_id, key, values).await
        }
    }

    pub async fn upsert(
        &self,
        txn_id: &TxnId,
        key: Vec<Value>,
        values: Vec<Value>,
    ) -> TCResult<()> {
        if let Some(row) = self.get(txn_id.clone(), key.to_vec()).await? {
            let row = self.primary.schema.row_from_values(row)?;
            self.delete_row(txn_id, row.clone()).await?;
        }

        let row = self.primary.schema().row_from_key_values(key, values)?;

        let mut inserts = Vec::with_capacity(self.auxiliary.len() + 1);
        inserts.push(self.primary.insert(txn_id, row.clone(), true));
        for index in self.auxiliary.values() {
            inserts.push(index.insert(txn_id, row.clone(), false));
        }

        try_join_all(inserts).await?;
        Ok(())
    }
}

impl Instance for TableIndex {
    type Class = TableBaseType;

    fn class(&self) -> Self::Class {
        Self::Class::Table
    }
}

#[async_trait]
impl TableInstance for TableIndex {
    type Stream = <Index as TableInstance>::Stream;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.primary.count(txn_id).await
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
        deletes.push(self.primary.delete(txn_id.clone()));
        for index in self.auxiliary.values() {
            deletes.push(index.clone().delete(txn_id.clone()));
        }

        try_join_all(deletes).await?;
        Ok(())
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        let row = self.primary.schema().validate_row(row)?;

        let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
        for index in self.auxiliary.values() {
            deletes.push(index.delete_row(txn_id, row.clone()));
        }
        deletes.push(self.primary.delete_row(txn_id, row));
        try_join_all(deletes).await?;

        Ok(())
    }

    async fn insert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        TableIndex::insert(self, txn_id, key, values).await
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.primary.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.primary.values()
    }

    fn order_by(&self, columns: Vec<Id>, reverse: bool) -> TCResult<Table> {
        self.validate_order(&columns)?;

        if self.primary.validate_order(&columns).is_ok() {
            let ordered = TableSlice::new(self.clone(), bounds::all())?;
            return if reverse {
                ordered.reversed()
            } else {
                Ok(ordered.into())
            };
        }

        let selection = TableSlice::new(self.clone(), bounds::all())?;
        let mut merge_source = MergeSource::Table(selection);

        let mut columns = &columns[..];
        loop {
            let initial = columns.to_vec();

            let mut i = columns.len();
            while i > 0 {
                let subset = &columns[..i];

                for index in iter::once(&self.primary).chain(self.auxiliary.values()) {
                    if index.validate_order(subset).is_ok() {
                        columns = &columns[i..];

                        let index_slice = index.index_slice(bounds::all())?;
                        let merged = Merged::new(merge_source, index_slice);

                        if columns.is_empty() {
                            return if reverse {
                                merged.reversed()
                            } else {
                                Ok(merged.into())
                            };
                        }

                        merge_source = MergeSource::Merge(Arc::new(merged));
                        break;
                    }
                }

                i = i - 1;
            }

            if columns == &initial[..] {
                return Err(error::bad_request(
                    "This table has no index to support the order",
                    Value::from_iter(columns.to_vec()),
                ));
            }
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Err(error::unsupported(
            "Cannot reverse a Table itself, consider reversing a slice of the table instead",
        ))
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        if self.primary.validate_bounds(&bounds).is_ok() {
            return TableSlice::new(self.clone(), bounds).map(|t| t.into());
        }

        let columns: Vec<Id> = self
            .primary
            .schema()
            .columns()
            .iter()
            .map(|c| c.name())
            .cloned()
            .collect();

        let bounds: Vec<(Id, ColumnBound)> = columns
            .into_iter()
            .filter_map(|name| bounds.get(&name).map(|bound| (name, bound.clone())))
            .collect();

        let selection = TableSlice::new(self.clone(), bounds::all())?;
        let mut merge_source = MergeSource::Table(selection);

        let mut bounds = &bounds[..];
        loop {
            let initial = bounds.len();
            let mut i = bounds.len();
            while i > 0 {
                let subset: Bounds = bounds[..i].iter().cloned().collect();

                for index in iter::once(&self.primary).chain(self.auxiliary.values()) {
                    if index.validate_bounds(&subset).is_ok() {
                        bounds = &bounds[i..];

                        let index_slice = index.index_slice(subset)?;
                        let merged = Merged::new(merge_source, index_slice);

                        if bounds.is_empty() {
                            return Ok(merged.into());
                        }

                        merge_source = MergeSource::Merge(Arc::new(merged));
                        break;
                    }
                }

                i = i - 1;
            }

            if bounds.len() == initial {
                return Err(error::bad_request(
                    "This table has no index to support selection bounds on",
                    Scalar::from_iter(bounds.to_vec()),
                ));
            }
        }
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.primary.stream(txn_id).await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds: Vec<(Id, ColumnBound)> = self
            .primary
            .schema()
            .columns()
            .iter()
            .filter_map(|c| {
                bounds
                    .get(c.name())
                    .map(|bound| (c.name().clone(), bound.clone()))
            })
            .collect();

        let mut bounds = &bounds[..];
        while !bounds.is_empty() {
            let initial = bounds.len();

            let mut i = bounds.len();
            loop {
                let subset: Bounds = bounds[..i].iter().cloned().collect();

                if self.primary.validate_bounds(&subset).is_ok() {
                    bounds = &bounds[i..];
                    break;
                }

                for index in self.auxiliary.values() {
                    if index.validate_bounds(&subset).is_ok() {
                        bounds = &bounds[i..];
                        break;
                    }
                }

                if bounds.is_empty() {
                    break;
                } else {
                    i = i - 1;
                }
            }

            if bounds.len() == initial {
                let order: Vec<String> = bounds.iter().map(|(name, _)| name.to_string()).collect();
                return Err(error::bad_request(
                    format!("This table has no index to support selection bounds on {}--available indices are", order.join(", ")),
                    Value::from_iter(self.auxiliary.keys().cloned()),
                ));
            }
        }

        Ok(())
    }

    fn validate_order(&self, mut order: &[Id]) -> TCResult<()> {
        while !order.is_empty() {
            let initial = order.to_vec();
            let mut i = order.len();
            loop {
                let subset = &order[..i];

                if self.primary.validate_order(subset).is_ok() {
                    order = &order[i..];
                    break;
                }

                for index in self.auxiliary.values() {
                    if index.validate_order(subset).is_ok() {
                        order = &order[i..];
                        break;
                    }
                }

                if order.is_empty() {
                    break;
                } else {
                    i = i - 1;
                }
            }

            if order == &initial[..] {
                let order: Vec<String> = order.iter().map(|id| id.to_string()).collect();
                return Err(error::bad_request(
                    "This table has no index to support the order",
                    order.join(", "),
                ));
            }
        }

        Ok(())
    }

    async fn update(self, txn: Txn, update: Row) -> TCResult<()> {
        for col in self.primary.schema().key() {
            if update.contains_key(col.name()) {
                return Err(error::bad_request(
                    "Cannot update the value of a primary key column",
                    col.name(),
                ));
            }
        }

        let schema = self.primary.schema();
        let update = schema.validate_row_partial(update)?;

        let index = self.clone().index(txn.clone(), None).await?;

        let txn_id = txn.id();
        index
            .stream(txn_id.clone())
            .await?
            .map(|values| schema.row_from_values(values))
            .map_ok(|row| self.update_row(txn_id.clone(), row, update.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, update: Row) -> TCResult<()> {
        let mut updated_row = row.clone();
        updated_row.extend(update);
        let (key, values) = self.primary.schema.key_values_from_row(updated_row)?;
        let txn_id_clone = txn_id.clone();
        self.delete_row(&txn_id, row)
            .and_then(|()| self.insert(txn_id_clone, key, values))
            .await
    }

    async fn upsert(&self, txn_id: &TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        TableIndex::upsert(self, txn_id, key, values).await
    }
}

impl From<TableIndex> for Table {
    fn from(index: TableIndex) -> Table {
        Table::Base(TableBase::from(index).into())
    }
}

#[async_trait]
impl Transact for TableIndex {
    async fn commit(&self, txn_id: &TxnId) {
        let mut commits = Vec::with_capacity(self.auxiliary.len() + 1);
        commits.push(self.primary.commit(txn_id));
        for index in self.auxiliary.values() {
            commits.push(index.commit(txn_id));
        }

        join_all(commits).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let mut rollbacks = Vec::with_capacity(self.auxiliary.len() + 1);
        rollbacks.push(self.primary.rollback(txn_id));
        for index in self.auxiliary.values() {
            rollbacks.push(index.rollback(txn_id));
        }

        join_all(rollbacks).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let mut cleanups = Vec::with_capacity(self.auxiliary.len() + 1);
        cleanups.push(self.primary.finalize(txn_id));
        for index in self.auxiliary.values() {
            cleanups.push(index.finalize(txn_id));
        }

        join_all(cleanups).await;
    }
}
