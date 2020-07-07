use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use arrayfire::Array;
use async_trait::async_trait;

use crate::transaction::{Txn, TxnId};

use crate::value::{TCResult, TCStream, TCType};

use super::base::*;

#[async_trait]
pub trait BlockTensorView: TensorView + Slice {
    async fn as_dtype(&self, txn: &Arc<Txn>, dtype: TCType) -> TCResult<BlockTensor>;

    async fn copy(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn sum(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn product(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn add<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: T) -> TCResult<BlockTensor>;

    async fn multiply<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: T)
        -> TCResult<BlockTensor>;

    async fn subtract<T: BlockTensorView>(
        &self,
        txn: &Arc<Txn>,
        other: &T,
    ) -> TCResult<BlockTensor>;

    async fn equals<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn and<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn or<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn xor<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn blocks(
        &self,
        txn_id: &Arc<TxnId>,
        len: usize,
    ) -> TCStream<Array<<Self as TensorView>::DType>>;
}

pub struct BlockTensor {
    shape: Vec<u64>,
    size: u64,
    ndim: usize,
}

pub struct DenseRebase<T: Rebase + 'static> {
    source: T,
}

#[async_trait]
impl<T: Rebase> TensorView for DenseRebase<T> {
    type DType = T::DType;

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.all(txn_id).await
    }

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.any(txn_id).await
    }
}

type DenseBroadcast<T> = DenseRebase<TensorBroadcast<T>>;
type DenseExpansion<T> = DenseRebase<Expansion<T>>;
type DensePermutation<T> = DenseRebase<Permutation<T>>;
type DenseTensorSlice<T> = DenseRebase<TensorSlice<T>>;

#[async_trait]
impl<T: TensorView + Slice> Slice for DensePermutation<T>
where
    <T as Slice>::Slice: Slice + Transpose,
{
    type Slice = <<T as Slice>::Slice as Transpose>::Permutation;

    async fn slice(&self, txn: &Arc<Txn>, coord: Index) -> TCResult<Self::Slice> {
        let mut permutation: BTreeMap<usize, usize> = self
            .source
            .permutation()
            .to_vec()
            .into_iter()
            .enumerate()
            .collect();

        let mut elided = HashSet::new();
        for axis in 0..coord.len() {
            if let AxisIndex::At(_) = coord[axis] {
                elided.insert(axis);
                permutation.remove(&axis);
            }
        }

        for axis in elided {
            permutation = permutation
                .into_iter()
                .map(|(s, d)| if d > axis { (s, d - 1) } else { (s, d) })
                .collect();
        }

        let permutation: Vec<usize> = permutation.values().cloned().collect();
        self.source
            .source()
            .slice(txn, self.source.invert_coord(coord))
            .await?
            .transpose(txn, Some(permutation))
            .await
    }
}
