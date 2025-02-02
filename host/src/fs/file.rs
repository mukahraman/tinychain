//! A transactional file

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::*;
use futures::future::{join_all, try_join_all, FutureExt, TryFutureExt};
use futures::try_join;
use log::debug;
use safecast::AsType;
use tokio::sync::{RwLock, RwLockWriteGuard};
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use tc_transact::{Transact, TxnId};

use super::{io_err, CacheBlock, VERSION};

type Blocks = HashMap<fs::BlockId, TxnLock<TxnId>>;

/// A transactional file
pub struct File<B> {
    canon: DirLock<CacheBlock>,

    // don't try to keep the block contents in a TxnLock, since the block versions
    // may need to be backed up to disk--just keep a lock on the TxnId of the last mutation
    // and guard the block content by acquiring a lock on the mutation ID
    // before allowing access to a block
    blocks: Arc<RwLock<Blocks>>,

    present: TxnLock<HashSet<fs::BlockId>>,
    versions: DirLock<CacheBlock>,
    phantom: PhantomData<B>,
}

impl<B> Clone for File<B> {
    fn clone(&self) -> Self {
        Self {
            canon: self.canon.clone(),
            blocks: self.blocks.clone(),
            present: self.present.clone(),
            versions: self.versions.clone(),
            phantom: PhantomData,
        }
    }
}

impl<B: fs::BlockData> File<B>
where
    CacheBlock: AsType<B>,
{
    pub async fn new(canon: DirLock<CacheBlock>) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;
        if fs_dir.len() > 0 {
            return Err(TCError::internal("new file is not empty"));
        }

        Ok(Self {
            canon,
            blocks: Arc::new(RwLock::new(HashMap::new())),
            present: TxnLock::new(format!("block listing for {:?}", &*fs_dir), HashSet::new()),
            versions: fs_dir.create_dir(VERSION.to_string()).map_err(io_err)?,
            phantom: PhantomData,
        })
    }

    pub(super) async fn load(canon: DirLock<CacheBlock>, txn_id: TxnId) -> TCResult<Self> {
        let mut fs_dir = canon.write().await;
        let versions = fs_dir
            .get_or_create_dir(VERSION.to_string())
            .map_err(io_err)?;

        let mut blocks = HashMap::new();
        let mut present = HashSet::new();
        let mut version = versions
            .write()
            .await
            .create_dir(txn_id.to_string())
            .map_err(io_err)?
            .write()
            .await;

        for (name, block) in fs_dir.iter() {
            if name.starts_with('.') {
                debug!("File::load skipping hidden filesystem entry");
                continue;
            }

            if name.len() < B::ext().len() + 1 || !name.ends_with(B::ext()) {
                return Err(TCError::internal(format!(
                    "block has invalid extension: {}",
                    name
                )));
            }

            let (size_hint, contents) = match block {
                DirEntry::File(block) => {
                    let size_hint = block.size_hint().await;
                    let contents = block
                        .read()
                        .map_ok(|contents| B::clone(&*contents))
                        .map_err(io_err)
                        .await?;

                    (size_hint, contents)
                }
                DirEntry::Dir(_) => {
                    return Err(TCError::internal(format!(
                        "expected block file but found directory: {}",
                        name
                    )))
                }
            };

            let block_id: fs::BlockId = name[..(name.len() - B::ext().len() - 1)].parse()?;

            present.insert(block_id.clone());

            let lock_name = format!("block {}", block_id);
            blocks.insert(block_id, TxnLock::new(lock_name, txn_id));

            version
                .create_file(name.clone(), contents, size_hint)
                .map_err(io_err)?;
        }

        Ok(Self {
            canon,
            blocks: Arc::new(RwLock::new(blocks)),
            present: TxnLock::new(format!("block listing for {:?}", &*fs_dir), present),
            versions,
            phantom: PhantomData,
        })
    }

    async fn block_read(
        &self,
        txn_id: TxnId,
        block_id: &fs::BlockId,
    ) -> TCResult<TxnLockReadGuard<TxnId>> {
        let present = self.present.read(txn_id).await?;
        let blocks = self.blocks.read().await;
        if !present.contains(block_id) {
            return Err(TCError::not_found(block_id));
        }

        let block = blocks
            .get(block_id)
            .cloned()
            .expect("block last mutation ID");

        block.read(txn_id).await
    }

    async fn block_write(
        &self,
        txn_id: TxnId,
        block_id: &fs::BlockId,
    ) -> TCResult<TxnLockWriteGuard<TxnId>> {
        let present = self.present.read(txn_id).await?;
        let blocks = self.blocks.read().await;
        if !present.contains(block_id) {
            return Err(TCError::not_found(block_id));
        }

        let block = blocks
            .get(block_id)
            .cloned()
            .expect("block last mutation ID");

        block.write(txn_id).await
    }

    async fn read_block_inner(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
        last_mutation: &TxnId,
    ) -> TCResult<FileLock<CacheBlock>> {
        debug!("File::read_block_inner {} at {}", block_id, txn_id);

        let name = Self::file_name(&block_id);
        let mut version_cache = self.version(&txn_id).await?;
        if let Some(block) = version_cache.get_file(&name) {
            debug!("read existing version of block {} at {}", block_id, txn_id);
            return Ok(block);
        }

        assert!(last_mutation < &txn_id);
        debug!("last mutation of block {} was at {}", block_id, txn_id);

        let (size_hint, value) = {
            let last_version = self
                .versions
                .read()
                .await
                .get_dir(&last_mutation.to_string())
                .expect("prior txn version dir")
                .read()
                .await;

            let block_version = last_version.get_file(&name).expect("block prior value");
            let size_hint = block_version.size_hint().await;

            let value = {
                let value = block_version.read().map_err(io_err).await?;
                B::clone(&*value)
            };

            (size_hint, value)
        };

        let block = version_cache
            .create_file(name, value, size_hint)
            .map_err(io_err)?;

        debug!("created new version of block {} at {}", block_id, txn_id);

        Ok(block)
    }

    async fn write_block_inner(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileLock<CacheBlock>> {
        let name = Self::file_name(&block_id);
        let mut version_cache = self.version(&txn_id).await?;
        if let Some(block) = version_cache.get_file(&name) {
            debug!("read existing version of block {} at {}", block_id, txn_id);
            return Ok(block);
        }

        // a write can only happen before a commit
        // therefore the canonical version must be current

        let block_canon = self
            .canon
            .read()
            .await
            .get_file(&name)
            .expect("canonical block");

        let size_hint = block_canon.size_hint().await;
        let value = {
            let value = block_canon.read().map_err(io_err).await?;
            B::clone(&*value)
        };

        let block = version_cache
            .create_file(name, value, size_hint)
            .map_err(io_err)?;

        debug!("created new version of block {} at {}", block_id, txn_id);

        Ok(block)
    }

    async fn version(&self, txn_id: &TxnId) -> TCResult<DirWriteGuard<CacheBlock>> {
        let mut versions = self.versions.write().await;
        let version = versions
            .get_or_create_dir(txn_id.to_string())
            .map_err(io_err)?;

        version.write().map(Ok).await
    }

    fn file_name(block_id: &fs::BlockId) -> String {
        format!("{}.{}", block_id, B::ext())
    }
}

#[async_trait]
impl<B: fs::BlockData> fs::Store for File<B> {
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        self.present
            .read(txn_id)
            .map_ok(|present| present.is_empty())
            .await
    }
}

#[async_trait]
impl<B> fs::File<B> for File<B>
where
    B: fs::BlockData,
    CacheBlock: AsType<B>,
{
    type Read = FileReadGuard<CacheBlock, B>;
    type Write = FileWriteGuard<CacheBlock, B>;

    async fn block_ids(&self, txn_id: TxnId) -> TCResult<HashSet<fs::BlockId>> {
        self.present
            .read(txn_id)
            .map_ok(|present| present.clone())
            .await
    }

    async fn contains_block(&self, txn_id: TxnId, name: &fs::BlockId) -> TCResult<bool> {
        self.present
            .read(txn_id)
            .map_ok(|present| present.contains(name))
            .await
    }

    async fn copy_from(&self, other: &Self, txn_id: TxnId) -> TCResult<()> {
        let (mut this_present, that_present) =
            try_join!(self.present.write(txn_id), other.present.read(txn_id))?;

        let (mut this_version, that_version) =
            try_join!(self.version(&txn_id), other.version(&txn_id))?;

        let mut blocks = self.blocks.write().await;
        let canon = other.canon.read().await;

        for block_id in that_present.iter() {
            let file_name = Self::file_name(block_id);
            let source = if let Some(version) = that_version.get_file(&file_name) {
                version
            } else {
                canon.get_file(&file_name).expect("block version")
            };

            this_present.insert(block_id.clone());

            if let Some(last_mutation) = blocks.get_mut(block_id) {
                *last_mutation.write(txn_id).await? = txn_id;
            } else {
                let lock_name = format!("block {}", block_id);
                blocks.insert(block_id.clone(), TxnLock::new(lock_name, txn_id));
            }

            let block = source.read().map_err(io_err).await?;
            let size_hint = source.size_hint().await;
            this_version.delete(file_name.clone());
            this_version
                .create_file(file_name, block.clone(), size_hint)
                .map_err(io_err)?;
        }

        Ok(())
    }

    async fn create_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<Self::Write> {
        debug!("File::create_block {}", block_id);

        let present = self.present.write(txn_id).await?;
        if present.contains(&block_id) {
            return Err(TCError::bad_request("block already exists", block_id));
        }

        let blocks = self.blocks.write().await;
        let version = self.version(&txn_id).await?;

        let block = create_block_inner(
            present,
            blocks,
            version,
            txn_id,
            block_id,
            initial_value,
            size_hint,
        )
        .await?;

        block.write().map_err(io_err).await
    }

    async fn create_block_unique(
        &self,
        txn_id: TxnId,
        initial_value: B,
        size_hint: usize,
    ) -> TCResult<(fs::BlockId, Self::Write)> {
        debug!("File::create_block_tmp");

        let present = self.present.write(txn_id).await?;
        let block_id = loop {
            let name = Uuid::new_v4().into();
            if !present.contains(&name) {
                break name;
            }
        };

        let blocks = self.blocks.write().await;
        let version = self.version(&txn_id).await?;

        let block = create_block_inner(
            present,
            blocks,
            version,
            txn_id,
            block_id.clone(),
            initial_value,
            size_hint,
        )
        .await?;

        let lock = block.write().map_err(io_err).await?;
        Ok((block_id, lock))
    }

    async fn delete_block(&self, txn_id: TxnId, block_id: fs::BlockId) -> TCResult<()> {
        debug!("File::delete_block {}", block_id);

        let mut present = self.present.write(txn_id).await?;
        let mut blocks = self.blocks.write().await;
        if let Some(last_mutation) = blocks.get_mut(&block_id) {
            *last_mutation.write(txn_id).await? = txn_id;

            // keep the version directory in sync in case create_block is called later
            // with the same block_id
            let mut version = self.version(&txn_id).await?;
            version.delete(Self::file_name(&block_id));
        }

        present.remove(&block_id);
        Ok(())
    }

    async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        debug!("File::read_block {}", block_id);
        let last_mutation = self.block_read(txn_id, &block_id).await?;
        let block = self
            .read_block_inner(txn_id, block_id, &*last_mutation)
            .await?;

        block.read().map_err(io_err).await
    }

    async fn read_block_owned(
        self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileReadGuard<CacheBlock, B>> {
        self.read_block(txn_id, block_id).await
    }

    async fn write_block(
        &self,
        txn_id: TxnId,
        block_id: fs::BlockId,
    ) -> TCResult<FileWriteGuard<CacheBlock, B>> {
        debug!("File::write_block {}", block_id);
        let mut last_mutation = self.block_write(txn_id, &block_id).await?;
        *last_mutation = txn_id;

        let block = self.write_block_inner(txn_id, block_id.clone()).await?;
        block.write().map_err(io_err).await
    }

    async fn truncate(&self, txn_id: TxnId) -> TCResult<()> {
        let mut contents = self.present.write(txn_id).await?;
        let mut version = self.version(&txn_id).await?;
        for block_id in contents.drain() {
            version.delete(Self::file_name(&block_id));
        }

        Ok(())
    }
}

#[async_trait]
impl<B> Transact for File<B>
where
    B: fs::BlockData,
    CacheBlock: AsType<B>,
{
    async fn commit(&self, txn_id: &TxnId) {
        debug!("File::commit");

        let mut blocks = self.blocks.write().await;

        self.present.commit(txn_id).await;
        debug!("File::commit committed block listing");

        let block_commits = blocks
            .values()
            .map(|last_mutation| last_mutation.commit(txn_id));

        join_all(block_commits).await;

        {
            let present = self.present.read(*txn_id).await.expect("file block list");
            let version = self.version(txn_id).await.expect("file block versions");

            let mut canon = self.canon.write().await;
            let mut deleted = Vec::with_capacity(blocks.len());
            let mut synchronize = Vec::with_capacity(present.len());
            for block_id in blocks.keys() {
                let name = Self::file_name(block_id);
                if present.contains(block_id) {
                    if let Some(version) = version.get_file(&name) {
                        let block = version.read().await.expect("block version");
                        let canon = if let Some(canon) = canon.get_file(&name) {
                            *canon.write().await.expect("canonical block") = block.clone();
                            canon
                        } else {
                            let size_hint = version.size_hint().await;
                            canon
                                .create_file(name, block.clone(), size_hint)
                                .expect("new canonical block")
                        };

                        synchronize.push(async move { canon.sync(true).await });
                    } else {
                        debug!("block {} has no version to commit at {}", block_id, txn_id);
                    }
                } else {
                    canon.delete(name);
                    deleted.push(block_id.clone());
                }
            }

            for block_id in deleted.into_iter() {
                assert!(blocks.remove(&block_id).is_some());
            }

            try_join_all(synchronize)
                .await
                .expect("sync block contents to disk");
        }

        try_join!(self.canon.sync(false), self.versions.sync(false))
            .expect("sync file commit to disk");
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let mut versions = self.versions.write().await;
        versions.delete(txn_id.to_string());

        let blocks = self.blocks.read().await;
        join_all(
            blocks
                .values()
                .map(|last_commit_id| last_commit_id.finalize(txn_id)),
        )
        .await;

        self.present.finalize(txn_id).await
    }
}

impl<B: Send + Sync + 'static> fmt::Display for File<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "file of {} blocks", std::any::type_name::<B>())
    }
}

async fn create_block_inner<'a, B: fs::BlockData + 'a>(
    mut present: TxnLockWriteGuard<HashSet<fs::BlockId>>,
    mut blocks: RwLockWriteGuard<'a, Blocks>,
    mut version: DirWriteGuard<CacheBlock>,
    txn_id: TxnId,
    block_id: fs::BlockId,
    value: B,
    size: usize,
) -> TCResult<FileLock<CacheBlock>>
where
    CacheBlock: AsType<B>,
{
    debug_assert!(!blocks.contains_key(&block_id));

    blocks.insert(
        block_id.clone(),
        TxnLock::new(format!("block {}", block_id), txn_id),
    );

    let name = format!("{}.{}", block_id, B::ext());
    let block = version
        .create_file(name, value, Some(size))
        .map_err(io_err)?;

    present.insert(block_id);

    Ok(block.into())
}
