pub(crate) mod index;
pub(crate) mod map;
pub(crate) mod material;
pub(crate) mod residency;

use crate::{
    BrickHubCell, CellID, CellKey, GenerationToken, MaterialID,
    brick::Brick,
    cell::{CellHandle, CellSlot},
};
use residency::{ResidencyDecision, ResidencyManager, classify_ring};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub(crate) enum StorageType {
    Brick,
    Material,
    #[default]
    Indices,
}

#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub(crate) enum DeltaState {
    #[default]
    Inactive,
    Active,
}

#[derive(Default)]
pub(crate) struct CellDeltaRange {
    range: sns_helper::prelude::Range<usize>,
    state: DeltaState,
}

impl CellDeltaRange {
    pub(crate) fn register(&mut self, index: usize) {
        self.register_range(index, index + 1);
    }

    pub(crate) fn register_range(&mut self, from: usize, to: usize) {
        if self.state == DeltaState::Active {
            self.range.set_start(self.range.start().min(from));
            self.range.set_end(self.range.end().max(to));
        } else {
            self.range.set_start(from);
            self.range.set_end(to);
            self.state = DeltaState::Active;
        }
    }

    pub(crate) fn take(&mut self) -> Option<(usize, usize)> {
        if self.state == DeltaState::Active {
            let range = (self.range.start(), self.range.end());
            *self = Self::default();
            Some(range)
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(crate) fn is_active(&self) -> bool {
        self.state == DeltaState::Active
    }
}

#[derive(Default)]
pub(crate) struct CellDelta {
    dirty_brick: CellDeltaRange,
    dirty_material: CellDeltaRange,
    dirty_indices: CellDeltaRange,
}

impl CellDelta {
    pub(crate) fn register(&mut self, storage: StorageType, index: usize) {
        self.range_mut(storage).register(index);
    }

    pub(crate) fn register_range(&mut self, storage: StorageType, from: usize, to: usize) {
        self.range_mut(storage).register_range(from, to);
    }

    pub(crate) fn take(&mut self, storage: StorageType) -> Option<(usize, usize)> {
        self.range_mut(storage).take()
    }

    #[cfg(test)]
    pub(crate) fn is_active(&self, storage: StorageType) -> bool {
        match storage {
            StorageType::Brick => self.dirty_brick.is_active(),
            StorageType::Material => self.dirty_material.is_active(),
            StorageType::Indices => self.dirty_indices.is_active(),
        }
    }

    fn range_mut(&mut self, storage: StorageType) -> &mut CellDeltaRange {
        match storage {
            StorageType::Brick => &mut self.dirty_brick,
            StorageType::Material => &mut self.dirty_material,
            StorageType::Indices => &mut self.dirty_indices,
        }
    }
}

pub(crate) enum UploadData {
    Bricks(Vec<Brick>),
    Materials(Vec<MaterialID>),
    Indices(Vec<u32>),
}

pub(crate) struct UploadRange {
    pub(crate) storage: StorageType,
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) destination: usize,
    pub(crate) data: UploadData,
}

pub(crate) struct CellUploadSnapshot {
    pub(crate) id: CellID,
    pub(crate) key: CellKey,
    pub(crate) ranges: Vec<UploadRange>,
}

/// Owns CPU cell state, generation tickets, and logical residency.
#[derive(Default)]
pub(crate) struct CellStore {
    cells: HashMap<CellID, Arc<Mutex<CellSlot>>>,
    residency: ResidencyManager,
    generation_epochs: HashMap<CellID, u64>,
    next_generation_epoch: u64,
    next_cell_generation: u64,
    pending_evictions: HashMap<CellKey, u8>,
}

impl CellStore {
    pub(crate) fn classify_ring(&self, id: CellID, view: CellID) -> Option<u8> {
        classify_ring(id, view)
    }

    pub(crate) fn contains(&self, id: CellID) -> bool {
        self.cells.contains_key(&id)
    }

    pub(crate) fn cell_key(&self, id: CellID) -> Option<CellKey> {
        let slot = self.cells.get(&id)?;
        let state = slot.lock().ok()?;
        state.get_resident().then(|| state.key())
    }

    pub(crate) fn validate_gpu_uses(&self, keys: &[CellKey]) -> Result<(), crate::BrickHubError> {
        let mut first_error = None;
        for &key in keys {
            let Some(slot) = self.cells.get(&key.id) else {
                first_error.get_or_insert(crate::BrickHubError::MissingCell(key.id));
                continue;
            };
            let state = match slot.lock() {
                Ok(state) => state,
                Err(_) => {
                    first_error.get_or_insert(crate::BrickHubError::CellPoisoned(key.id));
                    continue;
                }
            };
            if state.key() != key {
                first_error.get_or_insert(crate::BrickHubError::MissingCell(key.id));
            } else if !state.get_resident() {
                first_error.get_or_insert(crate::BrickHubError::CellEvicting(key.id));
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub(crate) fn cell_handle(&self, id: CellID) -> Option<CellHandle> {
        let slot = self.cells.get(&id)?.clone();
        let state = slot.lock().ok()?;
        if !state.get_resident() {
            return None;
        }
        let key = state.key();
        drop(state);
        Some(CellHandle::new(key, slot))
    }

    pub(crate) fn touch(&mut self, id: CellID, ring: u8, frame: u64) {
        self.residency.touch(id, ring, frame);
    }

    pub(crate) fn admit(&mut self, id: CellID, ring: u8, frame: u64) -> ResidencyDecision {
        self.residency.admit(id, ring, frame)
    }

    pub(crate) fn restore_admission(&mut self, id: CellID, ring: u8, evicted: CellID, frame: u64) {
        self.residency.remove(id, ring);
        self.residency.touch(evicted, ring, frame);
    }

    pub(crate) fn begin_generation(&mut self, id: CellID, view: CellID) -> Option<GenerationToken> {
        self.classify_ring(id, view)?;
        if self.cells.contains_key(&id) {
            return None;
        }
        self.next_generation_epoch = self
            .next_generation_epoch
            .checked_add(1)
            .expect("generation epoch exhausted");
        let token = GenerationToken::new(id, self.next_generation_epoch);
        self.generation_epochs.insert(id, token.epoch());
        Some(token)
    }

    pub(crate) fn cancel_generation(&mut self, token: GenerationToken) -> bool {
        if self.generation_epochs.get(&token.id()) == Some(&token.epoch()) {
            self.generation_epochs.remove(&token.id());
            true
        } else {
            false
        }
    }

    pub(crate) fn generation_is_current(&self, token: &GenerationToken) -> bool {
        self.generation_epochs.get(&token.id()) == Some(&token.epoch())
    }

    pub(crate) fn finish_generation(&mut self, id: CellID) {
        self.generation_epochs.remove(&id);
    }

    pub(crate) fn has_pending_eviction(&self, id: CellID) -> bool {
        self.pending_evictions.keys().any(|key| key.id == id)
    }

    pub(crate) fn pending_eviction_keys(&self) -> Vec<CellKey> {
        self.pending_evictions.keys().copied().collect()
    }

    pub(crate) fn has_pending_eviction_key(&self, key: CellKey) -> bool {
        self.pending_evictions.contains_key(&key)
    }

    pub(crate) fn complete_eviction(&mut self, key: CellKey) {
        if self
            .cells
            .get(&key.id)
            .is_some_and(|slot| slot.lock().is_ok_and(|slot| slot.key() == key))
        {
            self.cells.remove(&key.id);
        }
        self.pending_evictions.remove(&key);
    }

    pub(crate) fn begin_evict(
        &mut self,
        id: CellID,
        ring: u8,
    ) -> Result<CellKey, crate::BrickHubError> {
        self.generation_epochs.remove(&id);
        let Some(slot) = self.cells.get(&id) else {
            return Err(crate::BrickHubError::MissingCell(id));
        };
        let key = match slot.lock() {
            Ok(mut slot) => {
                slot.begin_eviction();
                slot.key()
            }
            Err(poisoned) => {
                let mut slot = poisoned.into_inner();
                slot.begin_eviction();
                slot.key()
            }
        };
        self.residency.remove(id, ring);
        self.pending_evictions.insert(key, ring);
        Ok(key)
    }

    pub(crate) fn insert_cell(&mut self, id: CellID, cell: BrickHubCell) -> CellKey {
        self.next_cell_generation = self
            .next_cell_generation
            .checked_add(1)
            .expect("cell generation exhausted");
        let key = CellKey {
            id,
            generation: self.next_cell_generation,
        };
        self.cells
            .insert(id, Arc::new(Mutex::new(CellSlot::new(key, true, cell))));
        key
    }

    pub(crate) fn snapshot_uploads(&self) -> Result<Vec<CellUploadSnapshot>, crate::BrickHubError> {
        let mut snapshots = Vec::new();
        for (&id, slot) in &self.cells {
            let mut state = match slot.lock() {
                Ok(state) => state,
                Err(poisoned) => {
                    poisoned.into_inner().begin_eviction();
                    self.restore_snapshot_ranges(&snapshots)?;
                    return Err(crate::BrickHubError::CellPoisoned(id));
                }
            };
            if !state.get_resident() {
                continue;
            }
            let mut ranges = Vec::with_capacity(3);
            for storage in [
                StorageType::Brick,
                StorageType::Material,
                StorageType::Indices,
            ] {
                let Some((from, to)) = state.cell().delta().take(storage) else {
                    continue;
                };
                let range = sns_helper::prelude::Range::new(from, to);
                let data = match storage {
                    StorageType::Brick => {
                        UploadData::Bricks(state.cell().map().get_brick(range).to_vec())
                    }
                    StorageType::Material => {
                        UploadData::Materials(state.cell().map().get_material(range).to_vec())
                    }
                    StorageType::Indices => {
                        UploadData::Indices(state.cell().map().get_indices(range).to_vec())
                    }
                };
                ranges.push(UploadRange {
                    storage,
                    from,
                    to,
                    destination: 0,
                    data,
                });
            }
            snapshots.push(CellUploadSnapshot {
                id,
                key: state.key(),
                ranges,
            });
        }
        Ok(snapshots)
    }

    pub(crate) fn restore_snapshot_ranges(
        &self,
        snapshots: &[CellUploadSnapshot],
    ) -> Result<(), crate::BrickHubError> {
        for snapshot in snapshots {
            let Some(slot) = self.cells.get(&snapshot.id) else {
                continue;
            };
            let mut state = match slot.lock() {
                Ok(state) => state,
                Err(poisoned) => {
                    let mut state = poisoned.into_inner();
                    state.begin_eviction();
                    return Err(crate::BrickHubError::CellPoisoned(snapshot.id));
                }
            };
            if !state.get_resident() || state.key() != snapshot.key {
                continue;
            }
            for range in &snapshot.ranges {
                state
                    .cell()
                    .delta()
                    .register_range(range.storage, range.from, range.to);
            }
        }
        Ok(())
    }

    pub(crate) fn is_resident_key(&self, key: CellKey) -> bool {
        self.cells.get(&key.id).is_some_and(|slot| {
            slot.lock()
                .is_ok_and(|slot| slot.key() == key && slot.get_resident())
        })
    }
}
