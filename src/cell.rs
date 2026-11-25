use crate::consts::{
    BRICKS_PER_BRICKMAP, BrickID, CELL_SIZE, CellID, DENSITY_CORNERS_PER_AXIS,
    DENSITY_WORDS_PER_CELL, INDEX_WORDS_PER_CELL, MaterialID, VOXELS_PER_BRICK,
};
use crate::cpu::{CellDelta, StorageType, map};
use bevy_math::prelude::*;
use sns_helper::prelude::{Noise3d, NoiseKind3d, NoiseSampler, SampledNoise3d};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct BrickHubCell {
    map: map::BrickMap,
    delta: CellDelta,
}
impl Default for BrickHubCell {
    fn default() -> Self {
        let mut delta = CellDelta::default();
        // Index ranges can be recycled while their old GPU contents remain. Upload the
        // zeroed map (brick grid and fill classifications)
        delta.register_range(StorageType::Indices, 0, INDEX_WORDS_PER_CELL);
        delta.register_range(StorageType::Density, 0, DENSITY_WORDS_PER_CELL);
        Self {
            map: map::BrickMap::default(),
            delta,
        }
    }
}
impl BrickHubCell {
    /// Creates an empty detached cell with a zeroed index grid ready to upload.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Builds a full cell containing a single material using a bulk BrickMap path.
    /// This constructor only mutates the owned cell and is suitable for a generation worker.
    pub fn solid(material: MaterialID) -> Self {
        let mut cell = Self::default();
        cell.map.fill_solid(material);
        cell.delta
            .register_range(StorageType::Brick, 0, BRICKS_PER_BRICKMAP);
        cell.delta.register_range(
            StorageType::Material,
            0,
            BRICKS_PER_BRICKMAP * VOXELS_PER_BRICK,
        );
        cell.delta
            .register_range(StorageType::Indices, 0, INDEX_WORDS_PER_CELL);
        cell.delta
            .register_range(StorageType::Density, 0, DENSITY_WORDS_PER_CELL);
        cell
    }

    /// Builds a detached cell from a local occupancy/material mask. The callback is
    /// called once for every coordinate in `0..CELL_SIZE` on each axis; `None` is empty.
    pub fn from_voxel_sampler(sample: impl FnMut(UVec3) -> Option<MaterialID>) -> Self {
        let mut cell = Self::default();
        let ranges = cell.map.build_from_sampler(sample);
        if ranges.brick_count > 0 {
            cell.delta
                .register_range(StorageType::Brick, 0, ranges.brick_count);
            cell.delta
                .register_range(StorageType::Material, 0, ranges.material_count);
        }
        cell.delta
            .register_range(StorageType::Density, 0, DENSITY_WORDS_PER_CELL);
        cell
    }

    /// Builds a detached cell from a raw-coordinate density sampler (below
    /// `isolevel` is inside; corner coordinates in `0..65` per axis). `isolevel`
    /// and `flag_partial_bricks` are renderer-supplied surface conventions; this
    /// crate only stores their results.
    pub fn from_density(
        &mut self,
        sample: impl FnMut(UVec3) -> f32,
        material: MaterialID,
        isolevel: f32,
        flag_partial_bricks: bool,
    ) {
        let ranges =
            self.map
                .build_from_density(sample, material, isolevel, flag_partial_bricks);
        self.delta
            .register_range(StorageType::Indices, 0, INDEX_WORDS_PER_CELL);
        self.delta
            .register_range(StorageType::Density, 0, DENSITY_WORDS_PER_CELL);
        if ranges.brick_count > 0 {
            self.delta
                .register_range(StorageType::Brick, 0, ranges.brick_count);
            self.delta
                .register_range(StorageType::Material, 0, ranges.material_count);
        }
    }

    /// Builds horizontal material bands with optional empty gaps. Ranges are local,
    /// half-open y intervals and must be non-empty, within `0..CELL_SIZE`, and disjoint.
    pub fn layered(layers: &[(Range<u32>, MaterialID)]) -> Result<Self, CellBuildError> {
        let mut materials_by_y = [None; CELL_SIZE];
        for (index, (range, material)) in layers.iter().enumerate() {
            if range.start >= range.end || range.end > CELL_SIZE as u32 {
                return Err(CellBuildError::InvalidLayerRange);
            }
            for (other_range, _) in &layers[..index] {
                if range.start < other_range.end && other_range.start < range.end {
                    return Err(CellBuildError::OverlappingLayers);
                }
            }
            materials_by_y[range.start as usize..range.end as usize].fill(Some(*material));
        }
        Ok(Self::from_voxel_sampler(|voxel| {
            materials_by_y[voxel.y as usize]
        }))
    }

    pub fn insert(&mut self, voxel: BrickID, material: MaterialID) {
        let events = self.map.insert(voxel, material);
        for event in events {
            self.apply_event(event);
        }
    }
    pub fn remove(&mut self, voxel: BrickID) {
        let events = self.map.remove(voxel);

        for event in events {
            self.apply_event(event);
        }
    }
    fn apply_event(&mut self, event: map::BrickEditEvent) {
        match event {
            map::BrickEditEvent::BrickAllocated { brick_index } => {
                self.delta.register(StorageType::Indices, brick_index);
                self.delta.register(StorageType::Brick, brick_index);
            }

            map::BrickEditEvent::MaterialBlockAllocated { base } => {
                self.delta.register_range(
                    StorageType::Material,
                    base as usize,
                    base as usize + VOXELS_PER_BRICK,
                );
            }

            map::BrickEditEvent::VoxelMaterialChanged { index } => {
                self.delta.register(StorageType::Material, index);
            }

            map::BrickEditEvent::BrickBecameEmpty { brick_index } => {
                self.delta.register(StorageType::Brick, brick_index);
                self.delta.register(StorageType::Indices, brick_index);
            }
        }
    }
    pub fn map(&self) -> &map::BrickMap {
        &self.map
    }
    pub(crate) fn delta(&mut self) -> &mut CellDelta {
        &mut self.delta
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellBuildError {
    InvalidLayerRange,
    OverlappingLayers,
}

pub(super) struct CellSlot {
    key: super::CellKey,
    state: CellState,
    cell: BrickHubCell,
}

impl CellSlot {
    pub fn key(&self) -> super::CellKey {
        self.key
    }
    pub fn begin_eviction(&mut self) {
        self.state = CellState::Evicting;
    }
    pub fn get_resident(&self) -> bool {
        self.state == CellState::Resident
    }
    pub fn cell(&mut self) -> &mut BrickHubCell {
        &mut self.cell
    }
    pub fn cell_ref(&self) -> &BrickHubCell {
        &self.cell
    }

    pub(crate) fn new(key: super::CellKey, resident: bool, cell: BrickHubCell) -> Self {
        Self {
            key,
            state: if resident {
                CellState::Resident
            } else {
                CellState::Evicting
            },
            cell,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum CellState {
    Resident,
    Evicting,
}

/// Shared access to one resident cell. Each edit closure holds the cell lock once,
/// including all map changes and their corresponding delta registrations.
#[derive(Clone)]
pub struct CellHandle {
    key: super::CellKey,
    slot: Arc<Mutex<CellSlot>>,
    dirty: Arc<AtomicBool>,
}

impl CellHandle {
    pub fn id(&self) -> CellID {
        self.key.id
    }

    pub fn key(&self) -> CellKey {
        self.key
    }

    /// Bulk-samples a 3D noise family across this cell's 65^3 density corner
    /// lattice in one backend call, positioned at the cell's integer origin. Feed
    /// the result to `from_density` through `sampled.get_vec3(corner)`. Lattice
    /// points sit `1.0` apart in noise space at `(id + corner) * frequency`, so
    /// the world scale belongs in `NoiseSettings::frequency` and neighboring
    /// cells join seamlessly (the lattice is global and integer-aligned).
    /// `kind` only selects the noise family at the type level (e.g. `Fbm3D`).
    pub fn sample_noise<K, T>(&self, noise: &NoiseSampler<T>, kind: K, seed: i32) -> SampledNoise3d
    where
        K: NoiseKind3d,
        T: Noise3d<K>,
    {
        let _ = kind;
        let pos = self.key.id.as_vec3();
        noise.sample_vec3::<K>(
            pos,
            UVec3::splat(DENSITY_CORNERS_PER_AXIS as u32),
            seed,
        )
    }

    pub fn with_mut<R>(
        &self,
        edit: impl FnOnce(&mut BrickHubCell) -> R,
    ) -> Result<R, CellAccessError> {
        let mut slot = self
            .slot
            .lock()
            .map_err(|_| CellAccessError::Poisoned(self.key.id))?;
        if slot.state != CellState::Resident {
            return Err(CellAccessError::Inactive(self.key.id));
        }
        let result = edit(&mut slot.cell);
        self.dirty.store(true, Ordering::Release);
        Ok(result)
    }

    pub fn with<R>(&self, read: impl FnOnce(&BrickHubCell) -> R) -> Result<R, CellAccessError> {
        let slot = self
            .slot
            .lock()
            .map_err(|_| CellAccessError::Poisoned(self.key.id))?;
        if slot.state != CellState::Resident {
            return Err(CellAccessError::Inactive(self.key.id));
        }
        Ok(read(&slot.cell))
    }

    pub(crate) fn new(
        key: CellKey,
        slot: Arc<Mutex<CellSlot>>,
        dirty: Arc<AtomicBool>,
    ) -> Self {
        Self { key, slot, dirty }
    }
}

/// Identifies one admission lifetime at a spatial coordinate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CellKey {
    pub id: CellID,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellAccessError {
    Inactive(CellID),
    Poisoned(CellID),
}
/// Opaque ticket identifying the currently valid generation job for a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenerationToken {
    id: CellID,
    epoch: u64,
}

impl GenerationToken {
    pub fn id(self) -> CellID {
        self.id
    }
    pub fn epoch(self) -> u64 {
        self.epoch
    }

    pub(crate) fn new(id: IVec3, epoch: u64) -> Self {
        Self { id, epoch }
    }
}

/// CPU-generated cell data returned from a worker to the BrickHub coordinator.
pub struct GeneratedCell {
    token: GenerationToken,
    cell: BrickHubCell,
}

impl GeneratedCell {
    pub fn new(token: GenerationToken, cell: BrickHubCell) -> Self {
        Self { token, cell }
    }
    pub(crate) fn token(&self) -> &GenerationToken {
        &self.token
    }

    pub fn cell_ref(&self) -> &BrickHubCell {
        &self.cell
    }

    pub(crate) fn cell(self) -> BrickHubCell {
        self.cell
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeneratedCellStatus {
    Committed,
    EvictionPending,
    Stale,
    OutsideResidency,
    AlreadyResident,
    RejectedByResidency,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeneratedCellResult {
    pub id: CellID,
    pub status: GeneratedCellStatus,
}
