use crate::consts::{
    BRICKS_PER_BRICKMAP, BrickID, CELL_SIZE, CellID, DENSITY_WORDS_PER_CELL, INDEX_WORDS_PER_CELL,
    MaterialID, VOXELS_PER_BRICK,
};
use crate::cpu::{CellDelta, StorageType, map};
use bevy_math::prelude::*;
use std::ops::Range;
use std::sync::{Arc, Mutex};

pub struct BrickHubCell {
    map: map::BrickMap,
    delta: CellDelta,
}
impl Default for BrickHubCell {
    fn default() -> Self {
        let mut delta = CellDelta::default();
        // Index ranges can be recycled while their old GPU contents remain. Upload the
        // zeroed map (brick grid and wedge states) so a new empty or sparse cell cannot
        // inherit stale brick references. Density uploads as entirely outside.
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
    /// No density field is written; a solid cell renders empty under the density baseline
    /// until edit paths are extended to update density (see `from_density`).
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

    /// Builds a detached cell from a raw-coordinate density sampler (negative inside,
    /// positive outside; corner coordinates in `0..65` per axis) plus the material every
    /// surface-touching voxel receives. Bricks are classified solid/shell from the corner
    /// field so GPU consumers can skip whole bricks. Later `insert`/`remove` edits update
    /// the occupancy path but not the density field or wedges; rebuild the cell after
    /// density-significant edits.
    pub fn from_density(&mut self, sample: impl FnMut(UVec3) -> f32, material: MaterialID) {
        let ranges = self.map.build_from_density(sample, material);
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
}

impl CellHandle {
    pub fn id(&self) -> CellID {
        self.key.id
    }

    pub fn key(&self) -> CellKey {
        self.key
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
        Ok(edit(&mut slot.cell))
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

    pub(crate) fn new(key: CellKey, slot: Arc<Mutex<CellSlot>>) -> Self {
        Self { key, slot }
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

#[cfg(test)]
mod tests {
    use super::*;
    use sns_helper::prelude::Range;

    #[test]
    fn empty_cell_has_no_bricks_and_uploads_a_cleared_index_grid() {
        let mut cell = BrickHubCell::empty();
        assert_eq!(cell.map().active_brick_count(), 0);
        assert_eq!(cell.delta().take(StorageType::Brick), None);
        assert_eq!(cell.delta().take(StorageType::Material), None);
        assert_eq!(
            cell.delta().take(StorageType::Indices),
            Some((0, INDEX_WORDS_PER_CELL))
        );
        assert_eq!(
            cell.delta().take(StorageType::Density),
            Some((0, DENSITY_WORDS_PER_CELL))
        );
    }

    #[test]
    fn density_mask_packs_occupied_bricks_and_initial_deltas() {
        let mut sample_count = 0;
        let mut cell = BrickHubCell::from_voxel_sampler(|voxel| {
            sample_count += 1;
            match voxel {
                UVec3 { x: 9, y: 9, z: 9 } => Some(17),
                UVec3 {
                    x: 63,
                    y: 63,
                    z: 63,
                } => Some(17),
                _ => None,
            }
        });

        assert_eq!(sample_count, CELL_SIZE * CELL_SIZE * CELL_SIZE);
        assert_eq!(cell.map().active_brick_count(), 2);
        assert_eq!(
            cell.map().get_brick(Range::new(0, 2))[0].occupied_voxel_count(),
            1
        );
        assert_eq!(
            cell.map().get_brick(Range::new(0, 2))[1].occupied_voxel_count(),
            1
        );
        assert_eq!(cell.map().used_material_ids(), vec![17]);
        assert_eq!(cell.delta().take(StorageType::Brick), Some((0, 2)));
        assert_eq!(
            cell.delta().take(StorageType::Material),
            Some((0, 2 * VOXELS_PER_BRICK))
        );
        let index_range = cell.delta().take(StorageType::Indices).unwrap();
        assert_eq!(index_range, (0, INDEX_WORDS_PER_CELL));
        assert_eq!(
            cell.delta().take(StorageType::Density),
            Some((0, DENSITY_WORDS_PER_CELL))
        );
    }

    #[test]
    fn layered_cell_fills_bands_and_leaves_gaps_empty() {
        let cell = BrickHubCell::layered(&[(0..2, 3), (4..8, 5)]).unwrap();
        assert_eq!(cell.map().active_brick_count(), 8 * 8);
        assert_eq!(cell.map().used_material_ids(), vec![3, 5]);

        let indices = cell.map().get_indices(Range::new(0, BRICKS_PER_BRICKMAP));
        assert!(indices.iter().any(|&index| index != 0));
        assert!(indices.iter().any(|&index| index == 0));
    }

    #[test]
    fn layered_cell_rejects_invalid_or_overlapping_ranges() {
        assert_eq!(
            BrickHubCell::layered(&[(3..3, 1)]).err().unwrap(),
            CellBuildError::InvalidLayerRange
        );
        assert_eq!(
            BrickHubCell::layered(&[(0..4, 1), (3..5, 2)])
                .err()
                .unwrap(),
            CellBuildError::OverlappingLayers
        );
        assert_eq!(
            BrickHubCell::layered(&[(62..65, 1)]).err().unwrap(),
            CellBuildError::InvalidLayerRange
        );
    }
}
