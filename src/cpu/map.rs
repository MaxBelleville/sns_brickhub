use bevy_math::*;
use smallvec::SmallVec;
use sns_helper::prelude::{LinearAllocator, Range};
use std::collections::HashSet;

use super::{index, material};
use crate::{
    brick::{Brick, BrickStatusGrid},
    consts::{
        BRICK_SIZE, BRICKMAP_SIZE, BRICKS_PER_BRICKMAP, BrickID, BrickIndex, CELL_SIZE,
        DENSITY_CORNERS_PER_AXIS, DENSITY_CORNERS_PER_CELL, MaterialID, VOXELS_PER_BRICK,
    },
    cpu::density::{DENSITY_UNSET, DensityField, MeshMode, WedgeState, corner_position},
    cpu::index::{IndexGrid, IndexKind},
};

#[inline]
pub fn voxel_in_brick(voxel: BrickID) -> BrickIndex {
    let bx = voxel.x & (BRICK_SIZE as u32 - 1);
    let by = voxel.y & (BRICK_SIZE as u32 - 1);
    let bz = voxel.z & (BRICK_SIZE as u32 - 1);
    bx + BRICK_SIZE as u32 * by + BRICK_SIZE as u32 * BRICK_SIZE as u32 * bz
}
#[inline]
pub fn brick_index_to_voxel(index: BrickIndex) -> BrickID {
    let x = index % BRICKMAP_SIZE as u32;
    let y = (index / BRICKMAP_SIZE as u32) % BRICKMAP_SIZE as u32;
    let z = index / (BRICKMAP_SIZE * BRICKMAP_SIZE) as u32;
    UVec3::new(x, y, z)
}

pub struct BrickMapStorage {
    bricks: LinearAllocator<Brick>,
    materials: material::MaterialAllocator,
    density: DensityField,
}

impl Default for BrickMapStorage {
    fn default() -> Self {
        let brick_capacity = BRICKS_PER_BRICKMAP;
        let bricks = LinearAllocator::new(brick_capacity);
        let materials = material::MaterialAllocator::new(brick_capacity * VOXELS_PER_BRICK);

        Self {
            bricks,
            materials,
            density: DensityField::default(),
        }
    }
}

impl BrickMapStorage {
    pub fn allocate_material<K: IndexKind>(
        &mut self,
        brick_index: usize,
        events: &mut SmallVec<[BrickEditEvent; 4]>,
    ) {
        let brick = self.bricks.get_mut(brick_index);
        if brick.get_raw_index().is_unset() {
            let base = self.materials.alloc_next_block();
            brick.set_start_index::<K>(base);
            events.push(BrickEditEvent::MaterialBlockAllocated { base });
        }
    }

    pub fn add_brick_occupancy(&mut self, brick_index: usize, voxel: UVec3, voxel_index: usize) {
        let brick = self.bricks.get_mut(brick_index);
        let word = voxel_index >> 6;
        let bit = voxel_index & 63;
        brick.set_occupancy(word, 1u64 << bit);
        brick.set_lod(voxel);
    }

    pub fn clear_brick_occupancy(
        &mut self,
        brick_index: usize,
        voxel_index: usize,
        voxel: UVec3,
    ) -> Option<bool> {
        let brick = self.bricks.get_mut(brick_index);
        let word = voxel_index >> 6;
        let bit = voxel_index & 63;
        if !brick.test_occupancy(word, bit) {
            return None; // already empty
        }
        let is_only_voxel = brick.occupied_voxel_count() == 1;
        brick.clear_occupancy(word, bit);
        brick.clear_lod(voxel);
        Some(is_only_voxel)
    }
    pub fn get_voxel(&self, brick_index: usize, voxel_index: usize) -> usize {
        let brick = self.bricks.get(brick_index);
        brick.get_raw_index().value() as usize + voxel_index
    }
}

pub struct BrickIndexGrid {
    index: IndexGrid,
    status: BrickStatusGrid,
}

impl Default for BrickIndexGrid {
    fn default() -> Self {
        let brick_capacity = BRICKS_PER_BRICKMAP;
        let index = IndexGrid::default();
        let status = BrickStatusGrid::new(brick_capacity);
        Self { index, status }
    }
}

#[derive(Default)]
pub struct BrickMap {
    storage: BrickMapStorage,
    grid: BrickIndexGrid,
    active_bricks: u32,
}

impl BrickMap {
    pub fn active_brick_count(&self) -> u32 {
        self.active_bricks
    }

    pub fn get_brick(&self, range: Range<usize>) -> &[Brick] {
        self.storage.bricks.slice_range(range.start(), range.end())
    }
    pub fn get_material(&self, range: Range<usize>) -> &[MaterialID] {
        self.storage
            .materials
            .slice_bucket(range.start(), range.end())
    }

    pub fn used_material_ids(&self) -> Vec<MaterialID> {
        let mut ids = HashSet::new();
        for z in 0..BRICKMAP_SIZE as u32 {
            for y in 0..BRICKMAP_SIZE as u32 {
                for x in 0..BRICKMAP_SIZE as u32 {
                    let Some(brick_index) = self.grid.index.get(UVec3::new(x, y, z)) else {
                        continue;
                    };
                    let brick = self.storage.bricks.get(brick_index as usize);
                    let material_base = brick.get_raw_index().value() as usize;
                    for voxel_index in 0..VOXELS_PER_BRICK {
                        if !brick.test_occupancy(voxel_index >> 6, voxel_index & 63) {
                            continue;
                        }
                        if let Some(material_id) = self
                            .storage
                            .materials
                            .get_material_id(material_base + voxel_index)
                        {
                            ids.insert(material_id);
                        }
                    }
                }
            }
        }
        let mut ids: Vec<_> = ids.into_iter().collect();
        ids.sort_unstable();
        ids
    }

    pub fn get_indices(&self, range: Range<usize>) -> &[u32] {
        self.grid.index.slice_range(range.start(), range.end())
    }

    /// The cell's corner density field (raw coordinate convention, see `cpu::density`).
    pub fn density(&self) -> &DensityField {
        &self.storage.density
    }

    pub fn density_mut(&mut self) -> &mut DensityField {
        &mut self.storage.density
    }

    /// Raw brick-coordinate cell-local brick grid coordinate mapping used by the
    /// wedge section of the index buffer.
    pub fn wedge_state(&self, raw_brick: UVec3) -> Option<WedgeState> {
        let word = BRICKS_PER_BRICKMAP + IndexGrid::index(raw_brick) / 16;
        let shift = (IndexGrid::index(raw_brick) % 16) * 2;
        let bits = (self.grid.index.slice_range(word, word + 1)[0] >> shift) & 0b11;
        WedgeState::from_bits(bits)
    }

    /// Whether the brick at raw coordinates is flagged for iso (Transvoxel) meshing.
    /// Outside the cell-local grid this is always false.
    pub fn brick_is_iso(&self, raw_brick: UVec3) -> bool {
        if raw_brick.x >= BRICKMAP_SIZE as u32
            || raw_brick.y >= BRICKMAP_SIZE as u32
            || raw_brick.z >= BRICKMAP_SIZE as u32
        {
            return false;
        }
        self.grid.index.iso(raw_brick)
    }

    /// Whether the voxel at raw coordinates is recorded as occupied. Outside the
    /// 64^3 cell it is always unoccupied.
    pub fn voxel_is_occupied(&self, raw_voxel: UVec3) -> bool {
        if raw_voxel.x >= CELL_SIZE as u32
            || raw_voxel.y >= CELL_SIZE as u32
            || raw_voxel.z >= CELL_SIZE as u32
        {
            return false;
        }
        let local_packed = UVec3::new(
            raw_voxel.x,
            BRICK_SIZE as u32 - 1 - raw_voxel.y % BRICK_SIZE as u32,
            raw_voxel.z,
        );
        let coord = UVec3::new(
            raw_voxel.x / BRICK_SIZE as u32,
            (BRICKMAP_SIZE as u32 - 1) - raw_voxel.y / BRICK_SIZE as u32,
            raw_voxel.z / BRICK_SIZE as u32,
        );
        let Some(slot) = self.grid.index.get(coord) else {
            return false;
        };
        let brick = self.storage.bricks.get(slot as usize);
        let voxel_index = voxel_in_brick(local_packed) as usize;
        brick.test_occupancy(voxel_index >> 6, voxel_index & 63)
    }

    /// Builds the whole cell from a raw-coordinate density sampler (negative inside,
    /// positive outside; corners at `0..65` per axis). Shell and solid bricks get brick
    /// records and per-voxel material entries; wedge states are stored per raw brick
    /// coordinate, and shell bricks are additionally flagged for iso meshing when
    /// `mode` is `MeshMode::Iso`. The `sample` function is called once per corner.
    pub(crate) fn build_from_density(
        &mut self,
        mut sample: impl FnMut(UVec3) -> f32,
        material: MaterialID,
        mode: MeshMode,
    ) -> MapBuildRanges {
        self.clear();
        self.storage.density.fill_from_fn(&mut sample);

        let axis = DENSITY_CORNERS_PER_AXIS as u32;
        let mut brick_count = 0usize;

        for raw_z in 0..BRICKMAP_SIZE as u32 {
            for raw_y in 0..BRICKMAP_SIZE as u32 {
                for raw_x in 0..BRICKMAP_SIZE as u32 {
                    let raw_coord = UVec3::new(raw_x, raw_y, raw_z);
                    let base = raw_coord * BRICK_SIZE as u32;

                    // Classification over the brick's 9^3 corner samples.
                    let mut minimum = f32::MAX;
                    let mut maximum = f32::MIN;
                    for cz in 0..=BRICK_SIZE as u32 {
                        for cy in 0..=BRICK_SIZE as u32 {
                            for cx in 0..=BRICK_SIZE as u32 {
                                let value =
                                    self.storage.density.corner(base + UVec3::new(cx, cy, cz));
                                minimum = minimum.min(value);
                                maximum = maximum.max(value);
                            }
                        }
                    }
                    let wedge = WedgeState::from_density(minimum, maximum);
                    if wedge == WedgeState::Empty {
                        continue;
                    }

                    let brick_index = self.storage.bricks.allocate(Brick::default());
                    let material_base = self.storage.materials.alloc_next_block();

                    // Shell corners can extend one voxel past the cell; clamp reads.
                    let clamp_axis = axis - 1;
                    if wedge == WedgeState::Solid {
                        self.storage
                            .bricks
                            .get_mut(brick_index)
                            .fill_solid::<index::VoxelKind>(material_base);
                        self.storage.materials.fill_bucket_range(
                            material_base as usize,
                            VOXELS_PER_BRICK,
                            material,
                        );
                    } else {
                        self.storage
                            .bricks
                            .get_mut(brick_index)
                            .set_start_index::<index::VoxelKind>(material_base);
                        for lz in 0..BRICK_SIZE as u32 {
                            for ly in 0..BRICK_SIZE as u32 {
                                for lx in 0..BRICK_SIZE as u32 {
                                    // A voxel belongs to the surface as soon as any of
                                    // its corners is inside; voxels fully outside are empty.
                                    let mut inside_any = false;
                                    for cz in 0..=1u32 {
                                        for cy in 0..=1u32 {
                                            for cx in 0..=1u32 {
                                                let corner = base
                                                    + UVec3::new(lx, ly, lz)
                                                    + UVec3::new(cx, cy, cz);
                                                let clamped = UVec3::new(
                                                    corner.x.min(clamp_axis),
                                                    corner.y.min(clamp_axis),
                                                    corner.z.min(clamp_axis),
                                                );
                                                if self.storage.density.corner(clamped) <= 0.0 {
                                                    inside_any = true;
                                                }
                                            }
                                        }
                                    }
                                    if !inside_any {
                                        continue;
                                    }

                                    // Occupancy and material storage use flipped coordinates.
                                    let local_packed =
                                        UVec3::new(lx, BRICK_SIZE as u32 - 1 - ly, lz);
                                    let voxel_index = voxel_in_brick(local_packed) as usize;
                                    let brick = self.storage.bricks.get_mut(brick_index);
                                    brick.set_voxel_occupied(local_packed, voxel_index);
                                    self.storage.materials.write_material_id(
                                        material_base as usize + voxel_index,
                                        material,
                                    );
                                }
                            }
                        }
                    }
                    self.grid.index.set(
                        UVec3::new(raw_x, BRICKMAP_SIZE as u32 - 1 - raw_y, raw_z),
                        brick_index as u32,
                    );
                    self.grid.status.set_active(brick_index);
                    self.grid.index.set_wedge(raw_coord, wedge.bits());
                    // Only shell bricks need polygonizing; solid interiors and empty
                    // space contribute no isosurface.
                    self.grid.index.set_iso(
                        raw_coord,
                        mode == MeshMode::Iso && wedge == WedgeState::Shell,
                    );
                    brick_count += 1;
                }
            }
        }

        self.active_bricks = brick_count as u32;
        MapBuildRanges {
            brick_count,
            material_count: brick_count * VOXELS_PER_BRICK,
        }
    }

    pub fn fill_solid(&mut self, material: MaterialID) {
        self.clear();
        for z in 0..BRICKMAP_SIZE as u32 {
            for y in 0..BRICKMAP_SIZE as u32 {
                for x in 0..BRICKMAP_SIZE as u32 {
                    let coord = UVec3::new(x, y, z);
                    let brick_index = self.storage.bricks.allocate(Brick::default());
                    let material_base = self.storage.materials.alloc_next_block();
                    self.storage
                        .bricks
                        .get_mut(brick_index)
                        .fill_solid::<index::VoxelKind>(material_base);
                    self.storage.materials.fill_bucket_range(
                        material_base as usize,
                        VOXELS_PER_BRICK,
                        material,
                    );
                    self.grid.index.set(coord, brick_index as u32);
                    self.grid.status.set_active(brick_index);
                }
            }
        }
        self.active_bricks = BRICKS_PER_BRICKMAP as u32;
    }

    /// Builds a cell directly from local voxel samples. Empty bricks are skipped and
    /// occupied bricks are packed densely in storage while the grid retains their positions.
    pub(crate) fn build_from_sampler(
        &mut self,
        mut sample: impl FnMut(UVec3) -> Option<MaterialID>,
    ) -> MapBuildRanges {
        self.clear();

        for brick_z in 0..BRICKMAP_SIZE as u32 {
            for brick_y in 0..BRICKMAP_SIZE as u32 {
                for brick_x in 0..BRICKMAP_SIZE as u32 {
                    let brick_coord = UVec3::new(brick_x, brick_y, brick_z);
                    let mut samples = [None; VOXELS_PER_BRICK];
                    let mut occupied_count = 0;

                    for local_y in 0..BRICK_SIZE as u32 {
                        for local_z in 0..BRICK_SIZE as u32 {
                            for local_x in 0..BRICK_SIZE as u32 {
                                let flipped_voxel = brick_coord * BRICK_SIZE as u32
                                    + UVec3::new(local_x, local_y, local_z);
                                let voxel = UVec3::new(
                                    flipped_voxel.x,
                                    (BRICKMAP_SIZE * BRICK_SIZE) as u32 - 1 - flipped_voxel.y,
                                    flipped_voxel.z,
                                );
                                let voxel_index = voxel_in_brick(flipped_voxel) as usize;
                                let material = sample(voxel);
                                occupied_count += usize::from(material.is_some());
                                samples[voxel_index] = material;
                            }
                        }
                    }

                    if occupied_count == 0 {
                        continue;
                    }

                    let brick_index = self.storage.bricks.allocate(Brick::default());
                    let material_base = self.storage.materials.alloc_next_block();
                    let brick = self.storage.bricks.get_mut(brick_index);
                    if occupied_count == VOXELS_PER_BRICK {
                        brick.fill_solid::<index::VoxelKind>(material_base);
                    } else {
                        brick.set_start_index::<index::VoxelKind>(material_base);
                        for voxel_index in 0..VOXELS_PER_BRICK {
                            if samples[voxel_index].is_some() {
                                let x = (voxel_index % BRICK_SIZE) as u32;
                                let y = ((voxel_index / BRICK_SIZE) % BRICK_SIZE) as u32;
                                let z = (voxel_index / (BRICK_SIZE * BRICK_SIZE)) as u32;
                                brick.set_voxel_occupied(UVec3::new(x, y, z), voxel_index);
                            }
                        }
                    }

                    for (voxel_index, material) in samples.into_iter().enumerate() {
                        if let Some(material) = material {
                            self.storage
                                .materials
                                .write_material_id(material_base as usize + voxel_index, material);
                        }
                    }

                    self.grid.index.set(brick_coord, brick_index as u32);
                    self.grid.status.set_active(brick_index);
                    self.active_bricks += 1;
                }
            }
        }

        self.derive_blocky_density();
        MapBuildRanges {
            brick_count: self.active_bricks as usize,
            material_count: self.active_bricks as usize * VOXELS_PER_BRICK,
        }
    }

    /// Derives a blocky density field from stored occupancy so cells built through the
    /// occupancy API also render through the density baseline: a corner is inside when
    /// any adjacent voxel is occupied (in-cell neighbors only). Wedge states classify
    /// each raw brick coordinate from the same field.
    fn derive_blocky_density(&mut self) {
        let cell_max = (CELL_SIZE - 1) as u32;
        let corner_count = DENSITY_CORNERS_PER_CELL;
        let mut values = vec![DENSITY_UNSET; corner_count];
        for index in 0..corner_count {
            let corner = corner_position(index);
            let voxel_min = UVec3::new(
                corner.x.saturating_sub(1),
                corner.y.saturating_sub(1),
                corner.z.saturating_sub(1),
            );
            let voxel_max = UVec3::new(
                corner.x.min(cell_max),
                corner.y.min(cell_max),
                corner.z.min(cell_max),
            );
            let mut occupied = false;
            for vz in voxel_min.z..=voxel_max.z {
                for vy in voxel_min.y..=voxel_max.y {
                    for vx in voxel_min.x..=voxel_max.x {
                        if self.voxel_is_occupied(UVec3::new(vx, vy, vz)) {
                            occupied = true;
                        }
                    }
                }
            }
            values[index] = if occupied { -1.0 } else { DENSITY_UNSET };
        }
        self.storage.density.fill_from_values(&values);

        for raw_z in 0..BRICKMAP_SIZE as u32 {
            for raw_y in 0..BRICKMAP_SIZE as u32 {
                for raw_x in 0..BRICKMAP_SIZE as u32 {
                    let base = UVec3::new(raw_x, raw_y, raw_z) * BRICK_SIZE as u32;
                    let mut minimum = f32::MAX;
                    let mut maximum = f32::MIN;
                    for cz in 0..=BRICK_SIZE as u32 {
                        for cy in 0..=BRICK_SIZE as u32 {
                            for cx in 0..=BRICK_SIZE as u32 {
                                let value =
                                    self.storage.density.corner(base + UVec3::new(cx, cy, cz));
                                minimum = minimum.min(value);
                                maximum = maximum.max(value);
                            }
                        }
                    }
                    let wedge = WedgeState::from_density(minimum, maximum);
                    self.grid
                        .index
                        .set_wedge(UVec3::new(raw_x, raw_y, raw_z), wedge.bits());
                }
            }
        }
    }

    pub fn insert(
        &mut self,
        voxel: BrickID,
        material: MaterialID,
    ) -> SmallVec<[BrickEditEvent; 4]> {
        let mut events = SmallVec::new();
        let flipped_voxel = self.flip_y(voxel);
        let brick_coord = flipped_voxel / BRICK_SIZE as u32;

        let (brick_index, _was_new) = self.get_or_alloc_brick(brick_coord, &mut events);

        let brick = self.storage.bricks.get_mut(brick_index);
        if brick.is_empty() {
            self.active_bricks += 1;
        }

        self.storage
            .allocate_material::<index::VoxelKind>(brick_index, &mut events);

        let voxel_index = voxel_in_brick(flipped_voxel) as usize;
        let local_voxel = flipped_voxel % BRICK_SIZE as u32;

        let voxel_offset = self.storage.get_voxel(brick_index, voxel_index);

        self.storage
            .materials
            .write_material_id(voxel_offset, material);

        events.push(BrickEditEvent::VoxelMaterialChanged {
            index: voxel_offset,
        });

        self.storage
            .add_brick_occupancy(brick_index, local_voxel, voxel_index);

        events
    }

    pub fn remove(&mut self, voxel: BrickID) -> SmallVec<[BrickEditEvent; 4]> {
        let mut events = SmallVec::new();
        let flipped_voxel = self.flip_y(voxel);

        let brick_coord = flipped_voxel / BRICK_SIZE as u32;
        let Some(brick_index) = self.grid.index.get(brick_coord).map(|index| index as usize) else {
            return events;
        };
        let local_voxel = flipped_voxel % BRICK_SIZE as u32;

        let brick = self.storage.bricks.get_mut(brick_index);
        let voxel_index = voxel_in_brick(flipped_voxel) as usize;

        let material_index = brick.get_raw_index().value() as usize + voxel_index;

        let Some(was_only_voxel) =
            self.storage
                .clear_brick_occupancy(brick_index, voxel_index, local_voxel)
        else {
            return events;
        };
        self.storage.materials.remove_bucket_index(material_index);
        events.push(BrickEditEvent::VoxelMaterialChanged {
            index: material_index,
        });

        if was_only_voxel {
            self.grid.index.clear(brick_coord);
            self.grid.status.set_inactive(brick_index);
            events.push(BrickEditEvent::BrickBecameEmpty { brick_index });
            self.active_bricks -= 1;
        }

        events
    }

    pub fn get_or_alloc_brick(
        &mut self,
        brick_coord: BrickID,
        events: &mut SmallVec<[BrickEditEvent; 4]>,
    ) -> (usize, bool) {
        match self.grid.index.get(brick_coord) {
            Some(idx) => (idx as usize, false),
            None => {
                let idx = self.storage.bricks.allocate(Brick::default());
                events.push(BrickEditEvent::BrickAllocated { brick_index: idx });
                self.grid.index.set(brick_coord, idx as u32);
                self.grid.status.set_active(idx);
                (idx, true)
            }
        }
    }

    pub fn clear(&mut self) {
        self.storage = BrickMapStorage::default();
        self.grid = BrickIndexGrid::default();
        self.active_bricks = 0;
    }

    fn flip_y(&self, voxel: BrickID) -> BrickID {
        let flipped_y = (BRICKMAP_SIZE * BRICK_SIZE) as u32 - 1 - voxel.y;
        UVec3::new(voxel.x, flipped_y, voxel.z)
    }
}

pub(crate) struct MapBuildRanges {
    pub(crate) brick_count: usize,
    pub(crate) material_count: usize,
}

pub enum BrickEditEvent {
    BrickAllocated { brick_index: usize },
    BrickBecameEmpty { brick_index: usize },
    MaterialBlockAllocated { base: u32 },
    VoxelMaterialChanged { index: usize },
}
