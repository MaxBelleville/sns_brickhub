use bevy_math::*;
use smallvec::SmallVec;
use sns_helper::prelude::{LinearAllocator, Range};
use std::collections::HashSet;

use super::{index, material};
use crate::{
    brick::{Brick, BrickStatusGrid},
    consts::{
        BRICK_SIZE, BRICKMAP_SIZE, BRICKS_PER_BRICKMAP, BrickID, BrickIndex, CELL_SIZE,
        DENSITY_CORNERS_PER_AXIS, DENSITY_CORNERS_PER_CELL, DENSITY_UNSET, MaterialID,
        VOXELS_PER_BRICK,
    },
    cpu::density::{BrickFill, DensityField, corner_position},
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

/// Sign bitmask for one x-row of 9 corners: bit `i` is set when the corner value
/// is at the isolevel or inside (`<= isolevel`). Rows are contiguous in the flat
/// scratch buffer, so the fold vectorizes.
#[inline]
fn inside_sign_row(corners: &[f32], start: usize, isolevel: f32) -> u16 {
    let row = &corners[start..start + BRICK_SIZE as usize + 1];
    let mut mask = 0u16;
    for (bit, &value) in row.iter().enumerate() {
        mask |= u16::from(value <= isolevel) << bit;
    }
    mask
}

/// Minimum and maximum over a brick's 9^3 corners in the flat scratch buffer,
/// read as 81 contiguous 9-wide x-rows.
#[inline]
fn brick_corner_min_max(
    corners: &[f32],
    base: UVec3,
    axis: usize,
    axis_sq: usize,
) -> (f32, f32) {
    let mut minimum = f32::MAX;
    let mut maximum = f32::MIN;
    for cz in 0..=BRICK_SIZE as u32 {
        let z_plane = axis_sq * (base.z + cz) as usize;
        for cy in 0..=BRICK_SIZE as u32 {
            let start = z_plane + axis * (base.y + cy) as usize + base.x as usize;
            let row = &corners[start..start + BRICK_SIZE as usize + 1];
            for &value in row {
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
        }
    }
    (minimum, maximum)
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

    /// Neutral corner classification stored for this brick (2-bit code).
    pub(crate) fn brick_fill(&self, raw_brick: UVec3) -> Option<BrickFill> {
        let word = BRICKS_PER_BRICKMAP + IndexGrid::index(raw_brick) / 16;
        let shift = (IndexGrid::index(raw_brick) % 16) * 2;
        let bits = (self.grid.index.slice_range(word, word + 1)[0] >> shift) & 0b11;
        BrickFill::from_bits(bits)
    }

    /// The general-purpose flag bit stored for this brick. The renderer uses it
    /// to mark bricks needing transvoxel polygonization. Outside the cell-local
    /// grid this is always false.
    pub fn brick_flag(&self, raw_brick: UVec3) -> bool {
        if raw_brick.x >= BRICKMAP_SIZE as u32
            || raw_brick.y >= BRICKMAP_SIZE as u32
            || raw_brick.z >= BRICKMAP_SIZE as u32
        {
            return false;
        }
        self.grid.index.brick_flag(raw_brick)
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

    /// Conservative occupancy of the stride-sized, stride-aligned block at raw
    /// coordinates, mirroring `voxel_mesh.wesl::block_occupied`. Stride 1 reads
    /// the exact voxel bit; stride 2 reads the brick's 4x4x4 mask; stride >= 4
    /// reads the coarser 2x2x2 mask.
    pub fn block_is_occupied(&self, raw_voxel: UVec3, stride: u32) -> bool {
        if raw_voxel.x >= CELL_SIZE as u32
            || raw_voxel.y >= CELL_SIZE as u32
            || raw_voxel.z >= CELL_SIZE as u32
        {
            return false;
        }
        let local_packed = UVec3::new(
            raw_voxel.x % BRICK_SIZE as u32,
            BRICK_SIZE as u32 - 1 - raw_voxel.y % BRICK_SIZE as u32,
            raw_voxel.z % BRICK_SIZE as u32,
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
        match stride {
            1 => {
                let voxel_index = voxel_in_brick(local_packed) as usize;
                brick.test_occupancy(voxel_index >> 6, voxel_index & 63)
            }
            2 => {
                let group = brick.get_lod_4x4x4_idx(local_packed);
                (brick.lod_mask_4x4x4() & (1u64 << group)) != 0
            }
            _ => {
                let group = brick.get_lod_2x2x2_idx(local_packed);
                (brick.lod_mask_2x2x2() & (1u8 << group)) != 0
            }
        }
    }

    /// Brick storage slot containing the voxel at raw coordinates; `None` when the
    /// voxel's brick is not allocated (which includes every out-of-cell coordinate).
    pub fn brick_slot(&self, raw_voxel: UVec3) -> Option<u32> {
        if raw_voxel.x >= CELL_SIZE as u32
            || raw_voxel.y >= CELL_SIZE as u32
            || raw_voxel.z >= CELL_SIZE as u32
        {
            return None;
        }
        let coord = UVec3::new(
            raw_voxel.x / BRICK_SIZE as u32,
            (BRICKMAP_SIZE as u32 - 1) - raw_voxel.y / BRICK_SIZE as u32,
            raw_voxel.z / BRICK_SIZE as u32,
        );
        self.grid.index.get(coord)
    }

    /// Builds the whole cell from a raw-coordinate density sampler (below the
    /// isolevel is inside; corners at `0..65` per axis). The isolevel and the
    /// meaning of the per-brick flag are supplied by the renderer; the storage
    /// library owns neither convention.
    pub(crate) fn build_from_density(
        &mut self,
        mut sample: impl FnMut(UVec3) -> f32,
        material: MaterialID,
        isolevel: f32,
        flag_partial_bricks: bool,
    ) -> MapBuildRanges {
        self.clear();
        self.storage.density.fill_from_fn(&mut sample);

        // Decode the packed fp16 corner field once into a flat x-fastest scratch
        // buffer: the classification and shell passes below touch corners many
        // times, and plain f32 row slices avoid re-decoding words (and re-doing
        // index arithmetic) on every visit.
        let axis = DENSITY_CORNERS_PER_AXIS;
        let axis_sq = axis * axis;
        let mut corners = vec![0.0f32; DENSITY_CORNERS_PER_CELL];
        for (index, value) in corners.iter_mut().enumerate() {
            *value = self.storage.density.corner_value(index);
        }

        let mut brick_count = 0usize;

        for raw_z in 0..BRICKMAP_SIZE as u32 {
            for raw_y in 0..BRICKMAP_SIZE as u32 {
                for raw_x in 0..BRICKMAP_SIZE as u32 {
                    let raw_coord = UVec3::new(raw_x, raw_y, raw_z);
                    let base = raw_coord * BRICK_SIZE as u32;

                    // Classification over the brick's 9^3 corners, read as 81
                    // contiguous 9-wide x-rows.
                    let (minimum, maximum) =
                        brick_corner_min_max(&corners, base, axis, axis_sq);
                    let fill = BrickFill::from_density(minimum, maximum, isolevel);
                    if fill == BrickFill::Empty {
                        continue;
                    }

                    let brick_index = self.storage.bricks.allocate(Brick::default());
                    let material_base = self.storage.materials.alloc_next_block();

                    if fill == BrickFill::Full {
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
                        // Per-row sign bitmasks collapse the 8-corner-per-voxel
                        // "inside" test to bit ops: voxel (lx, ly, lz) is inside
                        // iff any of its 8 corners is, which is bit lx of the OR
                        // of the four corner rows it spans, shifted 1 and masked
                        // to 8 voxels. Corner coordinates stay in 0..=64 for
                        // every in-cell voxel, so reads never clamp.
                        for lz in 0..BRICK_SIZE as u32 {
                            let z0 = axis_sq * (base.z + lz) as usize;
                            let z1 = axis_sq * (base.z + lz + 1) as usize;
                            for ly in 0..BRICK_SIZE as u32 {
                                let row_base = axis * (base.y + ly) as usize + base.x as usize;
                                // The voxel's 8 corners span z planes lz, lz+1 and
                                // y rows ly, ly+1.
                                let wide = inside_sign_row(&corners, z0 + row_base, isolevel)
                                    | inside_sign_row(&corners, z0 + row_base + axis, isolevel)
                                    | inside_sign_row(&corners, z1 + row_base, isolevel)
                                    | inside_sign_row(&corners, z1 + row_base + axis, isolevel);
                                let inside = (wide | (wide >> 1)) & 0x00FF;
                                if inside == 0 {
                                    continue;
                                }

                                let flipped_ly = BRICK_SIZE as u32 - 1 - ly;
                                for lx in 0..BRICK_SIZE as u32 {
                                    if inside & (1 << lx) == 0 {
                                        continue;
                                    }

                                    // Occupancy and material storage use flipped coordinates.
                                    let local_packed = UVec3::new(lx, flipped_ly, lz);
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
                    self.grid.index.set_fill_class(raw_coord, fill.bits());
                    // The renderer decides what the flag means; by default it
                    // marks partial bricks for transvoxel polygonization.
                    self.grid.index.set_brick_flag(
                        raw_coord,
                        flag_partial_bricks && fill == BrickFill::Partial,
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
    /// occupancy API also render through the density baseline
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

        // `values` already matches the flat x-fastest corner layout, so the
        // brick classification shares the vectorized row scan.
        let axis = DENSITY_CORNERS_PER_AXIS;
        let axis_sq = axis * axis;
        for raw_z in 0..BRICKMAP_SIZE as u32 {
            for raw_y in 0..BRICKMAP_SIZE as u32 {
                for raw_x in 0..BRICKMAP_SIZE as u32 {
                    let base = UVec3::new(raw_x, raw_y, raw_z) * BRICK_SIZE as u32;
                    let (minimum, maximum) =
                        brick_corner_min_max(&values, base, axis, axis_sq);
                    // Blocky-derived density is -1.0 inside / DENSITY_UNSET
                    // outside, so its natural classification level is 0.0.
                    let fill = BrickFill::from_density(minimum, maximum, 0.0);
                    self.grid
                        .index
                        .set_fill_class(UVec3::new(raw_x, raw_y, raw_z), fill.bits());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::CELL_SIZE;

    /// Sphere density (negative inside) centered in the cell, radius 18 voxels.
    pub(crate) fn sphere(corner: UVec3) -> f32 {
        (corner.as_vec3() - Vec3::splat(32.0)).length() - 18.0
    }

    /// Golden counts for build_from_density on the sphere: locks the classification
    /// and population behavior so performance refactors stay behavior-preserving.
    #[test]
    fn density_build_counts_match_sphere_baseline() {
        let mut map = BrickMap::default();
        let ranges = map.build_from_density(sphere, 7, 0.0, true);
        assert_eq!(ranges.brick_count, map.active_brick_count() as usize);

        let mut occupied = 0u32;
        for z in 0..CELL_SIZE as u32 {
            for y in 0..CELL_SIZE as u32 {
                for x in 0..CELL_SIZE as u32 {
                    if map.voxel_is_occupied(UVec3::new(x, y, z)) {
                        occupied += 1;
                    }
                }
            }
        }

        let (mut empty, mut solid, mut shell, mut iso) = (0u32, 0u32, 0u32, 0u32);
        for z in 0..BRICKMAP_SIZE as u32 {
            for y in 0..BRICKMAP_SIZE as u32 {
                for x in 0..BRICKMAP_SIZE as u32 {
                    let coord = UVec3::new(x, y, z);
                    match map.brick_fill(coord) {
                        Some(BrickFill::Empty) => empty += 1,
                        Some(BrickFill::Full) => solid += 1,
                        Some(BrickFill::Partial) => {
                            shell += 1;
                            if map.brick_flag(coord) {
                                iso += 1;
                            }
                        }
                        None => {}
                    }
                }
            }
        }

        println!(
            "sphere baseline: bricks={} occupied={} empty={} solid={} shell={} iso={}",
            ranges.brick_count, occupied, empty, solid, shell, iso
        );
        // Exact goldens captured from the pre-optimization implementation.
        assert_eq!(ranges.brick_count, 136);
        assert_eq!(occupied, 27544);
        assert_eq!(empty, 376);
        assert_eq!(solid, 8);
        assert_eq!(shell, 128);
        assert_eq!(iso, 128);
    }

    #[test]
    fn block_occupied_matches_any_voxel_in_each_lod_block() {
        let mut map = BrickMap::default();
        map.build_from_density(sphere, 7, 0.0, true);

        for stride in [1u32, 2, 4] {
            for z in (0..CELL_SIZE as u32).step_by(stride as usize) {
                for y in (0..CELL_SIZE as u32).step_by(stride as usize) {
                    for x in (0..CELL_SIZE as u32).step_by(stride as usize) {
                        let block = UVec3::new(x, y, z);
                        let mut any_voxel = false;
                        for dz in 0..stride {
                            for dy in 0..stride {
                                for dx in 0..stride {
                                    any_voxel |= map.voxel_is_occupied(block + UVec3::new(dx, dy, dz));
                                }
                            }
                        }
                        assert_eq!(
                            map.block_is_occupied(block, stride),
                            any_voxel,
                            "stride {stride} block at {block:?}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod bench {
    use super::tests::sphere;
    use super::*;
    use std::time::Instant;

    #[test]
    fn density_build_release_timing() {
        let mut map = BrickMap::default();
        // Warmup + measure; 30 builds on one map instance (build clears first).
        map.build_from_density(sphere, 7, 0.0, true);
        let started = Instant::now();
        let iterations = 30;
        for _ in 0..iterations {
            map.build_from_density(sphere, 7, 0.0, true);
        }
        let per_cell = started.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
        println!("from_density per-cell: {:.3} ms", per_cell);
        assert!(per_cell < 50.0);
    }
}
