use crate::consts::{
    BRICK_INDEX_ENCODING_BIAS, BRICKMAP_SIZE, BRICKS_PER_BRICKMAP, EMPTY_BRICK_INDEX,
    INDEX_WORDS_PER_CELL, WEDGE_WORDS_PER_CELL,
};
use bevy_math::*;
use bytemuck::{self, Pod, Zeroable};
use std::marker::PhantomData;

const _: () = assert!(BRICKS_PER_BRICKMAP + WEDGE_WORDS_PER_CELL == INDEX_WORDS_PER_CELL);

/// Cell-local index buffer: the brick lookup grid (`0..BRICKS_PER_BRICKMAP`, indexed in
/// *flipped* brick coordinates) followed by the per-brick density wedge states
/// (`BRICKS_PER_BRICKMAP..INDEX_WORDS_PER_CELL`, packed 2 bits per brick and indexed in
/// *raw* brick coordinates).
#[repr(C)]
pub struct IndexGrid {
    indices: [u32; INDEX_WORDS_PER_CELL],
}

impl Default for IndexGrid {
    fn default() -> Self {
        Self {
            indices: [0; INDEX_WORDS_PER_CELL],
        }
    }
}
impl IndexGrid {
    pub fn index(coord: UVec3) -> usize {
        let cell_size = BRICKMAP_SIZE as u32;
        (coord.x + coord.y * cell_size + coord.z * cell_size * cell_size) as usize
    }

    fn wedge_index(coord: UVec3) -> usize {
        BRICKS_PER_BRICKMAP + Self::index(coord) / 16
    }
    pub fn get(&self, coord: UVec3) -> Option<u32> {
        let v = self.indices[Self::index(coord)];
        if v == EMPTY_BRICK_INDEX {
            None
        } else {
            Some(v - BRICK_INDEX_ENCODING_BIAS)
        }
    }

    pub fn set(&mut self, coord: UVec3, brick_index: u32) {
        self.indices[Self::index(coord)] = brick_index
            .checked_add(BRICK_INDEX_ENCODING_BIAS)
            .expect("brick index cannot use the empty sentinel");
    }
    pub fn clear(&mut self, coord: UVec3) {
        self.indices[Self::index(coord)] = EMPTY_BRICK_INDEX;
    }
    /// Overwrites the 2-bit density wedge state of the brick at raw `coord`.
    pub fn set_wedge(&mut self, coord: UVec3, state: u32) {
        debug_assert!(state < 4);
        let word = Self::wedge_index(coord);
        let shift = (Self::index(coord) % 16) * 2;
        self.indices[word] = (self.indices[word] & !(0b11 << shift)) | (state << shift);
    }

    pub fn slice_range(&self, from: usize, to: usize) -> &[u32] {
        &self.indices[from..to]
    }
}

// 1. The "Raw" data type that bytemuck can easily handle
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct RawIndex(pub u32);
impl RawIndex {
    pub fn is_lod(&self) -> bool {
        self.0 & 0x8000_0000 != 0
    }
    pub fn is_unset(&self) -> bool {
        self.0 == 0
    }
    pub fn value(&self) -> u32 {
        self.0 & 0x7FFF_FFFF
    }
}

// 2. The Marker Trait
pub trait IndexKind: Copy + 'static {
    const FLAG: u32;
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct VoxelKind;
impl IndexKind for VoxelKind {
    const FLAG: u32 = 0x0;
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct LodKind;
impl IndexKind for LodKind {
    const FLAG: u32 = 0x8000_0000;
}

// 3. The Typed Wrapper (No PhantomData in the Pod struct!)
#[repr(transparent)]
pub struct StartIndex<K: IndexKind> {
    pub raw: RawIndex,
    _marker: PhantomData<K>,
}
impl<K: IndexKind> StartIndex<K> {
    pub fn new(value: u32) -> Self {
        debug_assert!(value & 0x8000_0000 == 0, "Index value too large!");
        Self {
            raw: RawIndex((value & 0x7FFF_FFFF) | K::FLAG),
            _marker: PhantomData,
        }
    }
    pub fn get_raw(&self) -> RawIndex {
        self.raw
    }
}

pub const UNSET_RAW_INDEX: RawIndex = RawIndex(0);
