use bevy_math::prelude::{IVec3, UVec3};
pub const CELL_GRID_HEADER_WORDS: usize = 8;
pub const CELL_GRID_MAX_ENTRIES: usize = 1_000_000;
pub const CELL_DESCRIPTOR_WORDS: usize = 10;

// Header word offsets.
pub const CELL_GRID_COUNT_WORD: usize = 0;
/// Words 1..4 hold the cell-grid origin. Encoded with `as u32`; consumers bitcast back
/// to signed coordinates.
pub const CELL_GRID_ORIGIN_WORD: usize = 1;
/// Words 4..7 hold the per-axis grid dimensions.
pub const CELL_GRID_DIMENSIONS_WORD: usize = 4;
/// Word 7 holds the word offset of the first cell descriptor.
pub const CELL_GRID_DESCRIPTOR_OFFSET_WORD: usize = 7;

// Descriptor word offsets.
pub const CELL_DESCRIPTOR_ID_WORD: usize = 0;
pub const CELL_DESCRIPTOR_BRICK_BASE_WORD: usize = 4;
pub const CELL_DESCRIPTOR_MATERIAL_BASE_WORD: usize = 5;
pub const CELL_DESCRIPTOR_INDEX_BASE_WORD: usize = 6;
pub const CELL_DESCRIPTOR_BRICK_COUNT_WORD: usize = 7;
pub const CELL_DESCRIPTOR_DENSITY_BASE_WORD: usize = 8;
/// Word count of the cell's corner density field; `0` means the cell has no density data.
pub const CELL_DESCRIPTOR_DENSITY_COUNT_WORD: usize = 9;

pub const BRICK_SIZE: usize = 8;
pub const VOXELS_PER_BRICK: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE; // 512
pub const MATERIAL_BUCKET_SIZE: usize = VOXELS_PER_BRICK;

pub const BRICKMAP_SIZE: usize = 8; // bricks per axis
pub const BRICKS_PER_BRICKMAP: usize = BRICKMAP_SIZE * BRICKMAP_SIZE * BRICKMAP_SIZE;
pub const CELL_SIZE: usize = BRICKMAP_SIZE * BRICK_SIZE;
/// Zero encodes an empty cell-local grid entry; occupied brick slots are stored as `slot + 1`.
pub const EMPTY_BRICK_INDEX: u32 = 0;
pub const BRICK_INDEX_ENCODING_BIAS: u32 = 1;

/// Density fields are sampled on the voxel grid corners plus one outer layer (a 65x65x65
/// field for a 64x64x64 cell), stored as fp16 with two corners per u32 word.
pub const DENSITY_CORNERS_PER_AXIS: usize = CELL_SIZE + 1;
pub const DENSITY_CORNERS_PER_CELL: usize =
    DENSITY_CORNERS_PER_AXIS * DENSITY_CORNERS_PER_AXIS * DENSITY_CORNERS_PER_AXIS;
pub const DENSITY_WORDS_PER_CELL: usize = DENSITY_CORNERS_PER_CELL.div_ceil(2);

/// Per-brick surface classification packed as 2 bits into the tail of the cell's
/// index buffer: 16 bricks per u32 word, `WEDGE_WORDS_PER_CELL` words after the
/// brick index grid. Wedge coordinates use raw (unflipped) brick coordinates; the
/// brick index grid in the same buffer uses flipped coordinates.
pub const WEDGE_BITS_PER_BRICK: usize = 2;
pub const BRICKS_PER_WEDGE_WORD: usize = 32 / WEDGE_BITS_PER_BRICK;
pub const WEDGE_WORDS_PER_CELL: usize = BRICKS_PER_BRICKMAP / BRICKS_PER_WEDGE_WORD;
/// Total words of the cell-local index buffer: brick lookup grid plus wedge states.
pub const INDEX_WORDS_PER_CELL: usize = BRICKS_PER_BRICKMAP + WEDGE_WORDS_PER_CELL;

pub const KIND_BITS: u16 = 2;
pub const PROPERTY_BITS: u16 = 6;
pub const COLOR_BITS: u16 = 8;

pub const KIND_MASK: u16 = (1 << KIND_BITS) - 1;
pub const PROPERTY_MASK: u16 = ((1 << (KIND_BITS + PROPERTY_BITS)) - 1) & !KIND_MASK;
pub const COLOR_MASK: u16 = u16::MAX & !KIND_MASK & !PROPERTY_MASK;

pub const MAX_PROPERTY_INDEX: u16 = (1 << PROPERTY_BITS) - 1;
pub const MAX_COLOR_INDEX: u16 = (1 << COLOR_BITS) - 1;

pub type CellID = IVec3;
pub type MapID = IVec3;
pub type BrickID = UVec3;

pub type MaterialID = u16;

pub type BrickIndex = u32;
