use crate::consts::{
    DENSITY_CORNERS_PER_AXIS, DENSITY_CORNERS_PER_CELL, DENSITY_ISOLEVEL, DENSITY_UNSET,
    DENSITY_WORDS_PER_CELL,
};
use bevy_math::UVec3;
use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WedgeState {
    /// Every corner of the brick is outside the material.
    Empty = 0,
    /// Every corner of the brick is inside the material.
    Solid = 1,
    /// The iso surface passes through the brick.
    Shell = 2,
}

impl WedgeState {
    pub fn from_density(minimum: f32, maximum: f32) -> Self {
        if minimum > DENSITY_ISOLEVEL {
            Self::Empty
        } else if maximum <= DENSITY_ISOLEVEL {
            Self::Solid
        } else {
            Self::Shell
        }
    }

    pub fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            0 => Some(Self::Empty),
            1 => Some(Self::Solid),
            2 => Some(Self::Shell),
            _ => None,
        }
    }

    pub fn bits(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshMode {
    /// Render as voxel cubes through the occupancy/density raymarcher.
    Blocky,
    /// Flag shell bricks so the Transvoxel meshing dispatch polygonizes them.
    Iso,
}

fn corner_index(corner: UVec3) -> usize {
    let axis = DENSITY_CORNERS_PER_AXIS;
    (corner.x as usize) + (corner.y as usize) * axis + (corner.z as usize) * axis * axis
}

/// Raw corner coordinates for a linear (x-fastest) corner index.
pub fn corner_position(index: usize) -> UVec3 {
    let axis = DENSITY_CORNERS_PER_AXIS;
    UVec3::new(
        (index % axis) as u32,
        ((index / axis) % axis) as u32,
        (index / (axis * axis)) as u32,
    )
}

/// Dense per-cell corner field, two fp16 corners per word.
pub struct DensityField {
    words: Vec<u32>,
}

impl Default for DensityField {
    fn default() -> Self {
        Self {
            words: vec![default_word(); DENSITY_WORDS_PER_CELL],
        }
    }
}

fn default_word() -> u32 {
    pack_word(DENSITY_UNSET, DENSITY_UNSET)
}

fn pack_word(even: f32, odd: f32) -> u32 {
    u32::from(f16::from_f32(even).to_bits()) | u32::from(f16::from_f32(odd).to_bits()) << 16
}

impl DensityField {
    /// Number of u32 words the field occupies (both host- and GPU-side).
    pub fn word_count() -> usize {
        DENSITY_WORDS_PER_CELL
    }

    /// Rewrites every corner; `sample` receives raw corner coordinates in `0..DENSITY_CORNERS_PER_AXIS`.
    pub fn fill_from_fn(&mut self, mut sample: impl FnMut(UVec3) -> f32) {
        let corner_count = DENSITY_CORNERS_PER_CELL;
        for word_index in 0..DENSITY_WORDS_PER_CELL {
            let even_index = word_index * 2;
            let even = sample(corner_position(even_index));
            let odd = if even_index + 1 < corner_count {
                sample(corner_position(even_index + 1))
            } else {
                DENSITY_UNSET
            };
            self.words[word_index] = pack_word(even, odd);
        }
    }

    /// Rewrites every corner from a full slice of f32 values in linear corner order.
    pub fn fill_from_values(&mut self, values: &[f32]) {
        assert_eq!(values.len(), DENSITY_CORNERS_PER_CELL);
        for word_index in 0..DENSITY_WORDS_PER_CELL {
            let even_index = word_index * 2;
            let even = values[even_index];
            let odd = if even_index + 1 < DENSITY_CORNERS_PER_CELL {
                values[even_index + 1]
            } else {
                DENSITY_UNSET
            };
            self.words[word_index] = pack_word(even, odd);
        }
    }

    /// Raw corner value at a raw corner index (x-fastest linear order).
    pub fn corner_value(&self, index: usize) -> f32 {
        let word = self.words[index / 2];
        let bits = if index % 2 == 0 {
            (word & 0xffff) as u16
        } else {
            (word >> 16) as u16
        };
        f16::from_bits(bits).to_f32()
    }

    /// Raw corner value at raw corner coordinates.
    pub fn corner(&self, corner: UVec3) -> f32 {
        self.corner_value(corner_index(corner))
    }

    /// Overwrites one corner and reports the word index that changed, for delta tracking.
    pub fn set_corner(&mut self, corner: UVec3, value: f32) -> usize {
        let index = corner_index(corner);
        let word = index / 2;
        let replacement = if index % 2 == 0 {
            let odd = f16::from_bits((self.words[word] >> 16) as u16).to_f32();
            pack_word(value, odd)
        } else {
            let even = f16::from_bits((self.words[word] & 0xffff) as u16).to_f32();
            pack_word(even, value)
        };
        self.words[word] = replacement;
        word
    }

    /// Half-open word range `[from, to)` of the packed corner words.
    pub fn word_slice(&self, from: usize, to: usize) -> &[u32] {
        &self.words[from..to]
    }

    pub fn word_len(&self) -> usize {
        self.words.len()
    }
}
