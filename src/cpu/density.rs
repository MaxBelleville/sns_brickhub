//! Per-cell corner density fields for iso-surface (Transvoxel-style) extraction.
//!
//! A density field is defined on the voxel grid corners plus one outer layer on
//! every axis: `DENSITY_CORNERS_PER_AXIS^3` corners per cell (65^3 = 274,625).
//! Corners are stored as IEEE binary16 values, two per `u32` word (low half is the
//! even corner index), matching the shader decode in `assets/shaders/share.wesl`.
//!
//! Corner coordinates and storage are in raw cell space (x fastest, then y, then z)
//! with no y-flip. The flipped-convention plumbing used by the occupancy path does
//! not apply to density; consumers sample raw coordinates directly.

use crate::consts::{DENSITY_CORNERS_PER_AXIS, DENSITY_CORNERS_PER_CELL, DENSITY_WORDS_PER_CELL};
use bevy_math::UVec3;
use half::f16;

/// Iso-surface level every density consumer assumes; strict sign conventions:
/// negative is inside the material, positive is outside, zero is on the surface.
pub const DENSITY_ISOLEVEL: f32 = 0.0;

/// Default corner value: unpopulated fields are entirely outside the material.
pub const DENSITY_UNSET: f32 = 1.0;

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

/// How a cell's bricks should be surfaced. This is a render-mode decision stored
/// per brick (iso flag in the index buffer) and is orthogonal to the BSDF kind of
/// the packed material: a Lambertian brick can be meshed, and a metal brick can
/// stay blocky.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_has_a_word_per_corner_pair() {
        assert_eq!(DENSITY_WORDS_PER_CELL, 137_313);
        assert_eq!(
            DENSITY_CORNERS_PER_AXIS * DENSITY_CORNERS_PER_AXIS * DENSITY_CORNERS_PER_AXIS,
            274_625
        );
        let field = DensityField::default();
        assert_eq!(field.word_len(), DENSITY_WORDS_PER_CELL);
        assert_eq!(field.corner_value(0), DENSITY_UNSET);
        assert_eq!(
            field.corner_value(DENSITY_CORNERS_PER_CELL - 1),
            DENSITY_UNSET
        );
    }

    #[test]
    fn set_and_read_roundtrip_at_word_boundaries() {
        let mut field = DensityField::default();
        let last_word = DENSITY_WORDS_PER_CELL - 1;
        // Even corner of the last word, odd corner of the last word, and a middle word.
        for index in [0, 1, 2, 136_000, 274_624] {
            // fp16-exact magnitude (multiples of 0.125 far below the fp16 range).
            let value = -0.125 * ((index % 1000) as f32);
            let word = field.set_corner(corner_position(index), value);
            assert_eq!(field.corner_value(index), f16::from_f32(value).to_f32());
            assert!(word <= last_word);
        }
        // Unset corners around the writes must not be corrupted.
        assert_eq!(field.corner_value(3), DENSITY_UNSET);
        assert_eq!(field.corner_value(274_623), DENSITY_UNSET);
    }

    #[test]
    fn wedge_state_classification_matches_sign_conventions() {
        assert_eq!(WedgeState::from_density(0.1, 2.0), WedgeState::Empty);
        assert_eq!(WedgeState::from_density(-2.0, -0.1), WedgeState::Solid);
        assert_eq!(WedgeState::from_density(-0.1, 0.1), WedgeState::Shell);
        // Zero counts as inside (strict > for empty, <= for solid).
        assert_eq!(WedgeState::from_density(0.0, 0.0), WedgeState::Solid);
    }

    #[test]
    fn fill_from_fn_visits_corners_in_x_fastest_order() {
        // fp16-exact coefficients so decode round-trips bit-exactly.
        let mut field = DensityField::default();
        field.fill_from_fn(|corner| {
            f32::from(corner.x as u16)
                + 0.5 * f32::from(corner.y as u16)
                + 0.25 * f32::from(corner.z as u16)
        });
        assert_eq!(field.corner(UVec3::new(0, 0, 0)), 0.0);
        assert_eq!(field.corner(UVec3::new(64, 1, 2)), 65.0);
        assert_eq!(field.corner(UVec3::new(64, 64, 64)), 112.0);
    }
}
