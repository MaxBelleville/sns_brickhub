//! Material palette storage: the color and per-kind property buffers that packed
//! material references (see `packed_material`) resolve against.
//!
//! The palette is stored and serialized by the hub but never interpreted; the
//! renderer owns the WGSL-side section layout. The serialized layout consumed by
//! the demo's shader (`assets/shaders/share.wesl`) is: one header vec4
//! (color/metal/dielectric counts as f32 bits), then vec4 entries for the color
//! buffer, the metal property buffer (x = fuzz), and the dielectric property
//! buffer (x = IOR).

use crate::consts::MaterialID;
use crate::cpu::material::{
    MaterialKind, color_index, encode_material, material_kind, property_index,
};
use bevy_math::prelude::*;

const HEADER_WORDS: usize = 4;
/// 8-bit color index space.
const MAX_COLORS: usize = 256;
/// 6-bit property index space.
const MAX_PROPERTIES: usize = 64;

/// Color + per-kind property definitions the packed voxel references point into.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialPalette {
    colors: Vec<Vec4>,
    metal_fuzz: Vec<f32>,
    dielectric_ior: Vec<f32>,
}

impl MaterialPalette {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn colors(&self) -> &[Vec4] {
        &self.colors
    }

    pub fn metal_fuzz(&self) -> &[f32] {
        &self.metal_fuzz
    }

    pub fn dielectric_ior(&self) -> &[f32] {
        &self.dielectric_ior
    }

    /// Registers a color and returns its 8-bit color index.
    pub fn add_color(&mut self, color: Vec4) -> Result<u16, MaterialPaletteError> {
        if self.colors.len() >= MAX_COLORS {
            return Err(MaterialPaletteError::ColorCapacity);
        }
        self.colors.push(color);
        Ok((self.colors.len() - 1) as u16)
    }

    /// Registers a pure-diffuse surface and returns its packed material reference.
    /// Lambertian has no property buffer, so the property index stays 0.
    pub fn add_lambertian(&mut self, color: Vec4) -> Result<MaterialID, MaterialPaletteError> {
        let color_index = self.add_color(color)?;
        Ok(encode_material(MaterialKind::Lambertian, 0, color_index)
            .expect("validated indices always encode"))
    }

    /// Registers a mirror-like metal surface (fuzz 0 = perfect mirror) and returns
    /// its packed material reference.
    pub fn add_metal(
        &mut self,
        color: Vec4,
        fuzz: f32,
    ) -> Result<MaterialID, MaterialPaletteError> {
        if self.metal_fuzz.len() >= MAX_PROPERTIES {
            return Err(MaterialPaletteError::PropertyCapacity);
        }
        let color_index = self.add_color(color)?;
        self.metal_fuzz.push(fuzz);
        let property = (self.metal_fuzz.len() - 1) as u16;
        Ok(encode_material(MaterialKind::Metal, property, color_index)
            .expect("validated indices always encode"))
    }

    /// Registers a glass-like dielectric surface and returns its packed material
    /// reference.
    pub fn add_dielectric(
        &mut self,
        color: Vec4,
        ior: f32,
    ) -> Result<MaterialID, MaterialPaletteError> {
        if self.dielectric_ior.len() >= MAX_PROPERTIES {
            return Err(MaterialPaletteError::PropertyCapacity);
        }
        let color_index = self.add_color(color)?;
        self.dielectric_ior.push(ior);
        let property = (self.dielectric_ior.len() - 1) as u16;
        Ok(
            encode_material(MaterialKind::Dielectric, property, color_index)
                .expect("validated indices always encode"),
        )
    }

    /// Registers one surface of every kind and returns their packed references in
    /// one batch. Capacity is checked for the whole batch before anything is pushed,
    /// so a failure leaves the palette untouched and callers handle a single error.
    pub fn add_palette(
        &mut self,
        lambertian_color: Vec4,
        metal_color: Vec4,
        metal_fuzz: f32,
        dielectric_color: Vec4,
        dielectric_ior: f32,
    ) -> Result<PaletteMaterials, MaterialPaletteError> {
        if self.colors.len() + 3 > MAX_COLORS {
            return Err(MaterialPaletteError::ColorCapacity);
        }
        if self.metal_fuzz.len() + 1 > MAX_PROPERTIES {
            return Err(MaterialPaletteError::PropertyCapacity);
        }
        if self.dielectric_ior.len() + 1 > MAX_PROPERTIES {
            return Err(MaterialPaletteError::PropertyCapacity);
        }

        let lambertian = self
            .add_lambertian(lambertian_color)
            .expect("batch capacity was checked");
        let metal = self
            .add_metal(metal_color, metal_fuzz)
            .expect("batch capacity was checked");
        let dielectric = self
            .add_dielectric(dielectric_color, dielectric_ior)
            .expect("batch capacity was checked");
        Ok(PaletteMaterials {
            lambertian,
            metal,
            dielectric,
        })
    }

    /// Validates one packed reference against the stored palette sections.
    pub fn validate_material(&self, material: MaterialID) -> Result<(), MaterialReferenceError> {
        let Some(kind) = material_kind(material) else {
            return Err(MaterialReferenceError::ReservedKind);
        };
        if usize::from(color_index(material)) >= self.colors.len() {
            return Err(MaterialReferenceError::ColorOutOfRange);
        }
        let property = usize::from(property_index(material));
        let valid_property = match kind {
            MaterialKind::Lambertian => property == 0,
            MaterialKind::Metal => property < self.metal_fuzz.len(),
            MaterialKind::Dielectric => property < self.dielectric_ior.len(),
        };
        if !valid_property {
            return Err(MaterialReferenceError::PropertyOutOfRange);
        }
        Ok(())
    }

    /// Flat word buffer in the platform-agnostic section layout documented above.
    pub fn to_words(&self) -> Vec<u32> {
        let section_vec4s = self.colors.len() + self.metal_fuzz.len() + self.dielectric_ior.len();
        let mut words = vec![0u32; HEADER_WORDS + section_vec4s * 4];
        // Consumers read the header as f32 values, so counts are stored as float bits.
        words[0] = (self.colors.len() as f32).to_bits();
        words[1] = (self.metal_fuzz.len() as f32).to_bits();
        words[2] = (self.dielectric_ior.len() as f32).to_bits();

        let mut cursor = HEADER_WORDS;
        for color in self.colors.iter() {
            for (offset, value) in color.to_array().iter().enumerate() {
                words[cursor + offset] = value.to_bits();
            }
            cursor += 4;
        }
        for fuzz in self.metal_fuzz.iter() {
            words[cursor] = fuzz.to_bits();
            cursor += 4;
        }
        for ior in self.dielectric_ior.iter() {
            words[cursor] = ior.to_bits();
            cursor += 4;
        }
        words
    }

    /// Little-endian bytes ready for a storage buffer upload.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let words = self.to_words();
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialPaletteError {
    ColorCapacity,
    PropertyCapacity,
}

/// The packed references produced by one `add_palette` batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaletteMaterials {
    pub lambertian: MaterialID,
    pub metal: MaterialID,
    pub dielectric: MaterialID,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialReferenceError {
    ReservedKind,
    ColorOutOfRange,
    PropertyOutOfRange,
}

/// One resident material reference rejected by palette validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidMaterialReference {
    pub material: MaterialID,
    pub reason: MaterialReferenceError,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-crafts a packed reference the palette API would not hand out, for
    /// validation tests of ghost property/color indices.
    fn probe(kind: MaterialKind, property: u16, color: u16) -> MaterialID {
        (kind as u16) | (property << 2) | (color << 8)
    }

    #[test]
    fn added_materials_roundtrip_through_their_references() {
        let mut palette = MaterialPalette::new();
        let lambertian = palette.add_lambertian(vec4(0.2, 0.35, 0.8, 1.0)).unwrap();
        let metal = palette
            .add_metal(vec4(0.85, 0.55, 0.35, 1.0), 0.05)
            .unwrap();
        let dielectric = palette
            .add_dielectric(vec4(0.9, 0.92, 0.95, 1.0), 1.5)
            .unwrap();

        assert_eq!(material_kind(lambertian), Some(MaterialKind::Lambertian));
        assert_eq!(material_kind(metal), Some(MaterialKind::Metal));
        assert_eq!(material_kind(dielectric), Some(MaterialKind::Dielectric));
        assert_eq!(palette.validate_material(lambertian), Ok(()));
        assert_eq!(palette.validate_material(metal), Ok(()));
        assert_eq!(palette.validate_material(dielectric), Ok(()));
    }

    #[test]
    fn validation_rejects_references_outside_the_stored_palette() {
        let mut palette = MaterialPalette::new();
        palette.add_metal(vec4(0.5, 0.5, 0.5, 1.0), 0.1).unwrap();

        // Property index 1 matches the paper's example bit pattern but has no entry.
        assert_eq!(
            palette.validate_material(probe(MaterialKind::Metal, 1, 0)),
            Err(MaterialReferenceError::PropertyOutOfRange)
        );
        // Color index 3 is beyond the single registered color.
        assert_eq!(
            palette.validate_material(probe(MaterialKind::Lambertian, 0, 3)),
            Err(MaterialReferenceError::ColorOutOfRange)
        );
    }

    #[test]
    fn lambertian_references_must_not_use_the_property_buffer() {
        let mut palette = MaterialPalette::new();
        let reference = palette.add_lambertian(vec4(1.0, 0.0, 0.0, 1.0)).unwrap();
        assert_eq!(palette.validate_material(reference), Ok(()));
        assert_eq!(
            palette.validate_material(probe(MaterialKind::Lambertian, 1, 0)),
            Err(MaterialReferenceError::PropertyOutOfRange)
        );
    }

    #[test]
    fn add_palette_registers_one_surface_per_kind() {
        let mut palette = MaterialPalette::new();
        let surfaces = palette
            .add_palette(
                vec4(0.2, 0.35, 0.8, 1.0),
                vec4(0.85, 0.55, 0.35, 1.0),
                0.05,
                vec4(0.9, 0.92, 0.95, 1.0),
                1.5,
            )
            .unwrap();

        assert_eq!(palette.validate_material(surfaces.lambertian), Ok(()));
        assert_eq!(palette.validate_material(surfaces.metal), Ok(()));
        assert_eq!(palette.validate_material(surfaces.dielectric), Ok(()));
        assert_eq!(
            material_kind(surfaces.lambertian),
            Some(MaterialKind::Lambertian)
        );
        assert_eq!(material_kind(surfaces.metal), Some(MaterialKind::Metal));
        assert_eq!(
            material_kind(surfaces.dielectric),
            Some(MaterialKind::Dielectric)
        );
    }

    #[test]
    fn add_palette_leaves_the_palette_untouched_on_capacity_failure() {
        let mut palette = MaterialPalette::new();
        // Leave room for fewer than a full batch's colors.
        for _ in 0..MAX_COLORS - 2 {
            palette.add_color(vec4(1.0, 1.0, 1.0, 1.0)).unwrap();
        }

        let result = palette.add_palette(
            vec4(1.0, 0.0, 0.0, 1.0),
            vec4(1.0, 1.0, 0.0, 1.0),
            0.25,
            vec4(0.0, 0.0, 1.0, 1.0),
            1.5,
        );
        assert_eq!(result, Err(MaterialPaletteError::ColorCapacity));
        assert_eq!(palette.colors().len(), MAX_COLORS - 2);
        assert_eq!(palette.metal_fuzz().len(), 0);
        assert_eq!(palette.dielectric_ior().len(), 0);
    }

    #[test]
    fn sections_serialize_in_the_documented_order() {
        let mut palette = MaterialPalette::new();
        palette.add_lambertian(vec4(0.1, 0.2, 0.3, 1.0)).unwrap();
        palette.add_metal(vec4(0.4, 0.5, 0.6, 1.0), 0.25).unwrap();

        let words = palette.to_words();
        assert_eq!(words[0], (2f32).to_bits()); // 2 colors
        assert_eq!(words[1], (1f32).to_bits()); // 1 metal property
        assert_eq!(words[2], (0f32).to_bits()); // 0 dielectric properties

        let metal_section = HEADER_WORDS + 2 * 4;
        assert_eq!(f32::from_bits(words[metal_section]), 0.25);
        // Header (1 vec4) + 2 colors + 1 property = 4 vec4s = 16 words = 64 bytes.
        assert_eq!(palette.to_le_bytes().len(), 64);
    }
}
