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
    /// one batch. Capacity is checked for the whole batch before anything is pushed
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
