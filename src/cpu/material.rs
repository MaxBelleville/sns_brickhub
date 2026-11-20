use crate::consts::{
    COLOR_MASK, KIND_BITS, KIND_MASK, MAX_COLOR_INDEX, MAX_PROPERTY_INDEX, MaterialID,
    PROPERTY_BITS, PROPERTY_MASK, VOXELS_PER_BRICK,
};

#[derive(Default)]
pub struct MaterialAllocator {
    material_bucket: Vec<MaterialID>,
    next_bucket_index: usize,
}

impl MaterialAllocator {
    pub fn new(capacity: usize) -> Self {
        Self {
            material_bucket: vec![0; capacity],
            next_bucket_index: 0,
        }
    }

    /// Allocates one brick's per-voxel global material ID entries.
    pub fn alloc_next_block(&mut self) -> u32 {
        let start = self.next_bucket_index;
        let end = start + VOXELS_PER_BRICK;
        assert!(end <= self.material_bucket.len());
        self.next_bucket_index = end;
        start as u32
    }

    pub fn write_material_id(&mut self, index: usize, material_id: MaterialID) {
        self.material_bucket[index] = material_id;
    }

    pub fn get_material_id(&self, index: usize) -> Option<MaterialID> {
        self.material_bucket.get(index).copied()
    }

    pub fn fill_bucket_range(&mut self, start: usize, count: usize, material_id: MaterialID) {
        self.material_bucket[start..start + count].fill(material_id);
    }

    pub fn remove_bucket_index(&mut self, index: usize) {
        self.material_bucket[index] = 0;
    }

    pub fn slice_bucket(&self, base: usize, count: usize) -> &[MaterialID] {
        &self.material_bucket[base..base + count]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MaterialKind {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
}

impl MaterialKind {
    pub fn from_bits(bits: u16) -> Option<Self> {
        match bits {
            0 => Some(Self::Lambertian),
            1 => Some(Self::Metal),
            2 => Some(Self::Dielectric),
            _ => None,
        }
    }

    pub fn bits(self) -> u16 {
        self as u16
    }
}

/// Decodes the BSDF kind; `None` for reserved/invalid bit patterns.
pub fn material_kind(material: MaterialID) -> Option<MaterialKind> {
    MaterialKind::from_bits(material & KIND_MASK)
}

pub fn property_index(material: MaterialID) -> u16 {
    (material & PROPERTY_MASK) >> KIND_BITS
}

pub fn color_index(material: MaterialID) -> u16 {
    (material & COLOR_MASK) >> (KIND_BITS + PROPERTY_BITS)
}

/// Encodes a packed reference. The property index means nothing for Lambertian
/// surfaces, but is still stored so sparse palettes do not need per-kind special cases.
pub fn encode_material(
    kind: MaterialKind,
    property: u16,
    color: u16,
) -> Result<MaterialID, PackedMaterialError> {
    if property > MAX_PROPERTY_INDEX {
        return Err(PackedMaterialError::PropertyOutOfRange(property));
    }
    if color > MAX_COLOR_INDEX {
        return Err(PackedMaterialError::ColorOutOfRange(color));
    }
    Ok(
        (kind.bits() & KIND_MASK)
            | (property << KIND_BITS)
            | (color << (KIND_BITS + PROPERTY_BITS)),
    )
}

/// Convenience constructor for material kinds that carry no per-kind properties.
pub fn solid_material(color: u16) -> Result<MaterialID, PackedMaterialError> {
    encode_material(MaterialKind::Lambertian, 0, color)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedMaterialError {
    PropertyOutOfRange(u16),
    ColorOutOfRange(u16),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_preserves_all_sections() {
        for kind in [
            MaterialKind::Lambertian,
            MaterialKind::Metal,
            MaterialKind::Dielectric,
        ] {
            for property in [0u16, 1, 31, 63] {
                for color in [0u16, 1, 128, 255] {
                    let packed = encode_material(kind, property, color).unwrap();
                    assert_eq!(material_kind(packed), Some(kind));
                    assert_eq!(property_index(packed), property);
                    assert_eq!(color_index(packed), color);
                }
            }
        }
    }

    #[test]
    fn encode_rejects_out_of_range_indices() {
        assert_eq!(
            encode_material(MaterialKind::Metal, 64, 0),
            Err(PackedMaterialError::PropertyOutOfRange(64))
        );
        assert_eq!(
            encode_material(MaterialKind::Metal, 0, 256),
            Err(PackedMaterialError::ColorOutOfRange(256))
        );
    }

    #[test]
    fn defaults_match_the_paper_example() {
        // "type of the material is metal, and the value of the next 6 bits is 32"
        let packed = encode_material(MaterialKind::Metal, 32, 17).unwrap();
        assert_eq!(material_kind(packed), Some(MaterialKind::Metal));
        assert_eq!(property_index(packed), 32);
    }

    #[test]
    fn zero_is_a_valid_lambertian_reference() {
        assert_eq!(material_kind(0), Some(MaterialKind::Lambertian));
        assert_eq!(property_index(0), 0);
        assert_eq!(color_index(0), 0);
    }
}
