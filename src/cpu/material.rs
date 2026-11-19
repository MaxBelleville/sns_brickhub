use std::collections::HashMap;

use crate::{MaterialID, MaterialIndex, VOXELS_PER_BRICK};

#[derive(Default)]
pub struct MaterialAllocator {
    material_bucket: Vec<MaterialID>,
    materials: Vec<MaterialID>, // flat array
    /// Lookup table for deduplication (CPU-only)
    material_lookup: HashMap<MaterialID, MaterialIndex>,
    next_bucket_index: usize,
}

impl MaterialAllocator {
    pub fn new(capacity: usize) -> Self {
        Self {
            material_bucket: vec![0; capacity],
            materials: Vec::new(),
            material_lookup: HashMap::new(),
            next_bucket_index: 0,
        }
    }
    /// Allocates one brick's 512 per-voxel material-index entries.
    pub fn alloc_next_block(&mut self) -> u32 {
        let start = self.next_bucket_index;
        let end = start + VOXELS_PER_BRICK;
        assert!(end <= self.material_bucket.len());
        self.next_bucket_index = end;
        start as u32
    }
    pub fn write_shared_mat(&mut self, material: MaterialID) -> MaterialIndex {
        if let Some(&idx) = self.material_lookup.get(&material) {
            return idx;
        }

        let idx = self.materials.len() as MaterialIndex;
        assert!(idx < u16::MAX);

        self.materials.push(material);
        self.material_lookup.insert(material, idx);
        idx
    }

    pub fn get_material(&self, index: MaterialIndex) -> Option<MaterialID> {
        self.materials.get(index as usize).copied()
    }

    pub fn write_bucket_index(&mut self, index: usize, material_index: MaterialIndex) {
        self.material_bucket[index] = material_index;
    }

    pub fn fill_bucket_range(&mut self, start: usize, count: usize, material: MaterialID) {
        self.material_bucket[start..start + count].fill(material);
    }

    pub fn remove_bucket_index(&mut self, index: usize) {
        self.material_bucket[index] = 0;
    }

    pub fn slice_bucket(&self, base: usize, count: usize) -> &[MaterialID] {
        &self.material_bucket[base..base + count]
    }
}
