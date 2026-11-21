use super::allocators::CellAllocation;
use crate::consts::*;
use bevy_math::IVec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellGridError {
    TooManyCells,
    InvalidBounds,
    GridTooLarge,
    OffsetOverflow,
    DescriptorTableOverflow,
    CellOutOfGrid,
    WordValueOutOfRange,
}

/// Axis-aligned bounds of the dense cell-coordinate lookup grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellGridBounds {
    origin: IVec3,
    dimensions: [usize; 3],
    grid_entries: usize,
}

impl CellGridBounds {
    /// Computes the smallest dense grid that covers every allocation's cell coordinate.
    pub fn new(allocations: &[(CellID, CellAllocation)]) -> Result<Self, CellGridError> {
        let Some((first, _)) = allocations.first() else {
            return Ok(Self {
                origin: IVec3::ZERO,
                dimensions: [1, 1, 1],
                grid_entries: 1,
            });
        };

        let mut min = *first;
        let mut max = *first;
        for (id, _) in allocations.iter() {
            min = min.min(*id);
            max = max.max(*id);
        }

        let origin = min;
        let dimensions = [
            cell_span(origin.x, max.x)?,
            cell_span(origin.y, max.y)?,
            cell_span(origin.z, max.z)?,
        ];
        let grid_entries = grid_entry_count(dimensions)?;

        Ok(Self {
            origin,
            dimensions,
            grid_entries,
        })
    }

    pub fn origin(&self) -> IVec3 {
        self.origin
    }

    pub fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }

    pub fn grid_entries(&self) -> usize {
        self.grid_entries
    }

    /// Pure x-fastest mapping from a cell coordinate inside the bounds to a grid index.
    pub fn grid_index(&self, id: CellID) -> Result<usize, CellGridError> {
        let local = [
            cell_local(self.origin.x, id.x)?,
            cell_local(self.origin.y, id.y)?,
            cell_local(self.origin.z, id.z)?,
        ];
        let (dx, dy, _dz) = (self.dimensions[0], self.dimensions[1], self.dimensions[2]);
        if local[0] >= dx || local[1] >= dy || local[2] >= _dz {
            return Err(CellGridError::CellOutOfGrid);
        }
        let z_stride = dx.checked_mul(dy).ok_or(CellGridError::InvalidBounds)?;
        let y_term = local[1]
            .checked_mul(dx)
            .ok_or(CellGridError::InvalidBounds)?;
        let z_term = local[2]
            .checked_mul(z_stride)
            .ok_or(CellGridError::InvalidBounds)?;
        let index = local[0]
            .checked_add(y_term)
            .and_then(|partial| partial.checked_add(z_term))
            .ok_or(CellGridError::InvalidBounds)?;
        (index < self.grid_entries)
            .then_some(index)
            .ok_or(CellGridError::CellOutOfGrid)
    }
}

fn cell_span(min: i32, max: i32) -> Result<usize, CellGridError> {
    i64::from(max)
        .checked_sub(i64::from(min))
        .and_then(|span| span.checked_add(1))
        .and_then(|span| usize::try_from(span).ok())
        .ok_or(CellGridError::InvalidBounds)
}

fn cell_local(origin: i32, coordinate: i32) -> Result<usize, CellGridError> {
    i64::from(coordinate)
        .checked_sub(i64::from(origin))
        .and_then(|local| usize::try_from(local).ok())
        .ok_or(CellGridError::CellOutOfGrid)
}

fn grid_entry_count(dimensions: [usize; 3]) -> Result<usize, CellGridError> {
    dimensions[0]
        .checked_mul(dimensions[1])
        .and_then(|count| count.checked_mul(dimensions[2]))
        .filter(|&count| count <= CELL_GRID_MAX_ENTRIES)
        .ok_or(CellGridError::GridTooLarge)
}

/// One resident cell's ranges in the shared GPU buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CellDescriptor {
    pub brick_base: u32,
    pub material_base: u32,
    pub index_base: u32,
    pub brick_count: u32,
    pub density_base: u32,
    /// Word count of the cell's density field; `0` means the field is absent.
    pub density_count: u32,
}

impl CellDescriptor {
    pub fn from_allocation(allocation: &CellAllocation) -> Result<Self, CellGridError> {
        Ok(Self {
            brick_base: word_value(allocation.get_brick_base())?,
            material_base: word_value(allocation.get_material_base())?,
            index_base: word_value(allocation.get_index_base())?,
            brick_count: word_value(allocation.get_brick_count())?,
            density_base: word_value(allocation.get_density_base())?,
            density_count: word_value(allocation.get_density_count())?,
        })
    }

    pub fn descriptor_words(&self, id: CellID) -> [u32; CELL_DESCRIPTOR_WORDS] {
        let mut words = [0u32; CELL_DESCRIPTOR_WORDS];
        for (offset, coord) in [id.x, id.y, id.z].into_iter().enumerate() {
            words[CELL_DESCRIPTOR_ID_WORD + offset] = coord as u32;
        }
        words[CELL_DESCRIPTOR_BRICK_BASE_WORD] = self.brick_base;
        words[CELL_DESCRIPTOR_MATERIAL_BASE_WORD] = self.material_base;
        words[CELL_DESCRIPTOR_INDEX_BASE_WORD] = self.index_base;
        words[CELL_DESCRIPTOR_BRICK_COUNT_WORD] = self.brick_count;
        words[CELL_DESCRIPTOR_DENSITY_BASE_WORD] = self.density_base;
        words[CELL_DESCRIPTOR_DENSITY_COUNT_WORD] = self.density_count;
        words
    }
}

fn word_value(value: usize) -> Result<u32, CellGridError> {
    u32::try_from(value).map_err(|_| CellGridError::WordValueOutOfRange)
}

/// An assembled cell-grid table: bounds, the dense lookup grid, and cell descriptors.
pub struct CellGridTable {
    bounds: CellGridBounds,
    grid: Vec<u32>,
    descriptors: Vec<(CellID, CellDescriptor)>,
}

impl CellGridTable {
    /// Builds a complete table from every resident cell's allocation.
    pub fn from_allocations(
        allocations: &[(CellID, CellAllocation)],
    ) -> Result<Self, CellGridError> {
        let bounds = CellGridBounds::new(allocations)?;
        let mut table = Self::new(bounds)?;
        for (id, allocation) in allocations.iter() {
            let descriptor = CellDescriptor::from_allocation(allocation)?;
            table.insert_cell(*id, descriptor)?;
        }
        Ok(table)
    }

    /// Creates an empty table over the given bounds, with all grid entries unset.
    pub fn new(bounds: CellGridBounds) -> Result<Self, CellGridError> {
        Ok(Self {
            bounds,
            grid: vec![0u32; bounds.grid_entries()],
            descriptors: Vec::new(),
        })
    }

    pub fn bounds(&self) -> CellGridBounds {
        self.bounds
    }

    pub fn count(&self) -> usize {
        self.descriptors.len()
    }

    /// Registers a cell at `id` with `descriptor`, growing the descriptor table.
    /// The grid entry is overwritten if the coordinate was already occupied.
    pub fn insert_cell(
        &mut self,
        id: CellID,
        descriptor: CellDescriptor,
    ) -> Result<(), CellGridError> {
        let descriptor_index = self
            .descriptors
            .len()
            .checked_add(1)
            .and_then(|index| u32::try_from(index).ok())
            .ok_or(CellGridError::TooManyCells)?;
        let grid_index = self.bounds.grid_index(id)?;
        self.grid[grid_index] = descriptor_index;
        self.descriptors.push((id, descriptor));
        Ok(())
    }

    /// Flat word buffer ready for platform-specific upload.
    pub fn to_words(&self) -> Result<Vec<u32>, CellGridError> {
        let descriptor_offset = CELL_GRID_HEADER_WORDS
            .checked_add(self.bounds.grid_entries())
            .ok_or(CellGridError::OffsetOverflow)?;
        let total_words = descriptor_offset
            .checked_add(
                self.descriptors
                    .len()
                    .checked_mul(CELL_DESCRIPTOR_WORDS)
                    .ok_or(CellGridError::DescriptorTableOverflow)?,
            )
            .ok_or(CellGridError::DescriptorTableOverflow)?;

        let mut words = vec![0u32; total_words];
        let count = word_value(self.descriptors.len())?;
        words[CELL_GRID_COUNT_WORD] = count;
        for (offset, coord) in [
            self.bounds.origin.x,
            self.bounds.origin.y,
            self.bounds.origin.z,
        ]
        .into_iter()
        .enumerate()
        {
            words[CELL_GRID_ORIGIN_WORD + offset] = coord as u32;
        }
        for (offset, dimension) in self.bounds.dimensions().into_iter().enumerate() {
            words[CELL_GRID_DIMENSIONS_WORD + offset] =
                word_value(dimension).map_err(|_| CellGridError::InvalidBounds)?;
        }
        words[CELL_GRID_DESCRIPTOR_OFFSET_WORD] =
            u32::try_from(descriptor_offset).map_err(|_| CellGridError::OffsetOverflow)?;

        for (index, entry) in self.grid.iter().enumerate() {
            words[CELL_GRID_HEADER_WORDS + index] = *entry;
        }
        for (descriptor_index, (id, descriptor)) in self.descriptors.iter().enumerate() {
            let base = descriptor_offset + descriptor_index * CELL_DESCRIPTOR_WORDS;
            words[base..base + CELL_DESCRIPTOR_WORDS]
                .copy_from_slice(&descriptor.descriptor_words(*id));
        }
        Ok(words)
    }

    /// Little-endian bytes ready for a storage buffer upload.
    pub fn to_le_bytes(&self) -> Result<Vec<u8>, CellGridError> {
        let words = self.to_words()?;
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// Serializes resident cell allocations into the platform-agnostic cell-grid table.
pub fn serialize_cell_allocations(
    allocations: &[(CellID, CellAllocation)],
) -> Result<Vec<u8>, CellGridError> {
    CellGridTable::from_allocations(allocations).and_then(|table| table.to_le_bytes())
}
