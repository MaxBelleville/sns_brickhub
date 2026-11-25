mod brick;
mod cell;
mod consts;
mod cpu;
mod gpu;
mod palette;

use bevy_math::Vec3;
use cpu::CellStore;
use cpu::residency::{SECTION_CELL_LAYERS, Section, SectionId, section_of_cell};
use gpu::GpuStore;
use palette::{InvalidMaterialReference, MaterialPalette};
use std::collections::HashSet;

use cell::{
    BrickHubCell, CellHandle, CellKey, GeneratedCell, GeneratedCellResult, GeneratedCellStatus,
    GenerationToken,
};

use sns_helper::prelude::{GpuCompletion, GpuDevice, GpuError};

use consts::{CellID, MaterialID};
use gpu::allocators::CellAllocation;
use gpu::device::GpuBufferHandles;

pub mod prelude {
    pub use crate::{BrickHub, SectionGeneration};
    pub use crate::cell::{
        BrickHubCell, CellAccessError, CellBuildError, CellHandle, CellKey, GeneratedCell,
        GeneratedCellResult, GeneratedCellStatus, GenerationToken,
    };
    pub use crate::consts::*;
    pub use crate::cpu::density::DensityField;
    pub use crate::cpu::index::LodKind;
    // Clipmap ring schedule: the shared source of truth for residency budgets,
    // camera-driven LOD ladders, and transition-side derivation.
    pub use crate::cpu::residency::{
        classify_ring, classify_section, section_cell, section_of_cell, ClipmapRing, Section,
        SectionId, SectionID, SectionResidencyDecision, CLIPMAP_RINGS, SECTION_CELL_LAYERS,
    };

    pub use crate::cpu::map::{BrickMap, brick_index_to_voxel, voxel_in_brick};
    pub use crate::cpu::material::{
        MaterialKind, color_index, encode_material, material_kind, property_index, solid_material,
    };
    pub use crate::gpu::allocators::CellAllocation;
    pub use crate::gpu::device::GpuBufferHandles;
    pub use crate::gpu::grid::{
        CellDescriptor, CellGridBounds, CellGridError, CellGridTable, serialize_cell_allocations,
    };
    pub use crate::gpu::scheduler::{MigrationBuffer, MigrationResolveError};
    pub use crate::palette::{
        InvalidMaterialReference, MaterialPalette, MaterialPaletteError, MaterialReferenceError,
        PaletteMaterials,
    };
}

#[derive(Debug)]
pub enum BrickHubError {
    Gpu(GpuError),
    Allocation(gpu::allocators::AllocationError),
    MigrationResolve(gpu::scheduler::MigrationResolveError),
    CellPoisoned(CellID),
    MissingAllocation(CellID),
    MissingCell(CellID),
    CellEvicting(CellID),
}

struct CompactionFailure {
    error: BrickHubError,
    uncertain_cells: HashSet<CellID>,
}

impl CompactionFailure {
    fn safe(error: BrickHubError) -> Self {
        Self {
            error,
            uncertain_cells: HashSet::new(),
        }
    }
}

impl From<GpuError> for BrickHubError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

/// Bundle of generation tokens for every cell in a vertical section. With
/// `SECTION_CELL_LAYERS = 1` it holds a single token; the type is already shaped
/// for sections that own several stacked cells.
#[derive(Clone, Debug)]
pub struct SectionGeneration {
    pub section: Section,
    pub tokens: Vec<GenerationToken>,
}

impl SectionGeneration {
    pub fn primary_token(&self) -> GenerationToken {
        self.tokens[0]
    }
}

#[derive(Default)]
pub struct BrickHub {
    cpu: CellStore,
    gpu: GpuStore,
    palette: MaterialPalette,
    allocations_dirty: bool,
}

impl BrickHub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether any CPU cells have pending delta uploads.
    pub fn has_pending_uploads(&self) -> bool {
        self.cpu.has_pending_uploads()
    }

    /// Whether residency or GPU cell allocation ranges changed since last cleared.
    pub fn allocations_dirty(&self) -> bool {
        self.allocations_dirty
    }

    /// True if either uploads or cell allocations need GPU synchronization.
    pub fn is_dirty(&self) -> bool {
        self.has_pending_uploads() || self.allocations_dirty
    }

    /// Clears the allocations dirty flag after reading and serializing them.
    pub fn clear_allocations_dirty(&mut self) {
        self.allocations_dirty = false;
    }

    /// The color and per-kind property buffers packed material references resolve
    /// against; the hub stores and serializes them but never interprets them.
    pub fn material_palette(&self) -> &MaterialPalette {
        &self.palette
    }

    pub fn material_palette_mut(&mut self) -> &mut MaterialPalette {
        &mut self.palette
    }

    /// Validates every material reference used by resident cells against the stored
    /// palette, returning one entry per invalid reference (empty result = all valid).
    pub fn validate_used_materials(&self) -> Result<Vec<InvalidMaterialReference>, BrickHubError> {
        Ok(self
            .cpu
            .used_material_ids()?
            .into_iter()
            .filter_map(|material| {
                self.palette
                    .validate_material(material)
                    .err()
                    .map(|reason| InvalidMaterialReference { material, reason })
            })
            .collect())
    }

    pub fn get_or_admit_cell(
        &mut self,
        id: CellID,
        view_cell: CellID,
        frame: u64,
    ) -> Option<CellHandle> {
        let ring = self.cpu.classify_ring(id, view_cell)?;

        if !self.cpu.contains(id) {
            match self.cpu.admit(id, ring, frame) {
                cpu::residency::ResidencyDecision::Admit => {
                    self.cpu.finish_generation(id);
                    self.cpu.insert_cell(id, BrickHubCell::default());
                    self.gpu.admit_cell(id, ring);
                    self.allocations_dirty = true;
                }

                cpu::residency::ResidencyDecision::EvictAndAdmit(evict_id) => {
                    if !self.unload_cell(evict_id).unwrap_or(false) {
                        self.cpu.restore_admission(id, ring, evict_id, frame);
                        return None;
                    }
                    self.cpu.finish_generation(id);
                    self.cpu.insert_cell(id, BrickHubCell::default());
                    self.gpu.admit_cell(id, ring);
                    self.allocations_dirty = true;
                }

                cpu::residency::ResidencyDecision::Reject => {
                    return None;
                }
            }
        } else if self.cell_key(id).is_some() {
            self.cpu.touch(id, ring, frame);
        } else {
            return None;
        }

        self.cpu.cell_handle(id)
    }

    /// Starts a CPU generation job without reserving residency or GPU memory.
    /// A newer request for the same cell invalidates any earlier token.
    pub fn begin_generation(&mut self, id: CellID, view_cell: CellID) -> Option<GenerationToken> {
        if self.cpu.contains(id) {
            return None;
        }
        self.cpu.begin_generation(id, view_cell)
    }

    /// Invalidates a queued or running generation job if its token is still current.
    pub fn cancel_generation(&mut self, token: GenerationToken) -> bool {
        self.cpu.cancel_generation(token)
    }

    /// Installs completed worker results serially, updating residency and GPU allocation
    /// metadata on the coordinator thread. Results whose tickets or residency are stale drop.
    pub fn commit_generated_cells(
        &mut self,
        generated: impl IntoIterator<Item = GeneratedCell>,
        view_cell: CellID,
        frame: u64,
    ) -> Vec<GeneratedCellResult> {
        let mut results = Vec::new();

        for generated in generated {
            let token = generated.token();
            let id = token.id();

            if !self.cpu.generation_is_current(token) {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::Stale,
                });
                continue;
            }

            self.cpu.finish_generation(id);

            let Some(ring) = self.cpu.classify_ring(id, view_cell) else {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::OutsideResidency,
                });
                continue;
            };

            if self.cell_key(id).is_some() {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::AlreadyResident,
                });
                continue;
            }

            if self.cpu.has_pending_eviction(id) {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::EvictionPending,
                });
                continue;
            }

            let status = match self.cpu.admit(id, ring, frame) {
                cpu::residency::ResidencyDecision::Admit => GeneratedCellStatus::Committed,
                cpu::residency::ResidencyDecision::EvictAndAdmit(evict_id) => {
                    if !self.unload_cell(evict_id).unwrap_or(false) {
                        self.cpu.restore_admission(id, ring, evict_id, frame);
                        results.push(GeneratedCellResult {
                            id,
                            status: GeneratedCellStatus::EvictionPending,
                        });
                        continue;
                    }
                    GeneratedCellStatus::Committed
                }
                cpu::residency::ResidencyDecision::Reject => {
                    results.push(GeneratedCellResult {
                        id,
                        status: GeneratedCellStatus::RejectedByResidency,
                    });
                    continue;
                }
            };
            let cell = generated.cell();
            // Validate materials against stored palette on admission
            for material in cell.map().used_material_ids() {
                if let Err(reason) = self.palette.validate_material(material) {
                    eprintln!(
                        "Invalid material reference in cell {:?}: material {}, reason: {:?}",
                        id, material, reason
                    );
                }
            }
            self.cpu.insert_cell(id, cell);
            self.gpu.admit_cell(id, ring);
            self.allocations_dirty = true;
            results.push(GeneratedCellResult { id, status });
        }
        results
    }

    /// Starts a CPU generation job for a whole vertical section. The section is
    /// classified in 2D, so altitude never changes which sections are generated.
    pub fn begin_section_generation(
        &mut self,
        section: Section,
        view_section: SectionId,
        _frame: u64,
    ) -> Option<SectionGeneration> {
        self.cpu.classify_section(section.id, view_section)?;
        let mut tokens = Vec::with_capacity(section.cell_count as usize);
        for id in section.cell_ids() {
            if self.cpu.contains(id) {
                return None; // generate/evict the column as a unit
            }
            tokens.push(self.cpu.begin_generation_unchecked(id)?);
        }
        if tokens.is_empty() {
            return None;
        }
        Some(SectionGeneration { section, tokens })
    }

    /// Commits worker results using section residency. Results are still
    /// individual `GeneratedCell`s (the worker API is per cell), but LRU and
    /// eviction decisions are made per x/z section.
    pub fn commit_generated_sections(
        &mut self,
        generated: impl IntoIterator<Item = GeneratedCell>,
        view_section: SectionId,
        frame: u64,
    ) -> Vec<GeneratedCellResult> {
        let mut results = Vec::new();
        for generated in generated {
            let token = generated.token();
            let id = token.id();
            let section = Section {
                id: section_of_cell(id),
                y_base: id.y,
                cell_count: SECTION_CELL_LAYERS,
            };

            if !self.cpu.generation_is_current(token) {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::Stale,
                });
                continue;
            }
            self.cpu.finish_generation(id);

            let Some(ring) = self.cpu.classify_section(section.id, view_section) else {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::OutsideResidency,
                });
                continue;
            };
            if self.cell_key(id).is_some() {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::AlreadyResident,
                });
                continue;
            }
            if self.cpu.has_pending_eviction(id) {
                results.push(GeneratedCellResult {
                    id,
                    status: GeneratedCellStatus::EvictionPending,
                });
                continue;
            }

            let status = match self.cpu.admit_section(section.id, ring, frame) {
                cpu::residency::SectionResidencyDecision::Admit => GeneratedCellStatus::Committed,
                cpu::residency::SectionResidencyDecision::EvictAndAdmit(evicted_id) => {
                    let evicted = Section {
                        id: evicted_id,
                        y_base: 0,
                        cell_count: SECTION_CELL_LAYERS,
                    };
                    if !self.unload_section(evicted) {
                        self.cpu
                            .restore_section_admission(section.id, ring, evicted_id, frame);
                        results.push(GeneratedCellResult {
                            id,
                            status: GeneratedCellStatus::EvictionPending,
                        });
                        continue;
                    }
                    GeneratedCellStatus::Committed
                }
                cpu::residency::SectionResidencyDecision::Reject => {
                    results.push(GeneratedCellResult {
                        id,
                        status: GeneratedCellStatus::RejectedByResidency,
                    });
                    continue;
                }
            };

            let cell = generated.cell();
            for material in cell.map().used_material_ids() {
                if let Err(reason) = self.palette.validate_material(material) {
                    eprintln!(
                        "Invalid material reference in cell {:?}: material {}, reason: {:?}",
                        id, material, reason
                    );
                }
            }
            self.cpu.insert_cell(id, cell);
            self.gpu.admit_cell(id, ring);
            self.allocations_dirty = true;
            results.push(GeneratedCellResult { id, status });
        }
        results
    }

    /// Reserves (or touches) a whole section and returns one handle per cell.
    pub fn get_or_admit_section(
        &mut self,
        section: Section,
        view_section: SectionId,
        frame: u64,
    ) -> Option<Vec<CellHandle>> {
        let ring = self.cpu.classify_section(section.id, view_section)?;
        if self.cpu.section_resident(section) {
            self.cpu.touch_section(section.id, ring, frame);
        } else {
            match self.cpu.admit_section(section.id, ring, frame) {
                cpu::residency::SectionResidencyDecision::Admit => {}
                cpu::residency::SectionResidencyDecision::EvictAndAdmit(evicted_id) => {
                    let evicted = Section {
                        id: evicted_id,
                        y_base: 0,
                        cell_count: SECTION_CELL_LAYERS,
                    };
                    if !self.unload_section(evicted) {
                        self.cpu
                            .restore_section_admission(section.id, ring, evicted_id, frame);
                        return None;
                    }
                }
                cpu::residency::SectionResidencyDecision::Reject => return None,
            }
            for id in section.cell_ids() {
                self.cpu.finish_generation(id);
                self.cpu.insert_cell(id, BrickHubCell::default());
                self.gpu.admit_cell(id, ring);
            }
            self.allocations_dirty = true;
        }
        section
            .cell_ids()
            .iter()
            .map(|&id| self.cpu.cell_handle(id))
            .collect()
    }

    fn unload_section(&mut self, section: Section) -> bool {
        let mut ok = true;
        for id in section.cell_ids() {
            if self.cell_key(id).is_some() && !self.unload_cell(id).unwrap_or(false) {
                ok = false;
            }
        }
        ok
    }

    /// Starts eviction. Logical residency and new work stop immediately; GPU ranges remain
    /// allocated until `process_evictions` observes completion of all tracked submissions.
    pub fn begin_evict(&mut self, id: CellID) -> Result<CellKey, BrickHubError> {
        let ring = self
            .gpu
            .allocation(id)
            .ok_or(BrickHubError::MissingAllocation(id))?
            .get_ring();
        let key = self.cpu.begin_evict(id, ring)?;
        self.gpu.cancel_cell_migrations(key);
        self.allocations_dirty = true;
        Ok(key)
    }

    /// Completes evictions whose tracked GPU submissions have finished. This never blocks;
    /// cells still referenced by in-flight work remain pending and keep their allocations.
    pub fn process_evictions(&mut self) -> Result<Vec<CellKey>, BrickHubError> {
        let keys = self.cpu.pending_eviction_keys();
        let mut completed = Vec::new();
        let mut first_error = None;
        for key in keys {
            match self.gpu.complete_eviction(key) {
                Ok(false) => continue,
                Err(error) => {
                    first_error.get_or_insert(BrickHubError::Gpu(error));
                    continue;
                }
                Ok(true) => {}
            }
            self.cpu.complete_eviction(key);
            completed.push(key);
        }
        if !completed.is_empty() {
            self.allocations_dirty = true;
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(completed)
    }

    /// Convenience wrapper: starts eviction and polls completion once. Returns `false` while
    /// submitted GPU work still references the cell.
    pub fn unload_cell(&mut self, id: CellID) -> Result<bool, BrickHubError> {
        let key = if let Some(key) = self
            .cpu
            .pending_eviction_keys()
            .into_iter()
            .find(|key| key.id == id)
        {
            key
        } else if !self.cpu.contains(id) {
            return Ok(true);
        } else {
            self.begin_evict(id)?
        };
        self.process_evictions()?;
        Ok(!self.cpu.has_pending_eviction_key(key))
    }

    /// Returns the handles current at the time of the call. Callers must fetch them again
    /// after any operation that may grow the shared buffers.
    pub fn gpu_buffer_handles(&self) -> Option<GpuBufferHandles> {
        self.gpu.buffer_handles()
    }

    /// Returns the current GPU ranges for a resident cell. Consumers should rebuild any
    /// dispatch metadata after compaction and stop using it once eviction begins.
    pub fn gpu_allocation(&self, id: CellID) -> Option<CellAllocation> {
        self.cpu.cell_key(id)?;
        self.gpu.allocation(id)
    }

    /// Returns every global material ID referenced by an occupied voxel in a resident cell.
    pub fn used_material_ids(&self) -> Result<Vec<MaterialID>, BrickHubError> {
        self.cpu.used_material_ids()
    }

    /// Returns the current GPU allocation ranges for all resident cells.
    pub fn gpu_cell_allocations(&self) -> Result<Vec<(CellID, CellAllocation)>, BrickHubError> {
        self.cpu
            .resident_cell_ids()?
            .into_iter()
            .map(|id| {
                self.gpu
                    .allocation(id)
                    .map(|allocation| (id, allocation))
                    .ok_or(BrickHubError::MissingAllocation(id))
            })
            .collect()
    }

    pub fn set_look_at(&mut self, look_at: Option<Vec3>) {
        self.cpu.set_look_at(look_at);
    }

    pub fn cell_key(&self, id: CellID) -> Option<CellKey> {
        self.cpu.cell_key(id)
    }

    /// Associates a submitted render/compute operation with every cell whose GPU allocation
    /// it references. Eviction waits for this token before reclaiming those ranges.
    pub fn track_gpu_use(
        &mut self,
        cells: &[CellKey],
        completion: Box<dyn GpuCompletion>,
    ) -> Result<(), BrickHubError> {
        if cells.is_empty() {
            completion.wait()?;
            return Ok(());
        }
        let validation = self.cpu.validate_gpu_uses(cells);
        self.gpu.track_gpu_use(cells, completion);
        validation
    }

    /// Coordinates CPU snapshots with the GPU store while keeping the public API stable.
    pub fn flush(&mut self, gpu: &dyn GpuDevice) -> Result<(), BrickHubError> {
        self.gpu.ensure_capacity(gpu)?;
        let snapshots = self.cpu.snapshot_uploads()?;
        let outcome = self.gpu.flush_snapshots(snapshots, &self.cpu, gpu);
        let restore_error = self
            .cpu
            .restore_snapshot_ranges(&outcome.restore_snapshots)
            .err();
        match outcome.error.or(restore_error) {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Compacts GPU allocations independently of ordinary delta flushing. Dirty CPU edits
    /// remain tracked and will upload to the new allocation on the next `flush`.
    pub fn compact(&mut self, gpu: &dyn GpuDevice) -> Result<(), BrickHubError> {
        let result = self.gpu.compact(&self.cpu, gpu);
        self.allocations_dirty = true;
        result
    }
}
