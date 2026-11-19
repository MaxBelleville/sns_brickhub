mod brick;
mod cell;
mod cpu;
mod gpu;

use cpu::{CellStore, *};
use gpu::{GpuStore, allocators, scheduler};
use std::collections::HashSet;
#[cfg(test)]
use std::sync::{Arc, Mutex};

use cell::{
    BrickHubCell, CellHandle, CellKey, GeneratedCell, GeneratedCellResult, GeneratedCellStatus,
    GenerationToken,
};

use bevy_math::*;
use sns_helper::prelude::{GpuCompletion, GpuDevice, GpuError};

use allocators::CellAllocation;
use gpu::device::GpuBufferHandles;

pub mod prelude {
    pub use crate::allocators::CellAllocation;
    pub use crate::cell::{
        BrickHubCell, CellAccessError, CellBuildError, CellHandle, CellKey, GeneratedCell,
        GeneratedCellResult, GeneratedCellStatus, GenerationToken,
    };
    pub use crate::gpu::device::GpuBufferHandles;
    pub use crate::index::LodKind;
    pub use crate::map::{brick_index_to_voxel, voxel_in_brick};
    pub use crate::scheduler::{MigrationBuffer, MigrationResolveError};

    pub use crate::{
        BRICK_SIZE, BRICKMAP_SIZE, BRICKS_PER_BRICKMAP, BrickHub, BrickHubError, BrickID,
        BrickIndex, CELL_SIZE, CellID, INVALID_BRICK, MATERIAL_BUCKET_SIZE, MapID, MaterialID,
        MaterialIndex, VOXELS_PER_BRICK,
    };
}

pub const BRICK_SIZE: usize = 8;
pub const VOXELS_PER_BRICK: usize = BRICK_SIZE * BRICK_SIZE * BRICK_SIZE; // 512
pub const MATERIAL_BUCKET_SIZE: usize = VOXELS_PER_BRICK;

pub const BRICKMAP_SIZE: usize = 8; // bricks per axis
pub const BRICKS_PER_BRICKMAP: usize = BRICKMAP_SIZE * BRICKMAP_SIZE * BRICKMAP_SIZE;
pub const CELL_SIZE: usize = BRICKMAP_SIZE * BRICK_SIZE;
pub const INVALID_BRICK: u32 = u32::MAX;

pub type CellID = IVec3;
pub type MapID = IVec3;
pub type BrickID = UVec3;
pub type MaterialID = u16;

pub type MaterialIndex = u16;
pub type BrickIndex = u32;

#[derive(Debug)]
pub enum BrickHubError {
    Gpu(GpuError),
    Allocation(allocators::AllocationError),
    MigrationResolve(scheduler::MigrationResolveError),
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

#[derive(Default)]
pub struct BrickHub {
    cpu: CellStore,
    gpu: GpuStore,
}

impl BrickHub {
    pub fn new() -> Self {
        Self::default()
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
                residency::ResidencyDecision::Admit => {
                    self.cpu.finish_generation(id);
                    self.cpu.insert_cell(id, BrickHubCell::default());
                    self.gpu.admit_cell(id, ring);
                }

                residency::ResidencyDecision::EvictAndAdmit(evict_id) => {
                    if !self.unload_cell(evict_id).unwrap_or(false) {
                        self.cpu.restore_admission(id, ring, evict_id, frame);
                        return None;
                    }
                    self.cpu.finish_generation(id);
                    self.cpu.insert_cell(id, BrickHubCell::default());
                    self.gpu.admit_cell(id, ring);
                }

                residency::ResidencyDecision::Reject => {
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
                residency::ResidencyDecision::Admit => GeneratedCellStatus::Committed,
                residency::ResidencyDecision::EvictAndAdmit(evict_id) => {
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
                residency::ResidencyDecision::Reject => {
                    results.push(GeneratedCellResult {
                        id,
                        status: GeneratedCellStatus::RejectedByResidency,
                    });
                    continue;
                }
            };
            self.cpu.insert_cell(id, generated.cell());
            self.gpu.admit_cell(id, ring);
            results.push(GeneratedCellResult { id, status });
        }
        results
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
        self.gpu.compact(&self.cpu, gpu)
    }
}

#[cfg(test)]
mod tests;
