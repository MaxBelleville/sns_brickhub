pub(crate) mod allocators;
pub(crate) mod device;
pub mod grid;
pub(crate) mod scheduler;
use crate::consts::{BRICKS_PER_BRICKMAP, CellID};

use crate::cpu::{CellUploadSnapshot, StorageType};

use crate::{BrickHubError, CellKey, CellStore, CompactionFailure};
use allocators::DeviceAllocator;
use device::Buffers;
use scheduler::DeviceScheduler;
use sns_helper::prelude::{GpuCompletion, GpuDevice, GpuError};
use std::collections::HashSet;

/// Owns GPU allocations, buffers, and submitted GPU work.
#[derive(Default)]
pub(crate) struct GpuStore {
    buffers: Buffers,
    allocator: DeviceAllocator,
    scheduler: DeviceScheduler,
}

impl GpuStore {
    pub(crate) fn buffer_handles(&self) -> Option<device::GpuBufferHandles> {
        self.buffers.handles()
    }

    pub(crate) fn admit_cell(&mut self, id: CellID, ring: u8) {
        self.allocator.admit_cell(id, ring);
    }

    pub(crate) fn allocation(&self, id: CellID) -> Option<allocators::CellAllocation> {
        self.allocator.get(id).copied()
    }

    pub(crate) fn cancel_cell_migrations(&mut self, key: CellKey) {
        self.scheduler.cancel_cell_migrations(key);
    }

    pub(crate) fn track_gpu_use(&mut self, cells: &[CellKey], completion: Box<dyn GpuCompletion>) {
        self.scheduler.register_gpu_use(cells, completion);
    }

    pub(crate) fn complete_eviction(&mut self, key: CellKey) -> Result<bool, GpuError> {
        if !self.scheduler.cell_is_idle(key)? {
            return Ok(false);
        }
        self.remove_cell(key.id);
        Ok(true)
    }

    pub(crate) fn remove_cell(&mut self, id: CellID) {
        if let Some(allocation) = self.allocator.remove(id) {
            self.allocator.free_cell_allocation(allocation);
        }
    }

    pub(crate) fn ensure_capacity(&mut self, gpu: &dyn GpuDevice) -> Result<(), BrickHubError> {
        self.buffers.ensure_capacity(
            gpu,
            self.allocator.brick_capacity(),
            self.allocator.material_capacity(),
            self.allocator.index_capacity(),
            self.allocator.density_capacity(),
        )?;
        Ok(())
    }

    pub(crate) fn flush_snapshots(
        &mut self,
        snapshots: Vec<CellUploadSnapshot>,
        cells: &CellStore,
        gpu: &dyn GpuDevice,
    ) -> FlushOutcome {
        let mut first_error = None;
        let mut blocked_cells = HashSet::new();
        if let Err(failure) = self.run_compaction(cells, gpu) {
            first_error = Some(failure.error);
            blocked_cells = failure.uncertain_cells;
        }
        blocked_cells.extend(
            snapshots
                .iter()
                .filter(|snapshot| self.allocator.is_gpu_quarantined(snapshot.id))
                .map(|snapshot| snapshot.id),
        );

        let mut upload_snapshots = Vec::with_capacity(snapshots.len());
        let mut restore_snapshots = Vec::new();
        for snapshot in snapshots {
            let idle = match self.scheduler.cell_is_idle(snapshot.key) {
                Ok(idle) => idle,
                Err(error) => {
                    first_error.get_or_insert(BrickHubError::Gpu(error));
                    false
                }
            };
            if blocked_cells.contains(&snapshot.id) || !idle {
                restore_snapshots.push(snapshot);
            } else if !snapshot.ranges.is_empty() {
                upload_snapshots.push(snapshot);
            }
        }

        if let Err(error) = self.bind_snapshot_destinations(&mut upload_snapshots, cells) {
            first_error.get_or_insert(error);
            restore_snapshots.extend(upload_snapshots);
            return FlushOutcome {
                restore_snapshots,
                error: first_error,
            };
        }

        if let Err(error) =
            self.scheduler
                .execute_delta_uploads(&upload_snapshots, &self.buffers, gpu)
        {
            first_error.get_or_insert(error.into());
            restore_snapshots.extend(upload_snapshots);
        }
        FlushOutcome {
            restore_snapshots,
            error: first_error,
        }
    }

    pub(crate) fn compact(
        &mut self,
        cells: &CellStore,
        gpu: &dyn GpuDevice,
    ) -> Result<(), BrickHubError> {
        self.ensure_capacity(gpu)?;
        self.run_compaction(cells, gpu)
            .map_err(|failure| failure.error)
    }

    fn run_compaction(
        &mut self,
        cells: &CellStore,
        gpu: &dyn GpuDevice,
    ) -> Result<(), CompactionFailure> {
        let reservations = self
            .reserve_compaction_if_needed(cells)
            .map_err(CompactionFailure::safe)?;
        let Some(reservations) = reservations else {
            return Ok(());
        };
        self.execute_compaction_reservations(&reservations, cells, gpu)
    }

    fn execute_compaction_reservations(
        &mut self,
        reservations: &[allocators::CompactionReservation],
        cells: &CellStore,
        gpu: &dyn GpuDevice,
    ) -> Result<(), CompactionFailure> {
        if let Err(error) = self.ensure_capacity(gpu) {
            self.allocator
                .cancel_compaction(reservations)
                .map_err(|error| CompactionFailure::safe(BrickHubError::Allocation(error)))?;
            return Err(CompactionFailure::safe(error));
        }
        for reservation in reservations {
            let Some(key) = cells.cell_key(reservation.id()) else {
                self.allocator
                    .cancel_compaction(reservations)
                    .map_err(|error| CompactionFailure::safe(BrickHubError::Allocation(error)))?;
                return Err(CompactionFailure::safe(BrickHubError::MissingCell(
                    reservation.id(),
                )));
            };
            self.scheduler.enqueue_migration_for_key(reservation, key);
        }
        if let Err(error) = self.scheduler.resolve_migrations() {
            let cleanup = self.allocator.cancel_compaction(reservations);
            self.scheduler.discard_migrations();
            cleanup.map_err(|error| CompactionFailure::safe(BrickHubError::Allocation(error)))?;
            return Err(CompactionFailure::safe(BrickHubError::MigrationResolve(
                error,
            )));
        }
        if let Err(failure) = self.scheduler.execute_migrations(&self.buffers, gpu) {
            let uncertain_cells = if failure.destination_uncertain {
                let result = self.allocator.quarantine_compaction(reservations);
                self.scheduler.discard_migrations();
                result
                    .map_err(|error| CompactionFailure::safe(BrickHubError::Allocation(error)))?;
                reservations
                    .iter()
                    .map(|reservation| reservation.id())
                    .collect()
            } else {
                let result = self.allocator.cancel_compaction(reservations);
                self.scheduler.discard_migrations();
                result
                    .map_err(|error| CompactionFailure::safe(BrickHubError::Allocation(error)))?;
                HashSet::new()
            };
            return Err(CompactionFailure {
                error: BrickHubError::Gpu(failure.error),
                uncertain_cells,
            });
        }
        self.allocator
            .commit_compaction(reservations)
            .map_err(|error| CompactionFailure::safe(BrickHubError::Allocation(error)))?;
        Ok(())
    }

    fn bind_snapshot_destinations(
        &self,
        snapshots: &mut [CellUploadSnapshot],
        cells: &CellStore,
    ) -> Result<(), BrickHubError> {
        for snapshot in snapshots {
            let Some(allocation) = self.allocator.get(snapshot.id) else {
                return Err(BrickHubError::MissingAllocation(snapshot.id));
            };
            if !cells.is_resident_key(snapshot.key) {
                return Err(BrickHubError::CellEvicting(snapshot.id));
            }
            for range in &mut snapshot.ranges {
                let base = match range.storage {
                    StorageType::Brick => allocation.get_brick_base(),
                    StorageType::Material => allocation.get_material_base(),
                    StorageType::Indices => allocation.get_index_base(),
                    StorageType::Density => allocation.get_density_base(),
                };
                range.destination = base
                    .checked_add(range.from)
                    .ok_or(BrickHubError::Gpu(GpuError::CapacityOverflow))?;
            }
        }
        Ok(())
    }

    fn reserve_compaction_if_needed(
        &mut self,
        cells: &CellStore,
    ) -> Result<Option<Vec<allocators::CompactionReservation>>, BrickHubError> {
        let ring = 0;
        if !self.allocator.is_fragmented_for(BRICKS_PER_BRICKMAP) {
            return Ok(None);
        }
        let candidates = self.allocator.select_compaction_candidates(ring);
        let mut eligible = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let id = candidate.get_id();
            if cells.has_pending_eviction(id) {
                continue;
            }
            let Some(key) = cells.cell_key(id) else {
                continue;
            };
            match self.scheduler.cell_is_idle(key) {
                Ok(true) => eligible.push(candidate),
                Ok(false) => {}
                Err(error) => return Err(BrickHubError::Gpu(error)),
            }
        }
        if eligible.is_empty() {
            return Ok(None);
        }
        self.allocator
            .reserve_compaction(&eligible)
            .map(Some)
            .map_err(BrickHubError::Allocation)
    }
}

pub(crate) struct FlushOutcome {
    pub(crate) restore_snapshots: Vec<CellUploadSnapshot>,
    pub(crate) error: Option<BrickHubError>,
}
