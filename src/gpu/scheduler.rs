use super::{allocators, device};
use sns_helper::prelude::{GpuCompletion, GpuDevice, GpuError};
use std::collections::HashSet;

use crate::{
    CellID, CellKey,
    cpu::{CellUploadSnapshot, UploadData},
};

#[derive(Default)]
pub struct DeviceScheduler {
    migration_queue: Vec<GpuMigrationTask>,
    resolved_migrations: Vec<ResolvedMigration>,
    in_flight: Vec<InFlightUse>,
}

struct InFlightUse {
    cells: HashSet<CellKey>,
    completion: Box<dyn GpuCompletion>,
}

#[derive(Clone, Copy)]
pub(crate) struct GpuMigrationTask {
    pub id: CellID,
    pub key: CellKey,
    pub old_alloc: allocators::CellAllocation,
    pub new_alloc: allocators::CellAllocation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BufferCopy {
    pub src_offset: usize,
    pub dst_offset: usize,
    pub len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedMigration {
    pub id: CellID,
    pub key: CellKey,
    pub bricks: BufferCopy,
    pub materials: BufferCopy,
    pub indices: BufferCopy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MigrationBuffer {
    Bricks,
    Materials,
    Indices,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MigrationResolveError {
    pub id: CellID,
    pub key: CellKey,
    pub buffer: MigrationBuffer,
    pub old_count: usize,
    pub new_count: usize,
}

#[derive(Debug)]
pub(crate) struct MigrationError {
    pub error: GpuError,
    /// A submitted copy reported an error, so its destination cannot be reused safely.
    pub destination_uncertain: bool,
}

impl DeviceScheduler {
    pub fn register_gpu_use(&mut self, cells: &[CellKey], completion: Box<dyn GpuCompletion>) {
        self.in_flight.push(InFlightUse {
            cells: cells.iter().copied().collect(),
            completion,
        });
    }

    pub fn cell_is_idle(&mut self, key: CellKey) -> Result<bool, GpuError> {
        let mut remaining = Vec::with_capacity(self.in_flight.len());
        let mut first_error = None;
        let mut idle = true;
        for usage in self.in_flight.drain(..) {
            if usage.cells.contains(&key) {
                match usage.completion.is_complete() {
                    Ok(true) => {}
                    Ok(false) => {
                        idle = false;
                        remaining.push(usage);
                    }
                    Err(error) => {
                        idle = false;
                        first_error.get_or_insert(error);
                        remaining.push(usage);
                    }
                }
            } else {
                remaining.push(usage);
            }
        }
        self.in_flight = remaining;
        match first_error {
            Some(error) => Err(error),
            None => Ok(idle),
        }
    }

    pub fn cancel_cell_migrations(&mut self, key: CellKey) {
        self.migration_queue.retain(|task| task.key != key);
        self.resolved_migrations.retain(|task| task.key != key);
    }

    pub fn execute_delta_uploads(
        &mut self,
        snapshots: &[CellUploadSnapshot],
        buffers: &device::Buffers,
        gpu: &dyn GpuDevice,
    ) -> Result<(), GpuError> {
        let mut completions: Vec<Box<dyn GpuCompletion>> = Vec::new();
        let mut submission_error = None;
        'snapshots: for snapshot in snapshots {
            for range in &snapshot.ranges {
                let result = match &range.data {
                    UploadData::Bricks(data) => {
                        buffers.bricks().upload_range(gpu, range.destination, &data)
                    }
                    UploadData::Materials(data) => {
                        buffers
                            .materials()
                            .upload_range(gpu, range.destination, &data)
                    }
                    UploadData::Indices(data) => {
                        buffers
                            .indices()
                            .upload_range(gpu, range.destination, &data)
                    }
                };
                match result {
                    Ok(token) => completions.push(token),
                    Err(error) => {
                        submission_error = Some(error);
                        break 'snapshots;
                    }
                }
            }
        }
        finish_submissions(completions, submission_error)
    }

    pub fn enqueue_migration_for_key(
        &mut self,
        reservation: &allocators::CompactionReservation,
        key: CellKey,
    ) {
        self.migration_queue.push(GpuMigrationTask {
            id: reservation.id(),
            key,
            old_alloc: reservation.old(),
            new_alloc: reservation.new(),
        });
    }

    pub fn resolve_migrations(&mut self) -> Result<(), MigrationResolveError> {
        let resolved: Vec<_> = self
            .migration_queue
            .iter()
            .copied()
            .map(GpuMigrationTask::resolve)
            .collect::<Result<_, _>>()?;
        self.migration_queue.clear();
        self.resolved_migrations.extend(resolved);
        Ok(())
    }

    pub fn execute_migrations(
        &mut self,
        buffers: &device::Buffers,
        gpu: &dyn GpuDevice,
    ) -> Result<(), MigrationError> {
        let mut completions: Vec<Box<dyn GpuCompletion>> = Vec::new();
        let mut submission_error = None;
        for migration in &self.resolved_migrations {
            if submission_error.is_some() {
                break;
            }
            let brick_copy = buffers.bricks().copy_from(
                gpu,
                buffers.bricks(),
                migration.bricks.src_offset,
                migration.bricks.dst_offset,
                migration.bricks.len,
            );
            match brick_copy {
                Ok(token) => completions.push(token),
                Err(error) => {
                    submission_error = Some(error);
                    break;
                }
            }
            let material_copy = buffers.materials().copy_from(
                gpu,
                buffers.materials(),
                migration.materials.src_offset,
                migration.materials.dst_offset,
                migration.materials.len,
            );
            match material_copy {
                Ok(token) => completions.push(token),
                Err(error) => {
                    submission_error = Some(error);
                    break;
                }
            }
            let index_copy = buffers.indices().copy_from(
                gpu,
                buffers.indices(),
                migration.indices.src_offset,
                migration.indices.dst_offset,
                migration.indices.len,
            );
            match index_copy {
                Ok(token) => completions.push(token),
                Err(error) => {
                    submission_error = Some(error);
                    break;
                }
            }
        }
        let completion_error = wait_all(completions).err();
        let result = match (submission_error, completion_error) {
            (Some(error), Some(_)) => Err(MigrationError {
                error,
                destination_uncertain: true,
            }),
            (Some(error), None) => Err(MigrationError {
                error,
                destination_uncertain: false,
            }),
            (None, Some(error)) => Err(MigrationError {
                error,
                destination_uncertain: true,
            }),
            (None, None) => Ok(()),
        };
        if result.is_ok() {
            self.resolved_migrations.clear();
        }
        result
    }

    /// Discards transaction work after BrickHub has cancelled or quarantined its reservations.
    pub fn discard_migrations(&mut self) {
        self.migration_queue.clear();
        self.resolved_migrations.clear();
    }
}

impl GpuMigrationTask {
    fn resolve(self) -> Result<ResolvedMigration, MigrationResolveError> {
        let validate = |buffer, old_count, new_count| {
            if old_count == new_count {
                Ok(())
            } else {
                Err(MigrationResolveError {
                    id: self.id,
                    key: self.key,
                    buffer,
                    old_count,
                    new_count,
                })
            }
        };
        validate(
            MigrationBuffer::Bricks,
            self.old_alloc.get_brick_count(),
            self.new_alloc.get_brick_count(),
        )?;
        validate(
            MigrationBuffer::Materials,
            self.old_alloc.get_material_count(),
            self.new_alloc.get_material_count(),
        )?;
        validate(
            MigrationBuffer::Indices,
            self.old_alloc.get_index_count(),
            self.new_alloc.get_index_count(),
        )?;
        Ok(ResolvedMigration {
            id: self.id,
            key: self.key,
            bricks: BufferCopy {
                src_offset: self.old_alloc.get_brick_base(),
                dst_offset: self.new_alloc.get_brick_base(),
                len: self.old_alloc.get_brick_count(),
            },
            materials: BufferCopy {
                src_offset: self.old_alloc.get_material_base(),
                dst_offset: self.new_alloc.get_material_base(),
                len: self.old_alloc.get_material_count(),
            },
            indices: BufferCopy {
                src_offset: self.old_alloc.get_index_base(),
                dst_offset: self.new_alloc.get_index_base(),
                len: self.old_alloc.get_index_count(),
            },
        })
    }
}

fn wait_all(completions: Vec<Box<dyn GpuCompletion>>) -> Result<(), GpuError> {
    let mut first_error = None;
    for completion in completions {
        if let Err(error) = completion.wait() {
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn finish_submissions(
    completions: Vec<Box<dyn GpuCompletion>>,
    submission_error: Option<GpuError>,
) -> Result<(), GpuError> {
    let completion_error = wait_all(completions).err();
    match (submission_error, completion_error) {
        (Some(error), _) => Err(error),
        (None, Some(error)) => Err(error),
        (None, None) => Ok(()),
    }
}
