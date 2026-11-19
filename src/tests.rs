use super::*;
use crate::cell::CellAccessError;
use sns_helper::prelude::{GpuBufferHandle, GpuBufferUsage, GpuCompletion};
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    mpsc::{Receiver, Sender, channel},
};
struct CountCompletion {
    waits: Arc<AtomicUsize>,
    fail: bool,
}

struct ManualCompletion(Arc<std::sync::atomic::AtomicBool>);

impl GpuCompletion for ManualCompletion {
    fn is_complete(&self) -> Result<bool, GpuError> {
        Ok(self.0.load(Ordering::SeqCst))
    }

    fn wait(&self) -> Result<(), GpuError> {
        self.0.store(true, Ordering::SeqCst);
        Ok(())
    }
}

impl GpuCompletion for CountCompletion {
    fn is_complete(&self) -> Result<bool, GpuError> {
        Ok(true)
    }

    fn wait(&self) -> Result<(), GpuError> {
        self.waits.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            Err(GpuError::Backend("injected completion failure".into()))
        } else {
            Ok(())
        }
    }
}

struct TestGpu {
    next_handle: AtomicU64,
    upload_count: Arc<AtomicUsize>,
    wait_count: Arc<AtomicUsize>,
    fail_upload_at: Option<usize>,
    copy_count: AtomicUsize,
    fail_copy_at: Option<usize>,
    fail_copy_wait: bool,
    upload_offsets: Mutex<Vec<usize>>,
    first_upload_entered: Option<Sender<()>>,
    resume_first_upload: Option<Mutex<Receiver<()>>>,
    block_copy_at: Option<usize>,
    first_copy_entered: Option<Sender<()>>,
    resume_first_copy: Option<Mutex<Receiver<()>>>,
}

impl TestGpu {
    fn new(fail_upload_at: Option<usize>) -> Self {
        Self {
            next_handle: AtomicU64::new(1),
            upload_count: Arc::new(AtomicUsize::new(0)),
            wait_count: Arc::new(AtomicUsize::new(0)),
            fail_upload_at,
            copy_count: AtomicUsize::new(0),
            fail_copy_at: None,
            fail_copy_wait: false,
            upload_offsets: Mutex::new(Vec::new()),
            first_upload_entered: None,
            resume_first_upload: None,
            block_copy_at: None,
            first_copy_entered: None,
            resume_first_copy: None,
        }
    }

    fn blocking_upload() -> (Self, Receiver<()>, Sender<()>) {
        let (entered_tx, entered_rx) = channel();
        let (resume_tx, resume_rx) = channel();
        let mut gpu = Self::new(None);
        gpu.first_upload_entered = Some(entered_tx);
        gpu.resume_first_upload = Some(Mutex::new(resume_rx));
        (gpu, entered_rx, resume_tx)
    }

    fn completion(&self) -> Box<dyn GpuCompletion> {
        Box::new(CountCompletion {
            waits: Arc::clone(&self.wait_count),
            fail: false,
        })
    }

    fn copy_completion(&self) -> Box<dyn GpuCompletion> {
        Box::new(CountCompletion {
            waits: Arc::clone(&self.wait_count),
            fail: self.fail_copy_wait,
        })
    }
}

impl GpuDevice for TestGpu {
    fn create_buffer(
        &self,
        _size_bytes: usize,
        _usage: GpuBufferUsage,
    ) -> Result<GpuBufferHandle, GpuError> {
        Ok(GpuBufferHandle::new(
            self.next_handle.fetch_add(1, Ordering::SeqCst),
        ))
    }

    fn destroy_buffer(&self, _handle: GpuBufferHandle) -> Result<(), GpuError> {
        Ok(())
    }

    fn upload_buffer(
        &self,
        _handle: GpuBufferHandle,
        _offset_bytes: usize,
        _data: &[u8],
    ) -> Result<Box<dyn GpuCompletion>, GpuError> {
        self.upload_offsets.lock().unwrap().push(_offset_bytes);
        let index = self.upload_count.fetch_add(1, Ordering::SeqCst);
        if index == 0 {
            if let Some(entered) = &self.first_upload_entered {
                let _ = entered.send(());
            }
            if let Some(resume) = &self.resume_first_upload {
                resume
                    .lock()
                    .unwrap()
                    .recv()
                    .expect("test should release blocked upload");
            }
        }
        if self.fail_upload_at == Some(index) {
            return Err(GpuError::Backend("injected upload failure".to_owned()));
        }
        Ok(self.completion())
    }

    fn copy_buffer(
        &self,
        _src: GpuBufferHandle,
        _src_offset_bytes: usize,
        _dst: GpuBufferHandle,
        _dst_offset_bytes: usize,
        _size_bytes: usize,
    ) -> Result<Box<dyn GpuCompletion>, GpuError> {
        let index = self.copy_count.fetch_add(1, Ordering::SeqCst);
        if self.block_copy_at == Some(index) {
            if let Some(entered) = &self.first_copy_entered {
                let _ = entered.send(());
            }
            if let Some(resume) = &self.resume_first_copy {
                resume
                    .lock()
                    .unwrap()
                    .recv()
                    .expect("test should release blocked migration copy");
            }
        }
        if self.fail_copy_at == Some(index) {
            return Err(GpuError::Backend("injected copy submission failure".into()));
        }
        Ok(self.copy_completion())
    }
}

#[test]
fn solid_cell_builds_full_bricks_and_coalesced_deltas() {
    let mut cell = BrickHubCell::solid(17);
    let bricks = cell
        .map()
        .get_brick(sns_helper::prelude::Range::new(0, BRICKS_PER_BRICKMAP));
    assert_eq!(bricks.len(), BRICKS_PER_BRICKMAP);
    assert!(
        bricks
            .iter()
            .all(|brick| brick.occupied_voxel_count() == VOXELS_PER_BRICK)
    );

    let materials = cell.map().get_material(sns_helper::prelude::Range::new(
        0,
        BRICKS_PER_BRICKMAP * VOXELS_PER_BRICK,
    ));
    assert!(materials.iter().all(|&material_index| material_index == 0));
    assert_eq!(cell.map().active_brick_count(), BRICKS_PER_BRICKMAP as u32);
    let indices = cell
        .map()
        .get_indices(sns_helper::prelude::Range::new(0, BRICKS_PER_BRICKMAP));
    assert!(
        indices
            .iter()
            .enumerate()
            .all(|(index, &value)| value == index as u32 + 1)
    );

    assert_eq!(
        cell.delta().take(cpu::StorageType::Brick),
        Some((0, BRICKS_PER_BRICKMAP))
    );
    assert_eq!(
        cell.delta().take(cpu::StorageType::Material),
        Some((0, BRICKS_PER_BRICKMAP * VOXELS_PER_BRICK))
    );
    assert_eq!(
        cell.delta().take(cpu::StorageType::Indices),
        Some((0, BRICKS_PER_BRICKMAP))
    );
}

#[test]
fn worker_generated_cells_commit_as_a_batch_and_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<BrickHubCell>();
    assert_send::<GeneratedCell>();

    let mut hub = BrickHub::new();
    let view = IVec3::ZERO;
    let ids = [
        IVec3::new(0, 0, 0),
        IVec3::new(1, 0, 0),
        IVec3::new(0, 1, 0),
    ];
    let jobs = ids.map(|id| {
        let token = hub.begin_generation(id, view).unwrap();
        std::thread::spawn(move || {
            let cell = BrickHubCell::layered(&[(0..8, 3)]).unwrap();
            GeneratedCell::new(token, cell)
        })
    });
    let results =
        hub.commit_generated_cells(jobs.into_iter().map(|job| job.join().unwrap()), view, 1);

    assert_eq!(results.len(), ids.len());
    assert!(
        results
            .iter()
            .all(|result| result.status == GeneratedCellStatus::Committed)
    );
    assert!(ids.iter().all(|id| hub.cpu.contains(*id)));
}

#[test]
fn newer_generation_invalidates_old_worker_result() {
    let mut hub = BrickHub::new();
    let id = IVec3::ZERO;
    let view = IVec3::ZERO;
    let old = hub.begin_generation(id, view).unwrap();
    let current = hub.begin_generation(id, view).unwrap();

    let results = hub.commit_generated_cells(
        [
            GeneratedCell::new(old, BrickHubCell::solid(1)),
            GeneratedCell::new(current, BrickHubCell::solid(2)),
        ],
        view,
        1,
    );

    assert_eq!(results[0].status, GeneratedCellStatus::Stale);
    assert_eq!(results[1].status, GeneratedCellStatus::Committed);
    let handle = hub.cpu.cell_handle(id).unwrap();
    let (material_index, material_value) = handle
        .with(|cell| {
            let material_index = cell
                .map()
                .get_material(sns_helper::prelude::Range::new(0, 1))[0];
            (
                material_index,
                cell.map().get_material_definition(material_index),
            )
        })
        .unwrap();
    assert_eq!(material_index, 0);
    assert_eq!(material_value, Some(2));
}

#[test]
fn commit_drops_results_that_moved_out_of_residency() {
    let mut hub = BrickHub::new();
    let id = IVec3::ZERO;
    let token = hub.begin_generation(id, IVec3::ZERO).unwrap();
    let results = hub.commit_generated_cells(
        [GeneratedCell::new(token, BrickHubCell::solid(1))],
        IVec3::splat(100),
        2,
    );

    assert_eq!(results[0].status, GeneratedCellStatus::OutsideResidency);
    assert!(!hub.cpu.contains(id));
}

#[test]
fn concurrent_edits_to_one_cell_serialize_and_preserve_deltas() {
    let mut hub = BrickHub::new();
    let id = IVec3::ZERO;
    let handle = hub.get_or_admit_cell(id, id, 0).unwrap();
    let left = handle.clone();
    let right = handle.clone();
    let a = std::thread::spawn(move || {
        left.with_mut(|cell| {
            for x in 0..32 {
                cell.insert(UVec3::new(x, 0, 0), 4);
            }
        })
        .unwrap();
    });
    let b = std::thread::spawn(move || {
        right
            .with_mut(|cell| {
                for x in 32..64 {
                    cell.insert(UVec3::new(x, 0, 0), 4);
                }
            })
            .unwrap();
    });
    a.join().unwrap();
    b.join().unwrap();

    let (material_delta_active, occupied) = handle
        .with_mut(|cell| {
            (
                cell.delta().is_active(cpu::StorageType::Material),
                cell.map()
                    .get_brick(sns_helper::prelude::Range::new(0, 8))
                    .iter()
                    .map(|brick| brick.occupied_voxel_count())
                    .sum::<usize>(),
            )
        })
        .unwrap();
    assert_eq!(occupied, 64);
    assert!(material_delta_active);
}

#[test]
fn edits_during_gpu_submission_remain_dirty_for_the_next_flush() {
    let (gpu, entered, resume) = TestGpu::blocking_upload();
    let gpu = Arc::new(gpu);
    let mut hub = BrickHub::new();
    let id = IVec3::ZERO;
    let handle = hub.get_or_admit_cell(id, id, 0).unwrap();
    handle
        .with_mut(|cell| cell.insert(UVec3::new(0, 0, 0), 7))
        .unwrap();

    let flush_gpu = Arc::clone(&gpu);
    let flushing = std::thread::spawn(move || {
        let result = hub.flush(&*flush_gpu);
        (hub, result)
    });
    entered.recv().unwrap();

    let edit_handle = handle.clone();
    let (edited_tx, edited_rx) = channel();
    std::thread::spawn(move || {
        let result = edit_handle.with_mut(|cell| cell.insert(UVec3::new(1, 0, 0), 8));
        edited_tx.send(result.is_ok()).unwrap();
    });
    assert!(
        edited_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("cell edit should not wait for GPU completion")
    );
    resume.send(()).unwrap();

    let (mut hub, flush_result) = flushing.join().unwrap();
    flush_result.unwrap();
    assert_eq!(gpu.upload_count.load(Ordering::SeqCst), 3);
    hub.flush(&*gpu).unwrap();
    assert_eq!(gpu.upload_count.load(Ordering::SeqCst), 4);
}

#[test]
fn failed_delta_upload_waits_submissions_and_restores_ranges() {
    let (mut gpu, entered, resume) = TestGpu::blocking_upload();
    gpu.fail_upload_at = Some(1);
    let gpu = Arc::new(gpu);
    let mut hub = BrickHub::new();
    let id = IVec3::ZERO;
    let handle = hub.get_or_admit_cell(id, id, 0).unwrap();
    handle
        .with_mut(|cell| cell.insert(UVec3::new(0, 0, 0), 7))
        .unwrap();

    let flush_gpu = Arc::clone(&gpu);
    let flushing = std::thread::spawn(move || {
        let result = hub.flush(&*flush_gpu);
        (hub, result)
    });
    entered.recv().unwrap();
    handle
        .with_mut(|cell| cell.insert(UVec3::new(1, 0, 0), 8))
        .unwrap();
    resume.send(()).unwrap();

    let (mut hub, flush_result) = flushing.join().unwrap();
    assert!(flush_result.is_err());
    assert_eq!(gpu.wait_count.load(Ordering::SeqCst), 1);
    assert_eq!(gpu.upload_count.load(Ordering::SeqCst), 2);

    let retry_gpu = TestGpu::new(None);
    hub.flush(&retry_gpu).unwrap();
    assert_eq!(retry_gpu.upload_count.load(Ordering::SeqCst), 3);
}

#[test]
fn handles_cannot_edit_a_cell_after_unload() {
    let mut hub = BrickHub::new();
    let id = IVec3::ZERO;
    let handle = hub.get_or_admit_cell(id, id, 0).unwrap();
    assert!(hub.unload_cell(id).unwrap());

    assert_eq!(
        handle.with_mut(|cell| cell.insert(UVec3::ZERO, 1)),
        Err(CellAccessError::Inactive(id))
    );
}

#[test]
fn eviction_defers_range_reclamation_until_registered_use_completes() {
    let mut hub = BrickHub::new();
    let id = IVec3::new(2, 3, 4);
    let handle = hub.get_or_admit_cell(id, id, 0).unwrap();
    let key = handle.key();
    let allocation = hub.gpu_allocation(id).unwrap();
    let complete = Arc::new(std::sync::atomic::AtomicBool::new(false));
    hub.track_gpu_use(&[key], Box::new(ManualCompletion(Arc::clone(&complete))))
        .unwrap();

    assert_eq!(hub.begin_evict(id).unwrap(), key);
    assert!(matches!(
        handle.with_mut(|_| ()),
        Err(CellAccessError::Inactive(_))
    ));
    assert!(hub.process_evictions().unwrap().is_empty());
    assert_eq!(hub.gpu.allocation(id), Some(allocation));

    complete.store(true, Ordering::SeqCst);
    assert_eq!(hub.process_evictions().unwrap(), vec![key]);
    assert!(hub.gpu.allocation(id).is_none());

    let next = hub.get_or_admit_cell(id, id, 1).unwrap().key();
    assert_ne!(next.generation, key.generation);
}
