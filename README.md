# sns_brickhub

`sns_brickhub` stores voxel cells on the CPU, tracks edits, and uploads cell data into shared GPU buffers through an `sns_helper::GpuDevice` backend. It provides cell and brick storage plus residency and upload coordination; it does not generate a whole world, create a renderer, or schedule Bevy render-world systems for you.

## Data layout

A cell is 64×64×64 voxels, arranged as 8×8×8 bricks. Each brick covers 8×8×8 voxels. `CellID` is a `bevy_math::IVec3` cell coordinate, and local voxel coordinates are `UVec3` values in `0..64` on each axis. Material IDs are `u16` values. The map deduplicates material definitions within each cell.

## Create or edit cells

For a detached cell, use one of the builders and then install it through a generation ticket (see below), or use `get_or_admit_cell` to create an empty resident cell directly:

```rust,ignore
use bevy_math::{IVec3, UVec3};
use sns_brickhub::prelude::*;

let mut hub = BrickHub::new();
let id = IVec3::new(0, 0, 0);

let mut generated = BrickHubCell::layered(&[(0..8, 2), (12..16, 5)])?;
generated.insert(UVec3::new(3, 20, 3), 7);

let handle = hub.get_or_admit_cell(id, id, 0).expect("cell is in residency range");
handle.with_mut(|cell| {
    cell.insert(UVec3::new(1, 2, 3), 4);
    cell.remove(UVec3::new(3, 20, 3));
})?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`BrickHubCell` builders:

- `empty()` creates an empty cell.
- `solid(material)` fills every voxel with one material.
- `layered(&[(start..end, material), ...])` fills horizontal Y bands; ranges are half-open and may have gaps.
- `from_voxel_sampler(|local_voxel| Option<material>)` visits each of the 64³ local voxel positions and builds a packed map. Return `None` for empty voxels.

Builders are intended for detached CPU generation. `insert` and `remove` update the map and its dirty ranges; use `CellHandle::with_mut` for edits to a resident cell so the lock covers both together. `CellHandle::with` provides locked read access. Edits through one handle serialize; edits to separate cells can run in parallel.

## Generate cells on worker threads

Start generation on the coordinator with `begin_generation`. Move the returned ticket into a worker, build a complete detached cell there, and return it wrapped in `GeneratedCell`. Collect results on the coordinator and pass them to `commit_generated_cells` as a batch:

```rust,ignore
use bevy_math::IVec3;
use sns_brickhub::prelude::*;

let mut hub = BrickHub::new();
let view_cell = IVec3::ZERO;
let jobs: Vec<_> = cell_ids.into_iter().filter_map(|id| {
    let ticket = hub.begin_generation(id, view_cell)?;
    Some(std::thread::spawn(move || {
        let cell = BrickHubCell::from_voxel_sampler(|voxel| terrain_material(id, voxel));
        GeneratedCell::new(ticket, cell)
    }))
}).collect();

let completed = jobs.into_iter().map(|job| job.join().expect("generation worker panicked"));
let results = hub.commit_generated_cells(completed, view_cell, frame);
```

The example assumes `cell_ids`, `terrain_material`, and `frame` come from your world-generation system. The worker only owns CPU data; it does not touch the hub, residency state, allocators, or GPU. `begin_generation` does not reserve residency or GPU memory. A newer ticket invalidates an older result for the same cell; `cancel_generation` invalidates a ticket explicitly. Commit results report whether each cell was committed, stale, outside residency, already resident, rejected, or waiting for eviction.

For CPU-only use without tickets, call `get_or_admit_cell(id, view_cell, frame)` on the coordinator and build/edit the returned handle. If a worker needs to produce the initial contents, prefer tickets and detached cells to avoid exposing the hub to workers.

## Upload and use GPU data

Implement `sns_helper::GpuDevice` for your backend, then call `flush` on the coordinator thread. It snapshots dirty CPU ranges, ensures shared buffers have enough capacity, performs any needed migration, submits delta uploads, and waits for those submissions to finish before returning:

```rust,ignore
hub.flush(&gpu_device)?;

let buffers = hub.gpu_buffer_handles().expect("flush created GPU buffers");
let key = hub.cell_key(id).expect("cell is resident");
let allocation = hub.gpu_allocation(id).expect("cell has GPU ranges");
// Use `buffers` plus `allocation` to build backend-specific dispatch or draw metadata.
```

The buffers are global typed arrays for bricks, materials, and indices. `CellAllocation` provides each cell's base and count in those arrays. The library does not expose a standalone packed byte blob; the backend consumes the GPU buffer handles and per-cell ranges. The exact interpretation of index entries and brick records must match your shader.

Fetch handles and allocation metadata again after `flush` or `compact`: buffer growth replaces handles, and compaction can move cell ranges. If a render or compute submission continues using a cell's ranges asynchronously, register its completion token with `track_gpu_use(&[key], completion)`. This lets eviction wait before recycling those ranges. Call `begin_evict` to stop new cell access, then poll `process_evictions`; `unload_cell` is a convenience that begins eviction and polls once.

`flush` and `compact` perform GPU work synchronously from the caller's perspective. CPU generation and edits can happen on workers, but residency changes, allocation, compaction, uploads, and GPU completion polling are coordinated through mutable access to `BrickHub`.
