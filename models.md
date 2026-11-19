# BrickHub Models and Current Architecture

This document describes the implementation in `src/` and its concurrency boundary. CPU generation can run in parallel for detached cells; the `BrickHub` coordinator still serializes residency, GPU allocation, compaction, and uploads.

## Ownership model

| Model | Current owner | Role |
| --- | --- | --- |
| `BrickHub` | `lib.rs` | Public façade. Coordinates operations that span CPU residency/cells and GPU allocation/submission. |
| `CellStore` | `cpu/mod.rs` | Owns cells, generation tickets, cell lifetimes, residency, and CPU upload snapshots. |
| `GpuStore` | `gpu/mod.rs` | Owns global buffers, device allocations, scheduling, uploads, compaction, and deferred GPU-use reclamation. |
| `BrickHubCell` | `cell.rs` | Owns one `BrickMap` and its `CellDelta`; supports detached bulk construction and tracked edits. |
| `BrickMap` | `map.rs` | Stores brick occupancy, per-voxel material bucket indices, brick-to-index mapping, brick status, and active-brick count. |
| `DeviceAllocator` | `gpu/allocators.rs` | Maps `CellID` to `CellAllocation` and owns independent range allocators for bricks, materials, and indices. |
| `ResidencyManager` | `cpu/residency.rs` | Classifies clipmap rings and tracks a per-ring LRU with cell budgets. |
| `DeviceScheduler` | `gpu/scheduler.rs` | Tracks GPU-use completion and submits uploads/copies on the coordinator thread. |
| `Buffers` | `gpu/device.rs` | Holds the global GPU brick, material, and index buffers. |

Cell and allocation state are separate. `CellStore` is the owner of CPU cells and logical residency; `GpuStore` is the owner of GPU allocations and resources. They are private implementation details and are coordinated by `BrickHub` so admission, eviction, flush, and compaction preserve cross-store invariants. Resident cells are stored behind `Arc<Mutex<CellSlot>>`; `CellHandle::with_mut` holds one cell lock for a batch of operations, keeping map edits and delta registration together. Different cells can be edited on separate threads; edits to one cell serialize. `begin_generation` and `commit_generated_cells` support detached workers and reject stale results. `MaterialAllocator`'s atomic cursor is not the synchronization mechanism for map edits.

## CPU map and deltas

`BrickMap` is an 8×8×8 grid of bricks per cell. Each brick represents 8×8×8 voxels and stores occupancy bits, LOD masks, and a raw material-bucket start index. The spatial `IndexGrid` maps brick coordinates to brick slots. Material buckets hold per-voxel indices into a CPU-side deduplicated material table.

`BrickHubCell::insert` and `remove` apply `BrickEditEvent`s to three coalesced cell-local dirty ranges: bricks, materials, and indices. Detached workers can instead use `empty`, `solid`, `layered`, or `from_voxel_sampler` to construct a complete cell without per-voxel edit events. A cell spans 64 voxels on each axis; the sampler visits each local voxel once, skips empty bricks, and packs occupied bricks. New cells mark the full zero-initialized index grid dirty so recycled GPU ranges cannot retain stale brick references; brick and material dirty ranges cover only packed occupied data. Material IDs are deduplicated within each generated cell. Resident edits run through a `CellHandle` lock. `BrickHub::flush` takes the ranges and copies the matching CPU slices into owned snapshots while holding each cell lock once. GPU submission happens after those locks are released.

The solid-cell fast path constructs all 512 bricks, their material blocks, occupancy/LOD masks, index-grid entries, and the three full-cell dirty ranges directly. The generated-mask path handles empty, full, and partial bricks directly; empty bricks consume no brick or material slots. Layered cells are represented as vertical half-open material bands with optional gaps. Brick storage is packed, while the fixed per-cell index grid retains spatial locations. Remaining CPU-map concerns include:

- Insertion raises the occupancy bit without checking whether that voxel was already occupied, while material bucket indices are deduplicated independently. Confirm intended overwrite semantics.
- When a brick becomes empty, its slot remains part of the linear brick storage and is marked inactive; this code does not reclaim/reuse that brick slot.

## GPU allocation and buffers

Every admitted cell receives fixed ranges sized for a full brick map: `BRICKS_PER_BRICKMAP` brick records, that count times `VOXELS_PER_BRICK` material entries, and `BRICKS_PER_BRICKMAP` index entries. `RangeAllocator` first-fits free ranges and merges adjacent ranges. Its `next` value is the high-water capacity; freeing ranges does not reduce that value.

`BrickHub::flush` coordinates snapshots, eligible compaction, and delta uploads through the two stores. Snapshots own CPU payloads and cell-local ranges; GPU destinations are bound after migration so they point at the active allocation. The scheduler validates matching old/new counts for bricks, materials, and indices and submits the operations; allocation ownership stays in `GpuStore`. Successful synchronous execution drains resolved work; `GpuStore` discards retained work only after cancelling or quarantining failed reservations. `BrickHub::compact` uses the same migration path without consuming deltas. A clean migration failure cancels reservations and lets flush upload to the old allocation before returning the migration error. If a submitted copy reports a completion error, source and destination ranges are quarantined; flush restores deltas for those cells and does not upload to them. Also verify that uploaded index encoding matches the consuming shader.

## Residency

`classify_ring` uses Chebyshev distance and the configured radii/budgets (levels 0–2, radii 4/8/16, dimensions 9³/17³/33³). Admission updates the ring LRU; if full, it selects the least recently used cell. A replacement admission waits until that cell’s asynchronous eviction completes, while unrelated cells continue to be processed.

Each cell admission receives a unique `CellKey { id, generation }`. Handles, upload snapshots, migration tasks, and GPU-use completion records carry that key, so delayed work for an earlier occupant of a coordinate cannot attach to a later admission. `begin_evict` marks a slot `Evicting`, removes logical residency, and cancels unsubmitted migration work. `process_evictions` polls completion tokens without blocking and only then removes CPU state and frees GPU ranges. Existing handles reject edits as soon as eviction starts. Render/compute callers obtain current handles and allocations through `gpu_buffer_handles` and `gpu_allocation`, then register each submission with `track_gpu_use`; callers must fetch handles again after a capacity-growing operation.

The scheduler tracks completion tokens per cell generation rather than a single numeric `last_use` value. This covers independently completed queues when each use is registered with its own token. Uploads and migrations are held back for cells with incomplete external GPU use; compaction only selects idle cells.

`GpuStore` ensures buffer capacity during flush or compaction, after allocator high-water marks are known. Growth creates a replacement buffer, copies the existing contents, waits for the copy completion, swaps the owned wrapper, and then destroys the old wrapper. The helper backend contract still requires `destroy_buffer` to defer physical destruction when earlier asynchronous submissions may reference the handle.

## Flush and compaction

The coordinated flush sequence is:

1. Ensure global GPU buffers cover current allocator capacity.
2. Lock each resident cell once, take dirty ranges, and copy CPU slices into owned snapshots without binding GPU destinations.
3. Reserve eligible migrations, queue tasks in the scheduler, resolve typed copies, wait, and commit or roll back in the allocator.
4. Bind upload destinations from the resulting active allocations, submit serially, and wait for completion.

If snapshot collection or an upload fails, consumed ranges are restored under each matching cell's lock and merge with edits made since the snapshot. Edits made while migration or upload runs remain dirty for the next flush. Migration is synchronous; GPU calls always run on the coordinator thread. The existing fragmentation and candidate-selection policy remains unchanged.

## Intended flow versus current status

| Area | Current status |
| --- | --- |
| Separate CPU cells and GPU allocation metadata | Implemented with `HashMap`s. |
| Detached parallel CPU generation | Implemented for independently owned cells, with generation tickets and coordinator-side batch commit. |
| Detached cell generation | Solid, empty, layered-band, and procedural occupancy/material-mask builders are available; builders can run on workers and feed the existing generation-ticket batch commit. |
| Concurrent mutation of installed cells | Supported through per-cell handles; same-cell edits serialize and cell snapshots are atomic. |
| Residency/allocator mutation | Serialized by `&mut BrickHub`; LRU, cell map, and allocation map are not independently thread-safe. |
| Compaction reservation/commit lifecycle | Active, reserved, committed, and quarantined destination states are tracked; only validated successful migrations can commit. |
| GPU-to-GPU migration for all three buffers | Implemented with separate source/destination offsets for each typed buffer. |
| Delta preservation across migration | Compaction leaves CPU deltas intact; later flush uploads them at the current allocation. |
| Bevy GPU backend integration | Uses `sns_helper::GpuDevice` and completion tokens; render/compute callers can register cell use, but the crate does not itself schedule Bevy render-world work. |

## Follow-up priorities

1. Add same-cell parallel generation only if profiling shows cell-level jobs are insufficient; design it around disjoint output ranges or a dedicated concurrent builder.
2. Measure sparse-world memory use before changing fixed per-cell GPU reservations to variable brick storage.
3. Verify index encoding against the consuming shader.
