use bevy_math::*;

use crate::{BRICKS_PER_BRICKMAP, CellID, VOXELS_PER_BRICK};

use std::collections::HashMap;

#[derive(Default)]
pub struct DeviceAllocator {
    bricks: RangeAllocator,
    materials: RangeAllocator,
    indices: RangeAllocator,
    allocations: HashMap<CellID, CellAllocation>,
    pending_compactions: HashMap<CellID, PendingCompaction>,
    quarantined: Vec<CellAllocation>,
}

impl DeviceAllocator {
    pub fn admit_cell(&mut self, coord: IVec3, ring: u8) -> &CellAllocation {
        self.allocations.entry(coord).or_insert_with(|| {
            let brick_count = BRICKS_PER_BRICKMAP;
            let material_count = BRICKS_PER_BRICKMAP * VOXELS_PER_BRICK;
            let index_count = BRICKS_PER_BRICKMAP;

            // Allocate from free ranges or grow
            let brick_base = self
                .bricks
                .alloc(brick_count)
                .expect("brick allocation overflow");
            let material_base = self
                .materials
                .alloc(material_count)
                .expect("material allocation overflow");
            let index_base = self
                .indices
                .alloc(index_count)
                .expect("index allocation overflow");
            CellAllocation {
                brick_base,
                brick_count,
                material_base,
                material_count,
                index_base,
                index_count,
                ring,
            }
        })
    }

    pub fn reserve_compaction(
        &mut self,
        candidates: &[CompactionCandidate],
    ) -> Result<Vec<CompactionReservation>, AllocationError> {
        for candidate in candidates {
            if self.allocations.get(&candidate.id) != Some(&candidate.old_alloc)
                || self.pending_compactions.contains_key(&candidate.id)
            {
                return Err(AllocationError::StaleCandidate(candidate.id));
            }
        }

        let mut reservations: Vec<CompactionReservation> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let old = candidate.old_alloc;
            let brick_base = match self.bricks.alloc(old.brick_count) {
                Ok(base) => base,
                Err(error) => {
                    for reserved in reservations {
                        self.free_allocation(reserved.new);
                    }
                    return Err(error);
                }
            };
            let material_base = match self.materials.alloc(old.material_count) {
                Ok(base) => base,
                Err(error) => {
                    self.bricks.free(brick_base, old.brick_count);
                    for reserved in reservations {
                        self.free_allocation(reserved.new);
                    }
                    return Err(error);
                }
            };
            let index_base = match self.indices.alloc(old.index_count) {
                Ok(base) => base,
                Err(error) => {
                    self.bricks.free(brick_base, old.brick_count);
                    self.materials.free(material_base, old.material_count);
                    for reserved in reservations {
                        self.free_allocation(reserved.new);
                    }
                    return Err(error);
                }
            };
            reservations.push(CompactionReservation {
                id: candidate.id,
                old,
                new: CellAllocation {
                    brick_base,
                    brick_count: old.brick_count,
                    material_base,
                    material_count: old.material_count,
                    index_base,
                    index_count: old.index_count,
                    ring: old.ring,
                },
            });
        }
        for reservation in &reservations {
            self.pending_compactions.insert(
                reservation.id,
                PendingCompaction {
                    old: reservation.old,
                    new: reservation.new,
                    quarantined: false,
                },
            );
        }
        Ok(reservations)
    }

    pub fn commit_compaction(
        &mut self,
        reservations: &[CompactionReservation],
    ) -> Result<(), AllocationError> {
        self.validate_reservations(reservations, false)?;
        for reservation in reservations {
            self.pending_compactions.remove(&reservation.id);
            self.allocations.insert(reservation.id, reservation.new);
            self.free_allocation(reservation.old);
        }
        Ok(())
    }

    pub fn cancel_compaction(
        &mut self,
        reservations: &[CompactionReservation],
    ) -> Result<(), AllocationError> {
        self.validate_reservations(reservations, false)?;
        for reservation in reservations {
            self.pending_compactions.remove(&reservation.id);
            self.free_allocation(reservation.new);
        }
        Ok(())
    }

    pub fn quarantine_compaction(
        &mut self,
        reservations: &[CompactionReservation],
    ) -> Result<(), AllocationError> {
        self.validate_reservations(reservations, true)?;
        for reservation in reservations {
            self.pending_compactions
                .get_mut(&reservation.id)
                .expect("validated reservation")
                .quarantined = true;
        }
        Ok(())
    }

    fn validate_reservations(
        &self,
        reservations: &[CompactionReservation],
        allow_quarantined: bool,
    ) -> Result<(), AllocationError> {
        for reservation in reservations {
            let Some(pending) = self.pending_compactions.get(&reservation.id) else {
                return Err(AllocationError::MissingReservation(reservation.id));
            };
            if pending.old != reservation.old
                || pending.new != reservation.new
                || self.allocations.get(&reservation.id) != Some(&reservation.old)
                || (pending.quarantined && !allow_quarantined)
            {
                return Err(AllocationError::StaleReservation(reservation.id));
            }
        }
        Ok(())
    }

    fn free_allocation(&mut self, allocation: CellAllocation) {
        self.bricks
            .free(allocation.brick_base, allocation.brick_count);
        self.materials
            .free(allocation.material_base, allocation.material_count);
        self.indices
            .free(allocation.index_base, allocation.index_count);
    }
    pub fn is_fragmented_for(&self, size: usize) -> bool {
        // Fragmented if we have free ranges but ring is still full
        self.bricks.is_fragmented_for(size)
            || self.materials.is_fragmented_for(size)
            || self.indices.is_fragmented_for(size)
    }

    pub fn select_compaction_candidates(&self, ring: u8) -> Vec<CompactionCandidate> {
        self.allocations
            .iter()
            .filter(|(id, a)| a.get_ring() == ring && !self.pending_compactions.contains_key(id))
            .map(|(id, alloc)| CompactionCandidate::from_alloc(alloc, *id))
            .take(2) // start small
            .collect()
    }

    /// Remove a cell allocation and return it
    pub fn remove(&mut self, id: CellID) -> Option<CellAllocation> {
        if let Some(pending) = self.pending_compactions.get(&id).copied() {
            if !pending.quarantined {
                self.pending_compactions.remove(&id);
                self.free_allocation(pending.new);
            } else {
                self.pending_compactions.remove(&id);
                self.quarantined.push(pending.old);
                self.quarantined.push(pending.new);
            }
        }
        self.allocations.remove(&id)
    }

    /// Free ranges for a removed cell back to RangeAllocator
    pub fn free_cell_allocation(&mut self, alloc: CellAllocation) {
        if self.quarantined.contains(&alloc) {
            return;
        }
        self.free_allocation(alloc);
    }

    /// Lookup a cell allocation
    pub fn get(&self, id: CellID) -> Option<&CellAllocation> {
        self.allocations.get(&id)
    }

    pub fn is_gpu_quarantined(&self, id: CellID) -> bool {
        self.pending_compactions
            .get(&id)
            .is_some_and(|pending| pending.quarantined)
    }

    pub fn brick_capacity(&self) -> usize {
        self.bricks.capacity()
    }

    pub fn material_capacity(&self) -> usize {
        self.materials.capacity()
    }

    pub fn index_capacity(&self) -> usize {
        self.indices.capacity()
    }
}

#[derive(Default)]
pub struct RangeAllocator {
    next: usize,
    free: Vec<FreeRange>,
}

impl RangeAllocator {
    pub fn alloc(&mut self, count: usize) -> Result<usize, AllocationError> {
        if let Some((idx, range)) = self.free.iter().enumerate().find(|(_, r)| r.count >= count) {
            let base = range.base;

            if range.count == count {
                self.free.swap_remove(idx);
            } else {
                self.free[idx].base += count;
                self.free[idx].count -= count;
            }

            return Ok(base);
        }

        let base = self.next;
        self.next = self
            .next
            .checked_add(count)
            .ok_or(AllocationError::CapacityOverflow)?;
        Ok(base)
    }
    pub fn free(&mut self, base: usize, count: usize) {
        self.free.push(FreeRange { base, count });
        self.free.sort_by_key(|r| r.base);

        // Merge contiguous ranges
        let mut merged: Vec<FreeRange> = Vec::with_capacity(self.free.len());
        for r in self.free.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.base + last.count == r.base {
                    last.count += r.count;
                    continue;
                }
            }
            merged.push(r);
        }
        self.free = merged;
    }

    pub fn capacity(&self) -> usize {
        self.next
    }

    pub fn total_free(&self) -> usize {
        self.free.iter().map(|r| r.count).sum()
    }

    pub fn largest_free_block(&self) -> usize {
        self.free.iter().map(|r| r.count).max().unwrap_or(0)
    }

    pub fn is_fragmented_for(&self, size: usize) -> bool {
        self.total_free() >= size && self.largest_free_block() < size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellAllocation {
    brick_base: usize,
    brick_count: usize,

    material_base: usize,
    material_count: usize,

    index_base: usize,
    index_count: usize,
    ring: u8,
}
impl CellAllocation {
    pub fn get_brick_base(&self) -> usize {
        self.brick_base
    }

    pub fn get_material_base(&self) -> usize {
        self.material_base
    }

    pub fn get_index_base(&self) -> usize {
        self.index_base
    }
    pub fn get_ring(&self) -> u8 {
        self.ring
    }
    pub fn get_brick_count(&self) -> usize {
        self.brick_count
    }
    pub fn get_material_count(&self) -> usize {
        self.material_count
    }
    pub fn get_index_count(&self) -> usize {
        self.index_count
    }
}

#[derive(Debug)]
pub enum AllocationError {
    CapacityOverflow,
    StaleCandidate(CellID),
    MissingReservation(CellID),
    StaleReservation(CellID),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactionReservation {
    id: CellID,
    old: CellAllocation,
    new: CellAllocation,
}
impl CompactionReservation {
    pub fn id(&self) -> CellID {
        self.id
    }
    pub fn old(&self) -> CellAllocation {
        self.old
    }
    pub fn new(&self) -> CellAllocation {
        self.new
    }
}

#[derive(Clone, Copy)]
struct PendingCompaction {
    old: CellAllocation,
    new: CellAllocation,
    quarantined: bool,
}

pub struct CompactionCandidate {
    id: CellID, // cell id
    old_alloc: CellAllocation,
}
impl CompactionCandidate {
    pub fn get_id(&self) -> IVec3 {
        self.id
    }
    pub fn from_alloc(alloc: &CellAllocation, id: IVec3) -> Self {
        Self {
            id,
            old_alloc: alloc.clone(),
        }
    }
}

struct FreeRange {
    base: usize,
    count: usize,
}

#[cfg(test)]
mod compaction_tests {
    use super::*;

    #[test]
    fn reservation_keeps_active_allocation_until_commit() {
        let mut allocator = DeviceAllocator::default();
        let id = IVec3::ZERO;
        let old = *allocator.admit_cell(id, 0);
        let candidate = CompactionCandidate::from_alloc(&old, id);
        let reservations = allocator.reserve_compaction(&[candidate]).unwrap();
        let reservation = reservations[0];

        assert_eq!(allocator.get(id), Some(&old));
        assert_ne!(reservation.old, reservation.new);
        allocator.commit_compaction(&reservations).unwrap();
        assert_eq!(allocator.get(id), Some(&reservation.new));
        assert!(matches!(
            allocator.commit_compaction(&reservations),
            Err(AllocationError::MissingReservation(_))
        ));
    }

    #[test]
    fn cancel_preserves_active_ranges_and_allows_reuse() {
        let mut allocator = DeviceAllocator::default();
        let id = IVec3::ZERO;
        let old = *allocator.admit_cell(id, 0);
        let candidates = [CompactionCandidate::from_alloc(&old, id)];
        let reservations = allocator.reserve_compaction(&candidates).unwrap();
        let new = reservations[0].new;

        allocator.cancel_compaction(&reservations).unwrap();
        assert_eq!(allocator.get(id), Some(&old));
        let replacement = allocator.bricks.alloc(old.brick_count).unwrap();
        assert_eq!(replacement, new.brick_base);
    }

    #[test]
    fn quarantined_reservation_cannot_be_released_or_committed() {
        let mut allocator = DeviceAllocator::default();
        let id = IVec3::ZERO;
        let old = *allocator.admit_cell(id, 0);
        let candidates = [CompactionCandidate::from_alloc(&old, id)];
        let reservations = allocator.reserve_compaction(&candidates).unwrap();
        let new = reservations[0].new;

        allocator.quarantine_compaction(&reservations).unwrap();
        assert!(matches!(
            allocator.cancel_compaction(&reservations),
            Err(AllocationError::StaleReservation(_))
        ));
        assert!(matches!(
            allocator.commit_compaction(&reservations),
            Err(AllocationError::StaleReservation(_))
        ));
        assert_eq!(allocator.get(id), Some(&old));
        let removed = allocator.remove(id).unwrap();
        allocator.free_cell_allocation(removed);
        let next = allocator.bricks.alloc(old.brick_count).unwrap();
        assert_ne!(next, old.brick_base);
        assert_ne!(next, new.brick_base);
    }

    #[test]
    fn stale_candidate_does_not_reserve_ranges() {
        let mut allocator = DeviceAllocator::default();
        let id = IVec3::ZERO;
        let old = *allocator.admit_cell(id, 0);
        let candidate = CompactionCandidate::from_alloc(&old, id);
        allocator.remove(id).unwrap();
        allocator.admit_cell(id, 0);

        assert!(matches!(
            allocator.reserve_compaction(&[candidate]),
            Err(AllocationError::StaleCandidate(_))
        ));
        assert!(allocator.pending_compactions.is_empty());
    }

    #[test]
    fn partial_reservation_failure_rolls_back_all_ranges() {
        let mut allocator = DeviceAllocator::default();
        let id = IVec3::ZERO;
        let old = *allocator.admit_cell(id, 0);
        let candidate = CompactionCandidate::from_alloc(&old, id);
        allocator.materials.next = usize::MAX;

        assert!(matches!(
            allocator.reserve_compaction(&[candidate]),
            Err(AllocationError::CapacityOverflow)
        ));
        assert_eq!(allocator.get(id), Some(&old));
        assert!(allocator.pending_compactions.is_empty());
        assert_eq!(allocator.bricks.largest_free_block(), old.brick_count);
    }
}
