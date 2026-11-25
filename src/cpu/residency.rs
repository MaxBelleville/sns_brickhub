use crate::consts::{CellID, CELL_SIZE};
use bevy_math::*;
use lru::LruCache;
use std::{collections::HashMap, num::NonZeroUsize};

#[derive(Clone, Copy)]
pub struct ClipmapRing {
    pub level: u8,
    pub radius_cells: i32,
    pub max_cells: usize,
}

pub struct ResidencyManager {
    budgets: HashMap<u8, usize>, // ring → max allowed (in cells)
    lru_per_ring: HashMap<u8, LruCache<CellID, u64>>,
    /// Section LRU, keyed by x/z column. A section occupies the same ring budget
    /// as the cells it owns, so a one-layer section behaves exactly like a cell.
    section_lru_per_ring: HashMap<u8, LruCache<SectionId, u64>>,
}
impl Default for ResidencyManager {
    fn default() -> Self {
        let mut budgets = HashMap::new();

        for ring in CLIPMAP_RINGS {
            budgets.insert(ring.level, ring.max_cells);
        }

        Self {
            budgets,
            lru_per_ring: HashMap::new(),
            section_lru_per_ring: HashMap::new(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum ResidencyDecision {
    Admit,
    EvictAndAdmit(CellID),
    Reject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SectionResidencyDecision {
    Admit,
    EvictAndAdmit(SectionId),
    Reject,
}

impl ResidencyManager {
    pub fn admit(&mut self, id: CellID, ring: u8, frame: u64) -> ResidencyDecision {
        let budget = self.budgets[&ring];
        let default = LruCache::new(NonZeroUsize::new(budget).unwrap());
        let lru = self.lru_per_ring.entry(ring).or_insert(default);

        if lru.len() >= budget {
            if let Some((evict_id, _evict_frame)) = lru.pop_lru() {
                lru.push(id, frame);
                return ResidencyDecision::EvictAndAdmit(evict_id);
            }
            return ResidencyDecision::Reject;
        }

        lru.push(id, frame);
        ResidencyDecision::Admit
    }

    pub fn touch(&mut self, id: CellID, ring: u8, frame: u64) {
        if let Some(lru) = self.lru_per_ring.get_mut(&ring) {
            lru.push(id, frame);
        }
    }

    pub fn remove(&mut self, id: CellID, ring: u8) {
        if let Some(lru) = self.lru_per_ring.get_mut(&ring) {
            lru.pop_entry(&id);
        }
    }

    /// Ring budget measured in sections (cells divided by section height).
    fn section_budget(&self, ring: u8) -> usize {
        self.budgets
            .get(&ring)
            .copied()
            .unwrap_or(0)
            .div_ceil(SECTION_CELL_LAYERS as usize)
            .max(1)
    }

    pub fn admit_section(
        &mut self,
        id: SectionId,
        ring: u8,
        frame: u64,
    ) -> SectionResidencyDecision {
        let budget = self.section_budget(ring);
        let default = LruCache::new(NonZeroUsize::new(budget).unwrap());
        let lru = self.section_lru_per_ring.entry(ring).or_insert(default);
        if lru.len() >= budget {
            if let Some((evict_id, _)) = lru.pop_lru() {
                lru.push(id, frame);
                return SectionResidencyDecision::EvictAndAdmit(evict_id);
            }
            return SectionResidencyDecision::Reject;
        }
        lru.push(id, frame);
        SectionResidencyDecision::Admit
    }

    pub fn touch_section(&mut self, id: SectionId, ring: u8, frame: u64) {
        if let Some(lru) = self.section_lru_per_ring.get_mut(&ring) {
            lru.push(id, frame);
        }
    }

    pub fn remove_section(&mut self, id: SectionId, ring: u8) {
        if let Some(lru) = self.section_lru_per_ring.get_mut(&ring) {
            lru.pop_entry(&id);
        }
    }
    pub fn _ring_full(&self, ring: u8) -> bool {
        let max = match self.budgets.get(&ring) {
            Some(b) => *b,
            None => return false,
        };

        let current = self
            .lru_per_ring
            .get(&ring)
            .map(|lru| lru.len())
            .unwrap_or(0);

        current >= max
    }

    /// Active count for debugging/metrics
    pub fn _active_count(&self, ring: u8) -> usize {
        self.lru_per_ring
            .get(&ring)
            .map(|lru| lru.len())
            .unwrap_or(0)
    }

    /// Section count for debugging/metrics and invariant tests.
    pub fn _section_active_count(&self, ring: u8) -> usize {
        self.section_lru_per_ring
            .get(&ring)
            .map(|lru| lru.len())
            .unwrap_or(0)
    }
}

// Seven rings at 3-cell steps. The closer spacing narrows the jump between
// neighbouring LODs; the heightmap renderer takes over at radius 18/21 (or
// 15/18 when view-cone culling is enabled). Earlier schedules were
// 4 / 8 / 12 / 16 / 20 and 3 / 6 / 9 / 12 / 15; keep this array and consts.wesl
// in sync (the parity test in the renderer checks both).
pub const CLIPMAP_RINGS: &[ClipmapRing] = &[
    ClipmapRing {
        level: 0,
        radius_cells: 3,
        max_cells: 7 * 7 * 7,
    },
    ClipmapRing {
        level: 1,
        radius_cells: 6,
        max_cells: 13 * 13 * 13,
    },
    ClipmapRing {
        level: 2,
        radius_cells: 9,
        max_cells: 19 * 19 * 19,
    },
    ClipmapRing {
        level: 3,
        radius_cells: 12,
        max_cells: 25 * 25 * 25,
    },
    ClipmapRing {
        level: 4,
        radius_cells: 15,
        max_cells: 31 * 31 * 31,
    },
    ClipmapRing {
        level: 5,
        radius_cells: 18,
        max_cells: 37 * 37 * 37,
    },
    ClipmapRing {
        level: 6,
        radius_cells: 21,
        max_cells: 43 * 43 * 43,
    },
];

/// A vertical section is an x/z column of 3D cells. The demo currently uses a
/// single y=0 layer, but the helpers below carry an explicit layer index so a
/// section can later own several stacked cells without changing callers.
pub type SectionID = IVec2;

/// CamelCase alias matching the renderer-facing API name.
pub type SectionId = SectionID;

/// Number of 3D cells stacked in one section. Fixed at one for the y=0 demo
/// slab; `section_cell` already accepts the layer index.
pub const SECTION_CELL_LAYERS: u8 = 1;

/// One vertical column of 3D cells. `y_base` and `cell_count` are kept explicit
/// so a section can later own more than the single y=0 layer without changing
/// residency or LOD call sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Section {
    pub id: SectionId,
    pub y_base: i32,
    pub cell_count: u8,
}

impl Section {
    pub fn new(id: SectionId) -> Self {
        Self {
            id,
            y_base: 0,
            cell_count: SECTION_CELL_LAYERS,
        }
    }

    /// All 3D cells owned by this column (bottom to top).
    pub fn cell_ids(&self) -> Vec<CellID> {
        (0..self.cell_count)
            .map(|layer| IVec3::new(self.id.x, self.y_base + layer as i32, self.id.y))
            .collect()
    }

    pub fn primary_cell(&self) -> CellID {
        IVec3::new(self.id.x, self.y_base, self.id.y)
    }
}

/// The 3D cell at `layer` inside a section (layer 0 is the y=0 cell).
pub fn section_cell(section: SectionID, layer: u8) -> CellID {
    IVec3::new(section.x, layer as i32, section.y)
}

/// The section containing a 3D cell. Y is deliberately discarded: LOD and
/// residency are independent of altitude.
pub fn section_of_cell(cell: CellID) -> SectionID {
    IVec2::new(cell.x, cell.z)
}

/// Classify a section into a clipmap ring. Mirrors `classify_ring` but measures
/// Chebyshev distance in x/z only, so climbing changes no distant LOD. The
/// section's vertical extent is handled by `SECTION_CELL_LAYERS`, not here.
pub fn classify_section(
    section: SectionID,
    view_section: SectionID,
    look_at: Option<Vec3>,
) -> Option<u8> {
    let delta = (section - view_section).abs();
    let dist = delta.x.max(delta.y); // IVec2.y is world z

    let ring = CLIPMAP_RINGS
        .iter()
        .find(|r| dist <= r.radius_cells)?;

    // Inner rings (dist <= 12, lod <= 3): full area, no cone gating.
    if dist <= 12 {
        return Some(ring.level);
    }
    let Some(look_at) = look_at else {
        return Some(ring.level);
    };
    let cell_size = CELL_SIZE as f32;
    let cam = (view_section.as_vec2() + Vec2::splat(0.5)) * cell_size;
    let center = (section.as_vec2() + Vec2::splat(0.5)) * cell_size;
    // Horizontal look direction only: climbing or pitching must not change
    // which x/z sections are resident or what LOD they get. A straight-down
    // look has no horizontal direction, so skip cone gating entirely.
    let forward = Vec2::new(look_at.x, look_at.z) - cam;
    let forward = forward.normalize_or_zero();
    if forward.length_squared() < 0.5 {
        return Some(ring.level);
    }
    let dir = center - cam;
    let along = dir.dot(forward);
    if along < 0.0 {
        return None;
    }
    let along = along.max(0.01);
    let perp = (dir - forward * along).length();
    let cell_radius = cell_size * 0.5 * 3.0_f32.sqrt();
    if perp > 1.732 * along + cell_radius * 2.0 {
        return None;
    }
    Some(ring.level)
}

/// Classify a cell into a clipmap ring. When `look_at` (world-space point the
/// camera is looking at) is `Some`, outer rings (dist > 12, i.e. lod >= 4)
/// are gated by a view cone so cells behind/outside the view
/// are excluded entirely — avoiding the unload-on-rotation problem of hard
/// frustum culling. Inner rings stay a full cube.
pub fn classify_ring(cell: IVec3, view_cell: IVec3, look_at: Option<Vec3>) -> Option<u8> {
    let delta = (cell - view_cell).abs();
    let dist = delta.x.max(delta.y).max(delta.z);

    let ring = CLIPMAP_RINGS
        .iter()
        .find(|r| dist <= r.radius_cells)?;

    // Inner rings (dist <= 12, lod <= 3): full cube, no cone gating.
    if dist <= 12 {
        return Some(ring.level);
    }
    // Outer rings: gate by view cone when a look_at is provided.
    let Some(look_at) = look_at else {
        return Some(ring.level);
    };
    let cell_size = CELL_SIZE as f32;
    let cam = (view_cell.as_vec3() + Vec3::splat(0.5)) * cell_size;
    let cell_center = (cell.as_vec3() + Vec3::splat(0.5)) * cell_size;
    let forward = (look_at - cam).normalize_or_zero();
    if forward.length_squared() < 0.5 {
        return Some(ring.level);
    }
    let dir = cell_center - cam;
    let along = dir.dot(forward);
    if along < 0.0 {
        return None; // behind camera
    }
    let along = along.max(0.01);
    let perp = (dir - forward * along).length();
    // Generous cone (~60 deg half-angle) plus a cell-radius margin.
    let cell_radius = cell_size * 0.5 * 3.0_f32.sqrt();
    if perp > 1.732 * along + cell_radius * 2.0 {
        return None;
    }
    Some(ring.level)
}

#[cfg(test)]
mod section_tests {
    use super::*;

    #[test]
    fn section_lru_is_independent_and_evicts_whole_sections() {
        let mut manager = ResidencyManager::default();
        let ring = 0;
        let budget = manager.section_budget(ring);
        for i in 0..budget {
            assert_eq!(
                manager.admit_section(IVec2::new(i as i32, 0), ring, i as u64),
                SectionResidencyDecision::Admit
            );
        }
        assert_eq!(manager._section_active_count(ring), budget);
        // Reserving 3D cells does not touch the section LRU.
        assert!(matches!(
            manager.admit(IVec3::ZERO, ring, 999),
            ResidencyDecision::Admit
        ));
        assert_eq!(manager._section_active_count(ring), budget);
        // One more section evicts exactly one section, never a cell entry.
        assert!(matches!(
            manager.admit_section(IVec2::new(9000, 0), ring, 1000),
            SectionResidencyDecision::EvictAndAdmit(_)
        ));
        assert_eq!(manager._section_active_count(ring), budget);
        assert_eq!(manager._active_count(ring), 1);
    }

    #[test]
    fn section_helpers_cover_the_layered_cell_mapping() {
        let section = IVec2::new(-4, 7);
        assert_eq!(SECTION_CELL_LAYERS, 1);
        assert_eq!(section_cell(section, 0), IVec3::new(-4, 0, 7));
        assert_eq!(section_cell(section, 1), IVec3::new(-4, 1, 7));
        assert_eq!(section_of_cell(section_cell(section, 0)), section);
        assert_eq!(section_of_cell(IVec3::new(-4, 31, 7)), section);
    }

    #[test]
    fn section_classification_matches_y0_cells_and_ignores_altitude() {
        let view = IVec2::new(2, -3);
        let view_cell = IVec3::new(2, 0, -3);
        for section in [
            IVec2::new(2, -3),
            IVec2::new(5, -3),
            IVec2::new(2, 10),
            IVec2::new(-7, 4),
            IVec2::new(23, -19),
        ] {
            let expected = classify_ring(section_cell(section, 0), view_cell, None);
            assert_eq!(classify_section(section, view, None), expected, "{section:?}");
            // Sections never vary with Y; the old 3D rule could.
            let elevated = classify_ring(IVec3::new(section.x, 2, section.y), view_cell, None);
            assert_eq!(classify_section(section, view, None), expected);
            let _ = elevated;
        }
    }

    #[test]
    fn section_cone_is_independent_of_camera_altitude() {
        let view = IVec2::ZERO;
        let ground_look = Vec3::new(384.0, 20.0, 384.0);
        let high_look = Vec3::new(384.0, 640.0, 384.0);
        for section in [
            IVec2::new(0, 6),
            IVec2::new(9, 9),
            IVec2::new(-9, 9),
            IVec2::new(0, -12),
            IVec2::new(16, 16),
        ] {
            assert_eq!(
                classify_section(section, view, Some(ground_look)),
                classify_section(section, view, Some(high_look)),
                "{section:?}"
            );
        }
    }

    #[test]
    fn section_cone_matches_the_3d_rule_for_y0_cells() {
        let view = IVec2::new(0, 0);
        let view_cell = IVec3::ZERO;
        let look_at = Vec3::new(0.0, 20.0, 512.0);
        for section in [
            IVec2::new(0, 4),
            IVec2::new(8, 14),
            IVec2::new(-8, 14),
            IVec2::new(0, -14),
            IVec2::new(18, 0),
        ] {
            assert_eq!(
                classify_section(section, view, Some(look_at)),
                classify_ring(section_cell(section, 0), view_cell, Some(look_at)),
                "{section:?}"
            );
        }
    }
}
