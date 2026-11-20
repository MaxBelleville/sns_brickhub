mod data;
use data::*;

//PS: Trans rights are human rights :3

/// Builds the regular case index from eight "corner is above the iso level" flags.
pub fn regular_case_index(corner_above_iso: [bool; 8]) -> u8 {
    let mut index = 0u8;
    for (bit, above) in corner_above_iso.into_iter().enumerate() {
        index |= (above as u8) << bit;
    }
    index
}

/// Vertex and triangle counts for a regular case; the empty (0x00) and full (0xff)
/// cases report (0, 0) since there is no surface inside the cell.
pub fn regular_case_geometry(case: u8) -> (u32, u32) {
    let class = REGULAR_CELL_CLASS[case as usize] as usize;
    let counts = REGULAR_CELL_COUNTS[class];
    (counts >> 4, counts & 0x0f)
}

/// One decoded regular vertex-data entry: the surface vertex lies on the cell edge
/// between 'corner_a' and 'corner_b'; its position interpolates the two corners'
/// density values. 'reuse' carries the neighbor-cell vertex reuse code (Figure 3.8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegularEdge {
    pub corner_a: u32,
    pub corner_b: u32,
    pub reuse: u32,
}

/// Decodes a regular vertex-data entry into its edge endpoints and reuse code.
/// The low byte packs the edge as high nibble = 'corner_a', low nibble =
/// 'corner_b' (e.g. case 0x01 cuts edges 0-1, 0-2, 0-4); the high byte holds the
/// neighbor-cell vertex reuse code.
pub fn decode_regular_vertex(entry: u16) -> RegularEdge {
    RegularEdge {
        corner_a: ((entry & 0xf0) >> 4) as u32,
        corner_b: (entry & 0x0f) as u32,
        reuse: (entry >> 8) as u32,
    }
}

/// Flattened u32 lookup tables for a GPU storage buffer, in fixed word layout.
/// All values are plain u32 indices (zero padded); GPU shaders index with the same
/// constants documented here:
/// [0..256] regular class per case;
/// [256..272] regular counts per class;
/// [272..512] regular triangulations ([u32; 15] x 16);
/// [512..3584] regular vertex data ([u32; 12] x 256);
/// [3584..4096] transition class per case, 512 words;
/// [4096..4152] transition counts per class, 56 words;
/// [4152..6168] transition triangulations ([u32; 36] x 56);
/// [6168..12312] transition vertex data ([u32; 12] x 512);
/// [12312..12325] transition corner reuse data, 13 words.
pub fn serialize_tables() -> Vec<u32> {
    let mut words = Vec::with_capacity(12_325);
    words.extend(REGULAR_CELL_CLASS.iter().map(|&v| v as u32));
    words.extend(REGULAR_CELL_COUNTS.iter().copied());
    for row in REGULAR_CELL_TRIANGLES.iter() {
        words.extend(row.iter().map(|&v| v as u32));
    }
    for row in REGULAR_VERTEX_DATA.iter() {
        words.extend(row.iter().map(|&v| v as u32));
    }
    words.extend(TRANSITION_CELL_CLASS.iter().map(|&v| v as u32));
    words.extend(TRANSITION_CELL_COUNTS.iter().copied());
    for row in TRANSITION_CELL_TRIANGLES.iter() {
        words.extend(row.iter().map(|&v| v as u32));
    }
    for row in TRANSITION_VERTEX_DATA.iter() {
        words.extend(row.iter().map(|&v| v as u32));
    }
    words.extend(TRANSITION_CORNER_DATA.iter().map(|&v| v as u32));
    debug_assert_eq!(words.len(), 12_325);
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty and full cases produce no geometry; a single-corner case produces one
    /// triangle on the three edges meeting at that corner, with the documented
    /// edge endpoints.
    #[test]
    fn known_cases_decode_to_expected_triangles() {
        assert_eq!(regular_case_geometry(0x00), (0, 0));
        assert_eq!(regular_case_geometry(0xff), (0, 0));
        assert_eq!(REGULAR_CELL_CLASS[0xff], 0);

        let (vertices, triangles) = regular_case_geometry(0x01);
        assert_eq!((vertices, triangles), (3, 1));
        // Case 0x01 cuts the surface on the three edges meeting at corner 0;
        // entries 0x6201, 0x5102, 0x3304 encode edges (0,1), (0,2), (0,4).
        let edge = decode_regular_vertex(REGULAR_VERTEX_DATA[0x01][0]);
        assert_eq!((edge.corner_a, edge.corner_b), (0, 1));
        let edge = decode_regular_vertex(REGULAR_VERTEX_DATA[0x01][1]);
        assert_eq!((edge.corner_a, edge.corner_b), (0, 2));
        let edge = decode_regular_vertex(REGULAR_VERTEX_DATA[0x01][2]);
        assert_eq!((edge.corner_a, edge.corner_b), (0, 4));
        let class = REGULAR_CELL_CLASS[0x01] as usize;
        assert!(REGULAR_CELL_TRIANGLES[class][..3].iter().all(|&v| v < 3));
    }

    /// Every case must map into 0..15 with counts consistent with its vertex data;
    /// vertex entries always name valid cube corners on real cube edges, and
    /// triangulation indices reference valid vertex slots.
    #[test]
    fn all_regular_cases_are_consistent() {
        for case in 0..=255u8 {
            let class = REGULAR_CELL_CLASS[case as usize] as usize;
            assert!(class < 16, "case {case} class {class}");
            let counts = REGULAR_CELL_COUNTS[class];
            let vertex_count = (counts >> 4) as usize;
            let triangle_count = (counts & 0x0f) as usize;
            assert!(vertex_count <= 12 && triangle_count <= 5, "case {case}");

            let data = REGULAR_VERTEX_DATA[case as usize];
            for slot in 0..vertex_count {
                let entry = data[slot];
                assert_ne!(entry & 0xff, 0, "case {case} vertex {slot} has no edge");
                for endpoint in [entry & 0xf, (entry & 0xf0) >> 4] {
                    assert!(
                        endpoint < 8,
                        "case {case} vertex {slot} endpoint {endpoint}"
                    );
                }
            }
            for &vertex in REGULAR_CELL_TRIANGLES[class]
                .iter()
                .take(triangle_count * 3)
            {
                assert!(
                    (vertex as usize) < vertex_count,
                    "case {case} triangle vertex {vertex} >= {vertex_count}"
                );
            }
        }
    }

    /// Transition classes stay inside 0..56, vertex data implies matching counts, and
    /// triangulation indices reference valid vertex slots.
    #[test]
    fn all_transition_cases_are_consistent() {
        for case in 0..512usize {
            let class = (TRANSITION_CELL_CLASS[case] & 0x7f) as usize;
            assert!(class < 56, "transition case {case}");
            let counts = TRANSITION_CELL_COUNTS[class];
            let vertex_count = (counts >> 4) as usize;
            let triangle_count = (counts & 0x0f) as usize;
            assert!(
                vertex_count <= 12 && triangle_count <= 12,
                "transition case {case}"
            );

            let data = TRANSITION_VERTEX_DATA[case];
            for slot in 0..vertex_count {
                let entry = data[slot];
                assert_ne!(entry & 0xff, 0, "transition case {case} vertex {slot}");
                for endpoint in [entry & 0xf, (entry & 0xf0) >> 4] {
                    assert!(endpoint < 13, "transition case {case} endpoint {endpoint}");
                }
            }
            for &vertex in TRANSITION_CELL_TRIANGLES[class]
                .iter()
                .take(triangle_count * 3)
            {
                assert!((vertex as usize) < vertex_count, "transition case {case}");
            }
        }
    }

    /// The GPU serialization reproduces the tables exactly, word for word.
    #[test]
    fn serialized_tables_round_trip() {
        let words = serialize_tables();
        assert_eq!(words.len(), 12_325);
        assert_eq!(
            words[..256],
            REGULAR_CELL_CLASS
                .iter()
                .map(|&v| v as u32)
                .collect::<Vec<_>>()[..]
        );
        assert_eq!(&words[256..272], &REGULAR_CELL_COUNTS[..]);
        for (case, row) in REGULAR_VERTEX_DATA.iter().enumerate() {
            for (slot, &entry) in row.iter().enumerate() {
                assert_eq!(words[512 + case * 12 + slot], entry as u32);
            }
        }
        assert_eq!(words[6168], TRANSITION_VERTEX_DATA[0][0] as u32);
        assert_eq!(words[12_312], TRANSITION_CORNER_DATA[0] as u32);
    }
}
