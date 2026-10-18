//! Hierarchical JUMBF/C2PA box parser — mirrors `_parse_mp4_tree` /
//! `_scan_jpeg_segments` / `_carve_jumbf_superboxes` in
//! `src/modules/c2pa_verify.py`. Standard Stable only, no `unsafe`.
//!
//! Wire contract: JSON field names match `src/schemas.py` (`manifest_info`).

use std::collections::HashMap;

/// Canonical C2PA manifest-store UUIDs (ISO-BMFF `uuid` boxes).
pub const C2PA_UUIDS: [[u8; 16]; 2] = [
    hexlit(b"63326370610011001000800000AA00389B71"),
    hexlit(b"63326B6D610011001000800000AA00389B71"),
];

const fn hexval(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0,
    }
}

const fn hexlit(s: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        out[i] = (hexval(s[2 * i]) << 4) | hexval(s[2 * i + 1]);
        i += 1;
    }
    out
}

/// One parsed box: mirrors the Python `{"marker","offset","size","is_c2pa"}` nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JumbfNode {
    pub marker: String,
    pub offset: usize,
    pub size: usize,
    pub is_c2pa: bool,
}

fn u32be(b: &[u8]) -> Option<u32> {
    if b.len() < 4 {
        return None;
    }
    Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// Walk top-level ISO-BMFF boxes; flags `uuid` boxes carrying a C2PA UUID.
/// Never panics on truncated input — returns what parsed cleanly.
pub fn parse_mp4_tree(head: &[u8]) -> Vec<JumbfNode> {
    let mut nodes = Vec::new();
    let mut i = 0usize;
    while i + 8 <= head.len() && nodes.len() < 64 {
        let size = match u32be(&head[i..]) {
            Some(s) if s >= 8 => s as usize,
            _ => break,
        };
        let boxtype = &head[i + 4..i + 8];
        if boxtype == b"uuid" && i + 24 <= head.len() {
            let mut uuid = [0u8; 16];
            uuid.copy_from_slice(&head[i + 8..i + 24]);
            if C2PA_UUIDS.contains(&uuid) {
                nodes.push(JumbfNode {
                    marker: "mp4-uuid-C2PA".into(),
                    offset: i,
                    size: size.min(head.len() - i),
                    is_c2pa: true,
                });
                break;
            }
        }
        // Nested scan one level for moov/meta/udta containers.
        if (boxtype == b"moov" || boxtype == b"meta" || boxtype == b"udta") && size < 4_000_000 {
            let end = (i + size).min(head.len());
            let mut j = i + 8;
            while j + 8 <= end && nodes.len() < 64 {
                let sub = match u32be(&head[j..]) {
                    Some(s) if s >= 8 => s as usize,
                    _ => break,
                };
                if j + sub > end {
                    break;
                }
                if &head[j + 4..j + 8] == b"uuid" && j + 24 <= end {
                    let mut uuid = [0u8; 16];
                    uuid.copy_from_slice(&head[j + 8..j + 24]);
                    if C2PA_UUIDS.contains(&uuid) {
                        nodes.push(JumbfNode {
                            marker: "mp4-uuid-C2PA".into(),
                            offset: j,
                            size: sub,
                            is_c2pa: true,
                        });
                    }
                }
                j += if sub > 0 { sub } else { 4 };
            }
        }
        nodes.push(JumbfNode {
            marker: String::from_utf8_lossy(boxtype).into_owned(),
            offset: i,
            size: size.min(head.len().saturating_sub(i)),
            is_c2pa: false,
        });
        if boxtype == b"mdat" {
            i += 8; // skip payload, keep scanning for trailing moov
            continue;
        }
        if size == 0 {
            break;
        }
        i += size;
    }
    nodes
}

/// JPEG APPn walk; flags APP11 segments containing JUMBF/C2PA payloads.
pub fn scan_jpeg_segments(head: &[u8]) -> Vec<JumbfNode> {
    let mut nodes = Vec::new();
    if head.len() < 2 || head[0] != 0xFF || head[1] != 0xD8 {
        return nodes;
    }
    let mut i = 2usize;
    while i + 4 <= head.len() {
        if head[i] != 0xFF {
            break;
        }
        let marker = head[i + 1];
        if marker == 0xD9 || marker == 0xDA {
            nodes.push(JumbfNode {
                marker: "EOI/SOS".into(),
                offset: i,
                size: 2,
                is_c2pa: false,
            });
            break;
        }
        let seg_len = u16::from_be_bytes([head[i + 2], head[i + 3]]) as usize;
        let end = (i + 2 + seg_len).min(head.len());
        let payload = &head[i + 4..end];
        let is_c2pa = marker == 0xEB
            && (payload.windows(5).any(|w| w == b"JUMBF" || w == b"jumb")
                || payload.windows(4).any(|w| w.eq_ignore_ascii_case(b"c2pa")));
        let label = if is_c2pa {
            "jpeg-APP11-JUMBF".to_string()
        } else if (0xE0..=0xEF).contains(&marker) {
            format!("APP{}", marker - 0xE0)
        } else {
            format!("M{marker:02X}")
        };
        nodes.push(JumbfNode {
            marker: label,
            offset: i,
            size: seg_len + 2,
            is_c2pa,
        });
        if is_c2pa {
            break;
        }
        i += 2 + seg_len;
    }
    nodes
}

/// Summary counts per marker label (for logging parity with Python).
pub fn marker_counts(nodes: &[JumbfNode]) -> HashMap<&str, usize> {
    let mut m = HashMap::new();
    for n in nodes {
        *m.entry(n.marker.as_str()).or_insert(0) += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jpeg_app11_detected() {
        let mut jpg = vec![0xFF, 0xD8, 0xFF, 0xEB, 0x00, 0x0C];
        jpg.extend_from_slice(b"JUMBF\x00c2pa");
        jpg.extend_from_slice(&[0xFF, 0xD9]);
        let nodes = scan_jpeg_segments(&jpg);
        assert!(nodes
            .iter()
            .any(|n| n.is_c2pa && n.marker == "jpeg-APP11-JUMBF"));
    }

    #[test]
    fn mp4_uuid_detected() {
        let mut mp4 = vec![];
        mp4.extend_from_slice(&12u32.to_be_bytes());
        mp4.extend_from_slice(b"ftypisom");
        let size = (8u32 + 16 + 4).to_be_bytes();
        mp4.extend_from_slice(&size);
        mp4.extend_from_slice(b"uuid");
        mp4.extend_from_slice(&C2PA_UUIDS[0]);
        mp4.extend_from_slice(b"data");
        let nodes = parse_mp4_tree(&mp4);
        assert!(nodes.iter().any(|n| n.is_c2pa));
    }

    #[test]
    fn truncated_input_never_panics() {
        assert!(parse_mp4_tree(&[0xFF, 0xD8]).is_empty());
        assert!(scan_jpeg_segments(&[0x00]).is_empty());
    }
}
