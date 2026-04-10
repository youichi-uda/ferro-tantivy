//! Global Top-K result cache.
//!
//! Caches search results per `(SegmentId, query_hash)` to avoid re-scoring
//! identical queries against the same segment. When the cache exceeds its
//! capacity (1024 entries), it is cleared entirely.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::DocId;
use crate::Score;
use crate::index::SegmentId;

const MAX_ENTRIES: usize = 1024;

type CacheKey = (SegmentId, u64);
type CacheValue = Vec<(DocId, Score)>;

static CACHE: OnceLock<Mutex<HashMap<CacheKey, CacheValue>>> = OnceLock::new();

fn cache() -> &'static Mutex<HashMap<CacheKey, CacheValue>> {
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Look up cached top-k results for the given segment and query hash.
///
/// Returns `None` on cache miss or if the lock is poisoned.
pub fn get(segment_id: SegmentId, query_hash: u64) -> Option<Vec<(DocId, Score)>> {
    let map = cache().lock().ok()?;
    map.get(&(segment_id, query_hash)).cloned()
}

/// Insert top-k results into the cache.
///
/// If the cache has reached its capacity limit, it is cleared before
/// inserting the new entry.
pub fn put(segment_id: SegmentId, query_hash: u64, results: Vec<(DocId, Score)>) {
    let Ok(mut map) = cache().lock() else {
        return;
    };
    if map.len() >= MAX_ENTRIES {
        map.clear();
    }
    map.insert((segment_id, query_hash), results);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::SegmentId;

    #[test]
    fn test_put_and_get() {
        let seg = SegmentId::generate_random();
        let hash = 42u64;
        let results = vec![(1u32, 1.0f32), (2, 0.5)];

        put(seg, hash, results.clone());
        let cached = get(seg, hash);
        assert_eq!(cached, Some(results));
    }

    #[test]
    fn test_miss() {
        let seg = SegmentId::generate_random();
        assert_eq!(get(seg, 9999), None);
    }
}
