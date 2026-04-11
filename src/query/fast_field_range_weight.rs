//! A range query that operates directly on f64 fast fields using raw bounds.
//!
//! This is a simplified fast-field range query designed for use when the caller
//! already has f64 lower/upper bounds and a field name, avoiding the need to
//! construct `Term`-based `BoundsRange`.

use std::fmt;
use std::ops::Bound;

use common::bounds::BoundsRange;

use super::{EnableScoring, Query, Weight};
use crate::query::FastFieldRangeWeight;
use crate::schema::{Field, Term};

/// Field type hint for `FastFieldRangeQuery` to generate correct `Term` types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeFieldType {
    /// F64 field — bounds as `Term::from_field_f64`.
    F64,
    /// I64 field — bounds converted from f64 to i64 via `as i64`.
    I64,
    /// U64 field — bounds converted from f64 to u64 via `as u64`.
    U64,
}

/// `FastFieldRangeQuery` performs range filtering on a numeric fast field.
///
/// Unlike `RangeQuery`, this struct takes raw f64 bounds directly, which is
/// convenient when the caller has already parsed the bounds from a DSL
/// (e.g., Elasticsearch-compatible range queries with `gte`/`gt`/`lte`/`lt`).
/// Supports F64, I64, and U64 fields via the `field_type` hint.
#[derive(Clone)]
pub struct FastFieldRangeQuery {
    /// The tantivy field handle.
    pub field: Field,
    /// The field name string (used for fast field lookup).
    pub field_name: String,
    /// The lower bound value.
    pub lower: f64,
    /// The upper bound value.
    pub upper: f64,
    /// Whether the lower bound is inclusive.
    pub lower_inclusive: bool,
    /// Whether the upper bound is inclusive.
    pub upper_inclusive: bool,
    /// Field type for correct Term generation.
    pub field_type: RangeFieldType,
}

impl fmt::Debug for FastFieldRangeQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lower_bracket = if self.lower_inclusive { '[' } else { '{' };
        let upper_bracket = if self.upper_inclusive { ']' } else { '}' };
        write!(
            f,
            "FastFieldRangeQuery({}: {}{}, {}{})",
            self.field_name, lower_bracket, self.lower, self.upper, upper_bracket
        )
    }
}

impl FastFieldRangeQuery {
    fn make_term(&self, val: f64) -> Term {
        match self.field_type {
            RangeFieldType::F64 => Term::from_field_f64(self.field, val),
            RangeFieldType::I64 => Term::from_field_i64(self.field, val as i64),
            RangeFieldType::U64 => Term::from_field_u64(self.field, val as u64),
        }
    }

    fn to_bounds(&self) -> (Bound<Term>, Bound<Term>) {
        let lower_bound = if self.lower == f64::NEG_INFINITY {
            Bound::Unbounded
        } else if self.lower_inclusive {
            Bound::Included(self.make_term(self.lower))
        } else {
            Bound::Excluded(self.make_term(self.lower))
        };

        let upper_bound = if self.upper == f64::INFINITY {
            Bound::Unbounded
        } else if self.upper_inclusive {
            Bound::Included(self.make_term(self.upper))
        } else {
            Bound::Excluded(self.make_term(self.upper))
        };

        (lower_bound, upper_bound)
    }
}

impl Query for FastFieldRangeQuery {
    fn weight(&self, _enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let (lower_bound, upper_bound) = self.to_bounds();
        let bounds = BoundsRange::new(lower_bound, upper_bound);
        Ok(Box::new(FastFieldRangeWeight::new(bounds)))
    }
}
