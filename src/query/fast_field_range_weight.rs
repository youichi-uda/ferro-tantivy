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

/// `FastFieldRangeQuery` performs range filtering on an f64 fast field.
///
/// Unlike `RangeQuery`, this struct takes raw f64 bounds directly, which is
/// convenient when the caller has already parsed the bounds from a DSL
/// (e.g., Elasticsearch-compatible range queries with `gte`/`gt`/`lte`/`lt`).
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
    fn to_bounds(&self) -> (Bound<Term>, Bound<Term>) {
        let lower_bound = if self.lower == f64::NEG_INFINITY {
            Bound::Unbounded
        } else if self.lower_inclusive {
            Bound::Included(Term::from_field_f64(self.field, self.lower))
        } else {
            Bound::Excluded(Term::from_field_f64(self.field, self.lower))
        };

        let upper_bound = if self.upper == f64::INFINITY {
            Bound::Unbounded
        } else if self.upper_inclusive {
            Bound::Included(Term::from_field_f64(self.field, self.upper))
        } else {
            Bound::Excluded(Term::from_field_f64(self.field, self.upper))
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
