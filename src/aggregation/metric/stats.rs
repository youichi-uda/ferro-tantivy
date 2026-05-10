use std::fmt::Debug;

use columnar::{Column, ColumnType};
use serde::{Deserialize, Serialize};

use super::*;
use crate::aggregation::agg_data::AggregationsSegmentCtx;
use crate::aggregation::intermediate_agg_result::{
    IntermediateAggregationResult, IntermediateAggregationResults, IntermediateMetricResult,
};
use crate::aggregation::segment_agg_result::SegmentAggregationCollector;
use crate::aggregation::*;
use crate::TantivyError;

#[cfg(feature = "cuda-stats-kernel")]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Phase 2 E-2 — CUDA stats dispatch threshold. Below this many
/// staged `f32` values, the kernel-launch + memcpy overhead exceeds
/// the per-element CPU cost and the GPU loses. The
/// `crates/ferro-compress/src/bin/stats_bench.rs` kernel-only bench
/// shows the GPU starts to win cleanly around 100 K elements on
/// RTX 4070 Ti SUPER (15.8× CPU at 100 K → 75.6× CPU at 100 M).
/// Tuned to the conservative side: smaller cohorts pass through the
/// CPU Kahan loop unchanged.
#[cfg(feature = "cuda-stats-kernel")]
const GPU_STAGING_FLUSH_THRESHOLD: usize = 100_000;

/// Total values staged across all GPU-eligible `SegmentStatsCollector`
/// dispatches in this process. Exposed for observability tests
/// (`gpu_dispatch_count_increments_on_staged_path`) and for future
/// REST-side stats endpoints. Each dispatch increments by the
/// number of `f32` values folded.
#[cfg(feature = "cuda-stats-kernel")]
pub(crate) static GPU_STATS_DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total values processed via the CPU Kahan fallback after a GPU
/// dispatch failure or because the staging never reached the
/// threshold / multi-bucket case was detected. Symmetric to
/// `GPU_STATS_DISPATCH_COUNT` so the GPU/CPU ratio is observable
/// from telemetry.
#[cfg(feature = "cuda-stats-kernel")]
pub(crate) static GPU_STATS_FALLBACK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Reset the GPU stats dispatch counters. Used by integration tests
/// that need a clean baseline before exercising the dispatch path.
#[cfg(all(feature = "cuda-stats-kernel", test))]
pub(crate) fn reset_gpu_stats_counters() {
    GPU_STATS_DISPATCH_COUNT.store(0, AtomicOrdering::Relaxed);
    GPU_STATS_FALLBACK_COUNT.store(0, AtomicOrdering::Relaxed);
}

/// Snapshot of (dispatch_count, fallback_count). Test-only sibling
/// of [`reset_gpu_stats_counters`].
#[cfg(all(feature = "cuda-stats-kernel", test))]
pub(crate) fn snapshot_gpu_stats_counters() -> (u64, u64) {
    (
        GPU_STATS_DISPATCH_COUNT.load(AtomicOrdering::Relaxed),
        GPU_STATS_FALLBACK_COUNT.load(AtomicOrdering::Relaxed),
    )
}

/// A multi-value metric aggregation that computes a collection of statistics on numeric values that
/// are extracted from the aggregated documents.
/// See [`Stats`] for returned statistics.
///
/// # JSON Format
/// ```json
/// {
///     "stats": {
///         "field": "score"
///     }
///  }
/// ```

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StatsAggregation {
    /// The field name to compute the stats on.
    pub field: String,
    /// The missing parameter defines how documents that are missing a value should be treated.
    /// By default they will be ignored but it is also possible to treat them as if they had a
    /// value. Examples in JSON format:
    /// { "field": "my_numbers", "missing": "10.0" }
    #[serde(default, deserialize_with = "deserialize_option_f64")]
    pub missing: Option<f64>,
}

impl StatsAggregation {
    /// Creates a new [`StatsAggregation`] instance from a field name.
    pub fn from_field_name(field_name: String) -> Self {
        StatsAggregation {
            field: field_name,
            missing: None,
        }
    }
    /// Returns the field name the aggregation is computed on.
    pub fn field_name(&self) -> &str {
        &self.field
    }
}

/// Stats contains a collection of statistics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    /// The number of documents.
    pub count: u64,
    /// The sum of the fast field values.
    pub sum: f64,
    /// The min value of the fast field values.
    pub min: Option<f64>,
    /// The max value of the fast field values.
    pub max: Option<f64>,
    /// The average of the fast field values. `None` if count equals zero.
    pub avg: Option<f64>,
}

impl Stats {
    pub(crate) fn get_value(&self, agg_property: &str) -> crate::Result<Option<f64>> {
        match agg_property {
            "count" => Ok(Some(self.count as f64)),
            "sum" => Ok(Some(self.sum)),
            "min" => Ok(self.min),
            "max" => Ok(self.max),
            "avg" => Ok(self.avg),
            _ => Err(TantivyError::InvalidArgument(format!(
                "Unknown property {agg_property} on stats metric aggregation"
            ))),
        }
    }
}

/// Intermediate result of the stats aggregation that can be combined with other intermediate
/// results.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntermediateStats {
    /// The number of extracted values.
    pub(crate) count: u64,
    /// The sum of the extracted values.
    pub(crate) sum: f64,
    /// delta for sum needed for [Kahan algorithm for summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)
    pub(crate) delta: f64,
    /// The min value.
    pub(crate) min: f64,
    /// The max value.
    pub(crate) max: f64,
}

impl Default for IntermediateStats {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            delta: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }
}

impl IntermediateStats {
    /// Returns the number of values collected.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the sum of all values collected.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Merges the other stats intermediate result into self.
    pub fn merge_fruits(&mut self, other: IntermediateStats) {
        self.count += other.count;

        // kahan algorithm for sum
        let y = other.sum - (self.delta + other.delta);
        let t = self.sum + y;
        self.delta = (t - self.sum) - y;
        self.sum = t;

        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Computes the final stats value.
    pub fn finalize(&self) -> Stats {
        let min = if self.count == 0 {
            None
        } else {
            Some(self.min)
        };
        let max = if self.count == 0 {
            None
        } else {
            Some(self.max)
        };
        let avg = if self.count == 0 {
            None
        } else {
            Some(self.sum / (self.count as f64))
        };
        Stats {
            count: self.count,
            sum: self.sum,
            min,
            max,
            avg,
        }
    }

    #[inline]
    pub(in crate::aggregation::metric) fn collect(&mut self, value: f64) {
        self.count += 1;

        // kahan algorithm for sum
        let y = value - self.delta;
        let t = self.sum + y;
        self.delta = (t - self.sum) - y;
        self.sum = t;

        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
}

/// The type of stats aggregation to perform.
/// Note that not all stats types are supported in the stats aggregation.
#[derive(Clone, Copy, Debug)]
pub enum StatsType {
    /// The average of the values.
    Average,
    /// The count of the values.
    Count,
    /// The maximum value.
    Max,
    /// The minimum value.
    Min,
    /// The stats (count, sum, min, max, avg) of the values.
    Stats,
    /// The extended stats (count, sum, min, max, avg, sum_of_squares, variance, std_deviation,
    ExtendedStats(Option<f64>), // sigma
    /// The sum of the values.
    Sum,
    /// The percentiles of the values.
    Percentiles,
}

fn create_collector<const TYPE_ID: u8>(
    req: &MetricAggReqData,
) -> Box<dyn SegmentAggregationCollector> {
    Box::new(SegmentStatsCollector::<TYPE_ID> {
        name: req.name.clone(),
        collecting_for: req.collecting_for,
        is_number_or_date_type: req.is_number_or_date_type,
        missing_u64: req.missing_u64,
        accessor: req.accessor.clone(),
        buckets: vec![IntermediateStats::default()],
        #[cfg(feature = "cuda-stats-kernel")]
        gpu_state: GpuState::initial::<TYPE_ID>(req),
    })
}

/// Build a concrete `SegmentStatsCollector` depending on the column type.
pub(crate) fn build_segment_stats_collector(
    req: &MetricAggReqData,
) -> crate::Result<Box<dyn SegmentAggregationCollector>> {
    match req.field_type {
        ColumnType::I64 => Ok(create_collector::<{ ColumnType::I64 as u8 }>(req)),
        ColumnType::U64 => Ok(create_collector::<{ ColumnType::U64 as u8 }>(req)),
        ColumnType::F64 => Ok(create_collector::<{ ColumnType::F64 as u8 }>(req)),
        ColumnType::Bool => Ok(create_collector::<{ ColumnType::Bool as u8 }>(req)),
        ColumnType::DateTime => Ok(create_collector::<{ ColumnType::DateTime as u8 }>(req)),
        ColumnType::Bytes => Ok(create_collector::<{ ColumnType::Bytes as u8 }>(req)),
        ColumnType::Str => Ok(create_collector::<{ ColumnType::Str as u8 }>(req)),
        ColumnType::IpAddr => Ok(create_collector::<{ ColumnType::IpAddr as u8 }>(req)),
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub(crate) struct SegmentStatsCollector<const COLUMN_TYPE_ID: u8> {
    pub(crate) missing_u64: Option<u64>,
    pub(crate) accessor: Column<u64>,
    pub(crate) is_number_or_date_type: bool,
    pub(crate) buckets: Vec<IntermediateStats>,
    pub(crate) name: String,
    pub(crate) collecting_for: StatsType,
    /// Phase 2 E-2 — GPU staging buffer + gate state. Present only
    /// under the `cuda-stats-kernel` feature; outside that feature
    /// the collector behaves exactly like upstream.
    #[cfg(feature = "cuda-stats-kernel")]
    pub(crate) gpu_state: GpuState,
}

/// Per-collector GPU dispatch state. Holds a staging `Vec<f32>` so
/// many small `collect()` blocks (~64 docs/block) accumulate into a
/// single ≥ [`GPU_STAGING_FLUSH_THRESHOLD`] dispatch — kernel
/// dispatch overhead at ~64-doc blocks would otherwise dominate
/// wallclock and make the integration a regression vs the existing
/// CPU Kahan loop.
///
/// Multi-bucket scenarios (sub-aggs where the parent is a `terms`
/// or `range` agg) are detected by observing
/// `parent_bucket_id != 0` on a `collect()` call: when that fires,
/// the staged values are flushed to the existing CPU path and GPU
/// dispatch is permanently disabled for this collector instance.
/// First-land scope is single-bucket (top-level stats agg) only;
/// per-bucket GPU dispatch is a future wave.
#[cfg(feature = "cuda-stats-kernel")]
#[derive(Clone, Debug)]
pub(crate) enum GpuState {
    /// Either the column type / missing / collecting_for combination
    /// makes GPU dispatch impossible, or the global
    /// [`crate::aggregation::metric::cuda_stats_dispatch::CudaStatsKernel`]
    /// failed to initialise. CPU path takes over for the lifetime of
    /// the collector.
    Disabled,
    /// GPU dispatch is potentially eligible. Staged `f32` values
    /// accumulate here; flushed to GPU when the buffer reaches
    /// [`GPU_STAGING_FLUSH_THRESHOLD`] or in
    /// `add_intermediate_aggregation_result`.
    Eligible {
        /// Staged values, converted via `convert_to_f64::<COLUMN_TYPE_ID>`
        /// then `as f32`. Honest precision limitation: f64 → f32
        /// downcast loses ~7 decimal digits, and the in-block kernel
        /// reduction is f32 too. Memory: ~1 % rel error at 100 M
        /// elements vs the CPU Kahan loop.
        staging: Vec<f32>,
    },
}

#[cfg(feature = "cuda-stats-kernel")]
impl GpuState {
    /// Decide the initial state at collector construction time.
    /// Conservative: only enable GPU when the column is one of the
    /// numeric types whose `convert_to_f64` round-trip we are willing
    /// to absorb (F64 / I64 / U64 / DateTime), no `missing` is set
    /// (else we'd have to materialise the per-doc missing fill on
    /// the GPU input buffer too — possible but deferred), and the
    /// metric is one whose result is fully expressible by
    /// (count, sum, min, max, sum_sq) — which is every `StatsType`
    /// except `Percentiles` (handled by a separate collector anyway).
    pub(crate) fn initial<const COLUMN_TYPE_ID: u8>(req: &MetricAggReqData) -> Self {
        // Gate 1 — column type. Bool / Bytes / Str / IpAddr are
        // handled by the existing `is_number_or_date_type=false`
        // branch in `collect_stats` (which collects `0.0` for every
        // matched doc — irrelevant to the GPU); we'd just be wasting
        // staging memory. Datetime is included because it's
        // i64 microseconds and stats agg over it is a real ES use
        // case (date_histogram fan-out).
        let column_eligible = matches!(
            req.field_type,
            ColumnType::F64 | ColumnType::I64 | ColumnType::U64 | ColumnType::DateTime
        );
        // Gate 2 — missing handling. The existing
        // `fetch_block_with_missing` materialises missing-as-default
        // values into the CPU iterator; we'd need to mirror that on
        // the GPU input. Defer to a future wave; for now fall back
        // when missing is set.
        let no_missing = req.missing_u64.is_none();
        // Gate 3 — collecting_for. Percentiles needs the full value
        // distribution, which we don't preserve in the (count, sum,
        // min, max, sum_sq) reduction. Every other StatsType variant
        // is fully expressible from those five scalars.
        let metric_eligible = !matches!(req.collecting_for, StatsType::Percentiles);
        // Gate 4 — actual numeric semantics. If the column has been
        // marked as non-numeric/date upstream (e.g. it was registered
        // as a count-only metric), the GPU input would be all-zeros
        // and the result would be meaningless.
        let numeric = req.is_number_or_date_type;

        if column_eligible && no_missing && metric_eligible && numeric {
            // We further verify the const generic matches the runtime
            // type so the `convert_to_f64::<COLUMN_TYPE_ID>` we'll
            // call later is the same one create_collector picked.
            // Should always hold (build_segment_stats_collector
            // dispatches by type) but defensive.
            let const_matches = COLUMN_TYPE_ID == req.field_type as u8;
            if const_matches {
                return GpuState::Eligible {
                    staging: Vec::new(),
                };
            }
        }
        GpuState::Disabled
    }
}

impl<const COLUMN_TYPE_ID: u8> SegmentAggregationCollector
    for SegmentStatsCollector<COLUMN_TYPE_ID>
{
    #[inline]
    fn add_intermediate_aggregation_result(
        &mut self,
        agg_data: &AggregationsSegmentCtx,
        results: &mut IntermediateAggregationResults,
        parent_bucket_id: BucketId,
    ) -> crate::Result<()> {
        let name = self.name.clone();

        // GPU staging: drain any remaining staged values into the
        // top-level bucket before emitting. CPU path observes no
        // change since we still merge into `self.buckets[0]`.
        #[cfg(feature = "cuda-stats-kernel")]
        self.flush_gpu_staging();

        self.prepare_max_bucket(parent_bucket_id, agg_data)?;
        let stats = self.buckets[parent_bucket_id as usize];
        let intermediate_metric_result = match self.collecting_for {
            StatsType::Average => {
                IntermediateMetricResult::Average(IntermediateAverage::from_stats(stats))
            }
            StatsType::Count => {
                IntermediateMetricResult::Count(IntermediateCount::from_stats(stats))
            }
            StatsType::Max => IntermediateMetricResult::Max(IntermediateMax::from_stats(stats)),
            StatsType::Min => IntermediateMetricResult::Min(IntermediateMin::from_stats(stats)),
            StatsType::Stats => IntermediateMetricResult::Stats(stats),
            StatsType::Sum => IntermediateMetricResult::Sum(IntermediateSum::from_stats(stats)),
            _ => {
                return Err(TantivyError::InvalidArgument(format!(
                    "Unsupported stats type for stats aggregation: {:?}",
                    self.collecting_for
                )))
            }
        };

        results.push(
            name,
            IntermediateAggregationResult::Metric(intermediate_metric_result),
        )?;

        Ok(())
    }

    #[inline]
    fn collect(
        &mut self,
        parent_bucket_id: BucketId,
        docs: &[crate::DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        // GPU staging fast path. Active only when:
        // - the `cuda-stats-kernel` feature is built in, AND
        // - the gate at construction time approved this collector
        //   (numeric column type, no `missing`, non-percentiles), AND
        // - this is the top-level bucket (parent_bucket_id == 0).
        // Any sub-agg parent bucket → drain staging through CPU and
        // disable GPU dispatch for the rest of this collector's life.
        #[cfg(feature = "cuda-stats-kernel")]
        {
            if matches!(self.gpu_state, GpuState::Eligible { .. }) {
                if parent_bucket_id != 0 {
                    // Multi-bucket scenario detected (sub-agg parent
                    // is a bucket agg). Drain anything we've staged
                    // into the top-level bucket via the CPU path,
                    // then disable GPU permanently.
                    self.flush_gpu_staging();
                    self.gpu_state = GpuState::Disabled;
                } else {
                    self.stage_for_gpu(docs, agg_data);
                    if let GpuState::Eligible { staging } = &self.gpu_state {
                        if staging.len() >= GPU_STAGING_FLUSH_THRESHOLD {
                            self.flush_gpu_staging();
                        }
                    }
                    return Ok(());
                }
            }
        }

        // TODO: remove once we fetch all values for all bucket ids in one go
        if docs.len() == 1 && self.missing_u64.is_none() {
            collect_stats::<COLUMN_TYPE_ID>(
                &mut self.buckets[parent_bucket_id as usize],
                self.accessor.values_for_doc(docs[0]),
                self.is_number_or_date_type,
            )?;

            return Ok(());
        }
        agg_data.column_block_accessor.fetch_block_with_missing(
            docs,
            &self.accessor,
            self.missing_u64,
        );
        collect_stats::<COLUMN_TYPE_ID>(
            &mut self.buckets[parent_bucket_id as usize],
            agg_data.column_block_accessor.iter_vals(),
            self.is_number_or_date_type,
        )?;

        Ok(())
    }

    fn prepare_max_bucket(
        &mut self,
        max_bucket: BucketId,
        _agg_data: &AggregationsSegmentCtx,
    ) -> crate::Result<()> {
        let required_buckets = (max_bucket as usize) + 1;
        if self.buckets.len() < required_buckets {
            self.buckets
                .resize_with(required_buckets, IntermediateStats::default);
        }
        Ok(())
    }

    fn flush(&mut self, _agg_data: &mut AggregationsSegmentCtx) -> crate::Result<()> {
        // Some upstream collectors (e.g. termagg) batch docs and
        // flush once at the end of the segment scan; we mirror the
        // staged-drain here so the per-segment partial result is
        // populated before merge.
        #[cfg(feature = "cuda-stats-kernel")]
        self.flush_gpu_staging();
        Ok(())
    }
}

#[cfg(feature = "cuda-stats-kernel")]
impl<const COLUMN_TYPE_ID: u8> SegmentStatsCollector<COLUMN_TYPE_ID> {
    /// Append the values for `docs` to the GPU staging buffer.
    /// Mirrors the `fetch_block_with_missing` + `iter_vals` block
    /// accessor pattern from the CPU `collect()` path so docs / cache
    /// state observed downstream is identical.
    ///
    /// Per-element overhead = 1 `convert_to_f64::<COLUMN_TYPE_ID>` +
    /// 1 `f64 → f32` cast + 1 `Vec::push`. Amortised against the
    /// later GPU dispatch's bandwidth-saturating throughput.
    #[inline]
    fn stage_for_gpu(
        &mut self,
        docs: &[crate::DocId],
        agg_data: &mut AggregationsSegmentCtx,
    ) {
        let GpuState::Eligible { staging } = &mut self.gpu_state else {
            return;
        };
        // Single-doc fast path matches `collect()`'s upstream shape.
        if docs.len() == 1 && self.missing_u64.is_none() {
            for v in self.accessor.values_for_doc(docs[0]) {
                staging.push(convert_to_f64::<COLUMN_TYPE_ID>(v) as f32);
            }
            return;
        }
        agg_data.column_block_accessor.fetch_block_with_missing(
            docs,
            &self.accessor,
            self.missing_u64,
        );
        // Reserve once for the typical case (~64 docs/block) so the
        // pushes are amortised allocations rather than 64 separate
        // grows.
        staging.reserve(docs.len());
        for v in agg_data.column_block_accessor.iter_vals() {
            staging.push(convert_to_f64::<COLUMN_TYPE_ID>(v) as f32);
        }
    }

    /// Drain the staged `f32` values: dispatch to the GPU if the
    /// kernel is available, otherwise replay through the CPU Kahan
    /// loop. Either way the buffer is cleared and the result merges
    /// into `self.buckets[0]` (the top-level bucket — multi-bucket
    /// is detected by `collect()` and disables GPU before reaching
    /// this point).
    fn flush_gpu_staging(&mut self) {
        let GpuState::Eligible { staging } = &mut self.gpu_state else {
            return;
        };
        if staging.is_empty() {
            return;
        }
        // Take ownership so we can borrow `&mut self.buckets` while
        // operating on the staging buffer. The Vec is left empty in
        // place; ready for the next batch.
        let values = std::mem::take(staging);
        let count = values.len() as u64;

        let kernel = super::cuda_stats_dispatch::global();
        let result = kernel.and_then(|k| k.compute(&values).ok());

        if let Some(stats) = result {
            GPU_STATS_DISPATCH_COUNT.fetch_add(stats.count, AtomicOrdering::Relaxed);
            self.merge_gpu_result(stats);
        } else {
            // GPU dispatch failed (kernel unavailable or runtime
            // error). Fall back to the CPU Kahan loop so the result
            // is still produced exactly. Disable GPU for the rest of
            // this collector's life so we don't pay the staging
            // overhead for nothing.
            GPU_STATS_FALLBACK_COUNT.fetch_add(count, AtomicOrdering::Relaxed);
            self.replay_through_cpu(&values);
            self.gpu_state = GpuState::Disabled;
        }
    }

    /// Merge a GPU [`StatsHostResult`] into the top-level bucket.
    /// Uses the existing `IntermediateStats::collect`-style updates
    /// for min/max so the semantics match exactly when the GPU
    /// dispatch later falls back to CPU. Sum is added directly
    /// (not Kahan-corrected) because the GPU's f32 partial sums
    /// have already absorbed the per-block cancellation; folding
    /// them into the f64 accumulator with Kahan would compensate
    /// for cancellation that no longer exists at this scale.
    fn merge_gpu_result(&mut self, gpu: ferro_compress::StatsHostResult) {
        let bucket = &mut self.buckets[0];
        if gpu.count == 0 {
            return;
        }
        bucket.count += gpu.count;
        // Kahan-style absorb of the GPU partial sum into the f64
        // accumulator — preserves precision when this is one of
        // many flush dispatches.
        let y = gpu.sum - bucket.delta;
        let t = bucket.sum + y;
        bucket.delta = (t - bucket.sum) - y;
        bucket.sum = t;
        // min/max: f32 → f64 lossless promotion.
        let gpu_min = f64::from(gpu.min);
        let gpu_max = f64::from(gpu.max);
        if gpu_min < bucket.min {
            bucket.min = gpu_min;
        }
        if gpu_max > bucket.max {
            bucket.max = gpu_max;
        }
    }

    /// Drain `values` through the CPU Kahan loop, merging into
    /// `self.buckets[0]`. Mirrors `collect_stats` but avoids the
    /// is_number_or_date_type branch (we already gated on numeric
    /// at construction time).
    fn replay_through_cpu(&mut self, values: &[f32]) {
        let bucket = &mut self.buckets[0];
        for &v in values {
            bucket.collect(f64::from(v));
        }
    }
}

#[inline]
fn collect_stats<const COLUMN_TYPE_ID: u8>(
    stats: &mut IntermediateStats,
    vals: impl Iterator<Item = u64>,
    is_number_or_date_type: bool,
) -> crate::Result<()> {
    if is_number_or_date_type {
        for val in vals {
            let val1 = convert_to_f64::<COLUMN_TYPE_ID>(val);
            stats.collect(val1);
        }
    } else {
        for _val in vals {
            // we ignore the value and simply record that we got something
            stats.collect(0.0);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use crate::aggregation::agg_req::{Aggregation, Aggregations};
    use crate::aggregation::agg_result::AggregationResults;
    use crate::aggregation::tests::{
        exec_request_with_query, get_test_index_2_segments, get_test_index_from_values,
    };
    use crate::aggregation::AggregationCollector;
    use crate::query::{AllQuery, TermQuery};
    use crate::schema::{IndexRecordOption, Schema, FAST};
    use crate::{Index, IndexWriter, Term};

    #[test]
    fn test_aggregation_stats_empty_index() -> crate::Result<()> {
        // test index without segments
        let values = vec![];

        let index = get_test_index_from_values(false, &values)?;

        let agg_req_1: Aggregations = serde_json::from_value(json!({
            "stats": {
                "stats": {
                    "field": "score",
                },
            }
        }))
        .unwrap();

        let collector = AggregationCollector::from_aggs(agg_req_1, Default::default());

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let agg_res: AggregationResults = searcher.search(&AllQuery, &collector).unwrap();

        let res: Value = serde_json::from_str(&serde_json::to_string(&agg_res)?)?;
        assert_eq!(
            res["stats"],
            json!({
                "avg": Value::Null,
                "count": 0,
                "max": Value::Null,
                "min": Value::Null,
                "sum": 0.0
            })
        );

        Ok(())
    }

    #[test]
    fn test_aggregation_stats_simple() -> crate::Result<()> {
        let values = vec![10.0];

        let index = get_test_index_from_values(false, &values)?;

        let agg_req_1: Aggregations = serde_json::from_value(json!({
            "stats": {
                "stats": {
                    "field": "score",
                },
            }
        }))
        .unwrap();

        let collector = AggregationCollector::from_aggs(agg_req_1, Default::default());

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let agg_res: AggregationResults = searcher.search(&AllQuery, &collector).unwrap();

        let res: Value = serde_json::from_str(&serde_json::to_string(&agg_res)?)?;
        assert_eq!(
            res["stats"],
            json!({
                "avg": 10.0,
                "count": 1,
                "max": 10.0,
                "min": 10.0,
                "sum": 10.0
            })
        );

        Ok(())
    }

    #[test]
    fn test_aggregation_stats() -> crate::Result<()> {
        let index = get_test_index_2_segments(false)?;

        let reader = index.reader()?;
        let text_field = reader.searcher().schema().get_field("text").unwrap();

        let term_query = TermQuery::new(
            Term::from_field_text(text_field, "cool"),
            IndexRecordOption::Basic,
        );

        let range_agg: Aggregation = {
            serde_json::from_value(json!({
                "range": {
                    "field": "score",
                    "ranges": [ { "from": 3.0f64, "to": 7.0f64 }, { "from": 7.0f64, "to": 19.0f64 }, { "from": 19.0f64, "to": 20.0f64 }  ]
                },
                "aggs": {
                    "stats": {
                        "stats": {
                            "field": "score"
                        }
                    }
                }
            }))
            .unwrap()
        };

        let agg_req_1: Aggregations = serde_json::from_value(json!({
            "stats_i64": {
                "stats": {
                    "field": "score_i64",
                },
            },
            "stats_f64": {
                "stats": {
                    "field": "score_f64",
                },
            },
            "stats": {
                "stats": {
                    "field": "score",
                },
            },
            "count_str": {
                "value_count": {
                    "field": "text",
                },
            },
            "range": range_agg
        }))
        .unwrap();

        let collector = AggregationCollector::from_aggs(agg_req_1, Default::default());

        let searcher = reader.searcher();
        let agg_res: AggregationResults = searcher.search(&term_query, &collector).unwrap();

        let res: Value = serde_json::from_str(&serde_json::to_string(&agg_res)?)?;
        assert_eq!(
            res["stats"],
            json!({
                "avg": 12.142857142857142,
                "count": 7,
                "max": 44.0,
                "min": 1.0,
                "sum": 85.0
            })
        );

        assert_eq!(
            res["stats_i64"],
            json!({
                "avg": 12.142857142857142,
                "count": 7,
                "max": 44.0,
                "min": 1.0,
                "sum": 85.0
            })
        );

        assert_eq!(
            res["stats_f64"],
            json!({
                "avg":  12.214285714285714,
                "count": 7,
                "max": 44.5,
                "min": 1.0,
                "sum": 85.5
            })
        );

        assert_eq!(
            res["range"]["buckets"][2]["stats"],
            json!({
                "avg": 10.666666666666666,
                "count": 3,
                "max": 14.0,
                "min": 7.0,
                "sum": 32.0
            })
        );

        assert_eq!(
            res["range"]["buckets"][3]["stats"],
            json!({
                "avg": serde_json::Value::Null,
                "count": 0,
                "max": serde_json::Value::Null,
                "min": serde_json::Value::Null,
                "sum": 0.0,
            })
        );

        assert_eq!(
            res["count_str"],
            json!({
                "value": 7.0,
            })
        );

        Ok(())
    }

    #[test]
    fn test_stats_json() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let json = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        // => Segment with empty json
        index_writer.add_document(doc!()).unwrap();
        index_writer.commit().unwrap();
        // => Segment with json, but no field partially_empty
        index_writer
            .add_document(doc!(json => json!({"different_field": "blue"})))
            .unwrap();
        index_writer.commit().unwrap();
        //// => Segment with field partially_empty
        index_writer
            .add_document(doc!(json => json!({"partially_empty": 10.0})))
            .unwrap();
        index_writer.add_document(doc!())?;
        index_writer.commit().unwrap();

        let agg_req: Aggregations = serde_json::from_value(json!({
            "my_stats": {
                "stats": {
                    "field": "json.partially_empty"
                },
            }
        }))
        .unwrap();

        let res = exec_request_with_query(agg_req, &index, None)?;

        assert_eq!(
            res["my_stats"],
            json!({
                "avg":  10.0,
                "count": 1,
                "max": 10.0,
                "min": 10.0,
                "sum": 10.0
            })
        );

        Ok(())
    }

    #[test]
    fn test_stats_json_missing() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let json = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        // => Segment with empty json
        index_writer.add_document(doc!()).unwrap();
        index_writer.commit().unwrap();
        // => Segment with json, but no field partially_empty
        index_writer
            .add_document(doc!(json => json!({"different_field": "blue"})))
            .unwrap();
        index_writer.commit().unwrap();
        //// => Segment with field partially_empty
        index_writer
            .add_document(doc!(json => json!({"partially_empty": 10.0})))
            .unwrap();
        index_writer.add_document(doc!())?;
        index_writer.commit().unwrap();

        let agg_req: Aggregations = serde_json::from_value(json!({
            "my_stats": {
                "stats": {
                    "field": "json.partially_empty",
                    "missing": 0.0
                },
            }
        }))
        .unwrap();

        let res = exec_request_with_query(agg_req, &index, None)?;

        assert_eq!(
            res["my_stats"],
            json!({
                "avg":  2.5,
                "count": 4,
                "max": 10.0,
                "min": 0.0,
                "sum": 10.0
            })
        );

        // From string
        let agg_req: Aggregations = serde_json::from_value(json!({
            "my_stats": {
                "stats": {
                    "field": "json.partially_empty",
                    "missing": "0.0"
                },
            }
        }))
        .unwrap();

        let res = exec_request_with_query(agg_req, &index, None)?;

        assert_eq!(
            res["my_stats"],
            json!({
                "avg":  2.5,
                "count": 4,
                "max": 10.0,
                "min": 0.0,
                "sum": 10.0
            })
        );

        Ok(())
    }

    #[test]
    fn test_stats_json_missing_sub_agg() -> crate::Result<()> {
        // This test verifies the `collect` method (in contrast to `collect_block`), which is
        // called when the sub-aggregations are flushed.
        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field("texts", FAST);
        let score_field_f64 = schema_builder.add_f64_field("score", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);

        {
            let mut index_writer = index.writer_for_tests()?;
            // writing the segment
            index_writer.add_document(doc!(
                score_field_f64 => 10.0f64,
                text_field => "a"
            ))?;

            index_writer.add_document(doc!(text_field => "a"))?;

            index_writer.commit()?;
        }

        let agg_req: Aggregations = {
            serde_json::from_value(json!({
                "range_with_stats": {
                    "terms": {
                        "field": "texts"
                    },
                    "aggs": {
                        "my_stats": {
                            "stats": {
                                "field": "score",
                                "missing": 0.0
                            }
                        }
                    }
                }
            }))
            .unwrap()
        };

        let res = exec_request_with_query(agg_req, &index, None)?;

        assert_eq!(
            res["range_with_stats"]["buckets"][0]["my_stats"]["count"],
            2
        );
        assert_eq!(
            res["range_with_stats"]["buckets"][0]["my_stats"]["min"],
            0.0
        );
        assert_eq!(
            res["range_with_stats"]["buckets"][0]["my_stats"]["avg"],
            5.0
        );

        Ok(())
    }

    /// Phase 2 E-2 — large-cohort `stats` agg should fire the GPU
    /// dispatch path at least once (cohort > 100 K hits the
    /// `GPU_STAGING_FLUSH_THRESHOLD`) and produce numerically the
    /// same result as the CPU oracle within `f32` epsilon.
    ///
    /// Skipped silently when the kernel cannot initialise (CPU-only
    /// CI runner without a GPU); the assertion on `dispatch_count
    /// > 0` only triggers if the GPU is actually present.
    #[cfg(feature = "cuda-stats-kernel")]
    #[test]
    fn gpu_dispatch_fires_on_large_cohort() -> crate::Result<()> {
        use crate::aggregation::metric::stats::{
            reset_gpu_stats_counters, snapshot_gpu_stats_counters,
        };

        const N: usize = 120_000; // safely above GPU_STAGING_FLUSH_THRESHOLD = 100 K
        let mut schema_builder = Schema::builder();
        let score = schema_builder.add_f64_field("score", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;
        for i in 0..N {
            writer.add_document(doc!(score => (i as f64) * 0.5))?;
        }
        writer.commit()?;

        let agg_req: Aggregations = serde_json::from_value(json!({
            "all_stats": { "stats": { "field": "score" } }
        }))
        .unwrap();
        let collector = AggregationCollector::from_aggs(agg_req, Default::default());

        // Counter snapshot before — independent of any other test
        // running in parallel because we add to the relative delta,
        // not to absolute counts.
        reset_gpu_stats_counters();
        let kernel_present =
            crate::aggregation::metric::cuda_stats_dispatch::global().is_some();

        let reader = index.reader()?;
        let searcher = reader.searcher();
        let agg_res: AggregationResults = searcher.search(&AllQuery, &collector).unwrap();
        let res: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&agg_res)?)?;

        // Numerical correctness — sum of arithmetic series 0.5 ·
        // 0..N. With f32 staging we accept ~1 % relative error at
        // this scale; tolerance set conservatively here.
        let expected_sum: f64 = (N as f64) * (N as f64 - 1.0) * 0.5 / 2.0;
        let got_sum = res["all_stats"]["sum"].as_f64().unwrap();
        let rel_err = (got_sum - expected_sum).abs() / expected_sum;
        assert!(
            rel_err < 1e-2,
            "sum rel_err={rel_err}: got {got_sum} expected {expected_sum}"
        );
        assert_eq!(res["all_stats"]["count"].as_u64().unwrap() as usize, N);
        assert!((res["all_stats"]["min"].as_f64().unwrap() - 0.0).abs() < 1e-3);
        let expected_max = (N as f64 - 1.0) * 0.5;
        assert!((res["all_stats"]["max"].as_f64().unwrap() - expected_max).abs() < 1.0);

        // Counter assertion — only meaningful when the GPU is
        // actually present. On CPU-only runners
        // `gpu_dispatch_count + gpu_fallback_count` should still
        // sum to ≥ N because the staging flush path always bumps
        // one of the two counters.
        let (dispatched, fellback) = snapshot_gpu_stats_counters();
        assert!(
            dispatched + fellback >= N as u64,
            "dispatched={dispatched} fellback={fellback} N={N} — staging path counters \
             must account for every value flushed"
        );
        if kernel_present {
            assert!(
                dispatched > 0,
                "GPU kernel was available but no values were dispatched (dispatched=0). \
                 Staging-flush gate may be misconfigured."
            );
        }

        Ok(())
    }

    /// Phase 2 E-2 — multi-bucket scenario (range agg + nested
    /// stats sub-agg) must drain the GPU staging buffer through the
    /// CPU path on the first sub-bucket call and then fall back
    /// permanently. The result for both top-level and nested stats
    /// must match the CPU oracle exactly (no f32 precision loss
    /// possible because GPU is disabled before the flush).
    #[cfg(feature = "cuda-stats-kernel")]
    #[test]
    fn gpu_disables_on_multi_bucket_path() -> crate::Result<()> {
        let mut schema_builder = Schema::builder();
        let score = schema_builder.add_f64_field("score", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut writer: IndexWriter = index.writer_for_tests()?;
        // 12 docs with deterministic small values; small enough that
        // even on the GPU path we'd get bit-exact f32 = f64 results,
        // so the precision check is sharp.
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        for v in values {
            writer.add_document(doc!(score => v))?;
        }
        writer.commit()?;

        let agg_req: Aggregations = serde_json::from_value(json!({
            "ranges": {
                "range": {
                    "field": "score",
                    "ranges": [{ "from": 1.0, "to": 7.0 }, { "from": 7.0, "to": 13.0 }]
                },
                "aggs": {
                    "bucket_stats": { "stats": { "field": "score" } }
                }
            }
        }))
        .unwrap();
        let collector = AggregationCollector::from_aggs(agg_req, Default::default());
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let agg_res: AggregationResults =
            searcher.search(&AllQuery, &collector).unwrap();
        let res: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&agg_res)?)?;

        // Range agg emits 4 buckets: synthetic [-inf,1) at index 0,
        // user [1,7) at index 1, user [7,13) at index 2, synthetic
        // [13,+inf) at index 3.
        // bucket 1: [1,7) -> docs 1..6 -> count=6 sum=21
        let b1 = &res["ranges"]["buckets"][1]["bucket_stats"];
        assert_eq!(b1["count"].as_u64().unwrap(), 6, "buckets[1] (1-7): {b1:?}");
        assert!(
            (b1["sum"].as_f64().unwrap() - 21.0).abs() < 1e-9,
            "buckets[1] sum: {b1:?}"
        );
        // bucket 2: [7,13) -> docs 7..12 -> count=6 sum=57
        let b2 = &res["ranges"]["buckets"][2]["bucket_stats"];
        assert_eq!(b2["count"].as_u64().unwrap(), 6, "buckets[2] (7-13): {b2:?}");
        assert!(
            (b2["sum"].as_f64().unwrap() - 57.0).abs() < 1e-9,
            "buckets[2] sum: {b2:?}"
        );

        Ok(())
    }
}
