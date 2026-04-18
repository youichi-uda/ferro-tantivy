use std::fmt;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use time::format_description::well_known::Rfc3339;
use time::{OffsetDateTime, PrimitiveDateTime, UtcOffset};

use crate::BinarySerializable;

/// Precision with which datetimes are truncated when stored in fast fields. This setting is only
/// relevant for fast fields. In the docstore, datetimes are always saved with nanosecond precision.
#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "lowercase")]
pub enum DateTimePrecision {
    /// Second precision.
    #[default]
    Seconds,
    /// Millisecond precision.
    Milliseconds,
    /// Microsecond precision.
    Microseconds,
    /// Nanosecond precision.
    Nanoseconds,
}

/// A date/time value stored with microseconds precision.
///
/// Internally we use an `i64` microseconds-since-epoch counter. This gives
/// us a usable range of ±292,471 years (vs ±292 years for a nanoseconds
/// counter), which is required to represent the full proleptic Gregorian
/// calendar that Elasticsearch exposes through the `uuuu` / `yyyy` date
/// formats. Conversions to nanosecond precision clamp to `i64::MIN`/`MAX`
/// for values outside the ~1677-2262 nanos epoch window so that relative
/// ordering is preserved.
///
/// This timestamp does not carry any explicit time zone information.
/// Users are responsible for applying the provided conversion
/// functions consistently. Internally the time zone is assumed
/// to be UTC, which is also used implicitly for JSON serialization.
///
/// All constructors and conversions are provided as explicit
/// functions and not by implementing any `From`/`Into` traits
/// to prevent unintended usage.
#[derive(Clone, Default, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DateTime {
    // Timestamp in microseconds. Micros gives us a ±292,471 year range, which
    // is enough to represent any date expressible in the proleptic Gregorian
    // calendar used by Elasticsearch's `uuuu` format. Conversions to nanos
    // clamp to `i64::MIN`/`MAX` outside the nanos-representable window.
    pub(crate) timestamp_micros: i64,
}

impl DateTime {
    /// Minimum possible `DateTime` value.
    pub const MIN: DateTime = DateTime {
        timestamp_micros: i64::MIN,
    };

    /// Maximum possible `DateTime` value.
    pub const MAX: DateTime = DateTime {
        timestamp_micros: i64::MAX,
    };

    /// Clamp-then-multiply: ensures the result stays within `i64` range
    /// without losing relative ordering for values that would overflow.
    /// This preserves sort order for dates outside the representable
    /// epoch window by mapping them to the min/max representable value.
    const fn clamp_mul(value: i64, factor: i64) -> i64 {
        // Pre-compute the safe input range to avoid overflow
        let min_safe = i64::MIN / factor;
        let max_safe = i64::MAX / factor;
        if value < min_safe {
            i64::MIN
        } else if value > max_safe {
            i64::MAX
        } else {
            value * factor
        }
    }

    /// Create new from UNIX timestamp in seconds
    pub const fn from_timestamp_secs(seconds: i64) -> Self {
        Self {
            timestamp_micros: Self::clamp_mul(seconds, 1_000_000),
        }
    }

    /// Create new from UNIX timestamp in milliseconds
    pub const fn from_timestamp_millis(milliseconds: i64) -> Self {
        Self {
            timestamp_micros: Self::clamp_mul(milliseconds, 1_000),
        }
    }

    /// Create new from UNIX timestamp in microseconds.
    pub const fn from_timestamp_micros(microseconds: i64) -> Self {
        Self {
            timestamp_micros: microseconds,
        }
    }

    /// Create new from UNIX timestamp in nanoseconds.
    ///
    /// Nanosecond precision below 1µs is truncated (floor division). Callers
    /// that need sub-microsecond precision should keep their own `i128` value.
    pub const fn from_timestamp_nanos(nanoseconds: i64) -> Self {
        Self {
            timestamp_micros: nanoseconds / 1_000,
        }
    }

    /// Create new from `OffsetDateTime`
    ///
    /// The given date/time is converted to UTC and the actual
    /// time zone is discarded.
    pub fn from_utc(dt: OffsetDateTime) -> Self {
        // `OffsetDateTime::unix_timestamp_nanos` returns an `i128` to avoid
        // overflow outside the nanos-representable window. We store micros,
        // so divide by 1000 after casting.
        let timestamp_micros = (dt.unix_timestamp_nanos() / 1_000) as i64;
        Self { timestamp_micros }
    }

    /// Create new from `PrimitiveDateTime`
    ///
    /// Implicitly assumes that the given date/time is in UTC!
    /// Otherwise the original value must only be reobtained with
    /// [`Self::into_primitive()`].
    pub fn from_primitive(dt: PrimitiveDateTime) -> Self {
        Self::from_utc(dt.assume_utc())
    }

    /// Convert to UNIX timestamp in seconds.
    pub const fn into_timestamp_secs(self) -> i64 {
        // Use Euclidean division so that sub-second values for negative
        // timestamps round towards `-inf` (same direction as truncating
        // micros to seconds in ES / Java semantics).
        self.timestamp_micros.div_euclid(1_000_000)
    }

    /// Convert to UNIX timestamp in milliseconds.
    pub const fn into_timestamp_millis(self) -> i64 {
        self.timestamp_micros.div_euclid(1_000)
    }

    /// Convert to UNIX timestamp in microseconds.
    pub const fn into_timestamp_micros(self) -> i64 {
        self.timestamp_micros
    }

    /// Convert to UNIX timestamp in nanoseconds.
    ///
    /// Clamps to `i64::MIN`/`MAX` when the underlying micros value is
    /// outside the ~1677-2262 nanos-representable window.
    pub const fn into_timestamp_nanos(self) -> i64 {
        Self::clamp_mul(self.timestamp_micros, 1_000)
    }

    /// Convert to UTC `OffsetDateTime`
    pub fn into_utc(self) -> OffsetDateTime {
        // Work in i128 nanos to stay within the `time` crate's supported
        // range even for pre-1677 / post-2262 dates.
        let timestamp_nanos_i128 = (self.timestamp_micros as i128) * 1_000;
        let utc_datetime = OffsetDateTime::from_unix_timestamp_nanos(timestamp_nanos_i128)
            .expect("valid UNIX timestamp");
        debug_assert_eq!(UtcOffset::UTC, utc_datetime.offset());
        utc_datetime
    }

    /// Convert to `OffsetDateTime` with the given time zone
    pub fn into_offset(self, offset: UtcOffset) -> OffsetDateTime {
        self.into_utc().to_offset(offset)
    }

    /// Convert to `PrimitiveDateTime` without any time zone
    ///
    /// The value should have been constructed with [`Self::from_primitive()`].
    /// Otherwise the time zone is implicitly assumed to be UTC.
    pub fn into_primitive(self) -> PrimitiveDateTime {
        let utc_datetime = self.into_utc();
        // Discard the UTC time zone offset
        debug_assert_eq!(UtcOffset::UTC, utc_datetime.offset());
        PrimitiveDateTime::new(utc_datetime.date(), utc_datetime.time())
    }

    /// Truncates the timestamp to the corresponding precision.
    pub fn truncate(self, precision: DateTimePrecision) -> Self {
        let truncated_timestamp_micros = match precision {
            // Truncate using `div_euclid` so that timestamps in the pre-epoch
            // half of the number line round towards `-inf` rather than `0`,
            // which matches ES / Java `Math.floorDiv` semantics.
            DateTimePrecision::Seconds => self.timestamp_micros.div_euclid(1_000_000) * 1_000_000,
            DateTimePrecision::Milliseconds => self.timestamp_micros.div_euclid(1_000) * 1_000,
            DateTimePrecision::Microseconds => self.timestamp_micros,
            // Sub-microsecond precision is not representable in our i64 micros
            // counter; treat as a no-op (the value was already floor-truncated
            // at construction time).
            DateTimePrecision::Nanoseconds => self.timestamp_micros,
        };
        Self {
            timestamp_micros: truncated_timestamp_micros,
        }
    }
}

impl fmt::Debug for DateTime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let utc_rfc3339 = self.into_utc().format(&Rfc3339).map_err(|_| fmt::Error)?;
        f.write_str(&utc_rfc3339)
    }
}

impl BinarySerializable for DateTime {
    fn serialize<W: Write + ?Sized>(&self, writer: &mut W) -> std::io::Result<()> {
        let timestamp_micros = self.into_timestamp_micros();
        <i64 as BinarySerializable>::serialize(&timestamp_micros, writer)
    }

    fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let timestamp_micros = <i64 as BinarySerializable>::deserialize(reader)?;
        Ok(Self::from_timestamp_micros(timestamp_micros))
    }
}
