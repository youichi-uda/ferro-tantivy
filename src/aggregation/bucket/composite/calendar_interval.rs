use time::convert::{Day, Microsecond};
use time::{Time, UtcDateTime};

const US_IN_DAY: i64 = Microsecond::per_t::<i64>(Day);

/// Computes the timestamp in microseconds corresponding to the beginning of the
/// year (January 1st at midnight UTC).
pub(super) fn try_year_bucket(timestamp_us: i64) -> crate::Result<i64> {
    year_bucket_using_time_crate(timestamp_us).map_err(|e| {
        crate::TantivyError::InvalidArgument(format!(
            "Failed to compute year bucket for timestamp {}: {e}",
            timestamp_us
        ))
    })
}

/// Computes the timestamp in microseconds corresponding to the beginning of the
/// month (1st at midnight UTC).
pub(super) fn try_month_bucket(timestamp_us: i64) -> crate::Result<i64> {
    month_bucket_using_time_crate(timestamp_us).map_err(|e| {
        crate::TantivyError::InvalidArgument(format!(
            "Failed to compute month bucket for timestamp {}: {e}",
            timestamp_us
        ))
    })
}

/// Computes the timestamp in microseconds corresponding to the beginning of the
/// week (Monday at midnight UTC).
pub(super) fn week_bucket(timestamp_us: i64) -> i64 {
    // 1970-01-01 was a Thursday (weekday = 4)
    let days_since_epoch = timestamp_us.div_euclid(US_IN_DAY);
    // Find the weekday: 0=Monday, ..., 6=Sunday
    let weekday = (days_since_epoch + 3).rem_euclid(7);
    let monday_days_since_epoch = days_since_epoch - weekday;
    monday_days_since_epoch * US_IN_DAY
}

/// Computes the timestamp in microseconds corresponding to the beginning of the
/// day (midnight UTC).
pub(super) fn day_bucket(timestamp_us: i64) -> i64 {
    timestamp_us.div_euclid(US_IN_DAY) * US_IN_DAY
}

fn year_bucket_using_time_crate(timestamp_us: i64) -> Result<i64, time::Error> {
    let timestamp_nanos = (timestamp_us as i128).saturating_mul(1_000);
    let timestamp_ns = UtcDateTime::from_unix_timestamp_nanos(timestamp_nanos)?
        .replace_ordinal(1)?
        .replace_time(Time::MIDNIGHT)
        .unix_timestamp_nanos();
    // Convert back to micros (safe: the replaced time is on a day boundary
    // which fits in i64 micros for the full proleptic Gregorian range).
    Ok((timestamp_ns / 1_000) as i64)
}

fn month_bucket_using_time_crate(timestamp_us: i64) -> Result<i64, time::Error> {
    let timestamp_nanos = (timestamp_us as i128).saturating_mul(1_000);
    let timestamp_ns = UtcDateTime::from_unix_timestamp_nanos(timestamp_nanos)?
        .replace_day(1)?
        .replace_time(Time::MIDNIGHT)
        .unix_timestamp_nanos();
    Ok((timestamp_ns / 1_000) as i64)
}

#[cfg(test)]
mod tests {
    use time::format_description::well_known::Iso8601;
    use time::UtcDateTime;

    use super::*;

    fn ts_us(iso: &str) -> i64 {
        (UtcDateTime::parse(iso, &Iso8601::DEFAULT)
            .unwrap()
            .unix_timestamp_nanos()
            / 1_000) as i64
    }

    #[test]
    fn test_year_bucket() {
        let ts = ts_us("1970-01-01T00:00:00Z");
        let res = try_year_bucket(ts).unwrap();
        assert_eq!(res, ts_us("1970-01-01T00:00:00Z"));

        let ts = ts_us("1970-06-01T10:00:01.010Z");
        let res = try_year_bucket(ts).unwrap();
        assert_eq!(res, ts_us("1970-01-01T00:00:00Z"));

        let ts = ts_us("2008-12-31T23:59:59.999999Z"); // leap year
        let res = try_year_bucket(ts).unwrap();
        assert_eq!(res, ts_us("2008-01-01T00:00:00Z"));

        let ts = ts_us("2008-01-01T00:00:00Z"); // leap year
        let res = try_year_bucket(ts).unwrap();
        assert_eq!(res, ts_us("2008-01-01T00:00:00Z"));

        let ts = ts_us("2010-12-31T23:59:59.999999Z");
        let res = try_year_bucket(ts).unwrap();
        assert_eq!(res, ts_us("2010-01-01T00:00:00Z"));

        let ts = ts_us("1972-06-01T00:10:00Z");
        let res = try_year_bucket(ts).unwrap();
        assert_eq!(res, ts_us("1972-01-01T00:00:00Z"));
    }

    #[test]
    fn test_month_bucket() {
        let ts = ts_us("1970-01-15T00:00:00Z");
        let res = try_month_bucket(ts).unwrap();
        assert_eq!(res, ts_us("1970-01-01T00:00:00Z"));

        let ts = ts_us("1970-02-01T00:00:00Z");
        let res = try_month_bucket(ts).unwrap();
        assert_eq!(res, ts_us("1970-02-01T00:00:00Z"));

        let ts = ts_us("2000-01-31T23:59:59.999999Z");
        let res = try_month_bucket(ts).unwrap();
        assert_eq!(res, ts_us("2000-01-01T00:00:00Z"));
    }

    #[test]
    fn test_week_bucket() {
        let ts = ts_us("1970-01-05T00:00:00Z"); // Monday
        let res = week_bucket(ts);
        assert_eq!(res, ts_us("1970-01-05T00:00:00Z"));

        let ts = ts_us("1970-01-05T23:59:59Z"); // Monday
        let res = week_bucket(ts);
        assert_eq!(res, ts_us("1970-01-05T00:00:00Z"));

        let ts = ts_us("1970-01-07T01:13:00Z"); // Wednesday
        let res = week_bucket(ts);
        assert_eq!(res, ts_us("1970-01-05T00:00:00Z"));

        let ts = ts_us("1970-01-11T23:59:59.999999Z"); // Sunday
        let res = week_bucket(ts);
        assert_eq!(res, ts_us("1970-01-05T00:00:00Z"));

        let ts = ts_us("2025-10-16T10:41:59.010Z"); // Thursday
        let res = week_bucket(ts);
        assert_eq!(res, ts_us("2025-10-13T00:00:00Z"));

        let ts = ts_us("1970-01-01T00:00:00Z"); // Thursday
        let res = week_bucket(ts);
        assert_eq!(res, ts_us("1969-12-29T00:00:00Z")); // Negative
    }

    #[test]
    fn test_day_bucket() {
        let ts = ts_us("2017-10-20T03:08:45Z");
        let res = day_bucket(ts);
        assert_eq!(res, ts_us("2017-10-20T00:00:00Z"));

        let ts = ts_us("2017-10-21T07:00:00Z");
        let res = day_bucket(ts);
        assert_eq!(res, ts_us("2017-10-21T00:00:00Z"));

        let ts = ts_us("2017-10-20T00:00:00Z");
        let res = day_bucket(ts);
        assert_eq!(res, ts_us("2017-10-20T00:00:00Z"));
    }
}
