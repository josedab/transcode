//! PNG filter operations.

/// PNG filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// No filter.
    None = 0,
    /// Sub filter (difference from left pixel).
    Sub = 1,
    /// Up filter (difference from pixel above).
    Up = 2,
    /// Average filter (average of left and above).
    Average = 3,
    /// Paeth filter (predictor based on left, above, upper-left).
    Paeth = 4,
}

impl FilterType {
    /// Create from byte value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(FilterType::None),
            1 => Some(FilterType::Sub),
            2 => Some(FilterType::Up),
            3 => Some(FilterType::Average),
            4 => Some(FilterType::Paeth),
            _ => None,
        }
    }
}

/// Unfilter a row of pixels.
pub fn unfilter_row(
    filter_type: FilterType,
    current: &mut [u8],
    previous: Option<&[u8]>,
    bytes_per_pixel: usize,
) {
    match filter_type {
        FilterType::None => {}
        FilterType::Sub => {
            for i in bytes_per_pixel..current.len() {
                current[i] = current[i].wrapping_add(current[i - bytes_per_pixel]);
            }
        }
        FilterType::Up => {
            if let Some(prev) = previous {
                for i in 0..current.len() {
                    current[i] = current[i].wrapping_add(prev[i]);
                }
            }
        }
        FilterType::Average => {
            if let Some(prev) = previous {
                for i in 0..current.len() {
                    let left = if i >= bytes_per_pixel {
                        current[i - bytes_per_pixel] as u16
                    } else {
                        0
                    };
                    let above = prev[i] as u16;
                    current[i] = current[i].wrapping_add(((left + above) / 2) as u8);
                }
            } else {
                for i in bytes_per_pixel..current.len() {
                    current[i] = current[i].wrapping_add(current[i - bytes_per_pixel] / 2);
                }
            }
        }
        FilterType::Paeth => {
            for i in 0..current.len() {
                let a = if i >= bytes_per_pixel {
                    current[i - bytes_per_pixel]
                } else {
                    0
                };
                let b = previous.map(|p| p[i]).unwrap_or(0);
                let c = if i >= bytes_per_pixel {
                    previous.map(|p| p[i - bytes_per_pixel]).unwrap_or(0)
                } else {
                    0
                };
                current[i] = current[i].wrapping_add(paeth_predictor(a, b, c));
            }
        }
    }
}

/// Filter a row of pixels for encoding.
pub fn filter_row(
    filter_type: FilterType,
    current: &[u8],
    previous: Option<&[u8]>,
    bytes_per_pixel: usize,
    output: &mut [u8],
) {
    match filter_type {
        FilterType::None => {
            output.copy_from_slice(current);
        }
        FilterType::Sub => {
            for i in 0..current.len() {
                let left = if i >= bytes_per_pixel {
                    current[i - bytes_per_pixel]
                } else {
                    0
                };
                output[i] = current[i].wrapping_sub(left);
            }
        }
        FilterType::Up => {
            if let Some(prev) = previous {
                for i in 0..current.len() {
                    output[i] = current[i].wrapping_sub(prev[i]);
                }
            } else {
                output.copy_from_slice(current);
            }
        }
        FilterType::Average => {
            for i in 0..current.len() {
                let left = if i >= bytes_per_pixel {
                    current[i - bytes_per_pixel] as u16
                } else {
                    0
                };
                let above = previous.map(|p| p[i] as u16).unwrap_or(0);
                output[i] = current[i].wrapping_sub(((left + above) / 2) as u8);
            }
        }
        FilterType::Paeth => {
            for i in 0..current.len() {
                let a = if i >= bytes_per_pixel {
                    current[i - bytes_per_pixel]
                } else {
                    0
                };
                let b = previous.map(|p| p[i]).unwrap_or(0);
                let c = if i >= bytes_per_pixel {
                    previous.map(|p| p[i - bytes_per_pixel]).unwrap_or(0)
                } else {
                    0
                };
                output[i] = current[i].wrapping_sub(paeth_predictor(a, b, c));
            }
        }
    }
}

/// Paeth predictor function.
#[inline]
pub fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let pa = (b as i16 - c as i16).abs();
    let pb = (a as i16 - c as i16).abs();
    let pc = (a as i16 + b as i16 - 2 * c as i16).abs();

    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Select the best filter type for a row.
pub fn select_filter(
    current: &[u8],
    previous: Option<&[u8]>,
    bytes_per_pixel: usize,
) -> FilterType {
    // Simple heuristic: try all filters and pick the one with minimum sum of absolute values
    let mut best_filter = FilterType::None;
    let mut best_sum = u64::MAX;

    for filter in [FilterType::None, FilterType::Sub, FilterType::Up, FilterType::Average, FilterType::Paeth] {
        let sum = calculate_filter_sum(filter, current, previous, bytes_per_pixel);
        if sum < best_sum {
            best_sum = sum;
            best_filter = filter;
        }
    }

    best_filter
}

/// Calculate the sum of absolute values for a filtered row.
fn calculate_filter_sum(
    filter_type: FilterType,
    current: &[u8],
    previous: Option<&[u8]>,
    bytes_per_pixel: usize,
) -> u64 {
    let mut sum = 0u64;

    match filter_type {
        FilterType::None => {
            for &byte in current {
                sum += byte as u64;
            }
        }
        FilterType::Sub => {
            for i in 0..current.len() {
                let left = if i >= bytes_per_pixel {
                    current[i - bytes_per_pixel]
                } else {
                    0
                };
                let filtered = current[i].wrapping_sub(left);
                sum += (filtered as i8).unsigned_abs() as u64;
            }
        }
        FilterType::Up => {
            if let Some(prev) = previous {
                for i in 0..current.len() {
                    let filtered = current[i].wrapping_sub(prev[i]);
                    sum += (filtered as i8).unsigned_abs() as u64;
                }
            } else {
                for &byte in current {
                    sum += byte as u64;
                }
            }
        }
        FilterType::Average => {
            for i in 0..current.len() {
                let left = if i >= bytes_per_pixel {
                    current[i - bytes_per_pixel] as u16
                } else {
                    0
                };
                let above = previous.map(|p| p[i] as u16).unwrap_or(0);
                let filtered = current[i].wrapping_sub(((left + above) / 2) as u8);
                sum += (filtered as i8).unsigned_abs() as u64;
            }
        }
        FilterType::Paeth => {
            for i in 0..current.len() {
                let a = if i >= bytes_per_pixel {
                    current[i - bytes_per_pixel]
                } else {
                    0
                };
                let b = previous.map(|p| p[i]).unwrap_or(0);
                let c = if i >= bytes_per_pixel {
                    previous.map(|p| p[i - bytes_per_pixel]).unwrap_or(0)
                } else {
                    0
                };
                let filtered = current[i].wrapping_sub(paeth_predictor(a, b, c));
                sum += (filtered as i8).unsigned_abs() as u64;
            }
        }
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_type() {
        assert_eq!(FilterType::from_u8(0), Some(FilterType::None));
        assert_eq!(FilterType::from_u8(4), Some(FilterType::Paeth));
        assert_eq!(FilterType::from_u8(5), None);
    }

    #[test]
    fn test_unfilter_none() {
        let mut row = vec![10, 20, 30, 40];
        unfilter_row(FilterType::None, &mut row, None, 3);
        assert_eq!(row, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_unfilter_sub() {
        let mut row = vec![10, 20, 30, 5];
        unfilter_row(FilterType::Sub, &mut row, None, 3);
        assert_eq!(row, vec![10, 20, 30, 15]); // 5 + 10 = 15
    }

    #[test]
    fn test_unfilter_up() {
        let prev = vec![5, 10, 15, 20];
        let mut row = vec![1, 2, 3, 4];
        unfilter_row(FilterType::Up, &mut row, Some(&prev), 3);
        assert_eq!(row, vec![6, 12, 18, 24]);
    }

    #[test]
    fn test_paeth_predictor() {
        // When all inputs are 0, result should be 0
        assert_eq!(paeth_predictor(0, 0, 0), 0);

        // Test with specific values
        assert_eq!(paeth_predictor(100, 100, 100), 100);
    }

    #[test]
    fn test_filter_roundtrip() {
        let original = vec![100, 150, 200, 50, 75, 100];
        let previous = vec![50, 60, 70, 80, 90, 100];
        let bytes_per_pixel = 3;

        for filter in [FilterType::None, FilterType::Sub, FilterType::Up, FilterType::Average, FilterType::Paeth] {
            let mut filtered = vec![0u8; original.len()];
            filter_row(filter, &original, Some(&previous), bytes_per_pixel, &mut filtered);

            unfilter_row(filter, &mut filtered, Some(&previous), bytes_per_pixel);
            assert_eq!(filtered, original, "Roundtrip failed for {:?}", filter);
        }
    }
}
