//! Convex-hull optimizer for finding Pareto-optimal encoding points.

use serde::{Deserialize, Serialize};

/// A candidate encoding point with quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityPoint {
    pub width: u32,
    pub height: u32,
    pub bitrate_kbps: u32,
    pub framerate: f64,
    pub vmaf: f64,
    pub ssim: f64,
}

/// The Pareto frontier of quality-vs-bitrate tradeoffs.
pub struct ParetoFront {
    points: Vec<QualityPoint>,
}

impl ParetoFront {
    /// Select rungs from the Pareto front, evenly distributed.
    pub fn select_rungs(&self, min: usize, max: usize) -> Vec<QualityPoint> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let target = self.points.len().clamp(min, max);
        if self.points.len() <= target {
            return self.points.clone();
        }

        // Evenly sample from the sorted front
        let step = (self.points.len() as f64) / (target as f64);
        let mut result = Vec::with_capacity(target);
        for i in 0..target {
            let idx = (i as f64 * step) as usize;
            let idx = idx.min(self.points.len() - 1);
            result.push(self.points[idx].clone());
        }
        result
    }

    pub fn points(&self) -> &[QualityPoint] {
        &self.points
    }
}

/// Finds the Pareto-optimal set of encoding points.
pub struct ConvexHullOptimizer {
    min_vmaf: f64,
    target_vmaf: f64,
}

impl ConvexHullOptimizer {
    pub fn new(min_vmaf: f64, target_vmaf: f64) -> Self {
        Self {
            min_vmaf,
            target_vmaf,
        }
    }

    /// Find the Pareto front: points where no other point has both
    /// lower bitrate AND higher quality.
    pub fn find_pareto_front(&self, candidates: &[QualityPoint]) -> ParetoFront {
        if candidates.is_empty() {
            return ParetoFront { points: Vec::new() };
        }

        // Sort by bitrate ascending
        let mut sorted: Vec<_> = candidates
            .iter()
            .filter(|p| p.vmaf >= self.min_vmaf)
            .cloned()
            .collect();
        sorted.sort_by(|a, b| a.bitrate_kbps.cmp(&b.bitrate_kbps));

        // Extract Pareto front: keep points with strictly increasing VMAF
        let mut front = Vec::new();
        let mut best_vmaf = f64::MIN;

        for point in sorted {
            if point.vmaf > best_vmaf {
                best_vmaf = point.vmaf;
                front.push(point);
            }
        }

        // Deduplicate by resolution — keep lowest bitrate that meets target VMAF,
        // or best VMAF if none meets the target.
        let mut by_resolution: std::collections::HashMap<(u32, u32), QualityPoint> =
            std::collections::HashMap::new();
        for point in &front {
            let key = (point.width, point.height);
            let replace = by_resolution.get(&key).map(|existing| {
                let existing_meets = existing.vmaf >= self.target_vmaf;
                let point_meets = point.vmaf >= self.target_vmaf;
                match (existing_meets, point_meets) {
                    (true, true) => point.bitrate_kbps < existing.bitrate_kbps,
                    (false, true) => true,
                    (true, false) => false,
                    (false, false) => point.vmaf > existing.vmaf,
                }
            }).unwrap_or(true);
            if replace {
                by_resolution.insert(key, point.clone());
            }
        }

        let mut result: Vec<_> = by_resolution.into_values().collect();
        result.sort_by_key(|p| p.bitrate_kbps);

        // Keep all Pareto-optimal points (the rung selection will limit count)
        ParetoFront { points: result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pareto_front_filters_dominated() {
        let optimizer = ConvexHullOptimizer::new(70.0, 95.0);
        let candidates = vec![
            QualityPoint { width: 640, height: 360, bitrate_kbps: 400, framerate: 30.0, vmaf: 75.0, ssim: 0.85 },
            QualityPoint { width: 640, height: 360, bitrate_kbps: 800, framerate: 30.0, vmaf: 72.0, ssim: 0.83 }, // dominated
            QualityPoint { width: 1280, height: 720, bitrate_kbps: 1500, framerate: 30.0, vmaf: 88.0, ssim: 0.92 },
            QualityPoint { width: 1920, height: 1080, bitrate_kbps: 4000, framerate: 30.0, vmaf: 95.0, ssim: 0.97 },
        ];

        let front = optimizer.find_pareto_front(&candidates);
        // Should have 3 non-dominated points (one per resolution)
        assert!(front.points().len() >= 2);
        // Should be sorted by bitrate
        for w in front.points().windows(2) {
            assert!(w[0].bitrate_kbps <= w[1].bitrate_kbps);
        }
    }

    #[test]
    fn test_empty_candidates() {
        let optimizer = ConvexHullOptimizer::new(70.0, 95.0);
        let front = optimizer.find_pareto_front(&[]);
        assert!(front.points().is_empty());
    }

    #[test]
    fn test_select_rungs_limits() {
        let front = ParetoFront {
            points: (0..10)
                .map(|i| QualityPoint {
                    width: 640,
                    height: 360,
                    bitrate_kbps: (i + 1) * 500,
                    framerate: 30.0,
                    vmaf: 70.0 + i as f64 * 3.0,
                    ssim: 0.8 + i as f64 * 0.02,
                })
                .collect(),
        };

        let selected = front.select_rungs(3, 5);
        assert!(selected.len() >= 3);
        assert!(selected.len() <= 5);
    }
}
