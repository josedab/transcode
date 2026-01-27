//! Auto-scaling logic for the worker pool.

use serde::{Deserialize, Serialize};

/// Policy controlling worker pool auto-scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    pub min_replicas: u32,
    pub max_replicas: u32,
    /// Target pending jobs per worker before scaling up.
    pub target_queue_depth_per_worker: u32,
    /// Queue utilization ratio that triggers scale-up.
    pub scale_up_threshold: f64,
    /// Queue utilization ratio that triggers scale-down.
    pub scale_down_threshold: f64,
    /// Minimum seconds between scaling events.
    pub cooldown_secs: u64,
}

impl Default for ScalingPolicy {
    fn default() -> Self {
        Self {
            min_replicas: 1,
            max_replicas: 20,
            target_queue_depth_per_worker: 2,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_secs: 300,
        }
    }
}

/// Decision from the auto-scaler.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingDecision {
    NoChange,
    ScaleUp { target: u32 },
    ScaleDown { target: u32 },
}

/// Evaluates current load and decides whether to scale the worker pool.
pub struct AutoScaler {
    policy: ScalingPolicy,
}

impl AutoScaler {
    pub fn new(policy: ScalingPolicy) -> Self {
        Self { policy }
    }

    /// Evaluate current queue depth, active workers, and replicas.
    /// Returns a scaling decision.
    pub fn evaluate(
        &self,
        pending_jobs: u32,
        active_workers: u32,
        current_replicas: u32,
    ) -> ScalingDecision {
        if current_replicas == 0 && pending_jobs > 0 {
            return ScalingDecision::ScaleUp {
                target: self.policy.min_replicas.max(1),
            };
        }

        if current_replicas == 0 {
            return ScalingDecision::NoChange;
        }

        let utilization = active_workers as f64 / current_replicas as f64;
        let queue_pressure =
            pending_jobs as f64 / (current_replicas * self.policy.target_queue_depth_per_worker) as f64;

        // Scale up: high utilization AND queue pressure
        if utilization >= self.policy.scale_up_threshold || queue_pressure > 1.0 {
            let desired = ((pending_jobs as f64 / self.policy.target_queue_depth_per_worker as f64)
                .ceil() as u32)
                .max(current_replicas + 1);
            let target = desired.min(self.policy.max_replicas);
            if target > current_replicas {
                return ScalingDecision::ScaleUp { target };
            }
        }

        // Scale down: low utilization AND no queue pressure
        if utilization < self.policy.scale_down_threshold && queue_pressure < 0.5 {
            let desired = (active_workers + 1).max(self.policy.min_replicas);
            if desired < current_replicas {
                return ScalingDecision::ScaleDown { target: desired };
            }
        }

        ScalingDecision::NoChange
    }

    pub fn policy(&self) -> &ScalingPolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_scaler() -> AutoScaler {
        AutoScaler::new(ScalingPolicy {
            min_replicas: 1,
            max_replicas: 10,
            target_queue_depth_per_worker: 2,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_secs: 300,
        })
    }

    #[test]
    fn test_scale_up_on_high_queue() {
        let scaler = default_scaler();
        // 10 pending jobs, 4 active workers, 4 replicas => utilization 100%, queue_pressure > 1
        let decision = scaler.evaluate(10, 4, 4);
        assert!(matches!(decision, ScalingDecision::ScaleUp { .. }));
    }

    #[test]
    fn test_scale_down_on_low_utilization() {
        let scaler = default_scaler();
        // 0 pending, 1 active, 5 replicas => utilization 20%, low pressure
        let decision = scaler.evaluate(0, 1, 5);
        assert!(matches!(decision, ScalingDecision::ScaleDown { .. }));
    }

    #[test]
    fn test_no_change_normal_load() {
        let scaler = default_scaler();
        // 3 pending, 3 active, 5 replicas => utilization 60%, queue pressure 0.3
        let decision = scaler.evaluate(3, 3, 5);
        assert_eq!(decision, ScalingDecision::NoChange);
    }

    #[test]
    fn test_scale_up_from_zero() {
        let scaler = default_scaler();
        let decision = scaler.evaluate(5, 0, 0);
        assert!(matches!(decision, ScalingDecision::ScaleUp { target: 1 }));
    }

    #[test]
    fn test_respects_max_replicas() {
        let scaler = default_scaler();
        // Huge queue but max replicas = 10
        let decision = scaler.evaluate(100, 10, 10);
        // Already at max, can't scale up further
        assert_eq!(decision, ScalingDecision::NoChange);
    }

    #[test]
    fn test_respects_min_replicas() {
        let scaler = default_scaler();
        // No load, 1 active, 2 replicas, min = 1
        let decision = scaler.evaluate(0, 0, 2);
        match decision {
            ScalingDecision::ScaleDown { target } => assert!(target >= 1),
            _ => {} // NoChange is also acceptable
        }
    }
}
