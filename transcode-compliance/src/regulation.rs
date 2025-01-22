//! Regulatory compliance validation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Regulatory framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Regulation {
    /// GDPR (EU General Data Protection Regulation).
    Gdpr,
    /// CCPA (California Consumer Privacy Act).
    Ccpa,
    /// COPPA (Children's Online Privacy Protection Act).
    Coppa,
    /// HIPAA (Health Insurance Portability and Accountability Act).
    Hipaa,
    /// FCC broadcast regulations.
    Fcc,
    /// EBU broadcast standards.
    Ebu,
    /// ARIB (Japan broadcasting standards).
    Arib,
    /// ATSC (North American broadcast).
    Atsc,
    /// DVB (Digital Video Broadcasting - Europe).
    Dvb,
    /// PCI DSS (Payment Card Industry).
    PciDss,
}

impl Regulation {
    /// Get human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gdpr => "General Data Protection Regulation",
            Self::Ccpa => "California Consumer Privacy Act",
            Self::Coppa => "Children's Online Privacy Protection Act",
            Self::Hipaa => "Health Insurance Portability and Accountability Act",
            Self::Fcc => "FCC Broadcast Regulations",
            Self::Ebu => "EBU Broadcast Standards",
            Self::Arib => "ARIB Broadcasting Standards",
            Self::Atsc => "ATSC Broadcast Standards",
            Self::Dvb => "DVB Broadcasting Standards",
            Self::PciDss => "PCI Data Security Standard",
        }
    }

    /// Get jurisdiction.
    pub fn jurisdiction(&self) -> &'static str {
        match self {
            Self::Gdpr => "European Union",
            Self::Ccpa => "California, USA",
            Self::Coppa | Self::Hipaa | Self::Fcc | Self::Atsc => "United States",
            Self::Ebu | Self::Dvb => "Europe",
            Self::Arib => "Japan",
            Self::PciDss => "Global",
        }
    }

    /// Check if regulation is privacy-related.
    pub fn is_privacy_related(&self) -> bool {
        matches!(
            self,
            Self::Gdpr | Self::Ccpa | Self::Coppa | Self::Hipaa | Self::PciDss
        )
    }

    /// Check if regulation is broadcast-related.
    pub fn is_broadcast_related(&self) -> bool {
        matches!(
            self,
            Self::Fcc | Self::Ebu | Self::Arib | Self::Atsc | Self::Dvb
        )
    }
}

/// Compliance rule definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    /// Rule identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Applicable regulation.
    pub regulation: Regulation,
    /// Severity if violated.
    pub severity: RuleSeverity,
    /// Rule category.
    pub category: RuleCategory,
    /// Conditions that trigger this rule.
    pub conditions: Vec<RuleCondition>,
}

/// Rule severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuleSeverity {
    /// Informational.
    Info,
    /// Advisory - should consider.
    Advisory,
    /// Warning - should fix.
    Warning,
    /// Error - must fix.
    Error,
    /// Critical - blocks processing.
    Critical,
}

/// Rule category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleCategory {
    /// Privacy-related rules.
    Privacy,
    /// Content rating rules.
    ContentRating,
    /// Technical requirements.
    Technical,
    /// Accessibility rules.
    Accessibility,
    /// Metadata requirements.
    Metadata,
    /// Geographic restrictions.
    Geographic,
    /// Data retention rules.
    Retention,
}

/// Rule condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    /// Field to check.
    pub field: String,
    /// Operator.
    pub operator: ConditionOperator,
    /// Value to compare.
    pub value: String,
}

/// Condition operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    Contains,
    NotContains,
    Matches,
    GreaterThan,
    LessThan,
    Exists,
    NotExists,
}

/// Compliance validator.
pub struct ComplianceValidator {
    rules: Vec<ComplianceRule>,
    regulations: Vec<Regulation>,
}

impl ComplianceValidator {
    /// Create a new validator with specific regulations.
    pub fn new(regulations: impl IntoIterator<Item = Regulation>) -> Self {
        let regulations: Vec<_> = regulations.into_iter().collect();
        let rules = Self::default_rules(&regulations);

        Self { rules, regulations }
    }

    /// Create a validator for GDPR compliance.
    pub fn gdpr() -> Self {
        Self::new([Regulation::Gdpr])
    }

    /// Create a validator for FCC compliance.
    pub fn fcc() -> Self {
        Self::new([Regulation::Fcc])
    }

    /// Create a validator for broadcast compliance.
    pub fn broadcast() -> Self {
        Self::new([Regulation::Fcc, Regulation::Ebu, Regulation::Atsc])
    }

    /// Add a custom rule.
    pub fn add_rule(&mut self, rule: ComplianceRule) {
        self.rules.push(rule);
    }

    /// Validate video metadata.
    pub fn validate(&self, context: &ValidationContext) -> ValidationResult {
        let mut violations = Vec::new();

        for rule in &self.rules {
            if self.regulations.contains(&rule.regulation) {
                if let Some(violation) = self.check_rule(rule, context) {
                    violations.push(violation);
                }
            }
        }

        let passed = !violations
            .iter()
            .any(|v| matches!(v.severity, RuleSeverity::Error | RuleSeverity::Critical));

        ValidationResult {
            passed,
            regulations_checked: self.regulations.clone(),
            violations,
            timestamp: chrono::Utc::now(),
        }
    }

    fn check_rule(&self, rule: &ComplianceRule, context: &ValidationContext) -> Option<Violation> {
        for condition in &rule.conditions {
            let field_value = context.get_field(&condition.field);

            let violated = match (&condition.operator, field_value) {
                (ConditionOperator::Exists, None) => true,
                (ConditionOperator::NotExists, Some(_)) => true,
                (ConditionOperator::Equals, Some(v)) => v != condition.value,
                (ConditionOperator::NotEquals, Some(v)) => v == condition.value,
                (ConditionOperator::Contains, Some(v)) => !v.contains(&condition.value),
                (ConditionOperator::NotContains, Some(v)) => v.contains(&condition.value),
                (ConditionOperator::Matches, Some(v)) => {
                    !regex::Regex::new(&condition.value)
                        .map(|r| r.is_match(&v))
                        .unwrap_or(false)
                }
                (ConditionOperator::GreaterThan, Some(v)) => {
                    v.parse::<f64>().unwrap_or(0.0)
                        <= condition.value.parse::<f64>().unwrap_or(0.0)
                }
                (ConditionOperator::LessThan, Some(v)) => {
                    v.parse::<f64>().unwrap_or(0.0)
                        >= condition.value.parse::<f64>().unwrap_or(0.0)
                }
                _ => false,
            };

            if violated {
                return Some(Violation {
                    rule_id: rule.id.clone(),
                    rule_name: rule.name.clone(),
                    regulation: rule.regulation,
                    severity: rule.severity,
                    category: rule.category,
                    message: rule.description.clone(),
                    field: Some(condition.field.clone()),
                    expected: Some(format!("{:?} {}", condition.operator, condition.value)),
                    actual: context.get_field(&condition.field),
                });
            }
        }

        None
    }

    fn default_rules(regulations: &[Regulation]) -> Vec<ComplianceRule> {
        let mut rules = Vec::new();

        if regulations.contains(&Regulation::Gdpr) {
            rules.extend(Self::gdpr_rules());
        }

        if regulations.contains(&Regulation::Fcc) {
            rules.extend(Self::fcc_rules());
        }

        if regulations.contains(&Regulation::Coppa) {
            rules.extend(Self::coppa_rules());
        }

        rules
    }

    fn gdpr_rules() -> Vec<ComplianceRule> {
        vec![
            ComplianceRule {
                id: "GDPR-001".to_string(),
                name: "Consent Required".to_string(),
                description: "Processing personal data requires explicit consent".to_string(),
                regulation: Regulation::Gdpr,
                severity: RuleSeverity::Critical,
                category: RuleCategory::Privacy,
                conditions: vec![RuleCondition {
                    field: "consent_obtained".to_string(),
                    operator: ConditionOperator::Equals,
                    value: "true".to_string(),
                }],
            },
            ComplianceRule {
                id: "GDPR-002".to_string(),
                name: "Data Retention Policy".to_string(),
                description: "Personal data must have defined retention period".to_string(),
                regulation: Regulation::Gdpr,
                severity: RuleSeverity::Error,
                category: RuleCategory::Retention,
                conditions: vec![RuleCondition {
                    field: "retention_period_days".to_string(),
                    operator: ConditionOperator::Exists,
                    value: "".to_string(),
                }],
            },
            ComplianceRule {
                id: "GDPR-003".to_string(),
                name: "Purpose Limitation".to_string(),
                description: "Data processing purpose must be specified".to_string(),
                regulation: Regulation::Gdpr,
                severity: RuleSeverity::Error,
                category: RuleCategory::Privacy,
                conditions: vec![RuleCondition {
                    field: "processing_purpose".to_string(),
                    operator: ConditionOperator::Exists,
                    value: "".to_string(),
                }],
            },
        ]
    }

    fn fcc_rules() -> Vec<ComplianceRule> {
        vec![
            ComplianceRule {
                id: "FCC-001".to_string(),
                name: "Caption Accuracy".to_string(),
                description: "Closed captions must be 99% accurate".to_string(),
                regulation: Regulation::Fcc,
                severity: RuleSeverity::Error,
                category: RuleCategory::Accessibility,
                conditions: vec![RuleCondition {
                    field: "caption_accuracy".to_string(),
                    operator: ConditionOperator::GreaterThan,
                    value: "0.99".to_string(),
                }],
            },
            ComplianceRule {
                id: "FCC-002".to_string(),
                name: "Caption Sync".to_string(),
                description: "Captions must be synchronized within 100ms".to_string(),
                regulation: Regulation::Fcc,
                severity: RuleSeverity::Error,
                category: RuleCategory::Accessibility,
                conditions: vec![RuleCondition {
                    field: "caption_sync_offset_ms".to_string(),
                    operator: ConditionOperator::LessThan,
                    value: "100".to_string(),
                }],
            },
            ComplianceRule {
                id: "FCC-003".to_string(),
                name: "Content Rating".to_string(),
                description: "Broadcast content must include V-Chip rating".to_string(),
                regulation: Regulation::Fcc,
                severity: RuleSeverity::Warning,
                category: RuleCategory::ContentRating,
                conditions: vec![RuleCondition {
                    field: "content_rating".to_string(),
                    operator: ConditionOperator::Exists,
                    value: "".to_string(),
                }],
            },
        ]
    }

    fn coppa_rules() -> Vec<ComplianceRule> {
        vec![
            ComplianceRule {
                id: "COPPA-001".to_string(),
                name: "Parental Consent".to_string(),
                description: "Collecting data from children requires parental consent".to_string(),
                regulation: Regulation::Coppa,
                severity: RuleSeverity::Critical,
                category: RuleCategory::Privacy,
                conditions: vec![RuleCondition {
                    field: "parental_consent".to_string(),
                    operator: ConditionOperator::Equals,
                    value: "true".to_string(),
                }],
            },
            ComplianceRule {
                id: "COPPA-002".to_string(),
                name: "Age Verification".to_string(),
                description: "Age verification must be performed for child-directed content".to_string(),
                regulation: Regulation::Coppa,
                severity: RuleSeverity::Error,
                category: RuleCategory::Privacy,
                conditions: vec![RuleCondition {
                    field: "age_verified".to_string(),
                    operator: ConditionOperator::Equals,
                    value: "true".to_string(),
                }],
            },
        ]
    }
}

/// Context for validation.
#[derive(Debug, Clone, Default)]
pub struct ValidationContext {
    fields: HashMap<String, String>,
}

impl ValidationContext {
    /// Create a new context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a field value.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Get a field value.
    pub fn get_field(&self, key: &str) -> Option<String> {
        self.fields.get(key).cloned()
    }

    /// Check if a field exists.
    pub fn has_field(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    /// Builder pattern for setting fields.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.set(key, value);
        self
    }
}

/// Validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed.
    pub passed: bool,
    /// Regulations that were checked.
    pub regulations_checked: Vec<Regulation>,
    /// Violations found.
    pub violations: Vec<Violation>,
    /// Timestamp of validation.
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ValidationResult {
    /// Get critical violations.
    pub fn critical_violations(&self) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.severity == RuleSeverity::Critical)
            .collect()
    }

    /// Get violations by category.
    pub fn violations_by_category(&self, category: RuleCategory) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.category == category)
            .collect()
    }

    /// Get violations by regulation.
    pub fn violations_by_regulation(&self, regulation: Regulation) -> Vec<&Violation> {
        self.violations
            .iter()
            .filter(|v| v.regulation == regulation)
            .collect()
    }
}

/// A compliance violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Rule ID that was violated.
    pub rule_id: String,
    /// Rule name.
    pub rule_name: String,
    /// Regulation.
    pub regulation: Regulation,
    /// Severity.
    pub severity: RuleSeverity,
    /// Category.
    pub category: RuleCategory,
    /// Human-readable message.
    pub message: String,
    /// Field that caused the violation.
    pub field: Option<String>,
    /// Expected value.
    pub expected: Option<String>,
    /// Actual value.
    pub actual: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regulation_info() {
        let gdpr = Regulation::Gdpr;
        assert_eq!(gdpr.name(), "General Data Protection Regulation");
        assert_eq!(gdpr.jurisdiction(), "European Union");
        assert!(gdpr.is_privacy_related());
        assert!(!gdpr.is_broadcast_related());

        let fcc = Regulation::Fcc;
        assert!(fcc.is_broadcast_related());
        assert!(!fcc.is_privacy_related());
    }

    #[test]
    fn test_gdpr_validation() {
        let validator = ComplianceValidator::gdpr();

        // Missing consent - should fail
        let context = ValidationContext::new();
        let result = validator.validate(&context);
        assert!(!result.passed);

        // With consent - should pass
        let context = ValidationContext::new()
            .with("consent_obtained", "true")
            .with("retention_period_days", "30")
            .with("processing_purpose", "transcoding");

        let result = validator.validate(&context);
        assert!(result.passed);
    }

    #[test]
    fn test_fcc_validation() {
        let validator = ComplianceValidator::fcc();

        let context = ValidationContext::new()
            .with("caption_accuracy", "0.995")
            .with("caption_sync_offset_ms", "50")
            .with("content_rating", "TV-PG");

        let result = validator.validate(&context);
        assert!(result.passed);
    }

    #[test]
    fn test_custom_rule() {
        let mut validator = ComplianceValidator::new([]);

        validator.add_rule(ComplianceRule {
            id: "CUSTOM-001".to_string(),
            name: "Custom Check".to_string(),
            description: "Custom requirement".to_string(),
            regulation: Regulation::Gdpr,
            severity: RuleSeverity::Error,
            category: RuleCategory::Technical,
            conditions: vec![RuleCondition {
                field: "custom_field".to_string(),
                operator: ConditionOperator::Exists,
                value: "".to_string(),
            }],
        });

        // Enable GDPR to activate the rule
        validator.regulations.push(Regulation::Gdpr);

        let context = ValidationContext::new();
        let result = validator.validate(&context);
        assert!(!result.passed);
    }

    #[test]
    fn test_violation_filtering() {
        let validator = ComplianceValidator::gdpr();
        // Set consent_obtained to "false" to trigger GDPR-001 critical violation
        let context = ValidationContext::new()
            .with("consent_obtained", "false");
        let result = validator.validate(&context);

        let critical = result.critical_violations();
        assert!(!critical.is_empty(), "Expected critical violations for GDPR consent rule");

        let privacy = result.violations_by_category(RuleCategory::Privacy);
        assert!(!privacy.is_empty(), "Expected privacy violations for GDPR");
    }
}
