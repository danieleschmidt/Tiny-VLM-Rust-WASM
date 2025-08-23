//! Advanced Security Framework
//!
//! Enterprise-grade security features including threat detection, anomaly analysis,
//! and multi-layer defense mechanisms.

use crate::{Result, TinyVlmError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Advanced security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSecurityConfig {
    pub enable_anomaly_detection: bool,
    pub enable_behavioral_analysis: bool,
    pub enable_content_scanning: bool,
    pub threat_score_threshold: f64,
    pub quarantine_duration_hours: u64,
    pub max_request_frequency: u32,
    pub enable_geofencing: bool,
    pub allowed_countries: Vec<String>,
    pub enable_model_poisoning_detection: bool,
}

impl Default for AdvancedSecurityConfig {
    fn default() -> Self {
        Self {
            enable_anomaly_detection: true,
            enable_behavioral_analysis: true,
            enable_content_scanning: true,
            threat_score_threshold: 0.7,
            quarantine_duration_hours: 24,
            max_request_frequency: 100,
            enable_geofencing: false,
            allowed_countries: vec!["US".to_string(), "CA".to_string(), "EU".to_string()],
            enable_model_poisoning_detection: true,
        }
    }
}

/// Security threat classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThreatLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}

/// Security event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEvent {
    AnomalousInput {
        request_id: String,
        anomaly_score: f32,
        features: Vec<String>,
    },
    RateLimitExceeded {
        client_id: String,
        current_rate: u32,
        window_minutes: u32,
    },
    SuspiciousGeolocation {
        client_id: String,
        country: String,
        previous_countries: Vec<String>,
    },
    ModelPoisoningAttempt {
        request_id: String,
        poison_indicators: Vec<String>,
        confidence: f64,
    },
    DataExfiltrationAttempt {
        client_id: String,
        data_volume_mb: f32,
        time_window_minutes: u32,
    },
}

/// Comprehensive security analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysis {
    pub threat_level: ThreatLevel,
    pub threat_score: f64,
    pub events: Vec<SecurityEvent>,
    pub recommendations: Vec<String>,
    pub should_block: bool,
    pub quarantine_until: Option<String>,
}

/// Advanced security manager with multi-layer protection
pub struct AdvancedSecurityManager {
    config: AdvancedSecurityConfig,
    anomaly_detector: AnomalyDetector,
    behavioral_analyzer: BehavioralAnalyzer,
    content_scanner: ContentScanner,
    threat_database: ThreatDatabase,
    quarantined_clients: HashMap<String, Instant>,
    client_profiles: HashMap<String, ClientProfile>,
}

impl AdvancedSecurityManager {
    pub fn new(config: AdvancedSecurityConfig) -> Self {
        Self {
            anomaly_detector: AnomalyDetector::new(),
            behavioral_analyzer: BehavioralAnalyzer::new(),
            content_scanner: ContentScanner::new(),
            threat_database: ThreatDatabase::new(),
            quarantined_clients: HashMap::new(),
            client_profiles: HashMap::new(),
            config,
        }
    }

    /// Comprehensive security analysis of an inference request
    pub fn analyze_request(&mut self, request: &SecurityRequest) -> Result<SecurityAnalysis> {
        let mut events = Vec::new();
        let mut threat_score = 0.0_f32;
        let mut recommendations = Vec::new();

        // Check if client is quarantined
        if let Some(quarantine_time) = self.quarantined_clients.get(&request.client_id) {
            let quarantine_duration = Duration::from_secs(self.config.quarantine_duration_hours * 3600);
            if quarantine_time.elapsed() < quarantine_duration {
                return Ok(SecurityAnalysis {
                    threat_level: ThreatLevel::High,
                    threat_score: 1.0,
                    events: vec![],
                    recommendations: vec!["Client is quarantined".to_string()],
                    should_block: true,
                    quarantine_until: Some("quarantined".to_string()),
                });
            } else {
                self.quarantined_clients.remove(&request.client_id);
            }
        }

        // Anomaly detection
        if self.config.enable_anomaly_detection {
            if let Some(anomaly_result) = self.anomaly_detector.detect_anomalies(request)? {
                threat_score += anomaly_result.score * 0.3;
                events.push(SecurityEvent::AnomalousInput {
                    request_id: request.request_id.clone(),
                    anomaly_score: anomaly_result.score,
                    features: anomaly_result.anomalous_features,
                });
                recommendations.push("Input shows anomalous patterns".to_string());
            }
        }

        // Behavioral analysis
        if self.config.enable_behavioral_analysis {
            let profile = self.client_profiles
                .entry(request.client_id.clone())
                .or_insert_with(ClientProfile::new);
            
            if let Some(behavioral_result) = self.behavioral_analyzer.analyze_behavior(request, profile)? {
                threat_score += behavioral_result.risk_score * 0.25;
                
                if behavioral_result.rate_limit_exceeded {
                    events.push(SecurityEvent::RateLimitExceeded {
                        client_id: request.client_id.clone(),
                        current_rate: behavioral_result.current_rate,
                        window_minutes: 5,
                    });
                    recommendations.push("Rate limiting triggered".to_string());
                }
                
                if behavioral_result.suspicious_geolocation {
                    events.push(SecurityEvent::SuspiciousGeolocation {
                        client_id: request.client_id.clone(),
                        country: request.country.clone().unwrap_or_default(),
                        previous_countries: profile.previous_countries.clone(),
                    });
                    recommendations.push("Geolocation change detected".to_string());
                }
            }
        }

        // Content scanning
        if self.config.enable_content_scanning {
            if let Some(content_result) = self.content_scanner.scan_content(request)? {
                threat_score += content_result.threat_score * 0.2;
                
                if content_result.contains_malicious_patterns {
                    recommendations.push("Malicious content patterns detected".to_string());
                }
                
                if content_result.potential_data_exfiltration {
                    events.push(SecurityEvent::DataExfiltrationAttempt {
                        client_id: request.client_id.clone(),
                        data_volume_mb: content_result.data_volume_mb,
                        time_window_minutes: 10,
                    });
                    recommendations.push("Potential data exfiltration attempt".to_string());
                }
            }
        }

        // Model poisoning detection
        if self.config.enable_model_poisoning_detection {
            if let Some(poisoning_result) = self.detect_model_poisoning(request)? {
                threat_score += poisoning_result.confidence * 0.25;
                events.push(SecurityEvent::ModelPoisoningAttempt {
                    request_id: request.request_id.clone(),
                    poison_indicators: poisoning_result.indicators,
                    confidence: poisoning_result.confidence,
                });
                recommendations.push("Model poisoning attempt detected".to_string());
            }
        }

        // Determine threat level and actions
        let threat_level = match threat_score {
            score if score >= 0.8 => ThreatLevel::Critical,
            score if score >= 0.6 => ThreatLevel::High,
            score if score >= 0.4 => ThreatLevel::Medium,
            score if score >= 0.2 => ThreatLevel::Low,
            _ => ThreatLevel::None,
        };

        let should_block = threat_score >= self.config.threat_score_threshold;
        let quarantine_until = if should_block && threat_score >= 0.8 {
            self.quarantined_clients.insert(request.client_id.clone(), Instant::now());
            Some("quarantine_activated".to_string())
        } else {
            None
        };

        // Update threat database
        self.threat_database.update_threat_intel(&request.client_id, threat_score);

        Ok(SecurityAnalysis {
            threat_level,
            threat_score,
            events,
            recommendations,
            should_block,
            quarantine_until,
        })
    }

    fn detect_model_poisoning(&self, request: &SecurityRequest) -> Result<Option<ModelPoisoningResult>> {
        // Advanced model poisoning detection algorithms
        let mut indicators = Vec::new();
        let mut confidence = 0.0;

        // Check for adversarial patterns in input
        if request.input_text.len() > 1000 && request.input_text.chars().filter(|c| !c.is_ascii()).count() > 100 {
            indicators.push("High non-ASCII character density".to_string());
            confidence += 0.3;
        }

        // Check for repeated patterns that might be poisoning attempts
        if self.has_repeated_suspicious_patterns(&request.input_text) {
            indicators.push("Repeated suspicious patterns".to_string());
            confidence += 0.4;
        }

        // Check for known poisoning signatures
        if self.contains_known_poison_signatures(&request.input_text) {
            indicators.push("Known poisoning signatures".to_string());
            confidence += 0.5;
        }

        if confidence > 0.0 {
            Ok(Some(ModelPoisoningResult {
                indicators,
                confidence,
            }))
        } else {
            Ok(None)
        }
    }

    fn has_repeated_suspicious_patterns(&self, text: &str) -> bool {
        // Simplified pattern detection
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 10 {
            return false;
        }

        let mut pattern_counts = HashMap::new();
        for word in &words {
            if word.len() > 3 {
                *pattern_counts.entry(*word).or_insert(0) += 1;
            }
        }

        pattern_counts.values().any(|&count| count > words.len() / 10)
    }

    fn contains_known_poison_signatures(&self, text: &str) -> bool {
        let poison_patterns = [
            "<!->",
            "\\x00",
            "union select",
            "javascript:",
            "<script>",
            "eval(",
        ];

        poison_patterns.iter().any(|pattern| text.to_lowercase().contains(&pattern.to_lowercase()))
    }

    /// Generate security report for monitoring
    pub fn generate_security_report(&self) -> SecurityReport {
        let total_clients = self.client_profiles.len();
        let quarantined_clients = self.quarantined_clients.len();
        let threat_level_distribution = self.calculate_threat_distribution();

        SecurityReport {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            total_clients,
            quarantined_clients,
            threat_level_distribution,
            active_threats: self.get_active_threats(),
            security_recommendations: self.generate_security_recommendations(),
        }
    }

    fn calculate_threat_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        distribution.insert("none".to_string(), 0);
        distribution.insert("low".to_string(), 0);
        distribution.insert("medium".to_string(), 0);
        distribution.insert("high".to_string(), 0);
        distribution.insert("critical".to_string(), 0);
        distribution
    }

    fn get_active_threats(&self) -> Vec<String> {
        vec!["None detected".to_string()]
    }

    fn generate_security_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if self.quarantined_clients.len() > 10 {
            recommendations.push("High number of quarantined clients - review security policies".to_string());
        }
        
        if self.client_profiles.len() > 10000 {
            recommendations.push("Consider implementing client profile archiving".to_string());
        }
        
        recommendations.push("Regular security audit recommended".to_string());
        recommendations
    }
}

/// Security request structure for analysis
#[derive(Debug, Clone)]
pub struct SecurityRequest {
    pub request_id: String,
    pub client_id: String,
    pub input_text: String,
    pub input_image_data: Option<Vec<u8>>,
    pub timestamp: Instant,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub country: Option<String>,
}

/// Client behavioral profile
#[derive(Debug, Clone)]
pub struct ClientProfile {
    pub request_count: usize,
    pub first_seen: Instant,
    pub last_seen: Instant,
    pub average_request_size: f64,
    pub request_frequency: f64,
    pub previous_countries: Vec<String>,
    pub typical_request_patterns: Vec<String>,
}

impl ClientProfile {
    pub fn new() -> Self {
        Self {
            request_count: 0,
            first_seen: Instant::now(),
            last_seen: Instant::now(),
            average_request_size: 0.0,
            request_frequency: 0.0,
            previous_countries: Vec::new(),
            typical_request_patterns: Vec::new(),
        }
    }
}

impl Default for ClientProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Anomaly detection engine
pub struct AnomalyDetector {
    baseline_stats: HashMap<String, f64>,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            baseline_stats: HashMap::new(),
        }
    }

    pub fn detect_anomalies(&mut self, request: &SecurityRequest) -> Result<Option<AnomalyResult>> {
        let mut anomalous_features = Vec::new();
        let mut anomaly_score = 0.0_f32;

        // Text length analysis
        let text_length = request.input_text.len() as f32;
        if text_length > 10000.0 {
            anomalous_features.push("Excessive text length".to_string());
            anomaly_score += 0.3;
        }

        // Character distribution analysis
        let non_ascii_ratio = request.input_text.chars()
            .filter(|c| !c.is_ascii())
            .count() as f64 / text_length;
        
        if non_ascii_ratio > 0.3 {
            anomalous_features.push("High non-ASCII character ratio".to_string());
            anomaly_score += 0.2;
        }

        // Entropy analysis
        if self.calculate_entropy(&request.input_text) > 4.5 {
            anomalous_features.push("High entropy in input".to_string());
            anomaly_score += 0.25;
        }

        if anomaly_score > 0.0 {
            Ok(Some(AnomalyResult {
                score: anomaly_score.min(1.0_f32),
                anomalous_features,
            }))
        } else {
            Ok(None)
        }
    }

    fn calculate_entropy(&self, text: &str) -> f64 {
        let mut char_counts = HashMap::new();
        let total_chars = text.len() as f64;

        for c in text.chars() {
            *char_counts.entry(c).or_insert(0) += 1;
        }

        char_counts.values()
            .map(|&count| {
                let probability = count as f64 / total_chars;
                -probability * probability.log2()
            })
            .sum()
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Behavioral analysis engine
pub struct BehavioralAnalyzer {
    rate_limits: HashMap<String, Vec<Instant>>,
}

impl BehavioralAnalyzer {
    pub fn new() -> Self {
        Self {
            rate_limits: HashMap::new(),
        }
    }

    pub fn analyze_behavior(&mut self, request: &SecurityRequest, profile: &mut ClientProfile) -> Result<Option<BehavioralResult>> {
        let mut risk_score = 0.0_f32;
        let mut rate_limit_exceeded = false;
        let mut suspicious_geolocation = false;

        // Update profile
        profile.request_count += 1;
        profile.last_seen = Instant::now();

        // Rate limiting analysis
        let request_times = self.rate_limits
            .entry(request.client_id.clone())
            .or_insert_with(Vec::new);
        
        request_times.push(Instant::now());
        
        // Keep only requests from the last 5 minutes
        let five_minutes_ago = Instant::now() - Duration::from_secs(300);
        request_times.retain(|&time| time > five_minutes_ago);
        
        let current_rate = request_times.len() as u32;
        if current_rate > 100 {
            rate_limit_exceeded = true;
            risk_score += 0.4;
        }

        // Geolocation analysis
        if let Some(country) = &request.country {
            if !profile.previous_countries.contains(country) {
                profile.previous_countries.push(country.clone());
                if profile.previous_countries.len() > 3 {
                    suspicious_geolocation = true;
                    risk_score += 0.3;
                }
            }
        }

        // Request size analysis
        let request_size = request.input_text.len() as f64;
        if profile.request_count > 10 {
            let size_deviation = (request_size - profile.average_request_size).abs() / profile.average_request_size;
            if size_deviation > 2.0 {
                risk_score += 0.2;
            }
        }
        
        // Update average request size
        profile.average_request_size = ((profile.average_request_size * (profile.request_count - 1) as f64) + request_size) / profile.request_count as f64;

        if risk_score > 0.0 || rate_limit_exceeded || suspicious_geolocation {
            Ok(Some(BehavioralResult {
                risk_score: risk_score.min(1.0_f32),
                rate_limit_exceeded,
                current_rate,
                suspicious_geolocation,
            }))
        } else {
            Ok(None)
        }
    }
}

impl Default for BehavioralAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Content scanning engine
pub struct ContentScanner {
    malicious_patterns: Vec<String>,
}

impl ContentScanner {
    pub fn new() -> Self {
        Self {
            malicious_patterns: vec![
                "script".to_string(),
                "eval".to_string(),
                "exec".to_string(),
                "system".to_string(),
                "shell".to_string(),
            ],
        }
    }

    pub fn scan_content(&self, request: &SecurityRequest) -> Result<Option<ContentScanResult>> {
        let mut threat_score = 0.0_f32;
        let mut contains_malicious_patterns = false;
        let mut potential_data_exfiltration = false;

        // Scan for malicious patterns
        for pattern in &self.malicious_patterns {
            if request.input_text.to_lowercase().contains(pattern) {
                contains_malicious_patterns = true;
                threat_score += 0.2;
            }
        }

        // Check for potential data exfiltration
        let data_volume_mb = request.input_text.len() as f32 / (1024.0 * 1024.0);
        if data_volume_mb > 10.0 {
            potential_data_exfiltration = true;
            threat_score += 0.3;
        }

        // Check for structured data patterns
        if self.contains_structured_data(&request.input_text) {
            threat_score += 0.15;
        }

        if threat_score > 0.0 {
            Ok(Some(ContentScanResult {
                threat_score: threat_score.min(1.0_f32),
                contains_malicious_patterns,
                potential_data_exfiltration,
                data_volume_mb,
            }))
        } else {
            Ok(None)
        }
    }

    fn contains_structured_data(&self, text: &str) -> bool {
        // Check for JSON, XML, SQL patterns that might indicate data exfiltration
        text.contains("SELECT") || 
        text.contains("{\"") || 
        text.contains("<?xml") ||
        text.matches(",").count() > text.len() / 50
    }
}

impl Default for ContentScanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Threat intelligence database
pub struct ThreatDatabase {
    threat_scores: HashMap<String, f64>,
    known_threats: Vec<String>,
}

impl ThreatDatabase {
    pub fn new() -> Self {
        Self {
            threat_scores: HashMap::new(),
            known_threats: Vec::new(),
        }
    }

    pub fn update_threat_intel(&mut self, client_id: &str, threat_score: f64) {
        self.threat_scores.insert(client_id.to_string(), threat_score);
        
        if threat_score > 0.8 {
            self.known_threats.push(client_id.to_string());
        }
    }

    pub fn get_threat_score(&self, client_id: &str) -> f64 {
        self.threat_scores.get(client_id).copied().unwrap_or(0.0)
    }
}

impl Default for ThreatDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// Result structures
#[derive(Debug)]
pub struct AnomalyResult {
    pub score: f32,
    pub anomalous_features: Vec<String>,
}

#[derive(Debug)]
pub struct BehavioralResult {
    pub risk_score: f32,
    pub rate_limit_exceeded: bool,
    pub current_rate: u32,
    pub suspicious_geolocation: bool,
}

#[derive(Debug)]
pub struct ContentScanResult {
    pub threat_score: f32,
    pub contains_malicious_patterns: bool,
    pub potential_data_exfiltration: bool,
    pub data_volume_mb: f32,
}

#[derive(Debug)]
pub struct ModelPoisoningResult {
    pub indicators: Vec<String>,
    pub confidence: f64,
}

/// Security monitoring report
#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityReport {
    pub timestamp: String,
    pub total_clients: usize,
    pub quarantined_clients: usize,
    pub threat_level_distribution: HashMap<String, usize>,
    pub active_threats: Vec<String>,
    pub security_recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_security_config_default() {
        let config = AdvancedSecurityConfig::default();
        assert!(config.enable_anomaly_detection);
        assert_eq!(config.threat_score_threshold, 0.7);
        assert_eq!(config.quarantine_duration_hours, 24);
    }

    #[test]
    fn test_security_manager_creation() {
        let config = AdvancedSecurityConfig::default();
        let manager = AdvancedSecurityManager::new(config);
        assert_eq!(manager.quarantined_clients.len(), 0);
        assert_eq!(manager.client_profiles.len(), 0);
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new();
        let request = SecurityRequest {
            request_id: "test".to_string(),
            client_id: "client1".to_string(),
            input_text: "a".repeat(15000), // Very long text
            input_image_data: None,
            timestamp: Instant::now(),
            ip_address: None,
            user_agent: None,
            country: None,
        };

        let result = detector.detect_anomalies(&request).unwrap();
        assert!(result.is_some());
        let anomaly = result.unwrap();
        assert!(anomaly.score > 0.0);
        assert!(!anomaly.anomalous_features.is_empty());
    }

    #[test]
    fn test_behavioral_analyzer() {
        let mut analyzer = BehavioralAnalyzer::new();
        let mut profile = ClientProfile::new();
        
        let request = SecurityRequest {
            request_id: "test".to_string(),
            client_id: "client1".to_string(),
            input_text: "Normal request".to_string(),
            input_image_data: None,
            timestamp: Instant::now(),
            ip_address: None,
            user_agent: None,
            country: Some("US".to_string()),
        };

        let result = analyzer.analyze_behavior(&request, &mut profile).unwrap();
        assert_eq!(profile.request_count, 1);
        assert!(profile.previous_countries.contains(&"US".to_string()));
    }

    #[test]
    fn test_content_scanner() {
        let scanner = ContentScanner::new();
        let request = SecurityRequest {
            request_id: "test".to_string(),
            client_id: "client1".to_string(),
            input_text: "This contains a script tag".to_string(),
            input_image_data: None,
            timestamp: Instant::now(),
            ip_address: None,
            user_agent: None,
            country: None,
        };

        let result = scanner.scan_content(&request).unwrap();
        assert!(result.is_some());
        let scan_result = result.unwrap();
        assert!(scan_result.contains_malicious_patterns);
        assert!(scan_result.threat_score > 0.0);
    }

    #[test]
    fn test_threat_database() {
        let mut db = ThreatDatabase::new();
        db.update_threat_intel("client1", 0.9);
        
        assert_eq!(db.get_threat_score("client1"), 0.9);
        assert_eq!(db.get_threat_score("unknown"), 0.0);
        assert!(db.known_threats.contains(&"client1".to_string()));
    }

    #[test]
    fn test_security_analysis() {
        let config = AdvancedSecurityConfig::default();
        let mut manager = AdvancedSecurityManager::new(config);
        
        let request = SecurityRequest {
            request_id: "test".to_string(),
            client_id: "client1".to_string(),
            input_text: "Normal request".to_string(),
            input_image_data: None,
            timestamp: Instant::now(),
            ip_address: Some("192.168.1.1".to_string()),
            user_agent: Some("TestAgent/1.0".to_string()),
            country: Some("US".to_string()),
        };

        let analysis = manager.analyze_request(&request).unwrap();
        assert!(matches!(analysis.threat_level, ThreatLevel::None | ThreatLevel::Low));
        assert!(!analysis.should_block);
    }

    #[test]
    fn test_malicious_request_detection() {
        let config = AdvancedSecurityConfig::default();
        let mut manager = AdvancedSecurityManager::new(config);
        
        let malicious_request = SecurityRequest {
            request_id: "test".to_string(),
            client_id: "attacker".to_string(),
            input_text: "<script>eval(malicious_code);</script>".repeat(1000),
            input_image_data: None,
            timestamp: Instant::now(),
            ip_address: Some("192.168.1.100".to_string()),
            user_agent: Some("AttackBot/1.0".to_string()),
            country: Some("XX".to_string()),
        };

        let analysis = manager.analyze_request(&malicious_request).unwrap();
        assert!(analysis.threat_score > 0.5);
        assert!(!analysis.events.is_empty());
        assert!(!analysis.recommendations.is_empty());
    }
}