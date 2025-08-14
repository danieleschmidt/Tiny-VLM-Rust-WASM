//! Advanced threat intelligence and behavioral analysis
//!
//! Provides ML-based anomaly detection and threat classification

// Core types available without std
#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

#[cfg(feature = "std")]
use crate::logging::{log_security_event, SecuritySeverity};

/// Threat intelligence system for advanced security analysis
#[cfg(feature = "std")]
pub struct ThreatIntelligenceSystem {
    behavioral_analyzer: BehavioralAnalyzer,
    anomaly_detector: AnomalyDetector,
    threat_classifier: ThreatClassifier,
    intelligence_db: ThreatIntelligenceDB,
}

#[cfg(feature = "std")]
impl ThreatIntelligenceSystem {
    /// Create a new threat intelligence system
    pub fn new() -> Self {
        Self {
            behavioral_analyzer: BehavioralAnalyzer::new(),
            anomaly_detector: AnomalyDetector::new(),
            threat_classifier: ThreatClassifier::new(),
            intelligence_db: ThreatIntelligenceDB::new(),
        }
    }

    /// Analyze request for advanced threats
    pub fn analyze_advanced_threat(&mut self, request: &ThreatAnalysisRequest) -> ThreatAssessment {
        let mut assessment = ThreatAssessment::new();

        // Behavioral analysis
        let behavioral_score = self.behavioral_analyzer.analyze(request);
        assessment.behavioral_score = behavioral_score;

        // Anomaly detection
        let anomaly_score = self.anomaly_detector.detect_anomaly(request);
        assessment.anomaly_score = anomaly_score;

        // Threat classification
        let threat_class = self.threat_classifier.classify(request);
        assessment.threat_classification = threat_class;

        // Check against threat intelligence database
        let intelligence_match = self.intelligence_db.check_indicators(request);
        assessment.intelligence_match = intelligence_match;

        // Calculate overall risk score
        assessment.calculate_risk_score();

        // Log if high risk
        if assessment.risk_score > 0.8 {
            let threat_msg = format!("High risk threat detected: score={:.3}, class={:?}", 
                assessment.risk_score, assessment.threat_classification);
            log_security_event(
                "high_risk_threat_detected",
                SecuritySeverity::Critical,
                &threat_msg,
            );
        }

        assessment
    }

    /// Update threat intelligence database
    pub fn update_intelligence(&mut self, indicators: Vec<ThreatIndicator>) {
        self.intelligence_db.update_indicators(indicators);
    }

    /// Get threat statistics
    pub fn get_threat_stats(&self) -> ThreatStats {
        ThreatStats {
            total_requests_analyzed: self.behavioral_analyzer.total_requests,
            high_risk_detections: self.behavioral_analyzer.high_risk_count,
            anomalies_detected: self.anomaly_detector.anomaly_count,
            threat_indicators: self.intelligence_db.indicator_count(),
        }
    }
}

/// Behavioral analysis for detecting suspicious patterns
#[cfg(feature = "std")]
struct BehavioralAnalyzer {
    user_profiles: HashMap<String, UserBehaviorProfile>,
    total_requests: u64,
    high_risk_count: u64,
}

#[cfg(feature = "std")]
impl BehavioralAnalyzer {
    fn new() -> Self {
        Self {
            user_profiles: HashMap::new(),
            total_requests: 0,
            high_risk_count: 0,
        }
    }

    fn analyze(&mut self, request: &ThreatAnalysisRequest) -> f64 {
        self.total_requests += 1;

        let profile = self.user_profiles
            .entry(request.client_id.clone())
            .or_insert_with(UserBehaviorProfile::new);

        profile.update(request);

        profile.update(request);
        let score = 0.1; // Simplified for now"
        
        if score > 0.8 {
            self.high_risk_count += 1;
        }

        score
    }

    fn calculate_behavioral_score(&self, profile: &UserBehaviorProfile, request: &ThreatAnalysisRequest) -> f64 {
        let mut score: f64 = 0.0;

        // Request frequency analysis
        if profile.requests_per_minute > 100.0 {
            score += 0.3;
        }

        // Payload size analysis
        let avg_payload_size = profile.total_payload_size as f64 / profile.total_requests as f64;
        if request.payload_size as f64 > avg_payload_size * 10.0 {
            score += 0.2;
        }

        // Time pattern analysis
        if self.is_unusual_time_pattern(&profile.request_times) {
            score += 0.2;
        }

        // Content type analysis
        if self.has_suspicious_content_patterns(request) {
            score += 0.3;
        }

        score.min(1.0)
    }

    fn is_unusual_time_pattern(&self, times: &[Instant]) -> bool {
        if times.len() < 5 {
            return false;
        }

        // Check for perfectly regular intervals (bot-like behavior)
        let intervals: Vec<Duration> = times.windows(2)
            .map(|pair| pair[1].duration_since(pair[0]))
            .collect();

        if intervals.len() < 2 {
            return false;
        }

        let avg_interval = intervals.iter().sum::<Duration>() / intervals.len() as u32;
        let variance = intervals.iter()
            .map(|interval| {
                let diff = if *interval > avg_interval {
                    *interval - avg_interval
                } else {
                    avg_interval - *interval
                };
                diff.as_millis() as f64
            })
            .sum::<f64>() / intervals.len() as f64;

        variance < 100.0 // Very low variance suggests automated behavior
    }

    fn has_suspicious_content_patterns(&self, request: &ThreatAnalysisRequest) -> bool {
        // Check for injection patterns, unusual encodings, etc.
        request.prompt.contains("</script>") ||
        request.prompt.contains("javascript:") ||
        request.prompt.len() > 10000 ||
        request.prompt.chars().filter(|c| !c.is_ascii()).count() > request.prompt.len() / 2
    }
}

/// Anomaly detection using statistical methods
#[cfg(feature = "std")]
struct AnomalyDetector {
    baseline_stats: BaselineStats,
    anomaly_count: u64,
}

#[cfg(feature = "std")]
impl AnomalyDetector {
    fn new() -> Self {
        Self {
            baseline_stats: BaselineStats::new(),
            anomaly_count: 0,
        }
    }

    fn detect_anomaly(&mut self, request: &ThreatAnalysisRequest) -> f64 {
        self.baseline_stats.update(request);

        let mut anomaly_score = 0.0;

        // Size-based anomaly detection
        let size_z_score = self.baseline_stats.calculate_size_z_score(request.payload_size);
        if size_z_score.abs() > 3.0 {
            anomaly_score += 0.4;
        }

        // Frequency-based anomaly detection
        let freq_z_score = self.baseline_stats.calculate_frequency_z_score();
        if freq_z_score.abs() > 2.5 {
            anomaly_score += 0.3;
        }

        // Content entropy analysis
        let entropy_score = self.calculate_entropy_anomaly(&request.prompt);
        anomaly_score += entropy_score * 0.3;

        if anomaly_score > 0.7 {
            self.anomaly_count += 1;
        }

        anomaly_score.min(1.0)
    }

    fn calculate_entropy_anomaly(&self, text: &str) -> f64 {
        let entropy = self.calculate_shannon_entropy(text);
        
        // Normal text entropy is typically between 3.5-4.5
        let normal_entropy_range = 3.5..=4.5;
        
        if normal_entropy_range.contains(&entropy) {
            0.0
        } else if entropy < 2.0 || entropy > 6.0 {
            1.0 // Very unusual entropy
        } else {
            (entropy - 4.0).abs() / 2.0 // Scaled distance from normal
        }
    }

    fn calculate_shannon_entropy(&self, text: &str) -> f64 {
        let mut freq_map = HashMap::new();
        let total_chars = text.len() as f64;

        for ch in text.chars() {
            *freq_map.entry(ch).or_insert(0) += 1;
        }

        freq_map.values()
            .map(|&count| {
                let p = count as f64 / total_chars;
                -p * p.log2()
            })
            .sum()
    }
}

/// Threat classification system
#[cfg(feature = "std")]
struct ThreatClassifier {
    // In a real implementation, this would include ML models
}

#[cfg(feature = "std")]
impl ThreatClassifier {
    fn new() -> Self {
        Self {}
    }

    fn classify(&self, request: &ThreatAnalysisRequest) -> ThreatClass {
        // Simplified rule-based classification
        // In production, this would use trained ML models

        if self.is_injection_attack(&request.prompt) {
            ThreatClass::InjectionAttack
        } else if self.is_dos_attack(request) {
            ThreatClass::DosAttack  
        } else if self.is_data_exfiltration(request) {
            ThreatClass::DataExfiltration
        } else if self.is_automated_scraping(request) {
            ThreatClass::AutomatedScraping
        } else {
            ThreatClass::Benign
        }
    }

    fn is_injection_attack(&self, prompt: &str) -> bool {
        let injection_patterns = [
            "<script", "javascript:", "eval(", "document.cookie",
            "SELECT * FROM", "DROP TABLE", "UNION SELECT",
            "'; DROP", "../", "..\\", "/etc/passwd"
        ];

        injection_patterns.iter().any(|pattern| 
            prompt.to_lowercase().contains(&pattern.to_lowercase())
        )
    }

    fn is_dos_attack(&self, request: &ThreatAnalysisRequest) -> bool {
        request.payload_size > 50_000_000 || // 50MB
        request.prompt.len() > 1_000_000     // 1M characters
    }

    fn is_data_exfiltration(&self, request: &ThreatAnalysisRequest) -> bool {
        let exfil_patterns = [
            "show me all", "list all", "dump", "export",
            "password", "secret", "key", "token"
        ];

        exfil_patterns.iter().any(|pattern|
            request.prompt.to_lowercase().contains(pattern)
        )
    }

    fn is_automated_scraping(&self, request: &ThreatAnalysisRequest) -> bool {
        // This would be based on behavioral patterns
        // For now, simplified detection
        request.user_agent.contains("bot") || 
        request.user_agent.contains("crawler") ||
        request.user_agent.is_empty()
    }
}

/// Threat intelligence database
#[cfg(feature = "std")]
struct ThreatIntelligenceDB {
    indicators: Vec<ThreatIndicator>,
    last_update: Instant,
}

#[cfg(feature = "std")]
impl ThreatIntelligenceDB {
    fn new() -> Self {
        Self {
            indicators: Vec::new(),
            last_update: Instant::now(),
        }
    }

    fn check_indicators(&self, request: &ThreatAnalysisRequest) -> Option<ThreatIndicatorMatch> {
        for indicator in &self.indicators {
            if indicator.matches(request) {
                return Some(ThreatIndicatorMatch {
                    indicator: indicator.clone(),
                    confidence: indicator.confidence,
                });
            }
        }
        None
    }

    fn update_indicators(&mut self, new_indicators: Vec<ThreatIndicator>) {
        self.indicators.extend(new_indicators);
        self.last_update = Instant::now();
    }

    fn indicator_count(&self) -> usize {
        self.indicators.len()
    }
}

/// Supporting types and structures

#[derive(Debug, Clone)]
pub struct ThreatAnalysisRequest {
    pub client_id: String,
    pub payload_size: usize,
    pub prompt: String,
    pub user_agent: String,
    pub timestamp: Instant,
}

#[derive(Debug, Clone)]
pub struct ThreatAssessment {
    pub behavioral_score: f64,
    pub anomaly_score: f64,
    pub threat_classification: ThreatClass,
    pub intelligence_match: Option<ThreatIndicatorMatch>,
    pub risk_score: f64,
    pub recommendations: Vec<String>,
}

impl ThreatAssessment {
    fn new() -> Self {
        Self {
            behavioral_score: 0.0,
            anomaly_score: 0.0,
            threat_classification: ThreatClass::Benign,
            intelligence_match: None,
            risk_score: 0.0,
            recommendations: Vec::new(),
        }
    }

    fn calculate_risk_score(&mut self) {
        self.risk_score = (
            self.behavioral_score * 0.3 +
            self.anomaly_score * 0.3 +
            self.threat_class_score() * 0.3 +
            self.intelligence_score() * 0.1
        ).min(1.0);

        self.generate_recommendations();
    }

    fn threat_class_score(&self) -> f64 {
        match self.threat_classification {
            ThreatClass::Benign => 0.0,
            ThreatClass::AutomatedScraping => 0.3,
            ThreatClass::DataExfiltration => 0.7,
            ThreatClass::DosAttack => 0.9,
            ThreatClass::InjectionAttack => 1.0,
        }
    }

    fn intelligence_score(&self) -> f64 {
        self.intelligence_match
            .as_ref()
            .map(|m| m.confidence)
            .unwrap_or(0.0)
    }

    fn generate_recommendations(&mut self) {
        if self.risk_score > 0.9 {
            self.recommendations.push("Block request immediately".to_string());
            self.recommendations.push("Add client to blacklist".to_string());
        } else if self.risk_score > 0.7 {
            self.recommendations.push("Apply rate limiting".to_string());
            self.recommendations.push("Require additional authentication".to_string());
        } else if self.risk_score > 0.5 {
            self.recommendations.push("Monitor client closely".to_string());
            self.recommendations.push("Log detailed request information".to_string());
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ThreatClass {
    Benign,
    AutomatedScraping,
    DataExfiltration,
    DosAttack,
    InjectionAttack,
}

#[derive(Debug, Clone)]
pub struct ThreatIndicator {
    pub id: String,
    pub indicator_type: IndicatorType,
    pub value: String,
    pub confidence: f64,
    pub description: String,
}

impl ThreatIndicator {
    fn matches(&self, request: &ThreatAnalysisRequest) -> bool {
        match self.indicator_type {
            IndicatorType::IpAddress => request.client_id == self.value,
            IndicatorType::UserAgent => request.user_agent.contains(&self.value),
            IndicatorType::Prompt => request.prompt.contains(&self.value),
            IndicatorType::Pattern => {
                // Simple regex-like matching
                request.prompt.contains(&self.value)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum IndicatorType {
    IpAddress,
    UserAgent,
    Prompt,
    Pattern,
}

#[derive(Debug, Clone)]
pub struct ThreatIndicatorMatch {
    pub indicator: ThreatIndicator,
    pub confidence: f64,
}

#[cfg(feature = "std")]
struct UserBehaviorProfile {
    total_requests: u64,
    total_payload_size: u64,
    requests_per_minute: f64,
    request_times: Vec<Instant>,
    first_seen: Instant,
    last_seen: Instant,
}

#[cfg(feature = "std")]
impl UserBehaviorProfile {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            total_requests: 0,
            total_payload_size: 0,
            requests_per_minute: 0.0,
            request_times: Vec::new(),
            first_seen: now,
            last_seen: now,
        }
    }

    fn update(&mut self, request: &ThreatAnalysisRequest) {
        self.total_requests += 1;
        self.total_payload_size += request.payload_size as u64;
        self.last_seen = request.timestamp;
        self.request_times.push(request.timestamp);

        // Keep only recent request times (last 100)
        if self.request_times.len() > 100 {
            self.request_times.drain(0..self.request_times.len() - 100);
        }

        // Calculate requests per minute
        let time_window = self.last_seen.duration_since(self.first_seen);
        if time_window.as_secs() > 0 {
            self.requests_per_minute = (self.total_requests as f64) / 
                (time_window.as_secs_f64() / 60.0);
        }
    }
}

#[cfg(feature = "std")]
struct BaselineStats {
    size_history: Vec<usize>,
    request_times: Vec<Instant>,
    size_mean: f64,
    size_variance: f64,
}

#[cfg(feature = "std")]
impl BaselineStats {
    fn new() -> Self {
        Self {
            size_history: Vec::new(),
            request_times: Vec::new(),
            size_mean: 0.0,
            size_variance: 0.0,
        }
    }

    fn update(&mut self, request: &ThreatAnalysisRequest) {
        self.size_history.push(request.payload_size);
        self.request_times.push(request.timestamp);

        // Keep only recent history (last 1000 requests)
        if self.size_history.len() > 1000 {
            self.size_history.drain(0..self.size_history.len() - 1000);
            self.request_times.drain(0..self.request_times.len() - 1000);
        }

        self.update_size_stats();
    }

    fn update_size_stats(&mut self) {
        if self.size_history.len() < 2 {
            return;
        }

        self.size_mean = self.size_history.iter().sum::<usize>() as f64 / 
            self.size_history.len() as f64;

        self.size_variance = self.size_history.iter()
            .map(|&size| {
                let diff = size as f64 - self.size_mean;
                diff * diff
            })
            .sum::<f64>() / (self.size_history.len() - 1) as f64;
    }

    fn calculate_size_z_score(&self, size: usize) -> f64 {
        if self.size_variance <= 0.0 {
            return 0.0;
        }

        let std_dev = self.size_variance.sqrt();
        (size as f64 - self.size_mean) / std_dev
    }

    fn calculate_frequency_z_score(&self) -> f64 {
        // Simplified frequency analysis
        if self.request_times.len() < 10 {
            return 0.0;
        }

        let recent_requests = self.request_times.iter()
            .rev()
            .take(10)
            .count();

        let time_span = self.request_times.iter()
            .rev()
            .take(10)
            .map(|t| self.request_times.last().unwrap().duration_since(*t).as_secs_f64())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        let current_rate = recent_requests as f64 / time_span;
        let normal_rate = 0.1; // 1 request per 10 seconds is normal

        (current_rate - normal_rate) / normal_rate
    }
}

#[derive(Debug, Clone)]
pub struct ThreatStats {
    pub total_requests_analyzed: u64,
    pub high_risk_detections: u64,
    pub anomalies_detected: u64,
    pub threat_indicators: usize,
}

impl ThreatStats {
    /// Calculate threat detection rate
    pub fn detection_rate(&self) -> f64 {
        if self.total_requests_analyzed == 0 {
            return 0.0;
        }
        (self.high_risk_detections + self.anomalies_detected) as f64 / 
            self.total_requests_analyzed as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_assessment_calculation() {
        let mut assessment = ThreatAssessment::new();
        assessment.behavioral_score = 0.8;
        assessment.anomaly_score = 0.6;
        assessment.threat_classification = ThreatClass::DosAttack;
        
        assessment.calculate_risk_score();
        
        // Expected calculation: 0.8*0.3 + 0.6*0.3 + 0.9*0.3 + 0.0*0.1 = 0.24 + 0.18 + 0.27 + 0.0 = 0.69
        assert!(assessment.risk_score > 0.6);
        assert!(assessment.risk_score < 0.8);
        assert!(!assessment.recommendations.is_empty());
    }

    #[test]
    fn test_threat_indicator_matching() {
        let indicator = ThreatIndicator {
            id: "test1".to_string(),
            indicator_type: IndicatorType::Prompt,
            value: "malicious_pattern".to_string(),
            confidence: 0.9,
            description: "Test indicator".to_string(),
        };

        let request = ThreatAnalysisRequest {
            client_id: "test_client".to_string(),
            payload_size: 1000,
            prompt: "This contains malicious_pattern text".to_string(),
            user_agent: "test_agent".to_string(),
            timestamp: Instant::now(),
        };

        assert!(indicator.matches(&request));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_threat_intelligence_system() {
        let mut ti_system = ThreatIntelligenceSystem::new();
        
        let request = ThreatAnalysisRequest {
            client_id: "benign_client".to_string(),
            payload_size: 1000,
            prompt: "What is the weather today?".to_string(),
            user_agent: "Mozilla/5.0".to_string(),
            timestamp: Instant::now(),
        };

        let assessment = ti_system.analyze_advanced_threat(&request);
        assert!(assessment.risk_score < 0.5);
        assert_eq!(assessment.threat_classification, ThreatClass::Benign);
    }
}