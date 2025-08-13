//! Error recovery and resilience mechanisms for Tiny-VLM
//!
//! Provides automatic error recovery, circuit breakers, and fault tolerance

use crate::{Result, TinyVlmError};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

#[cfg(feature = "std")]
use crate::logging::{log_security_event, SecuritySeverity};

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (blocking requests)
    Open,
    /// Circuit is half-open (testing if service recovered)
    HalfOpen,
}

/// Circuit breaker for fault tolerance
pub struct CircuitBreaker {
    state: CircuitState,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
    failure_threshold: usize,
    recovery_timeout: Duration,
    success_threshold: usize,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(failure_threshold: usize, recovery_timeout: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            failure_threshold,
            recovery_timeout,
            success_threshold: 3, // Require 3 successes to close circuit
        }
    }

    /// Execute operation with circuit breaker protection
    pub fn execute<T, F>(&mut self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        match self.state {
            CircuitState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() >= self.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        self.success_count = 0;
                    } else {
                        return Err(TinyVlmError::inference("Circuit breaker is open"));
                    }
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => {}
        }

        match operation() {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(error) => {
                self.on_failure();
                Err(error)
            }
        }
    }

    fn on_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                }
            }
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(Instant::now());

        match self.state {
            CircuitState::Closed => {
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    
                    #[cfg(feature = "std")]
                    {
                        let error_msg = format!("Circuit breaker opened after {} failures", self.failure_count);
                        log_security_event(
                            "circuit_breaker_opened",
                            SecuritySeverity::High,
                            &error_msg,
                        );
                    }
                }
            }
            CircuitState::HalfOpen => {
                self.state = CircuitState::Open;
            }
            CircuitState::Open => {}
        }
    }

    /// Get current circuit breaker state
    pub fn state(&self) -> CircuitState {
        self.state.clone()
    }
}

/// Retry policy configuration
pub struct RetryPolicy {
    max_attempts: usize,
    base_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy
    pub fn new(max_attempts: usize, base_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }

    /// Execute operation with retry logic
    pub fn execute<T, F>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let mut last_error = None;

        for attempt in 1..=self.max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    // Don't retry on certain error types
                    if self.should_not_retry(&error) {
                        return Err(error);
                    }

                    last_error = Some(error);

                    // Don't sleep after the last attempt
                    if attempt < self.max_attempts {
                        let delay = self.calculate_delay(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            TinyVlmError::inference("All retry attempts failed")
        }))
    }

    fn should_not_retry(&self, error: &TinyVlmError) -> bool {
        match error {
            TinyVlmError::InvalidInput(_) | TinyVlmError::Config(_) => true,
            _ => false,
        }
    }

    fn calculate_delay(&self, attempt: usize) -> Duration {
        let delay_ms = (self.base_delay.as_millis() as f64) 
            * self.backoff_multiplier.powi((attempt - 1) as i32);
        
        let delay = Duration::from_millis(delay_ms as u64);
        std::cmp::min(delay, self.max_delay)
    }
}

/// Recovery manager for coordinating fault tolerance
pub struct RecoveryManager {
    circuit_breakers: std::collections::HashMap<String, CircuitBreaker>,
    retry_policy: RetryPolicy,
}

impl RecoveryManager {
    /// Create a new recovery manager
    pub fn new() -> Self {
        Self {
            circuit_breakers: std::collections::HashMap::new(),
            retry_policy: RetryPolicy::default(),
        }
    }

    /// Execute operation with full recovery protection
    pub fn execute_with_recovery<T, F>(&mut self, component: &str, operation: F) -> Result<T>
    where
        F: FnMut() -> Result<T>,
    {
        let circuit_breaker = self.circuit_breakers
            .entry(component.to_string())
            .or_insert_with(|| CircuitBreaker::new(5, Duration::from_secs(60)));

        circuit_breaker.execute(|| {
            self.retry_policy.execute(operation)
        })
    }

    /// Get recovery statistics
    pub fn get_stats(&self) -> RecoveryStats {
        let open_circuits = self.circuit_breakers
            .values()
            .filter(|cb| cb.state() == CircuitState::Open)
            .count();

        let half_open_circuits = self.circuit_breakers
            .values()
            .filter(|cb| cb.state() == CircuitState::HalfOpen)
            .count();

        RecoveryStats {
            total_components: self.circuit_breakers.len(),
            open_circuits,
            half_open_circuits,
            healthy_circuits: self.circuit_breakers.len() - open_circuits - half_open_circuits,
        }
    }
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Recovery system statistics
#[derive(Debug, Clone)]
pub struct RecoveryStats {
    pub total_components: usize,
    pub open_circuits: usize,
    pub half_open_circuits: usize,
    pub healthy_circuits: usize,
}

impl RecoveryStats {
    /// Check if the system is healthy
    pub fn is_healthy(&self) -> bool {
        self.open_circuits == 0 && self.half_open_circuits <= 1
    }

    /// Get system health score (0.0 to 1.0)
    pub fn health_score(&self) -> f64 {
        if self.total_components == 0 {
            return 1.0;
        }

        self.healthy_circuits as f64 / self.total_components as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(1));
        assert_eq!(cb.state(), CircuitState::Closed);

        // Should execute successfully
        let result = cb.execute(|| Ok("success"));
        assert!(result.is_ok());
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(1));
        
        // First failure
        let _ = cb.execute(|| Err::<(), _>(TinyVlmError::inference("fail1")));
        assert_eq!(cb.state(), CircuitState::Closed);

        // Second failure should open circuit
        let _ = cb.execute(|| Err::<(), _>(TinyVlmError::inference("fail2")));
        assert_eq!(cb.state(), CircuitState::Open);

        // Next call should be blocked
        let result = cb.execute(|| Ok("success"));
        assert!(result.is_err());
    }

    #[test]
    fn test_retry_policy() {
        let policy = RetryPolicy::new(3, Duration::from_millis(10));
        let mut attempts = 0;

        let result = policy.execute(|| {
            attempts += 1;
            if attempts < 3 {
                Err(TinyVlmError::inference("temporary failure"))
            } else {
                Ok("success")
            }
        });

        assert!(result.is_ok());
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_recovery_manager() {
        let mut manager = RecoveryManager::new();
        
        let result = manager.execute_with_recovery("test_component", || {
            Ok("success")
        });

        assert!(result.is_ok());

        let stats = manager.get_stats();
        assert_eq!(stats.total_components, 1);
        assert_eq!(stats.healthy_circuits, 1);
        assert!(stats.is_healthy());
    }

    #[test]
    fn test_recovery_stats() {
        let stats = RecoveryStats {
            total_components: 4,
            open_circuits: 1,
            half_open_circuits: 1,
            healthy_circuits: 2,
        };

        assert!(!stats.is_healthy());
        assert_eq!(stats.health_score(), 0.5);
    }
}