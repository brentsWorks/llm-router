"""
API Logging and Monitoring Utilities
====================================

Centralized logging, error tracking, and monitoring capabilities for the LLM Router API.
Provides structured logging, error metrics collection, and monitoring utilities.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ErrorMetrics:
    """Container for error metrics and monitoring data."""
    validation_errors: int = 0
    internal_errors: int = 0
    routing_errors: int = 0
    classification_errors: int = 0
    timeout_errors: int = 0
    rate_limit_errors: int = 0
    last_error_time: Optional[str] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "validation_errors": self.validation_errors,
            "internal_errors": self.internal_errors,
            "routing_errors": self.routing_errors,
            "classification_errors": self.classification_errors,
            "timeout_errors": self.timeout_errors,
            "rate_limit_errors": self.rate_limit_errors,
            "last_error_time": self.last_error_time,
            "total_errors": self.get_total_errors(),
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }
    
    def get_total_errors(self) -> int:
        """Get total error count across all categories."""
        return (
            self.validation_errors + self.internal_errors + 
            self.routing_errors + self.classification_errors + 
            self.timeout_errors + self.rate_limit_errors
        )


class APILogger:
    """Enhanced logger for API requests, errors, and monitoring."""
    
    def __init__(self, name: str = "llm_router_api"):
        """Initialize the API logger."""
        self.logger = logging.getLogger(name)
        self.error_metrics = ErrorMetrics()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup enhanced logging configuration."""
        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def log_request_start(self, request_id: str, method: str, path: str, 
                         client_ip: str = "unknown", user_agent: str = "unknown") -> None:
        """Log the start of an API request with enhanced context."""
        self.logger.info(
            f"[{request_id}] {method} {path} - Started "
            f"(IP: {client_ip}, UA: {user_agent[:50]}...)"
        )
    
    def log_request_end(self, request_id: str, method: str, path: str, 
                       status_code: int, duration: float) -> None:
        """Log the completion of an API request."""
        self.logger.info(
            f"[{request_id}] {method} {path} - "
            f"{status_code} ({duration:.3f}s)"
        )
    
    def log_error_metrics(self, error_type: str, error_message: str, 
                         request_id: str, path: str) -> None:
        """Log error metrics for monitoring and alerting."""
        current_time = datetime.now(timezone.utc)
        
        # Update error counters
        if hasattr(self.error_metrics, error_type):
            current_count = getattr(self.error_metrics, error_type)
            setattr(self.error_metrics, error_type, current_count + 1)
        
        self.error_metrics.last_error_time = current_time.isoformat()
        
        # Add to error history (keep last 100)
        error_entry = {
            "timestamp": current_time.isoformat(),
            "type": error_type,
            "message": error_message,
            "request_id": request_id,
            "path": path
        }
        
        self.error_metrics.error_history.append(error_entry)
        if len(self.error_metrics.error_history) > 100:
            self.error_metrics.error_history.pop(0)
        
        # Log with appropriate severity
        if error_type in ["internal_errors", "routing_errors"]:
            self.logger.error(f"CRITICAL ERROR [{request_id}] {error_type}: {error_message}")
        else:
            self.logger.warning(f"ERROR [{request_id}] {error_type}: {error_message}")
    
    def log_validation_error(self, request_id: str, path: str, field_errors: List[str]) -> None:
        """Log validation errors with detailed field information."""
        error_message = f"Validation failed for field(s): {', '.join(field_errors)}"
        self.log_error_metrics("validation_errors", error_message, request_id, path)
    
    def log_internal_error(self, request_id: str, path: str, error: Exception) -> None:
        """Log internal server errors with stack trace."""
        error_message = f"Internal error: {str(error)}"
        self.log_error_metrics("internal_errors", error_message, request_id, path)
        
        # Log full exception with stack trace for debugging
        self.logger.error(
            f"[{request_id}] Internal server error in {path}: {str(error)}", 
            exc_info=True
        )
    
    def log_routing_error(self, request_id: str, error_message: str) -> None:
        """Log routing-specific errors."""
        self.log_error_metrics("routing_errors", error_message, request_id, "/route")
    
    def log_classification_error(self, request_id: str, error_message: str) -> None:
        """Log classification-specific errors."""
        self.log_error_metrics("classification_errors", error_message, request_id, "/classify")
    
    def log_middleware_error(self, request_id: str, method: str, path: str, 
                           error: Exception, duration: float) -> None:
        """Log middleware errors with timing context."""
        self.logger.error(
            f"[{request_id}] Middleware error in {method} {path} "
            f"after {duration:.3f}s: {str(error)}", 
            exc_info=True
        )
        self.log_error_metrics("internal_errors", f"Middleware error: {str(error)}", 
                              request_id, path)
    
    def log_large_request_blocked(self, request_id: str, content_length: str) -> None:
        """Log when large requests are blocked."""
        self.logger.warning(f"[{request_id}] Request too large: {content_length} bytes")
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get current error metrics for monitoring endpoints."""
        return self.error_metrics.to_dict()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on error rates."""
        total_errors = self.error_metrics.get_total_errors()
        critical_errors = (self.error_metrics.internal_errors + 
                         self.error_metrics.routing_errors)
        
        # Simple health check based on error counts
        if critical_errors > 10:
            status = "unhealthy"
        elif total_errors > 50:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "total_errors": total_errors,
            "critical_errors": critical_errors,
            "last_error": self.error_metrics.last_error_time
        }


# Global logger instance
api_logger = APILogger()


def get_api_logger() -> APILogger:
    """Get the global API logger instance."""
    return api_logger
