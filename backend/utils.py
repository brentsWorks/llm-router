"""Utility functions for the LLM Router project."""

from typing import Any
from pydantic import ValidationError


def format_validation_error(error: ValidationError) -> str:
    """
    Convert Pydantic validation error to a clear, user-friendly message.
    
    Args:
        error: Pydantic ValidationError instance
        
    Returns:
        Clear, user-friendly error message
    """
    if not error.errors():
        return "Validation error occurred"
    
    # Get the first error for simplicity
    first_error = error.errors()[0]
    field_name = first_error.get("loc", ["unknown"])[-1] if first_error.get("loc") else "unknown"
    error_type = first_error.get("type", "validation_error")
    
    # Custom error messages based on error type
    if error_type == "greater_than_equal":
        return f"{field_name} must be greater than or equal to {first_error.get('ctx', {}).get('ge', 'the minimum value')}"
    elif error_type == "less_than_equal":
        return f"{field_name} must be less than or equal to {first_error.get('ctx', {}).get('le', 'the maximum value')}"
    elif error_type == "value_error":
        return f"{field_name}: {first_error.get('msg', 'Invalid value')}"
    else:
        return f"{field_name}: {first_error.get('msg', 'Validation failed')}"


def safe_float_compare(a: float, b: float, tolerance: float = 1e-10) -> bool:
    """
    Safely compare two floats with a tolerance for floating-point precision issues.
    
    Args:
        a: First float value
        b: Second float value
        tolerance: Tolerance for comparison (default: 1e-10)
        
    Returns:
        True if values are approximately equal within tolerance
    """
    return abs(a - b) <= tolerance


def validate_percentage(value: float, field_name: str = "value") -> None:
    """
    Validate that a value is a valid percentage (0.0 to 1.0).
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Raises:
        ValueError: If value is not a valid percentage
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")
