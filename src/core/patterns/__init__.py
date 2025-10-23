"""
Design Patterns Module
Provides Singleton and Base patterns
"""

from src.core.patterns.singleton import Singleton
from src.core.patterns.base_model import BaseModel, Field, field_validator

__all__ = [
    "Singleton",
    "BaseModel",
    "Field",
    "field_validator",
]