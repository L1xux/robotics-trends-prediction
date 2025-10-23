"""
Enhanced BaseModel with validation and serialization
Extends Pydantic BaseModel with additional functionality
"""
from pydantic import (
    BaseModel as PydanticBaseModel,
    ConfigDict,
    Field,
    field_validator
)
from typing import Dict, Any, Optional
import json
from datetime import datetime


class BaseModel(PydanticBaseModel):
    """
    Enhanced BaseModel with common functionality

    Features:
    - JSON serialization/deserialization
    - Dictionary conversion
    - Validation
    - Immutability options
    """

    model_config = ConfigDict(
        # Allow arbitrary types (for LangChain objects, etc.)
        arbitrary_types_allowed=True,
        # Validate on assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # Populate by name
        populate_by_name=True,
        # Allow extra fields (for flexibility with LLM outputs)
        extra="allow",
    )

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Convert model to dictionary

        Args:
            exclude_none: Exclude None values

        Returns:
            Dictionary representation
        """
        return self.model_dump(
            exclude_none=exclude_none,
            mode="python",
        )

    def to_json(self, exclude_none: bool = True, indent: Optional[int] = None) -> str:
        """
        Convert model to JSON string

        Args:
            exclude_none: Exclude None values
            indent: JSON indentation level

        Returns:
            JSON string
        """
        data = self.to_dict(exclude_none=exclude_none)
        return json.dumps(data, indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """
        Create model from dictionary

        Args:
            data: Dictionary data

        Returns:
            Model instance
        """
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseModel":
        """
        Create model from JSON string

        Args:
            json_str: JSON string

        Returns:
            Model instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def update_from_dict(self, data: Dict[str, Any]) -> "BaseModel":
        """
        Update model from dictionary (returns new instance)

        Args:
            data: Dictionary with updates

        Returns:
            New model instance with updates
        """
        current = self.to_dict(exclude_none=False)
        current.update(data)
        return self.__class__.from_dict(current)

    def __repr__(self) -> str:
        """String representation"""
        fields = ", ".join(
            f"{k}={repr(v)}"
            for k, v in self.to_dict(exclude_none=True).items()
            if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({fields})"

    def __str__(self) -> str:
        """User-friendly string representation"""
        return self.to_json(indent=2)


class TimestampedModel(BaseModel):
    """
    BaseModel with automatic timestamps

    Adds created_at and updated_at fields
    """

    created_at: datetime = None
    updated_at: datetime = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def update_timestamp(self):
        """Update the updated_at timestamp"""
        object.__setattr__(self, "updated_at", datetime.now())


class ImmutableModel(BaseModel):
    """
    Immutable BaseModel

    Cannot be modified after creation
    """

    model_config = ConfigDict(
        frozen=True,  # Make immutable
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=True,
        populate_by_name=True,
        extra="allow",
    )
