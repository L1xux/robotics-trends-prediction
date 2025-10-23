"""
Singleton Pattern Implementation
Thread-safe singleton metaclass
"""
import threading
from typing import Dict, Any


class Singleton(type):
    """
    Thread-safe Singleton metaclass

    Usage:
        class MyClass(metaclass=Singleton):
            def __init__(self):
                pass
    """

    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        """
        Thread-safe singleton instance creation
        """
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance

        return cls._instances[cls]

    @classmethod
    def clear_instances(cls):
        """Clear all singleton instances (useful for testing)"""
        with cls._lock:
            cls._instances.clear()