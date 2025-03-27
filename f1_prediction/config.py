"""
Configuration module for F1 Predictions.

This module handles configuration settings for the F1 predictions package.
It follows the Singleton pattern to ensure consistent configuration across the application.
"""

import os
import json
from typing import Dict, Any, Optional
import logging


class Config:
    """
    Configuration manager for F1 predictions.

    This class manages configuration settings, including paths, model parameters,
    and other settings. It follows the Singleton pattern to ensure there's only
    one configuration instance.
    """

    _instance = None

    def __new__(cls):
        """Creation of singleton method"""
        if cls._instance is None:
            print("Creating a new object")
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self):
        if self._initialised:
            return

        self._config = {
            # Paths
            "data_dir": os.path.join(os.getcwd(), "data"),
            "models_dir": os.path.join(os.getcwd(), "models"),
            "cache_dir": os.path.join(os.getcwd(), "f1_cache"),
            "results_dir": os.path.join(os.getcwd(), "results"),

            # FastF1 settings
            "enable_cache": True,

            # Model settings
            "default_model": "gradient_boosting",
            "model_params": {
                "gradient_boosting": {
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42
                }
            },

            # Logging settings
            "log_level": "INFO",
            "log_file": "f1_predictions.log"
        }

        for dir_key in ["data_dir", "models_dir", "cache_dir", "results_dir"]:
            os.makedirs(self._config[dir_key], exist_ok=True)

        self._initialised = True

    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary

        :returns:
            Dict[str, Any]: Config Dict
        """
        return self._config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value
        :param key : Configuration key
        :param default: Default value if key not found. Defaults to None.
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set the config value
        :param key: Configuration key
        :param value: Configuration value
        """
        self._config[key] = value

        if key.endswith("_dir") and isinstance(value, str):
            os.makedirs(value, exist_ok=True)






