"""
Configuration loader for YAML and JSON files.

Enables loading pipeline configurations from declarative files.
"""

import json
from pathlib import Path
from typing import Any

import yaml

from ondine.core.specifications import PipelineSpecifications


class ConfigLoader:
    """
    Loads pipeline configurations from YAML or JSON files.

    Follows Single Responsibility: only handles config file loading.
    """

    @staticmethod
    def from_yaml(file_path: str | Path) -> PipelineSpecifications:
        """
        Load configuration from YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            PipelineSpecifications

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If invalid YAML or configuration
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return ConfigLoader._dict_to_specifications(config_dict)

    @staticmethod
    def from_json(file_path: str | Path) -> PipelineSpecifications:
        """
        Load configuration from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            PipelineSpecifications

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If invalid JSON or configuration
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = json.load(f)

        return ConfigLoader._dict_to_specifications(config_dict)

    @staticmethod
    def _dict_to_specifications(config: dict[str, Any]) -> PipelineSpecifications:
        """
        Convert configuration dictionary to PipelineSpecifications.

        Args:
            config: Configuration dictionary

        Returns:
            PipelineSpecifications
        """
        return PipelineSpecifications(**config)

    @staticmethod
    def to_yaml(specifications: PipelineSpecifications, file_path: str | Path) -> None:
        """
        Save specifications to YAML file.

        Args:
            specifications: Pipeline specifications
            file_path: Destination file path
        """
        path = Path(file_path)

        # Convert to dict
        config_dict = specifications.model_dump(mode="json")

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    @staticmethod
    def to_json(specifications: PipelineSpecifications, file_path: str | Path) -> None:
        """
        Save specifications to JSON file.

        Args:
            specifications: Pipeline specifications
            file_path: Destination file path
        """
        path = Path(file_path)

        # Convert to dict
        config_dict = specifications.model_dump(mode="json")

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
