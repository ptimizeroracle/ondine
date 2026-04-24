"""
Configuration loader for YAML and JSON files.

Enables loading pipeline configurations from declarative files.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

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
    def _expand_env_vars(obj: Any) -> Any:
        """Recursively expand ${VAR} and $VAR patterns in string values."""
        if isinstance(obj, str):
            expanded = os.path.expandvars(obj)
            unresolved = re.findall(r"\$\{([^}]+)\}", expanded)
            if unresolved:
                raise ValueError(
                    f"Environment variable(s) not set: {', '.join(unresolved)}"
                )
            return expanded
        if isinstance(obj, dict):
            return {k: ConfigLoader._expand_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ConfigLoader._expand_env_vars(item) for item in obj]
        return obj

    @staticmethod
    def _dict_to_specifications(config: dict[str, Any]) -> PipelineSpecifications:
        """
        Convert configuration dictionary to PipelineSpecifications.

        Maps user-friendly YAML field names to internal Pydantic field names.

        Args:
            config: Configuration dictionary

        Returns:
            PipelineSpecifications
        """
        config = ConfigLoader._expand_env_vars(config)

        # Map YAML field names to Pydantic field names
        # YAML uses 'data' but Pydantic expects 'dataset'
        if "data" in config:
            data_config = config.pop("data")

            # Map data.source.type to dataset.source_type
            if "source" in data_config and isinstance(data_config["source"], dict):
                source = data_config.pop("source")
                if "type" in source:
                    data_config["source_type"] = source["type"]
                if "path" in source:
                    data_config["source_path"] = source["path"]

            config["dataset"] = data_config

        # Map output.format to output.destination_type
        if "output" in config and isinstance(config["output"], dict):
            if "format" in config["output"]:
                format_value = config["output"].pop("format")
                # Map string to enum (YAML 'format' → Pydantic 'destination_type')
                config["output"]["destination_type"] = format_value

        # Map processing field names
        if "processing" in config and isinstance(config["processing"], dict):
            # Map rate_limit to rate_limit_rpm
            if "rate_limit" in config["processing"]:
                config["processing"]["rate_limit_rpm"] = config["processing"].pop(
                    "rate_limit"
                )

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
