"""Configuration loading for DeepStream applications.

Provides YAML parsing and a base config class that resolves file paths
relative to the application directory.
"""

from pathlib import Path
import yaml


def parse_yaml(file_path):
    """Load a YAML file and return its contents as a dict."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


class AppConfig:
    """Base configuration loaded from a YAML file.

    Resolves config-file paths relative to the application directory
    so apps can be run from any working directory.

    Usage:
        class Config(AppConfig):
            def __init__(self):
                super().__init__(__file__)
                self.source = self.data["source"]
                self.pgie_config = self.resolve("pgie", "config_file")
    """

    def __init__(self, app_file, yaml_filename="config.yaml"):
        self.app_dir = Path(app_file).resolve().parent
        self.data = parse_yaml(str(self.app_dir / yaml_filename))

    def resolve(self, *keys):
        """Navigate nested keys, resolve final value as absolute path relative to app_dir."""
        value = self.data
        for key in keys:
            value = value[key]
        return str(self.app_dir / value)

    def get(self, *keys, default=None):
        """Safely get nested value with default."""
        value = self.data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value
