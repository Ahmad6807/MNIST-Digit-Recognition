"""Configuration loader utility."""
import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict with all configuration values.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(str(path)) as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict, config_path: str = "config.yaml"):
    """Save configuration dict to YAML file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save the YAML file.
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
