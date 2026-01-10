"""
Configuration loader for diarization system.
"""

import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_diarizer_from_config(config_path: str, **kwargs):
    """
    Create a Diarizer instance from a configuration file.

    Args:
        config_path: Path to YAML configuration file
        **kwargs: Additional parameters to override config values

    Returns:
        Diarizer instance
    """
    from onlinediar.diarizer import Diarizer
    
    config = load_config(config_path)
    
    # Override config with kwargs
    config.update(kwargs)
    
    # Handle prepared speakers if specified
    prepared_speakers = config.pop("prepared_speakers", None)
    
    # Create diarizer
    diarizer = Diarizer(**config)
    
    # Register prepared speakers
    if prepared_speakers:
        for speaker in prepared_speakers:
            diarizer.register_speaker(
                audio_path=speaker["audio_path"],
                speaker_name=speaker["name"]
            )
    
    return diarizer
