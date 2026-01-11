"""
Configuration-based diarization initialization for PSDA and CSEA backends.
"""

import logging
from onlinediar.utils.config_loader import load_diarizer_from_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_csea_config():
    """Example: Load CSEA diarizer from configuration file."""
    logger = logging.getLogger(__name__)
    logger.info("=== CSEA Config-based Example ===")
    
    diarizer = load_diarizer_from_config("configs/csea_pca.yaml")
    
    # Alternative: Override specific parameters
    # diarizer = load_diarizer_from_config(
    #     "configs/csea_pca.yaml",
    #     device="cuda",
    #     csea_threshold=0.70
    # )
    
    logger.info("Starting diarization")
    annotation = diarizer.diarize("path/to/audio.wav")
    
    output_path = "output_csea.rttm"
    diarizer.save_rttm(annotation, output_path)
    logger.info(f"Results saved: {output_path}")


def example_psda_config():
    """Example: Load PSDA diarizer from configuration file."""
    logger = logging.getLogger(__name__)
    logger.info("=== PSDA Config-based Example ===")
    
    diarizer = load_diarizer_from_config("configs/psda_pca.yaml")
    
    # Alternative: Override specific parameters
    # diarizer = load_diarizer_from_config(
    #     "configs/psda_pca.yaml",
    #     device="cuda",
    #     online_psda_threshold=-25.0
    # )
    
    logger.info("Starting diarization")
    annotation = diarizer.diarize("path/to/audio.wav")
    
    output_path = "output_psda.rttm"
    diarizer.save_rttm(annotation, output_path)
    logger.info(f"Results saved: {output_path}")


if __name__ == "__main__":
    example_csea_config()
    # example_psda_config()
