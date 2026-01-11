# Online Speaker Diarization

A Python library for real-time and file-based speaker diarization supporting both **PSDA** and **CSEA**  backends.
## Installation
### Setup

```bash
# Clone the repository
# python 3.10.0
git clone https://github.com/NikiPshg/online-diarization
cd online-diarization
export PYTHONPATH=./

# Install dependencies
uv pip install -r requirements.txt

# Download required models
# Place models in the following structure:
# models/
# ├── voxblink2_samresnet34_ft.onnx          # Embedding model
# ├── voxblinkR34_ft/
# │   └── psda_vox_128_2sec/                 # PSDA model (for PSDA backend)
# └── pca_model_128d.joblib                  # PCA model (optional)
```

## Quick Start

### Basic Usage

```python
from onlinediar.diarizer import Diarizer

# Initialize diarizer
diarizer = Diarizer(
    backend="csea",  # or "psda"
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    csea_threshold=0.68,
    device="cpu"
)

# Diarize audio file
annotation = diarizer.diarize("audio.wav")

# Save results
diarizer.save_rttm(annotation, "output.rttm")
```

### Real-time Streaming

```python
diarizer.stream_from_microphone(
    save_audio=True,
    output_prefix="stream_output",
    show_realtime_results=True
)
```

## Backend Comparison

### CSEA (Cosine Similarity Embedding Aggregation)

**Best for**: Dynamic scenarios with unknown number of speakers

**Advantages**:
- No pre-trained model required (beyond embeddings)
- Adaptive to speaker changes
- Lower memory footprint

**Key Parameters**:
- `csea_threshold`: Similarity threshold (0.68 recommended)
- `max_embeddings_per_speaker`: Rolling buffer size (40 recommended)

**Example**:
```python
diarizer = Diarizer(
    backend="csea",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    csea_threshold=0.68,
    max_embeddings_per_speaker=40,
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)
```

### PSDA

**Best for**: Controlled environments with stable speakers

**Advantages**:
- Statistical modeling for robustness
- Better handling of speaker overlaps
- Proven performance on benchmarks

**Key Parameters**:
- `online_psda_threshold`: Log-likelihood threshold (-30.0 recommended)
- `psda_path`: Path to PSDA model directory

**Example**:
```python
diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/voxblinkR34_ft/psda_vox_128_2sec",
    online_psda_threshold=-30.0,
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)
```

## Advanced Features

### Speaker Registration

Pre-register known speakers for identification:

```python
diarizer = Diarizer(
    backend="csea",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    freeze=False  # Allow new speakers
)

# Register speakers
diarizer.register_speaker("john.wav", "John")
diarizer.register_speaker("alice.wav", "Alice")

# Diarize with speaker names
annotation = diarizer.diarize("meeting.wav")

# Results will include registered speaker names
for segment, _, speaker in annotation.itertracks(yield_label=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker}")
```

### Configuration-based Initialization

Use YAML configuration files for reproducible experiments:

**config.yaml**:
```yaml
backend: csea
emb_model_path: models/voxblink2_samresnet34_ft.onnx
csea_threshold: 0.68
max_embeddings_per_speaker: 40
use_pca: true
pca_path: models/pca_model_128d.joblib
resample_rate: 16000
device: cpu
```

**Python code**:
```python
from onlinediar.utils.config_loader import load_diarizer_from_config

diarizer = load_diarizer_from_config("config.yaml")
annotation = diarizer.diarize("audio.wav")
```

## Examples

The `examples/` directory contains comprehensive examples:

| Example | Description |
|---------|-------------|
| `file_diarization.py` | Batch processing of multiple files |
| `streaming.py` | Real-time microphone streaming |
| `with_registered_speakers.py` | Speaker identification |
| `with_config.py` | Configuration-based setup |

Run an example:
```bash
python examples/streaming.py
```

## API Reference

### Diarizer Class

```python
Diarizer(
    backend: str = "csea",              # Backend type: "csea" or "psda"
    emb_model_path: str = None,         # Path to embedding model
    psda_path: str = None,              # Path to PSDA model (PSDA only)
    online_psda_threshold: float = -30.0,  # PSDA threshold
    csea_threshold: float = 0.68,       # CSEA threshold
    max_embeddings_per_speaker: int = 40,  # CSEA buffer size
    resample_rate: int = 16000,         # Audio sample rate
    device: str = "cpu",                # Device: "cpu" or "cuda"
    freeze: bool = False,               # Only use registered speakers
    use_pca: bool = False,              # Enable PCA
    pca_path: str = None,               # Path to PCA model
    log_level: int = logging.INFO       # Logging level
)
```

### Methods

**`diarize(audio_path: str) -> Annotation`**

Process an audio file and return diarization results.

**`register_speaker(audio_path: str, speaker_name: str)`**

Register a known speaker from an audio sample.

**`stream_from_microphone(save_audio: bool, output_prefix: str, show_realtime_results: bool, block_size: int)`**

Stream audio from microphone and perform real-time diarization.

**`save_rttm(annotation: Annotation, output_path: str, audio_uri: str)`**

Save diarization results in RTTM format.

**`reset()`**

Reset diarizer state (clear speakers and buffers).

**`get_speaker_map() -> Dict[str, str]`**

Get current speaker ID to name mapping.

## Output Format

### RTTM (Rich Transcription Time Marked)

```
SPEAKER audio_file 1 0.000 2.500 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER audio_file 1 2.500 3.200 <NA> <NA> SPEAKER_01 <NA> <NA>
SPEAKER audio_file 1 5.700 4.100 <NA> <NA> SPEAKER_00 <NA> <NA>
```

Format: `SPEAKER <audio_uri> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>`

## Logging

The library uses Python's standard logging module:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

diarizer = Diarizer(backend="csea", log_level=logging.INFO)
```

**Log output example**:
```
2025-01-11 15:30:45 - Diarizer - INFO - Initializing CSEA diarization backend
2025-01-11 15:30:46 - Diarizer - INFO - Starting streaming diarization
2025-01-11 15:30:47 - Diarizer - INFO - Speaker changed: SPEAKER_00 (similarity: 0.850)
2025-01-11 15:30:50 - Diarizer - INFO - Speaker: SPEAKER_00, Duration: 3.20s
```

## Performance Optimization

### GPU Acceleration

```python
diarizer = Diarizer(
    backend="csea",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    device="cuda"  # Use GPU
)
```

### PCA Dimensionality Reduction

Reduce embedding dimensions for faster processing:

```python
diarizer = Diarizer(
    backend="csea",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Speaker embedding model: [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
- PSDA algorithm: Based on probabilistic speaker modeling
- VAD: [Silero VAD](https://github.com/snakers4/silero-vad)


