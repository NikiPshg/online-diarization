#!/usr/bin/env python3
import logging
from pathlib import Path
from onlinediar.diarizer import Diarizer
import argparse
import yaml


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diarization_batch.log'),
        logging.StreamHandler()
    ]
)


parser = argparse.ArgumentParser(description="Run diarization on a directory of audio files.")
parser.add_argument('--audio_dir', type=str, required=True, help="Path to directory containing audio files.")
parser.add_argument('--run_name', type=str, required=True, choices=["csea+pca", "csea", "psda", "psda+pca"], help="Select diarization run mode")
parser.add_argument('--config', type=str, default=None, help="Path to config.yaml file (optional)")
args = parser.parse_args()

audio_dir = Path(args.audio_dir)
audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma']


audio_files = []
for ext in audio_extensions:
    audio_files.extend(audio_dir.glob(f"*{ext}"))


audio_files = sorted(audio_files)

logging.info(f"Found {len(audio_files)} audio file(s) to process")

# Load config if provided
config = {}
if args.config:
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            all_config = yaml.safe_load(f)
            config = all_config.get(args.run_name, {})
            logging.info(f"Loaded config from {config_path} for run_name: {args.run_name}")
    else:
        logging.warning(f"Config file {config_path} not found, using defaults")

run_name = args.run_name
model_dir = Path("./models")
emb_model_path = model_dir / "voxblink2_samresnet34_ft.onnx"

if run_name == "csea+pca":
    pca_path = model_dir / "pca_model_128d.joblib"

    diarizer = Diarizer(
        backend="csea",
        emb_model_path=str(emb_model_path),
        csea_threshold=config.get("csea_threshold", 0.68),
        max_embeddings_per_speaker=config.get("max_embeddings_per_speaker", 40),
        resample_rate=config.get("resample_rate", 16000),
        device=config.get("device", "cuda"),
        use_pca=True,
        pca_path=str(pca_path),
        log_level=logging.INFO
    )
elif run_name == "csea":
    diarizer = Diarizer(
        backend="csea",
        emb_model_path=str(emb_model_path),
        csea_threshold=config.get("csea_threshold", 0.68),
        max_embeddings_per_speaker=config.get("max_embeddings_per_speaker", 40),
        resample_rate=config.get("resample_rate", 16000),
        device=config.get("device", "cuda"),
        use_pca=False,
        log_level=logging.INFO
    )
elif run_name == "psda":
    psda_path = model_dir / "voxblinkR34_ft/psda_vox_256_2sec"

    diarizer = Diarizer(
        backend="psda",
        emb_model_path=str(emb_model_path),
        online_psda_threshold=config.get("online_psda_threshold", -73),
        min_silence_dur_ms=config.get("min_silence_dur_ms", 100),
        min_duration_for_change=config.get("min_duration_for_change", 1000),
        max_embeddings_per_speaker=config.get("max_embeddings_per_speaker", 40),
        resample_rate=config.get("resample_rate", 16000),
        device=config.get("device", "cuda"),
        use_pca=False,
        psda_path=str(psda_path),
        log_level=logging.INFO
    )
elif run_name == "psda+pca":
    pca_path = model_dir / "pca_model_128d.joblib"
    psda_path = model_dir / "voxblinkR34_ft/psda_vox_128_2sec"

    diarizer = Diarizer(
        backend="psda",
        emb_model_path=str(emb_model_path),
        online_psda_threshold=config.get("online_psda_threshold", -73),
        min_silence_dur_ms=config.get("min_silence_dur_ms", 100),
        min_duration_for_change=config.get("min_duration_for_change", 1000),
        max_embeddings_per_speaker=config.get("max_embeddings_per_speaker", 40),
        resample_rate=config.get("resample_rate", 16000),
        device=config.get("device", "cuda"),
        use_pca=True,
        pca_path=str(pca_path),
        psda_path=str(psda_path),
        log_level=logging.INFO
    )


# Build params string for directory name based on run_name
params_parts = []
if run_name in ["psda", "psda+pca"]:
    psda_thresh = config.get("online_psda_threshold", -73)
    silence_ms = config.get("min_silence_dur_ms", 100)
    min_dur = config.get("min_duration_for_change", 1000)
    params_parts.append(f"psda{psda_thresh}")
    params_parts.append(f"sil{silence_ms}")
    params_parts.append(f"mindur{min_dur}")
elif run_name in ["csea", "csea+pca"]:
    csea_thresh = config.get("csea_threshold", 0.68)
    max_emb = config.get("max_embeddings_per_speaker", 40)
    params_parts.append(f"csea{csea_thresh}")
    params_parts.append(f"maxemb{max_emb}")

params_str = "_".join(params_parts) if params_parts else "default"

audio_dir_name = str(audio_dir).replace('/', '-')
output_dir = Path("exps") / f"{run_name}_{params_str}_{audio_dir_name}"
output_dir.mkdir(parents=True, exist_ok=True)

for audio_file in audio_files:
    logging.info(f"Diarization is started for file: {audio_file.name}")
    annotation = diarizer.diarize(str(audio_file))

    audio_uri = audio_file.stem
    rttm_path = output_dir / f"{audio_uri}.rttm"

    diarizer.save_rttm(annotation, str(rttm_path), audio_uri=audio_uri)
    logging.info(f"Save RTTM: {rttm_path}")
