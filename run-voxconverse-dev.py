#!/usr/bin/env python3
import logging
from pathlib import Path
from onlinediar.diarizer import Diarizer
import argparse


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
args = parser.parse_args()

audio_dir = Path(args.audio_dir)
audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma']


audio_files = []
for ext in audio_extensions:
    audio_files.extend(audio_dir.glob(f"*{ext}"))


audio_files = sorted(audio_files)

logging.info(f"Found {len(audio_files)} audio file(s) to process")

run_name = args.run_name
if run_name == "csea+pca":
    model_dir = Path("./models")
    emb_model_path = model_dir / "voxblink2_samresnet34_ft.onnx"
    pca_path = model_dir / "pca_model_128d.joblib"

    diarizer = Diarizer(
        backend="csea",
        emb_model_path=str(emb_model_path),
        csea_threshold=0.68,
        max_embeddings_per_speaker=40,
        resample_rate=16000,
        device="cuda",
        use_pca=True,
        pca_path=str(pca_path),
        log_level=logging.INFO
    )
elif run_name == "csea":
    model_dir = Path("./models")
    emb_model_path = model_dir / "voxblink2_samresnet34_ft.onnx"

    diarizer = Diarizer(
        backend="csea",
        emb_model_path=str(emb_model_path),
        csea_threshold=0.68,
        max_embeddings_per_speaker=40,
        resample_rate=16000,
        device="cuda",
        use_pca=False,
        log_level=logging.INFO
    )
elif run_name == "psda":
    model_dir = Path("./models")
    emb_model_path = model_dir / "voxblink2_samresnet34_ft.onnx"
    psda_path = model_dir / "voxblinkR34_ft/psda_vox_256_2sec"

    diarizer = Diarizer(
        backend="psda",
        emb_model_path=str(emb_model_path),
        max_embeddings_per_speaker=40,
        resample_rate=16000,
        device="cuda",
        use_pca=False,
        psda_path=str(psda_path),
        log_level=logging.INFO
    )
elif run_name == "psda+pca":
    model_dir = Path("./models")
    emb_model_path = model_dir / "voxblink2_samresnet34_ft.onnx"
    pca_path = model_dir / "pca_model_128d.joblib"
    psda_path = model_dir / "voxblinkR34_ft/psda_vox_128_2sec"

    diarizer = Diarizer(
        backend="psda",
        emb_model_path=str(emb_model_path),
        max_embeddings_per_speaker=40,
        resample_rate=16000,
        device="cuda",
        use_pca=True,
        pca_path=str(pca_path),
        psda_path=str(psda_path),
        log_level=logging.INFO
    )


audio_dir_name = str(audio_dir).replace('/', '-')
output_dir = Path("exps") / f"{run_name}_{audio_dir_name}"
output_dir.mkdir(parents=True, exist_ok=True)

for audio_file in audio_files:
    logging.info(f"Diarization is started for file: {audio_file.name}")
    annotation = diarizer.diarize(str(audio_file))

    audio_uri = audio_file.stem
    rttm_path = output_dir / f"{audio_uri}.rttm"

    diarizer.save_rttm(annotation, str(rttm_path), audio_uri=audio_uri)
    logging.info(f"Save RTTM: {rttm_path}")
