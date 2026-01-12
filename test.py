from onlinediar.diarizer import Diarizer

# Initialize diarizer
diarizer = Diarizer(
    backend="csea",  # or "psda"
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    csea_threshold=0.68,
    device="cpu"
)

# Diarize audio file
annotation = diarizer.diarize("data/ru-mss.wav")

# Save results
diarizer.save_rttm(annotation, "output.rttm")