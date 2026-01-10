"""
Простой пример использования библиотеки для диаризации.
"""

from onlinediar.diarizer import Diarizer

# Вариант 1: Использование PSDA с PCA (рекомендуется для лучшего качества)
diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib",
    online_psda_threshold=-30.0,
    device="cpu"  # или "cuda" для GPU
)

# Вариант 2: Использование CSEA (быстрее, но менее точно)
# diarizer = Diarizer(
#     backend="csea",
#     emb_model_path="models/voxblink2_samresnet34_ft.onnx",
#     use_pca=True,
#     pca_path="models/pca_model_128d.joblib",
#     csea_threshold=0.68,
#     device="cpu"
# )

# Диаризация аудио файла
annotation = diarizer.diarize("path/to/audio.wav")

# Сохранение результатов в RTTM
diarizer.save_rttm(annotation, "output.rttm", audio_uri="audio")

# Вывод результатов
print("Результаты диаризации:")
for segment, _, speaker in annotation.itertracks(yield_label=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker}")
