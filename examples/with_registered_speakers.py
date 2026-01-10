"""
Пример использования с предварительной регистрацией известных спикеров.
"""

from onlinediar.diarizer import Diarizer

# Создание диаризатора
diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib",
    online_psda_threshold=-30.0,
    freeze=False  # False = разрешить новых спикеров, True = только зарегистрированные
)

# Регистрация известных спикеров
print("Регистрация спикеров...")
diarizer.register_speaker("speakers/john.wav", "Джон")
diarizer.register_speaker("speakers/alice.wav", "Алиса")
diarizer.register_speaker("speakers/bob.wav", "Боб")

# Посмотреть зарегистрированных спикеров
speaker_map = diarizer.get_speaker_map()
print(f"Зарегистрированные спикеры: {speaker_map}")

# Диаризация аудио файла
print("\nЗапуск диаризации...")
annotation = diarizer.diarize("path/to/audio.wav")

# Сохранение результатов с именами спикеров
diarizer.save_rttm(annotation, "output.rttm")

# Вывод результатов с именами
print("\nРезультаты диаризации:")
for segment, _, speaker in annotation.itertracks(yield_label=True):
    speaker_name = speaker_map.get(speaker, speaker)
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker_name}")
