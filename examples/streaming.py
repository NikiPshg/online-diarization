"""
Пример использования в режиме стриминга (реального времени).
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
    device="cpu"
)

# Опционально: регистрация известных спикеров
# diarizer.register_speaker("speakers/john.wav", "Джон")
# diarizer.register_speaker("speakers/alice.wav", "Алиса")

print("Запуск стриминговой диаризации...")
print("Говорите в микрофон. Нажмите Ctrl+C для остановки.")

# Запуск стриминга с микрофона
# После остановки автоматически сохранится аудио и результаты диаризации
diarizer.stream_from_microphone(
    save_audio=True,
    output_prefix="stream_output"
)

print("Стриминг завершен!")
print("Результаты сохранены в stream_output.wav и stream_output.rttm")
