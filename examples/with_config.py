"""
Пример использования с конфигурационным файлом.
"""

from onlinediar.utils.config_loader import load_diarizer_from_config

# Загрузка диаризатора из конфигурационного файла
diarizer = load_diarizer_from_config("configs/psda_pca.yaml")

# Или с переопределением параметров
# diarizer = load_diarizer_from_config(
#     "configs/psda_pca.yaml",
#     device="cuda",
#     online_psda_threshold=-25.0
# )

# Диаризация аудио файла
annotation = diarizer.diarize("path/to/audio.wav")

# Сохранение результатов
diarizer.save_rttm(annotation, "output.rttm")

print("Диаризация завершена!")
