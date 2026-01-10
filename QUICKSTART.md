# Быстрый старт

## Установка

```bash
cd online-diarization
pip install -r requirements.txt
```

## Минимальный пример

```python
from onlinediar import Diarizer

# Создание диаризатора
diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)

# Диаризация
annotation = diarizer.diarize("audio.wav")

# Сохранение
diarizer.save_rttm(annotation, "output.rttm")

# Вывод результатов
for segment, _, speaker in annotation.itertracks(yield_label=True):
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {speaker}")
```

## С конфигурацией

```python
from onlinediar import load_diarizer_from_config

# Загрузка из конфига
diarizer = load_diarizer_from_config("configs/psda_pca.yaml")

# Диаризация
annotation = diarizer.diarize("audio.wav")
diarizer.save_rttm(annotation, "output.rttm")
```

## Регистрация спикеров

```python
from onlinediar import Diarizer

diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)

# Регистрация
diarizer.register_speaker("john.wav", "Джон")
diarizer.register_speaker("alice.wav", "Алиса")

# Диаризация
annotation = diarizer.diarize("meeting.wav")

# Результаты с именами
speaker_map = diarizer.get_speaker_map()
for segment, _, speaker in annotation.itertracks(yield_label=True):
    name = speaker_map.get(speaker, speaker)
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {name}")
```

## Стриминг

```python
from onlinediar import Diarizer

diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)

# Запуск стриминга (Ctrl+C для остановки)
diarizer.stream_from_microphone(save_audio=True)
```

## Основные параметры

### PSDA (лучшее качество)
- `backend="psda"`
- `online_psda_threshold=-30.0` (от -20 до -40, ниже = строже)

### CSEA (быстрее)
- `backend="csea"`
- `csea_threshold=0.68` (от 0 до 1, выше = строже)

### Общие
- `device="cpu"` или `"cuda"`
- `use_pca=True` для улучшения качества
- `freeze=True` для использования только зарегистрированных спикеров
- `step_size=1.0` - шаг для оверлапов (секунды)

## Скачивание моделей

1. **Embedding model**: 
   - Скачать voxblink2_samresnet34_ft.onnx
   - Источник: https://github.com/wenet-e2e/wespeaker

2. **PSDA model** (если используете PSDA):
   - Обучить на своих данных или использовать предобученную

3. **PCA model** (опционально):
   - Обучить на своих данных

Поместите в папку `models/`.

## Примеры

Все примеры находятся в папке `examples/`:
- `simple_usage.py`
- `with_config.py`
- `with_registered_speakers.py`
- `streaming.py`

## Подробная документация

См. [README.md](README.md)
