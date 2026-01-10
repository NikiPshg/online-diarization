# Online Speaker Diarization

Библиотека для онлайн-диаризации говорящих с поддержкой PCA, оверлапов и предварительной регистрации спикеров.

## Особенности

- 🎯 **Два алгоритма кластеризации**: PSDA (высокое качество) и CSEA (высокая скорость)
- 🔄 **Обработка оверлапов**: Улучшенная обработка перекрывающихся сегментов
- 📊 **Поддержка PCA**: Снижение размерности эмбеддингов для улучшения качества
- 👤 **Регистрация спикеров**: Возможность заранее зарегистрировать известных говорящих
- 🎤 **Стриминг**: Диаризация в реальном времени с микрофона
- 📁 **Простой API**: Удобный интерфейс для использования
- ⚙️ **Конфигурационные файлы**: Гибкая настройка параметров через YAML

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Простое использование

```python
from onlinediar.diarizer import Diarizer

# Создание диаризатора
diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)

# Диаризация аудио файла
annotation = diarizer.diarize("audio.wav")

# Сохранение результатов в RTTM
diarizer.save_rttm(annotation, "output.rttm")
```

### 2. Использование с конфигурацией

```python
from onlinediar.utils.config_loader import load_diarizer_from_config

# Загрузка из конфига
diarizer = load_diarizer_from_config("configs/psda_pca.yaml")

# Диаризация
annotation = diarizer.diarize("audio.wav")
diarizer.save_rttm(annotation, "output.rttm")
```

### 3. С регистрацией спикеров

```python
from onlinediar.diarizer import Diarizer

diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)

# Регистрация известных спикеров
diarizer.register_speaker("john.wav", "Джон")
diarizer.register_speaker("alice.wav", "Алиса")

# Диаризация
annotation = diarizer.diarize("audio.wav")

# Результаты с именами
speaker_map = diarizer.get_speaker_map()
for segment, _, speaker in annotation.itertracks(yield_label=True):
    name = speaker_map.get(speaker, speaker)
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {name}")
```

### 4. Стриминг с микрофона

```python
from onlinediar.diarizer import Diarizer

diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib"
)

# Стриминг (Ctrl+C для остановки)
diarizer.stream_from_microphone(save_audio=True, output_prefix="stream")
```

## Конфигурационные файлы

В папке `configs/` находятся готовые конфигурации:

- `psda_pca.yaml` - PSDA с PCA (лучшее качество)
- `psda.yaml` - PSDA без PCA
- `csea_pca.yaml` - CSEA с PCA (быстрее)
- `csea.yaml` - CSEA без PCA

### Пример конфигурации

```yaml
backend: "psda"

# Пути к моделям
emb_model_path: "models/voxblink2_samresnet34_ft.onnx"
psda_path: "models/psda_vox_128_2sec"
pca_path: "models/pca_model_128d.joblib"

# Параметры
online_psda_threshold: -30.0
resample_rate: 16000
step_size: 1.0
duration: 2.015
vad_threshold: 0.5
min_silence_dur_ms: 1000
min_duration_for_change: 1000
device: "cpu"
use_pca: true
freeze: false

# Предварительные спикеры (опционально)
# prepared_speakers:
#   - name: "Джон"
#     audio_path: "speakers/john.wav"
```

## Параметры

### Основные параметры

- `backend`: "psda" или "csea" - алгоритм кластеризации
- `emb_model_path`: путь к модели эмбеддингов (.onnx или папка с .pt)
- `device`: "cpu" или "cuda"

### PSDA параметры

- `psda_path`: путь к модели PSDA
- `online_psda_threshold`: порог схожести (-20 до -40, ниже = строже)

### CSEA параметры

- `csea_threshold`: порог косинусного сходства (0.0 до 1.0, выше = строже)
- `max_embeddings_per_speaker`: максимум эмбеддингов на спикера

### Обработка аудио

- `resample_rate`: частота дискретизации (обычно 16000)
- `step_size`: шаг для оверлапающихся окон (секунды)
- `duration`: длительность окна для эмбеддингов (секунды)

### VAD параметры

- `vad_threshold`: чувствительность VAD (0.0 до 1.0)
- `min_silence_dur_ms`: минимальная длительность тишины (мс)
- `min_duration_for_change`: минимальная длительность для смены спикера (мс)

### Дополнительно

- `use_pca`: использовать PCA для снижения размерности
- `pca_path`: путь к модели PCA (.joblib)
- `freeze`: если True, использовать только зарегистрированных спикеров

## Требуемые модели

Для работы требуются следующие модели:

1. **Модель эмбеддингов**: `voxblink2_samresnet34_ft.onnx`
   - Скачать: [WeSpeaker Models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)

2. **PSDA модель** (если используете PSDA): `psda_vox_128_2sec`
   - Обучается на ваших данных или используйте предобученную

3. **PCA модель** (опционально): `pca_model_128d.joblib`
   - Обучается на ваших данных для снижения размерности с 256 до 128

Поместите модели в папку `models/`.

## Примеры

Все примеры находятся в папке `examples/`:

- `simple_usage.py` - базовое использование
- `with_config.py` - использование с конфигом
- `with_registered_speakers.py` - с регистрацией спикеров
- `streaming.py` - стриминг с микрофона

## API Reference

### Класс Diarizer

```python
Diarizer(
    backend="csea",
    emb_model_path=None,
    psda_path=None,
    online_psda_threshold=-30.0,
    csea_threshold=0.68,
    max_embeddings_per_speaker=40,
    resample_rate=16000,
    step_size=1.0,
    min_silence_dur_ms=1000,
    min_duration_for_change=1000,
    vad_threshold=0.5,
    duration=2.015,
    device="cpu",
    freeze=False,
    use_pca=False,
    pca_path=None
)
```

#### Методы

- `diarize(audio_path: str) -> Annotation` - диаризация аудио файла
- `register_speaker(audio_path: str, speaker_name: str)` - регистрация спикера
- `save_rttm(annotation: Annotation, output_path: str, audio_uri: str)` - сохранение в RTTM
- `stream_from_microphone(save_audio: bool, output_prefix: str)` - стриминг с микрофона
- `get_speaker_map() -> Dict[str, str]` - получить маппинг спикеров
- `reset()` - сброс состояния

## Улучшения по сравнению со старой версией

1. ✅ **Оверлапы**: Обработка перекрывающихся сегментов через `step_size` и `duration`
2. ✅ **PCA**: Снижение размерности эмбеддингов для улучшения качества
3. ✅ **Регистрация спикеров**: Метод `recognize()` для предварительной регистрации
4. ✅ **Улучшенная обработка**: Метод `_process_diarization_segments()` для объединения и разделения сегментов
5. ✅ **Простой API**: Класс `Diarizer` для удобного использования
6. ✅ **Конфигурации**: Готовые YAML конфиги для разных сценариев

## Лицензия

Apache License 2.0

## Контакты

Если есть вопросы или предложения, создайте issue в репозитории.
