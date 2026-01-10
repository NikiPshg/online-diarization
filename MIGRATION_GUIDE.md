# Руководство по миграции

## Что нового в версии 2.0

Эта версия включает все улучшения из `sync_speech_recognition` и делает библиотеку полностью автономной.

### Основные улучшения

1. **✅ Поддержка PCA**
   - Снижение размерности эмбеддингов с 256 до 128
   - Значительное улучшение качества диаризации
   - Опциональное использование (можно работать и без PCA)

2. **✅ Обработка оверлапов**
   - Параметр `step_size` для создания перекрывающихся окон
   - Улучшенный метод `_process_diarization_segments()` для обработки пересечений
   - Автоматическое объединение соседних сегментов одного спикера
   - Разделение оверлапов между разными спикерами по середине

3. **✅ Регистрация спикеров**
   - Новый метод `recognize()` для предварительной регистрации
   - Поддержка режима `freeze` (только зарегистрированные спикеры)
   - Маппинг ID спикеров на их имена

4. **✅ Улучшенная извлечение эмбеддингов**
   - Правильная обработка коротких сегментов
   - Использование контекста (предыдущие 2 секунды) для коротких фрагментов
   - Нормализация аудио

5. **✅ Простой API**
   - Класс `Diarizer` для удобного использования
   - Загрузка из конфигурационных файлов
   - Готовые примеры для всех сценариев

6. **✅ Автономность**
   - Удалены зависимости от `streaming_nemo`
   - Все импорты локальные
   - Готовый к использованию репозиторий

## Миграция со старого API

### Было (старая версия)

```python
from onlinediar.cli.online_speaker_psda import OnlineSpeakerPSDA

online_speaker = OnlineSpeakerPSDA(
    model_path="models/model.pt",
    psda_path="models/psda",
    online_vad_threshold=0.5,
    online_psda_threshold=-300.0,
    min_silence_duration_vad_ms=350.0,
    min_dur_for_changes=1500.0,
    min_silence_dur=1000,
    device="cpu"
)

result = online_speaker.online_diarization("audio.wav", save=True)
```

### Стало (новая версия)

```python
from onlinediar import Diarizer

diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    online_psda_threshold=-30.0,  # с PCA используем меньшее значение
    vad_threshold=0.5,
    min_silence_dur_ms=1000,
    min_duration_for_change=1000,
    device="cpu",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib",
    step_size=1.0  # новый параметр для оверлапов
)

annotation = diarizer.diarize("audio.wav")
diarizer.save_rttm(annotation, "output.rttm")
```

### Ключевые изменения

1. **Класс**: `OnlineSpeakerPSDA` → `Diarizer`
2. **Параметры**:
   - `model_path` → `emb_model_path`
   - `min_silence_duration_vad_ms` → `min_silence_dur_ms`
   - `min_dur_for_changes` → `min_duration_for_change`
   - Добавлен `step_size` для оверлапов
   - Добавлены `use_pca` и `pca_path`

3. **Методы**:
   - `online_diarization()` → `diarize()`
   - Добавлен `register_speaker()` для регистрации
   - Добавлен `save_rttm()` для сохранения

## Новые возможности

### 1. Использование с конфигурацией

```python
from onlinediar import load_diarizer_from_config

# Просто укажите путь к конфигу
diarizer = load_diarizer_from_config("configs/psda_pca.yaml")
annotation = diarizer.diarize("audio.wav")
```

### 2. Регистрация известных спикеров

```python
diarizer = Diarizer(...)

# Регистрируем известных спикеров
diarizer.register_speaker("john.wav", "Джон")
diarizer.register_speaker("alice.wav", "Алиса")

# Теперь результаты будут использовать эти имена
annotation = diarizer.diarize("meeting.wav")
```

### 3. Работа с результатами

```python
# Получение маппинга спикеров
speaker_map = diarizer.get_speaker_map()
print(speaker_map)  # {'SPEAKER_00': 'Джон', 'SPEAKER_01': 'Алиса'}

# Вывод результатов
for segment, _, speaker in annotation.itertracks(yield_label=True):
    name = speaker_map.get(speaker, speaker)
    print(f"{segment.start:.2f}s - {segment.end:.2f}s: {name}")
```

### 4. Два бэкенда на выбор

```python
# PSDA (лучшее качество, медленнее)
diarizer_psda = Diarizer(backend="psda", ...)

# CSEA (быстрее, хорошее качество)
diarizer_csea = Diarizer(backend="csea", ...)
```

## Рекомендуемые настройки

### Для лучшего качества (PSDA + PCA)

```python
diarizer = Diarizer(
    backend="psda",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    psda_path="models/psda_vox_128_2sec",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib",
    online_psda_threshold=-30.0,
    step_size=1.0,
    duration=2.015,
    device="cpu"
)
```

### Для скорости (CSEA + PCA)

```python
diarizer = Diarizer(
    backend="csea",
    emb_model_path="models/voxblink2_samresnet34_ft.onnx",
    use_pca=True,
    pca_path="models/pca_model_128d.joblib",
    csea_threshold=0.68,
    step_size=1.0,
    device="cpu"
)
```

## Тонкая настройка параметров

### PSDA threshold

- С PCA: от -20 до -40 (рекомендуется -30)
- Без PCA: от -200 до -400 (рекомендуется -300)
- Ниже = строже (меньше ошибок, но больше "ложных" спикеров)

### CSEA threshold

- С PCA: 0.65 - 0.75 (рекомендуется 0.68)
- Без PCA: 0.70 - 0.80 (рекомендуется 0.75)
- Выше = строже

### Step size (оверлапы)

- 1.0 сек: хороший баланс качества и скорости
- 0.5 сек: лучшее качество, медленнее
- 1.5 сек: быстрее, но может пропускать быстрые смены спикеров

## Получение моделей

### 1. Embedding модель

```bash
# Скачать voxblink2_samresnet34_ft.onnx
# Источник: https://github.com/wenet-e2e/wespeaker
```

### 2. PSDA модель

Обучить на ваших данных или использовать предобученную.

### 3. PCA модель

```python
# Обучение PCA модели на ваших эмбеддингах
from sklearn.decomposition import PCA
import joblib

# embeddings - ваши эмбеддинги (N x 256)
pca = PCA(n_components=128)
pca.fit(embeddings)
joblib.dump(pca, "pca_model_128d.joblib")
```

## Часто задаваемые вопросы

**Q: Нужен ли PCA?**
A: Не обязательно, но настоятельно рекомендуется. PCA значительно улучшает качество.

**Q: Какой бэкенд лучше?**
A: PSDA дает лучшее качество, CSEA быстрее. Для большинства задач рекомендуется PSDA с PCA.

**Q: Можно ли использовать без регистрации спикеров?**
A: Да, регистрация опциональна. Система работает и без нее.

**Q: Как работать со стримингом?**
A: Используйте `diarizer.stream_from_microphone()` или реализуйте свою логику через методы класса.

**Q: Совместим ли со старыми RTTM файлами?**
A: Да, формат RTTM стандартный и полностью совместим.
