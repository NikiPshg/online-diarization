# Структура проекта

```
online-diarization/
│
├── onlinediar/                      # Основная библиотека
│   ├── __init__.py                  # Экспорт основных классов
│   ├── diarizer.py                  # Простой API класс (НОВОЕ)
│   │
│   ├── cli/                         # CLI классы для диаризации
│   │   ├── __init__.py
│   │   ├── base.py                  # Базовый класс (ОБНОВЛЕН)
│   │   ├── psda_diarizate.py        # PSDA бэкенд (ОБНОВЛЕН)
│   │   ├── csea_diarizate.py        # CSEA бэкенд (ОБНОВЛЕН)
│   │   └── hub.py                   # Загрузка моделей
│   │
│   ├── clustering/                  # Алгоритмы кластеризации
│   │   ├── backend/                 # Бэкенды для скоринга
│   │   ├── online_clustering.py     # Онлайн кластеризация
│   │   └── PSDA/                    # PSDA модель
│   │
│   ├── models/                      # Модели эмбеддингов
│   │   ├── speaker_model.py         # Основной класс
│   │   └── ...                      # Различные архитектуры
│   │
│   └── utils/                       # Утилиты
│       ├── config_loader.py         # Загрузка конфигов (НОВОЕ)
│       ├── checkpoint.py            # Загрузка чекпоинтов
│       └── utils.py                 # Общие утилиты
│
├── configs/                         # Конфигурационные файлы (НОВОЕ)
│   ├── psda_pca.yaml               # PSDA + PCA (рекомендуется)
│   ├── psda.yaml                   # PSDA без PCA
│   ├── csea_pca.yaml               # CSEA + PCA
│   └── csea.yaml                   # CSEA без PCA
│
├── examples/                        # Примеры использования (НОВОЕ)
│   ├── simple_usage.py             # Базовое использование
│   ├── with_config.py              # С конфигурационным файлом
│   ├── with_registered_speakers.py # С регистрацией спикеров
│   └── streaming.py                # Стриминг с микрофона
│
├── models/                          # Папка для моделей (создать)
│   ├── voxblink2_samresnet34_ft.onnx  # Модель эмбеддингов
│   ├── psda_vox_128_2sec/             # PSDA модель
│   └── pca_model_128d.joblib          # PCA модель (опционально)
│
├── README.md                        # Основная документация (ОБНОВЛЕН)
├── QUICKSTART.md                    # Быстрый старт (НОВОЕ)
├── MIGRATION_GUIDE.md               # Руководство по миграции (НОВОЕ)
├── CHANGELOG.md                     # История изменений (НОВОЕ)
├── PROJECT_STRUCTURE.md             # Этот файл (НОВОЕ)
├── requirements.txt                 # Зависимости (ОБНОВЛЕН)
├── test_simple.py                   # Простой тест (НОВОЕ)
└── .gitignore                       # Git ignore файл
```

## Ключевые файлы

### 1. Основной API

**onlinediar/diarizer.py** - Главный класс для использования
```python
from onlinediar import Diarizer

diarizer = Diarizer(backend="psda", ...)
annotation = diarizer.diarize("audio.wav")
```

### 2. Базовые классы

**onlinediar/cli/base.py** - Базовый класс с общей логикой
- Загрузка моделей (ONNX и PyTorch)
- VAD (Voice Activity Detection)
- Извлечение эмбеддингов
- Поддержка PCA
- Обработка оверлапов

**onlinediar/cli/psda_diarizate.py** - PSDA бэкенд
- Использует PSDA модель для кластеризации
- Метод `processing_embedding()` для обработки эмбеддингов
- Метод `recognize()` для регистрации спикеров

**onlinediar/cli/csea_diarizate.py** - CSEA бэкенд
- Использует косинусное сходство для кластеризации
- Быстрее PSDA, но может быть менее точным

### 3. Конфигурации

**configs/*** - Готовые конфигурации для разных сценариев
- `psda_pca.yaml` - Лучшее качество
- `psda.yaml` - Без PCA
- `csea_pca.yaml` - Быстрее с хорошим качеством
- `csea.yaml` - Самый быстрый

### 4. Утилиты

**onlinediar/utils/config_loader.py** - Загрузка конфигураций
```python
from onlinediar import load_diarizer_from_config

diarizer = load_diarizer_from_config("configs/psda_pca.yaml")
```

## Основные компоненты

### 1. Класс Diarizer

Главный класс для взаимодействия с библиотекой.

**Методы:**
- `diarize(audio_path)` - диаризация аудио файла
- `register_speaker(audio_path, name)` - регистрация спикера
- `save_rttm(annotation, output_path)` - сохранение результатов
- `stream_from_microphone()` - стриминг с микрофона
- `get_speaker_map()` - получить маппинг спикеров
- `reset()` - сброс состояния

### 2. Базовый класс BaseSpeakerDiarization

**Основные методы:**
- `_init_embedding_model()` - загрузка модели эмбеддингов
- `_init_vad()` - инициализация VAD
- `_init_pca_model()` - загрузка PCA модели
- `extract_embedding()` - извлечение эмбеддингов
- `extract_embedding_from_pcm()` - извлечение из аудио файла
- `online_diarization()` - основной метод диаризации
- `_get_start_end()` - определение сегментов с оверлапами
- `_process_diarization_segments()` - обработка оверлапов
- `run()` - обработка аудио потока (переопределяется в подклассах)

### 3. PSDA/CSEA классы

**PsdaSpeakerDiarization:**
- Использует PSDA для вычисления схожести
- `processing_embedding()` - обработка эмбеддинга

**CseaSpeakerDiarization:**
- Использует косинусное сходство
- `find_speaker()` - поиск ближайшего спикера

## Поток данных

```
Аудио файл
    ↓
[Загрузка и ресэмплинг]
    ↓
[VAD - детекция речи]
    ↓
[Создание оверлапающихся окон] ← step_size, duration
    ↓
[Извлечение эмбеддингов]
    ↓
[PCA (опционально)]
    ↓
[Кластеризация (PSDA/CSEA)]
    ↓
[Обработка оверлапов]
    ↓
[Объединение соседних сегментов]
    ↓
[Маппинг на имена спикеров]
    ↓
Annotation (pyannote.core)
    ↓
RTTM файл
```

## Ключевые улучшения версии 2.0

### 1. Оверлапы

**Старая версия:**
```
Сегмент 1: [0.0 - 2.0]
Сегмент 2:       [2.0 - 4.0]
Сегмент 3:             [4.0 - 6.0]
```

**Новая версия (step_size=1.0):**
```
Сегмент 1: [0.0 - 2.0]
Сегмент 2:   [1.0 - 3.0]
Сегмент 3:     [2.0 - 4.0]
Сегмент 4:       [3.0 - 5.0]
Сегмент 5:         [4.0 - 6.0]
```

Преимущества:
- Лучше ловит смены спикеров
- Более плавные переходы
- Меньше ошибок на границах

### 2. PCA

**Без PCA:**
- Эмбеддинги: 256 измерений
- Больше шума
- Сложнее разделять спикеров

**С PCA:**
- Эмбеддинги: 128 измерений
- Меньше шума
- Лучшее разделение спикеров
- Меньше вычислений

### 3. Регистрация спикеров

**Старая версия:**
- Только автоматическая кластеризация
- Спикеры: SPEAKER_00, SPEAKER_01, ...

**Новая версия:**
```python
diarizer.register_speaker("john.wav", "Джон")
diarizer.register_speaker("alice.wav", "Алиса")

# Результат:
# SPEAKER_00 → Джон
# SPEAKER_01 → Алиса
```

## Зависимости

### Основные
- torch, torchaudio - работа с аудио и моделями
- onnxruntime - для ONNX моделей
- silero-vad - Voice Activity Detection
- pyannote.core - для объекта Annotation

### Для обработки
- numpy - вычисления
- scipy - научные вычисления
- scikit-learn - PCA и другие утилиты

### Для конфигураций
- PyYAML - загрузка конфигов
- pydantic - валидация
- joblib - загрузка PCA моделей

## Тестирование

Запустите простой тест:
```bash
python test_simple.py
```

Проверяет:
- Импорты
- Структуру файлов
- Наличие конфигураций
- Наличие примеров

## Следующие шаги

1. ✅ Скачайте модели (см. README.md)
2. ✅ Запустите test_simple.py для проверки
3. ✅ Попробуйте примеры из папки examples/
4. ✅ Настройте параметры под свои задачи
5. ✅ При необходимости обучите PCA модель

## Поддержка

- README.md - основная документация
- QUICKSTART.md - быстрый старт
- MIGRATION_GUIDE.md - миграция со старой версии
- examples/ - примеры кода
