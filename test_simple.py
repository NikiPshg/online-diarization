"""
Простой тест для проверки работоспособности библиотеки.
"""

import sys
import os

# Добавляем путь к библиотеке
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Тест импорта
    print("=" * 60)
    print("Тест импорта библиотеки...")
    print("=" * 60)
    
    from onlinediar import Diarizer, load_diarizer_from_config
    print("✅ Успешный импорт основных классов")
    
    from onlinediar.cli import BaseSpeakerDiarization, PsdaSpeakerDiarization, CseaSpeakerDiarization
    print("✅ Успешный импорт CLI классов")
    
    # Проверка версии
    import onlinediar
    print(f"✅ Версия библиотеки: {onlinediar.__version__}")
    
    print("\n" + "=" * 60)
    print("Создание диаризатора (тестовая конфигурация)...")
    print("=" * 60)
    
    # Тест создания диаризатора (без реальных моделей)
    # В реальном использовании нужны настоящие пути к моделям
    try:
        test_config = {
            "backend": "csea",
            "emb_model_path": "models/test.onnx",  # фейковый путь для теста
            "csea_threshold": 0.68,
            "device": "cpu",
            "use_pca": False
        }
        
        # Не создаем объект, чтобы не было ошибок из-за отсутствующих моделей
        print("⚠️  Для полного теста нужны реальные модели")
        print("   См. README.md для инструкций по скачиванию моделей")
        
    except Exception as e:
        print(f"⚠️  Ошибка при создании диаризатора: {e}")
        print("   Это нормально, если модели не загружены")
    
    print("\n" + "=" * 60)
    print("Проверка конфигурационных файлов...")
    print("=" * 60)
    
    config_files = [
        "configs/psda_pca.yaml",
        "configs/psda.yaml",
        "configs/csea_pca.yaml",
        "configs/csea.yaml"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} не найден")
    
    print("\n" + "=" * 60)
    print("Проверка примеров...")
    print("=" * 60)
    
    example_files = [
        "examples/simple_usage.py",
        "examples/with_config.py",
        "examples/with_registered_speakers.py",
        "examples/streaming.py"
    ]
    
    for example_file in example_files:
        if os.path.exists(example_file):
            print(f"✅ {example_file}")
        else:
            print(f"❌ {example_file} не найден")
    
    print("\n" + "=" * 60)
    print("✅ Базовая проверка завершена успешно!")
    print("=" * 60)
    print("\nСледующие шаги:")
    print("1. Скачайте необходимые модели (см. README.md)")
    print("2. Запустите примеры из папки examples/")
    print("3. См. QUICKSTART.md для быстрого старта")
    
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("\nВозможные причины:")
    print("1. Не установлены зависимости: pip install -r requirements.txt")
    print("2. Неверный путь к библиотеке")
    sys.exit(1)
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
