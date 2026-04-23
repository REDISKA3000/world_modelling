# World Model Research Project

Исследовательский проект по `world models` и `chunked video generation`.

Текущий основной сценарий живёт в ноутбуке [ftiad_cursach_proj_v4ipynb.ipynb](/Users/egorgladilin/vscodeProjects/world_model/ftiad_cursach_proj_v4ipynb.ipynb), а логика постепенно выносится в модульный Python-код внутри [src/](/Users/egorgladilin/vscodeProjects/world_model/src).

## Pipeline

Текущий rollout pipeline устроен так:

1. на вход подаются `image + prompt`
2. модель генерирует один `video chunk`
3. из chunk video сохраняется `last frame`
4. `last frame` становится входом для следующего чанка
5. в итоге собирается rollout из нескольких чанков

Ноутбук остаётся orchestration-layer, а крупные логические блоки постепенно выносятся в `src/`.

## Current Status

В ноутбуке уже есть:

- chunk video generation
- `run_chunk_loop`
- `run_chunk_loop_with_metrics`
- `clip_log` / `dataframe` / `csv` логирование
- Omni-Metric блок
- VBench интеграция
- FVD расчёт
- full-rollout summary

На текущем этапе в `src/` уже вынесен FVD-блок.

## Repository Structure

```text
world_model/
├── ftiad_cursach_proj_v4ipynb.ipynb
├── README.md
├── requirements.txt
└── src/
    ├── metrics/
    │   ├── __init__.py
    │   └── fvd/
    │       ├── __init__.py
    │       ├── features.py
    │       ├── metric.py
    │       ├── pairs.py
    │       └── utils.py
    ├── pipeline/
    │   └── __init__.py
    └── utils/
        └── __init__.py
```

## FVD Module

FVD-код вынесен в пакет [src/metrics/fvd/](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/__init__.py).

Главный entry point:

- [FVDMetric](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/metric.py:27)

Что он инкапсулирует:

- manifest-based pairing
- explicit list pairing
- validation входных пар
- I3D-based feature extraction
- Fréchet distance
- standard FVD
- full-rollout FVD
- chunk-consistency metrics
- online history update/save
- structured result format

### FVD Files

- [src/metrics/fvd/metric.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/metric.py:27)
  Основной класс `FVDMetric` и orchestration логика FVD.
- [src/metrics/fvd/features.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/features.py:19)
  `I3DFeatureExtractor`, lazy init/unload и hook-based feature extraction.
- [src/metrics/fvd/pairs.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/pairs.py:12)
  Pairing и validation для manifest/list inputs.
- [src/metrics/fvd/utils.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/utils.py:16)
  Video loading, frame sampling, stats и Fréchet helpers.

## Using FVD From Python

### Standard FVD

```python
from src.metrics.fvd import FVDMetric

fvd_metric = FVDMetric(
    device="cuda",
    batch_size=4,
    num_frames=16,
    resize_to=(224, 224),
    sampling_strategy="uniform",
    cache_features=False,
    verbose=True,
)

result = fvd_metric.run(
    manifest_csv="generated_manifest.csv",
)

print(result)
```

`run(...)` умеет работать и с явными списками путей:

```python
result = fvd_metric.run(
    real_video_paths=["real_001.mp4", "real_002.mp4"],
    generated_video_paths=["gen_001.mp4", "gen_002.mp4"],
)
```

### Full-Rollout FVD

```python
result = fvd_metric.run_on_rollout(
    real_video_path="real_reference_video.mp4",
    generated_video_path="full_rollout.mp4",
)
```

### Callback For Notebook Pipeline

```python
compute_full_rollout_fvd_callback = fvd_metric.build_full_rollout_callback(
    real_video_path=real_video_path,
    num_frames=16,
    resize=(224, 224),
)
```

## FVD Result Format

И success-case, и final failure-case возвращают structured dict.

Основные поля результата:

- `fvd`
- `status`
- `error`
- `pairing_mode`
- `num_pairs`
- `num_valid_pairs`
- `num_failed_pairs`
- `used_real_paths`
- `used_generated_paths`
- `details`

Для совместимости также сохранены:

- `n_used`
- `n_failed`
- `bad_examples`

Пример failure-case:

```python
{
    "fvd": None,
    "status": "failed",
    "error": "No valid feature pairs were extracted.",
    "num_pairs": 4,
    "num_valid_pairs": 0,
    "num_failed_pairs": 4,
    "pairing_mode": "manifest",
    ...
}
```

## Using FVD From The Notebook

Сейчас ноутбук использует FVD так:

1. импортирует `FVDMetric`
2. создаёт один общий объект `fvd_metric`
3. передаёт этот же объект в `run_chunk_loop_with_metrics(..., fvd_metric=fvd_metric)`
4. использует `build_full_rollout_callback(...)` для full-rollout FVD

То есть FVD в ноутбуке больше не живёт как набор отдельных helper-функций. Ноутбук использует уже вынесенный модульный API.

## Design Principle

Целевая архитектура проекта:

- ноутбук = orchestration-layer
- `src/` = reusable logic
- крупные блоки метрик и evaluation постепенно переезжают из notebook cells в Python modules

FVD уже приведён к этой модели. Следующие кандидаты на аналогичный вынос:

- Omni
- VBench-adjacent helpers
- дополнительные rollout metrics

## Notes

- Текущий FVD extractor опирается на `pytorch-i3d`
- По умолчанию код ожидает I3D repo и checkpoint в среде выполнения
- Полный runtime smoke-test с реальным I3D и реальными rollout-видео стоит прогонять после значимых изменений в evaluation pipeline
