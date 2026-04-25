# World Model Research Project

Исследовательский проект по `world models` и `chunked video generation`.

Текущий основной сценарий живёт в ноутбуке [ftiad_cursach_proj_v4ipynb.ipynb](/Users/egorgladilin/vscodeProjects/world_model/ftiad_cursach_proj_v4ipynb.ipynb), а основная evaluation-логика уже вынесена в модульный Python-код внутри [src/](/Users/egorgladilin/vscodeProjects/world_model/src).

## Pipeline

Текущий rollout pipeline устроен так:

1. на вход подаются `image + prompt`
2. модель генерирует один `video chunk`
3. из chunk video сохраняется `last frame`
4. `last frame` становится входом для следующего чанка
5. в итоге собирается rollout из нескольких чанков

Ноутбук остаётся orchestration-layer: он отвечает за запуск, конфигурацию, chunk loop, логирование и визуализацию, а ядро evaluation теперь живёт в `src/`.

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

На текущем этапе в `src/` уже вынесены:

- FVD evaluation stack
- Omni evaluation stack
- VBench evaluation stack
- WorldScore data adapter layer
- Omni rollout summary / agentic score helpers
- Omni embedding backend layer

Текущий pipeline при этом не переписывался: benchmark/data integrations добавляются через adapter layer, а не через изменения `run_chunk_loop_with_metrics(...)`.

## Quick Start

1. Склонировать репозиторий и установить зависимости из `requirements.txt`.
2. Подготовить внешние runtime backend’ы:
   - `pytorch-i3d` для FVD;
   - `VBench` repo/runtime для VBench.
3. Открыть [ftiad_cursach_proj_v4ipynb.ipynb](/Users/egorgladilin/vscodeProjects/world_model/ftiad_cursach_proj_v4ipynb.ipynb).
4. Проверить актуальные runtime paths в env/launch ячейках.
5. Сначала прогнать облегчённый smoke-run.
6. Только после этого запускать более тяжёлые rollout-эксперименты.

## Current Architecture

- `src/data/worldscore/` — data adapter layer for benchmark inputs
- `src/metrics/fvd/` — FVD evaluation stack
- `src/metrics/omni/` — Omni evaluation stack
- `src/metrics/vbench/` — VBench evaluation stack
- notebook — orchestration, launch, logging, visualization

## Repository Structure

```text
world_model/
├── ftiad_cursach_proj_v4ipynb.ipynb
├── README.md
├── requirements.txt
└── src/
    ├── data/
    │   ├── __init__.py
    │   └── worldscore/
    │       ├── __init__.py
    │       └── adapter.py
    ├── metrics/
    │   ├── __init__.py
    │   ├── fvd/
    │   │   ├── __init__.py
    │   │   ├── features.py
    │   │   ├── metric.py
    │   │   ├── pairs.py
    │   │   └── utils.py
    │   ├── omni/
    │   │   ├── __init__.py
    │   │   ├── agentic_score.py
    │   │   ├── runner.py
    │   │   ├── spec.py
    │   │   ├── summary.py
    │   │   ├── common/
    │   │   ├── interstab_l/
    │   │   ├── interstab_n/
    │   │   ├── intercov/
    │   │   ├── interorder/
    │   │   ├── transitions_detect/
    │   │   ├── object_control/
    │   │   └── camera_control/
    │   └── vbench/
    │       ├── __init__.py
    │       ├── metric.py
    │       ├── parser.py
    │       └── summary.py
    ├── pipeline/
    │   └── __init__.py
    └── utils/
        └── __init__.py
```

## Evaluation Stack

Основные entry points evaluation-части теперь такие:

- [FVDMetric](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/metric.py)  
  Считает стандартный FVD, full-rollout FVD и chunk-consistency вспомогательные метрики.
- [OmniMetricRunner](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/omni/runner.py)  
  Оркестрирует modular Omni metrics, готовит shared inputs и собирает unified Omni result.
- [VBenchMetric](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/vbench/metric.py)  
  Инкапсулирует VBench backend lifecycle, parsing и chunk/full-rollout execution.

Вынесенные Omni components включают:

- `build_omni_spec(...)`
- `InterStabLMetric`
- `InterStabNMetric`
- `InterCovMetric`
- `InterOrderMetric`
- `TransitionsDetectMetric`
- `ObjectControlMetric`
- `CameraControlMetric`
- `compute_agentic_score(...)`
- `add_optional_full_rollout_omni_summary(...)`

## WorldScore Data Adapter

WorldScore adapter layer вынесен в [src/data/worldscore/](/Users/egorgladilin/vscodeProjects/world_model/src/data/worldscore/__init__.py).

Основные сущности:

- `WorldScoreDatasetAdapter`  
  Загружает split `Howieeeee/WorldScore`, сохраняет изображение в cache-dir, нормализует запись и собирает pipeline-compatible `world_spec`.
- `WorldScoreSample`  
  Нормализованный sample object с `image_path`, `positive_prompt`, `negative_prompt`, `world_spec` и `metadata`.

Adapter layer не меняет существующий pipeline и не требует менять `run_chunk_loop_with_metrics(...)`. Он только преобразует benchmark row в тот data shape, который уже понимает текущий orchestration path.

WorldScore в текущей постановке рассматривается как основной benchmark-input прежде всего для:

- Omni
- VBench

Для FVD WorldScore используется только косвенно: датасет не даёт GT-video, поэтому FVD здесь трактуется как слой chunk consistency, adjacent cross-chunk Fréchet и drift from first chunk, а не как strict comparison against reference real video.

### WorldScore Usage Example

```python
from src.data.worldscore import WorldScoreDatasetAdapter

adapter = WorldScoreDatasetAdapter(
    split_name="dynamic",
    cache_dir="/content/worldscore_cache",
    num_chunks=1,
)

sample = adapter.get_sample(0)

videos, current_image, clip_log, metric_history, rollout_record = run_chunk_loop_with_metrics(
    image_path=sample.image_path,
    positive_prompt=sample.positive_prompt,
    negative_prompt=sample.negative_prompt,
    world_spec=sample.world_spec,
    ...
)
```

Этот workflow добавляет только data adaptation layer. Сам evaluation stack и launch path остаются прежними.

## FVD Module

FVD-код вынесен в пакет [src/metrics/fvd/](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/__init__.py).

Главный entry point:

- [FVDMetric](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/metric.py)

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

- [src/metrics/fvd/metric.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/metric.py)
  Основной класс `FVDMetric` и orchestration логика FVD.
- [src/metrics/fvd/features.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/features.py)
  `I3DFeatureExtractor`, lazy init/unload и hook-based feature extraction.
- [src/metrics/fvd/pairs.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/pairs.py)
  Pairing и validation для manifest/list inputs.
- [src/metrics/fvd/utils.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/fvd/utils.py)
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

## Omni Module

Omni-код вынесен в пакет [src/metrics/omni/](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/omni/__init__.py).

Главные entry points:

- [OmniMetricRunner](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/omni/runner.py)
- [build_omni_spec](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/omni/spec.py)

Runner modular считает:

- `InterStab-L`
- `InterStab-N`
- `InterCov`
- `InterOrder`
- `Transitions Detect`
- `Object Control`
- `Camera Control`

Common helpers и notebook-side хвосты тоже вынесены в `src/metrics/omni/...`, включая:

- standardized result protocol
- embedding backend layer
- rollout summary helper
- agentic score helper

Ноутбук больше не держит legacy Omni metric implementations и использует только modular Omni path.

WorldScore хорошо подходит как benchmark-input для Omni path, потому что adapter сразу даёт `world_spec`, совместимый с существующим rollout/evaluation workflow.

## VBench Module

VBench-код вынесен в пакет [src/metrics/vbench/](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/vbench/__init__.py).

Главный entry point:

- [VBenchMetric](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/vbench/metric.py)

Что он делает:

- lazy-load VBench backend
- запускает VBench на одном chunk video
- запускает VBench на full rollout
- парсит raw VBench output
- нормализует scores в плоские `vbench_*` поля
- возвращает structured result, не роняя основной pipeline при backend/runtime failure

Parser и rollout summary helpers живут в:

- [src/metrics/vbench/parser.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/vbench/parser.py)
- [src/metrics/vbench/summary.py](/Users/egorgladilin/vscodeProjects/world_model/src/metrics/vbench/summary.py)

Ноутбук теперь использует import-based path через `VBenchMetric`, а не локальные VBench helper functions.

WorldScore также хорошо подходит как benchmark-input для VBench path: adapter даёт готовый `image_path + prompt + metadata` слой, а VBench execution остаётся неизменным.

## Using The Evaluation Stack From Python

```python
from src.metrics import FVDMetric, OmniMetricRunner, VBenchMetric

fvd_metric = FVDMetric(device="cuda")
omni_runner = OmniMetricRunner(verbose=False)
vbench_metric = VBenchMetric(device="cpu")
```

Практически это те три объекта, вокруг которых теперь строится notebook orchestration path.

## Colab Usage

Короткий практический сценарий для Colab:

1. склонировать репозиторий проекта;
2. установить notebook/runtime зависимости;
3. отдельно убедиться, что доступны внешние backend-зависимости:
   - `VBench` repo/runtime;
   - `pytorch-i3d` для FVD;
4. открыть [ftiad_cursach_proj_v4ipynb.ipynb](/Users/egorgladilin/vscodeProjects/world_model/ftiad_cursach_proj_v4ipynb.ipynb);
5. проверить runtime paths в launch/env ячейках;
6. сначала прогнать облегчённый smoke-run, и только потом тяжёлый rollout.

Для Colab-окружения важно проверять фактические пути до внешних runtime-папок. На рабочих прогонах использовались, например:

- VBench repo: `/content/world_modelling/VBench`
- VBench full info json: `/content/world_modelling/VBench/vbench/VBench_full_info.json`
- FVD backend repo: `/content/pytorch-i3d`
- VBench temp input root: `/content/vbench_inputs`

Эти пути не стоит считать универсальными для любой среды: их нужно сверять с фактическим layout окружения.

## Known Runtime Dependencies

- `pytorch-i3d` должен быть доступен для FVD runtime.
- `VBench` repo/runtime должен быть доступен для VBench execution.
- Часть тяжёлых зависимостей поднимается lazy-import’ом только в момент реального запуска.
- Пути могут отличаться между local environment, Colab и HPC, поэтому runtime paths нужно проверять явно.
- Live loading `Howieeeee/WorldScore` через Hugging Face зависит от runtime/network availability.

## Runtime Notes

- `FVDMetric` опирается на `pytorch-i3d`; без доступного I3D backend FVD path не будет полноценно работать.
- `VBenchMetric` использует lazy import для VBench backend и ряда тяжёлых runtime dependencies. Это сделано специально, чтобы ноутбук открывался и импортировался без VBench noise до фактического вызова `run(...)`.
- Часть heavy backend’ов намеренно поднимается только в runtime, а не на module import stage.
- Ноутбук ориентирован прежде всего на Colab-like execution environment.
- `src/` теперь является основным местом жизни evaluation-логики; notebook-side код в основном ограничен orchestration и запуском.

## Smoke-Test Checklist

- репозиторий клонирован, а `src` доступен для импортов;
- создаётся `WorldScoreDatasetAdapter`;
- `adapter.get_sample(...)` возвращает pipeline-compatible sample;
- `sample.world_spec` передаётся в launch path без дополнительных изменений;
- создаются `FVDMetric`, `OmniMetricRunner`, `VBenchMetric`;
- VBench backend repo доступен по актуальному runtime path;
- `pytorch-i3d` доступен по ожидаемому пути;
- проходит облегчённая launch/smoke-run ячейка в ноутбуке;
- chunk generation завершается без раннего runtime failure;
- `clip_log` содержит `omni_*`, `vbench_*` и chunk-level `fvd_*`/consistency fields;
- rollout summary собирается без падения;
- только после этого имеет смысл включать более тяжёлый full-rollout path.

## WorldScore Validation Note

На текущем этапе adapter-level compatibility уже подтверждена:

- `WorldScoreDatasetAdapter` и `WorldScoreSample` проходят API smoke-checks;
- `_build_world_spec_compat(...)` совпадает с notebook-side `build_world_spec(...)`;
- `sample.world_spec` совместим с текущим pipeline call без изменения `run_chunk_loop_with_metrics(...)`.

При этом live Hugging Face loading и полноценный end-to-end generation run всё ещё зависят от runtime/network условий и должны рассматриваться как execution-level checks, а не как архитектурный риск адаптера.

## Design Principle

Целевая архитектура проекта:

- ноутбук = orchestration-layer
- `src/` = reusable logic
- основная evaluation-логика уже живёт в `src/metrics/...`
- notebook-side code может всё ещё содержать отдельные utility / visualization / environment setup blocks, но не core metric implementations

FVD, Omni и VBench уже приведены к этой модели. Не полностью вынесенными пока могут оставаться отдельные notebook-side utility blocks, не являющиеся ядром evaluation stack.

## Notes

- Текущий FVD extractor опирается на `pytorch-i3d`
- По умолчанию код ожидает I3D repo и checkpoint в среде выполнения
- VBench требует доступного backend/runtime repo
- Полный runtime smoke-test с реальными backend’ами и реальными rollout-видео стоит прогонять после значимых изменений в evaluation pipeline
