# JEV Benchmark — русский

[English version](README.md)

**Воспроизводимый benchmark для ответа на практический инженерный вопрос: когда очень маленькая модель становится действительно полезным компонентом системы принятия решений?**

## North Star

**Сколько полезного принятия решений можно перенести в чрезвычайно маленькую модель, если дорогие интеллектуальные операции — perception, planning, supervision и проверку ограничений — вынести за пределы runtime-модели?**

Практическая цель проекта — **не маленькие модели сами по себе**. Мы хотим понять, можно ли вынести сложную работу во внешний слой, а на самом устройстве оставить очень маленькую модель, которая выполняет частые, локальные и чувствительные к latency решения.

Это особенно важно для edge/embedded-сценариев: внешний компьютер или сервер может выполнять perception, planning, генерацию кандидатов или supervision, а устройство уровня ESP32 и другие сильно ограниченные устройства могут запускать только компактную decision model.

JEV — один из возможных механизмов такой архитектуры. Во время обучения или оценки LLM/JEV judge может давать supervision или reward signal, а при deployment дорогой judge не обязан находиться внутри runtime loop устройства.

## Зачем нужен benchmark

Существует много benchmark'ов, которые отвечают на вопрос **«насколько хорошо работает модель?»**. Нам нужен другой ответ:

> **Если мы сознательно вынесем сложность за пределы tiny model, сколько полезного поведения всё ещё сможет обеспечить маленький компонент системы — и что практически даст такая декомпозиция?**

Маленькая модель интересна только тогда, когда полезна **система целиком**. Поэтому benchmark не останавливается на accuracy или количестве параметров.

Мы измеряем цепочку:

**построение состояния → генерация допустимых кандидатов → tiny decision → выполнение → конечный результат**

а для JEV-based learning:

**transition среды → JEV judgment → reward → обучение → конечное поведение**

Benchmark должен позволить разработчику понять:

- достаточно ли tiny model для конкретного слоя принятия решений;
- какой capacity ей действительно нужен;
- сколько качества теряется при уменьшении модели;
- компенсируют ли меньшие latency и размер модели эту потерю;
- помогают ли confidence и fallback сделать маленькую модель практически пригодной;
- сохраняется ли решение при изменении представления или порядка кандидатов;
- полезен ли LLM/JEV-generated learning signal для последующего обучения;
- и где такая архитектура перестаёт быть практически оправданной.

## Что именно оценивает benchmark

У benchmark есть четыре уровня оценки:

1. **Качество решений** — accuracy, reward, regret, survival, game return, legal-action rate и task-specific utility.
2. **Эффективность tiny model** — количество параметров, размер/память модели, inference latency и capacity.
3. **Надёжность при выборочном использовании** — confidence, abstention, coverage и fallback.
4. **Полезность системы целиком** — поведение при последовательных решениях, robustness, learning quality, model calls, cache hits и стоимость там, где это применимо.

Поэтому benchmark выдаёт **профиль характеристик**, а не один универсальный score.

Для конкретной задачи главный практический вопрос:

> **Какую минимальную/дешёвую operating point можно использовать, сохранив достаточно полезного downstream behavior для целевой системы?**

Именно этот результат должен быть главным выходом benchmark.

## Как это может выглядеть в реальной системе

Типовая архитектура:

    CLOUD / SERVER / TEACHER
    ┌──────────────────────────────┐
    │ perception                   │
    │ planning                     │
    │ candidate generation        │
    │ JEV / LLM supervision       │
    └──────────────┬───────────────┘
                   │ compact state / candidates
                   ▼
    EDGE DEVICE
    ┌──────────────────────────────┐
    │ tiny model                   │
    │ local decision               │
    └──────────────┬───────────────┘
                   ▼
             deterministic
               execution

Не предполагается, что каждое приложение обязано иметь именно такую архитектуру. Benchmark позволяет измерить саму **границу разделения**: какую работу можно вынести из tiny model и что после этого всё ещё возможно делать внутри неё.

## Основная идея

    raw data
        ↓
    compact state
        ↓
    tiny model
        ↓
    decision / candidate
        ↓
    deterministic executor
        ↓
    reward / feedback

В candidate-selection задачах:

**planner генерирует допустимые варианты; tiny model выбирает среди них.**

В J05:

**RL agent выбирает действие; JEV оценивает transition и формирует reward.**

## Для каких практических задач это может быть полезно

Одна и та же архитектура применима к вопросам, возникающим в:

- embedded и IoT;
- low-power edge AI;
- latency-sensitive control;
- robotics и локальной автоматизации;
- игровых агентах;
- scheduling, routing и resource allocation;
- системах, где большая модель доступна на этапе разработки, но не должна запускаться при каждом локальном решении.

Benchmark не утверждает, что tiny models универсально лучше больших моделей. Его задача — показать, **где такое разделение действительно полезно, какую цену оно имеет и где перестаёт работать**.

# Архитектура

Общий цикл:

    Raw data / state
            |
            v
    Compact state representation
            |
       +----+----+
       |         |
       v         v
    Perception  Planner
                |
                v
         Legal candidates
                |
                v
         Tiny scorer / policy
                |
                v
         Decision + confidence
             /       \
        confident   uncertain
           |           |
           v           v
        execute     fallback
             \       /
              \     /
                v
          reward / metrics

Ключевая граница:

**planner генерирует legal candidates; tiny model выбирает среди них.**

# Экспериментальные треки

| Трек | Вопрос | Пример |
|---|---|---|
| Perception | Может ли tiny model извлечь полезное состояние? | HARTH |
| Decision | Может ли выбрать среди явных альтернатив? | J03 |
| Sequential Decision | Даёт ли повторяющийся выбор полезное поведение? | J04 Tetris |
| Control | Может ли tiny policy стабилизировать среду? | J01 CartPole |
| Hierarchical Decision | Как tiny models встраиваются в большие системы? | будущий трек |

Цель — не победить один dataset, а понять, **в каких классах задач tiny decision models полезны и какая архитектура системы делает их полезными**.

# Реализованные задачи

## J01 — CartPole

Первый control vertical slice.

Есть:

- deterministic rule baseline;
- tiny fixed-weight MLP size probe;
- action latency;
- confidence;
- abstention/fallback instrumentation;
- parameter и FP32-size accounting.

J01 — базовый systems/measurement experiment. Tiny MLP здесь является fixed-weight inference-size probe, а не обученным claim о качестве на CartPole.

    jev-bench run --task j01_cartpole --policy rule --episodes 20 --seed 0
    jev-bench sweep --task j01_cartpole --hidden-units 1,2,4,8,16,32,64 --episodes 20 --seed 0

## J02 — HARTH perception

Первый real-world sensor/perception task.

HARTH преобразуется в фиксированные окна и оценивается через subject-disjoint LOSO.

По умолчанию:

- 6 acceleration channels;
- 128 samples;
- stride 128;
- только pure-label windows;
- subject identity сохраняется;
- train/test разделены по subjects.

Реализованы:

- streaming CSV loader;
- schema validation;
- manifest;
- window inspection;
- Tiny MLP;
- nearest-centroid baseline;
- LOSO;
- multi-seed LOSO;
- risk-coverage;
- hidden-size sweep;
- Pareto analysis;
- single-window inference latency.

Критическое правило:

**HARTH performance numbers не придумываются.**

Synthetic CSV используются только для тестирования loader/contracts. Публикуемый результат должен быть получен реальным runner на зафиксированном dataset.

    jev-bench harth-manifest --dataset-root /path/to/harth
    jev-bench harth-inspect --dataset-root /path/to/harth --subject S015

    jev-bench harth-loso \
      --dataset-root /path/to/harth \
      --model tiny_mlp \
      --hidden-units 8 \
      --output results/j02-loso-h8.json

    jev-bench harth-loso-multi-seed \
      --dataset-root /path/to/harth \
      --model tiny_mlp \
      --hidden-units 8 \
      --seeds 0,1,2,3,4 \
      --output results/j02-loso-multi-seed-h8.json

Подробнее: docs/experiment-j02-harth.md.

## J03 — Candidate Selection

J03 вводит центральный паттерн:

**planner → tiny selector → reward**

Planner предоставляет:

- compact context;
- конечный набор legal candidates;
- candidate features.

Tiny model ранжирует кандидатов и выбирает один.

    state
      |
    planner
      |
    legal candidates
      |
    tiny scorer
      |
    selected candidate
      |
    reward

J03 намеренно synthetic.

Каждая задача содержит:

- context размерности 6;
- 8 candidates;
- 6 candidate features;
- deterministic latent utility, известный evaluator, но не policy.

Метрики:

- selection accuracy;
- reward;
- oracle reward;
- regret;
- p95 regret;
- confidence;
- abstention;
- latency;
- parameter count;
- FP32 model bytes;
- permutation invariance.

Главный принцип: **accuracy не является единственной целью**.

    jev-bench j03-train --model tiny_mlp --hidden-units 8
    jev-bench j03-train --model linear
    jev-bench j03-sweep --hidden-units 1,2,4,8,16,32,64 --output results/j03-sweep.json

Подробнее: docs/experiment-j03-candidate-selection.md.

## J04 — Sequential Tetris Candidate Selection

J04 переносит candidate selection из synthetic tables в последовательную меняющуюся среду.

    board state + current piece
              |
        legal placements
              |
          tiny scorer
              |
           placement
              |
         board update
              |
            reward
              |
         next decision

Среда содержит:

- deterministic 10×20 board;
- семь tetromino types;
- collision checking;
- gravity/drop;
- line clearing;
- legal-placement generation;
- deterministic seeds.

Tiny model не придумывает произвольные координаты. Среда сначала строит legal placements, затем валидирует выбранный placement.

Метрики:

- game return;
- lines cleared;
- pieces survived;
- teacher agreement;
- teacher-relative regret;
- permutation invariance;
- legal-action rate;
- mean/p95 latency;
- parameter count;
- model size;
- confidence;
- coverage;
- fallback rate;
- ECE.

При низкой confidence tiny model может abstain, после чего выбор передаётся heuristic teacher.

    jev-bench j04-run --policy heuristic --episodes 20
    jev-bench j04-run --policy random --episodes 20

    jev-bench j04-train --model tiny_mlp --hidden-units 8

    jev-bench j04-sweep \
      --hidden-units 1,2,4,8,16,32,64 \
      --output results/j04-sweep.json

    jev-bench j04-risk-coverage \
      --model tiny_mlp \
      --hidden-units 8 \
      --output results/j04-risk-coverage.json

Подробнее: docs/experiment-j04-tetris.md.

> J04 — компактная исследовательская среда, а не доказательство качества на сторонней реализации Tetris. Следующий шаг — сохранить тот же candidate-selection contract и перейти к established benchmark или более богатому simulator.

## J05 — JEV Reward RL

J05 добавляет второй способ использования JEV: **не как selector действия, а как judge/reward provider, который формирует learning signal для RL-агента**.

Цикл benchmark:

    RL agent
        |
        v
      action
        |
        v
    environment transition
        |
        v
      JEV judge
        |
        v
      reward
        |
        v
     RL update

В этом эксперименте JEV **не выбирает действие**. Один и тот же learner и одна и та же среда сравниваются при разных reward providers.

### Что уже реализовано

J05 сейчас содержит:

- deterministic Key Quest environment;
- native environment reward;
- hand-written rule reward;
- OpenRouter-backed JEV-style probabilistic reward provider;
- persistent cache для reward judgments;
- judge evaluation CLI;
- tabular Q-learning runner;
- deterministic и cache/regression tests;
- документацию J05 по implementation и evaluation protocol.

JEV adapter использует `OPENROUTER_API_KEY` и явно заданную модель.

### Научное разделение

J05 разделяет три разных вопроса:

1. **Judge quality** — насколько полезно JEV оценивает held-out transitions?
2. **Learning quality** — способен ли один и тот же RL learner обучаться на этих rewards?
3. **Systems cost** — сколько требуется model calls, cache hits, времени и tokens/dollars?

Эти измерения не сворачиваются в один общий score.

### Следующие необходимые controls

Реализация расширяется в сторону:

- фиксированных train/holdout transition splits;
- независимых versioned transition labels;
- confidence-based abstention/escalation;
- adversarial reward-hacking cases;
- representation-robustness tests;
- multi-seed aggregation;
- проверки live-vs-cache equivalence;
- cost accounting в общем result schema.

Таким образом, J05 — это отдельный **reward-learning benchmark layer**: он проверяет не только способность LLM/JEV выдавать правдоподобное суждение, но и полезность этого learning signal для последующего обучения.

Подробнее: `docs/experiment-j05-jev-rl.md` и `docs/experiment-j05-implementation.md`.

# Что измеряем

Benchmark объединяет model-centric и system-centric metrics.

### Model

- parameter count;
- FP32 parameter bytes;
- serialized model size;
- hidden width.

### Inference

- single-decision latency;
- batch latency;
- p95 latency.

Для JEV особенно важна latency одного решения.

### Decision quality

В зависимости от задачи:

- accuracy;
- macro-F1;
- reward;
- oracle reward;
- regret;
- game return;
- survival;
- lines cleared;
- teacher agreement;
- legal-action rate.

### Confidence / selective prediction

- confidence;
- abstention rate;
- coverage;
- fallback rate;
- risk-coverage;
- calibration diagnostics.

Confidence нельзя автоматически считать calibrated probability.

### Robustness

- candidate permutation invariance;
- subject-disjoint evaluation;
- repeated seeds;
- deterministic seeds;
- explicit schema/class validation.

# Pareto analysis

В проекте нет предположения, что существует один «лучший» размер модели.

Capacity sweep строит Pareto frontier между:

- task performance;
- inference latency;
- model size.

Это позволяет искать operating points и performance plateaus.

# Confidence + fallback

Одна из центральных гипотез:

**tiny model не обязана быть правильной в 100% случаев, чтобы быть полезной.**

    confidence
         |
    +----+----+
    |         |
  above τ   below τ
    |         |
    v         v
 tiny-model  abstain
 action        |
                v
          fallback policy

Fallback может быть:

- deterministic rule;
- teacher;
- larger model;
- другая policy;
- в будущем человек.

Так появляется измеримый trade-off между coverage, accepted-decision risk, fallback rate, downstream reward и compute/latency.

# Воспроизводимость

Публикуемый результат должен фиксировать:

- task;
- protocol;
- dataset/version;
- exact train/test или held-out subject list;
- random seeds;
- windowing/sampling parameters;
- model architecture;
- hidden size;
- training hyperparameters;
- confidence threshold;
- result schema version.

Repeated experiments сохраняют отдельные runs.

## Versioned result artifacts

Используются:

- jev-benchmark.result/v1;
- jev-benchmark.sweep/v1.

Старый raw JSON можно нормализовать:

    jev-bench normalize-result --input old-result.json --output result-v1.json

Подробнее: docs/result-schema-v1.md.

# Data provenance

Datasets внешние по отношению к source repository, если явно не указано обратное.

Для HARTH repository предоставляет loaders, validation и download/materialization helpers, но не хранит полный dataset.

Главный вопрос provenance:

> **Какие именно данные породили это число?**

HARTH manifests могут фиксировать subject files, source information и SHA-256.

# Что это за проект — и чем он не является

## Это

- research benchmark;
- executable experimental framework;
- общий contract для tiny policies/selectors;
- сравнение model size, latency и downstream utility;
- framework для confidence-aware fallback;
- последовательность усложняющихся задач.

## Это не

- leaderboard, доказывающий превосходство tiny models над большими;
- набор придуманных benchmark numbers;
- обычный classification benchmark;
- утверждение, что одна tiny architecture оптимальна всегда;
- замена established real-world benchmarks.

Цель — сделать вопрос **измеримым и воспроизводимым**.

# Текущее состояние

**v0.1 — первые vertical slices, теперь расширенные reward-learning слоем**

Уже реализованы в `main`:

- общие policy / decision contracts;
- versioned result schema;
- J01 CartPole;
- J02 HARTH pipeline и LOSO;
- J02 multi-seed analysis;
- J02 capacity / Pareto analysis;
- HARTH Hugging Face archive/materialization helpers и provenance manifests;
- J03 candidate selection;
- J04 sequential candidate selection;
- confidence / abstention / fallback instrumentation;
- permutation-invariance tests;
- reproducibility-oriented validation;
- **J05 JEV Reward RL** с native/rules/JEV reward providers;
- J05 persistent cache, judge evaluation и tabular Q-learning;
- CLI runners, документация и automated tests.

### Текущий статус интеграции

- **PR #5 открыт:** real HARTH validation и Windows runner для 22 локальных HARTH subject files пользователя.
- PR #5 — это именно **validation/smoke-run этап**, а не опубликованный HARTH benchmark result.
- Реализация J05 уже находится в `main`; полный held-out/adversarial/multi-seed reward-learning protocol — следующий исследовательский increment.

J02 код готов работать с реальными HARTH CSV, но **число считается benchmark result только после фактического запуска runner на зафиксированном dataset**.

# Roadmap

## Ближайшие шаги

1. Завершить PR с validation/smoke-run реального HARTH и зафиксировать provenance dataset.
2. Выполнить J02 multi-seed LOSO на зафиксированном наборе из 22 subjects.
3. Выполнить полный hidden-size sweep.
4. Построить size / latency / macro-F1 Pareto front.
5. Расширить selective-prediction analysis.
6. Сохранять результаты как versioned artifacts.
7. Завершить J05 held-out reward-fidelity evaluation.
8. Добавить J05 confidence/abstention, reward-hacking и representation-robustness controls.
9. Добавить J05 multi-seed и live-vs-cache/cost reports.

## Следующий уровень

Перенести candidate selection из synthetic J03 и компактного J04 в established environments:

- scheduling;
- routing;
- games;
- robotics simulators;
- resource allocation;
- browser/tool action selection.

Сохранить общий contract:

    state → legal candidates → tiny selector → execution → reward

## Более долгосрочные вопросы

- Где полезная нижняя граница размера модели?
- Насколько важна state compression?
- Легче ли сжимать candidate selection, чем direct policy learning?
- Насколько confidence + fallback компенсируют ограниченную capacity?
- Становится ли tiny model полезнее, если planner гарантирует legal actions?
- Какие invariances критичны?
- Где появляются performance plateaus?
- Каков выигрыш по compute/energy при одинаковом downstream utility?
- Какие классы задач действительно подходят tiny decision models?

# Установка

Требуется Python 3.10+.

    pip install -e '.[dev]'
    pytest
    jev-bench list-tasks

Эталонные реализации используют NumPy, чтобы экспериментальная логика оставалась прозрачной.

# Структура репозитория

    src/jev_bench/
    ├── core/          # contracts, candidates, versioned results
    ├── datasets/      # loaders, manifests, validation
    ├── policies/      # tiny models и baselines
    ├── envs/          # research environments
    ├── tasks/         # J01–J05
    └── cli.py         # CLI benchmark runner

    docs/
    ├── experiment-j02-harth.md
    ├── experiment-j03-candidate-selection.md
    ├── experiment-j04-tetris.md
    ├── experiment-j05-jev-rl.md
    ├── experiment-j05-implementation.md
    └── result-schema-v1.md

    tests/             # deterministic unit / contract tests

# Статус

Проект находится в активной стадии research development. Protocol и task suite будут уточняться по мере того, как эксперименты покажут, какие controls и metrics действительно необходимы.
