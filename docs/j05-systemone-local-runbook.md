
# J05 + Jev System One — локальный запуск

Пошаговый runbook для текущего main репозитория designfordrink/jev-benchmark.

Цель первого запуска:

    Key Quest
      -> observable transition
      -> Jev System One
      -> choice + probabilities
      -> reward
      -> RL update

**Критическое требование:** hidden event не передаётся Jev. Jev получает полный observable context: environment geometry + state/action/next_state/terminated/step.

## 1. Требования

Нужно:

- Windows + PowerShell
- Python 3.10+
- Git
- OpenRouter API key

Проверка:

    python --version
    git --version

## 2. Получить актуальный код

Если репозитория ещё нет:

    git clone https://github.com/designfordrink/jev-benchmark.git
    cd jev-benchmark

Если уже есть:

    cd путь\к\jev-benchmark
    git checkout main
    git pull

Проверьте:

    git status

Не удаляйте незакоммиченные изменения через reset --hard.

## 3. Создать Python environment

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Если PowerShell блокирует скрипт:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    .\.venv\Scripts\Activate.ps1

Установить benchmark:

    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

Проверить CLI:

    jev-bench --help
    jev-bench list-tasks

## 4. Сначала запустить локальные тесты

До обращения к OpenRouter:

    pytest

Если локальные тесты не проходят, сначала исправьте это. Live JEV запуск на сломанном checkout неинформативен.

## 5. Создать локальный .env

В репозитории уже есть helper:

    .\scripts\setup-openrouter.ps1

Он попросит OpenRouter key скрытым вводом и создаст gitignored .env.

Проверьте только факт игнорирования:

    git status --short
    git check-ignore .env

**Никогда не выводите API key через echo и не отправляйте его в чат.**

## 6. Указать именно System One

После helper в .env может стоять:

    J05_JEV_MODEL=jev-latest

Для этого эксперимента задайте:

    J05_JEV_MODEL=typesafe/jev-1.13
    J05_JEV_PROVIDER=jev_systemone

Итог:

    OPENROUTER_API_KEY=ВАШ_КЛЮЧ
    J05_JEV_MODEL=typesafe/jev-1.13
    J05_JEV_PROVIDER=jev_systemone

Реальный key остаётся только локально.

## 7. Первый live-тест: только judge

Сначала не запускайте RL.

    jev-bench j05-judge `
      --provider jev_systemone `
      --model typesafe/jev-1.13 `
      --cache .cache/j05-jev-systemone.json

Эта команда проверяет:

1. J05 transition создаётся;
2. observable state формируется;
3. System One вызывается;
4. typed choice возвращается;
5. probabilities разбираются;
6. reward вычисляется;
7. результат попадает в cache.

Если команда успешно завершилась, базовый System One adapter работает.

## 8. Что System One должен получить

Observable input:

    environment:
      width, height, coordinate system
      walls, hazards, key_location, exit_location, max_steps
    transition:
      state: position, has_key, step
      action: id, name, delta
      next_state: position, has_key, step
      terminated

Choice criteria:

    lava
    timeout
    wall
    boundary
    move
    key
    exit

**Hidden event, reference reward и termination reason не должны присутствовать в prompt/request.**

Например, нельзя передавать:

    event = exit
    reference_reward = 1.0

Иначе benchmark фактически сообщает Jev правильный ответ.

## 9. Проверить cache

После первого вызова:

    Get-Item .cache\j05-jev-systemone.json

Cache нужен для:

- повторяемости;
- повторного анализа;
- сокращения повторных live calls;
- сохранения результатов judge.

Cache hit нельзя интерпретировать как доказательство нулевой стоимости live эксперимента.

Для отдельного нового live набора можно использовать другой файл:

    --cache .cache\j05-jev-systemone-fresh.json

## 10. RL smoke test

Теперь запускаем полный цикл, но всего на 10 эпизодах:

    jev-bench j05-run `
      --provider jev_systemone `
      --model typesafe/jev-1.13 `
      --cache .cache/j05-jev-systemone.json `
      --episodes 10 `
      --seed 0

Это **техническая проверка**, а не научный результат.

Проверяем:

- нет runtime error;
- reward возвращается;
- RL loop завершается;
- cache используется;
- результат сериализуется.

## 11. Один нормальный seed

После успешного smoke test:

    New-Item -ItemType Directory -Force -Path results\j05 | Out-Null

    jev-bench j05-run `
      --provider jev_systemone `
      --model typesafe/jev-1.13 `
      --cache .cache/j05-jev-systemone.json `
      --episodes 100 `
      --seed 0 |
      Tee-Object -FilePath results\j05\systemone-seed0.json

Не делайте вывод о качестве JEV только по одному seed.

## 12. Основной J05 protocol v1

После успешного single-seed:

    New-Item -ItemType Directory -Force -Path results\j05 | Out-Null

    jev-bench j05-protocol `
      --provider jev_systemone `
      --model typesafe/jev-1.13 `
      --cache .cache/j05-jev-systemone.json `
      --seeds 0,1,2,3,4 `
      --episodes 100 `
      --labels-output results\j05\labels-v1.json |
      Tee-Object -FilePath results\j05\protocol-v1.json

Это основной воспроизводимый запуск текущего J05 protocol v1.

## 13. Что проверяет protocol

### 13.1 Judge quality

На held-out transitions:

- reward MAE;
- maximum absolute error;
- exact reward rate.

### 13.2 Adversarial cases

Проверяется, не использует ли judge простой state-only shortcut вместо оценки transition.

### 13.3 Representation consistency

Одинаковое содержание должно приводить к согласованному решению.

Текущая версия protocol содержит инфраструктуру этой проверки; controlled paraphrase / field-order perturbations — следующий methodological increment.

### 13.4 Multi-seed learning

По умолчанию:

    seeds: 0,1,2,3,4
    episodes: 100

Нельзя выбирать лучший seed.

Смотрим:

- success rate;
- mean learned return;
- mean/std по seeds.

### 13.5 Systems cost

Для live provider сохраняются доступные сведения о:

- model;
- live calls;
- cache hits;
- latency;
- token/cost information, если provider её возвращает.

## 14. Результаты

После protocol ожидаются:

    results/
    └── j05/
        ├── labels-v1.json
        └── protocol-v1.json

Cache:

    .cache/
    └── j05-jev-systemone.json

Посмотреть JSON:

    Get-Content results\j05\protocol-v1.json

Или:

    Get-Content results\j05\protocol-v1.json |
      ConvertFrom-Json |
      ConvertTo-Json -Depth 20

## 15. Автоматический runner

Можно выполнить тот же five-seed protocol через:

    .\scripts\run-j05-openrouter.ps1

Явно указать модель:

    .\scripts\run-j05-openrouter.ps1 `
      -Model "typesafe/jev-1.13"

Runner:

1. загружает .env;
2. проверяет key;
3. выбирает provider;
4. создаёт results/j05;
5. запускает five-seed protocol;
6. пишет protocol-v1.json.

## 16. Рекомендуемый порядок — без пропусков

    1. git checkout main + git pull
    2. python -m pip install -e ".[dev]"
    3. pytest
    4. setup-openrouter.ps1
    5. .env: J05_JEV_PROVIDER=jev_systemone
    6. .env: J05_JEV_MODEL=typesafe/jev-1.13
    7. j05-judge
    8. j05-run --episodes 10
    9. j05-run --episodes 100 --seed 0
    10. j05-protocol --seeds 0,1,2,3,4

Не начинайте с пункта 10.

## 17. Частые ошибки

### OPENROUTER_API_KEY is missing

Проверьте:

    Test-Path .env
    git check-ignore .env

И повторите:

    .\scripts\run-j05-openrouter.ps1

### Authentication / 401

Проверьте локальный key и его права в OpenRouter. Не отправляйте key для диагностики.

### Использован не тот provider

Для этого эксперимента должно быть:

    --provider jev_systemone

Не:

    --provider jev

Это другой adapter и другой baseline.

### Результат появился слишком быстро

Возможно, сработал cache. Проверьте:

    Get-Item .cache\j05-jev-systemone.json

## 18. Что прислать после первого запуска

API key присылать не нужно.

Для диагностики достаточно:

1. команду;
2. stdout/stderr, если есть ошибка;
3. JSON от j05-judge;
4. JSON smoke test;
5. после полного запуска — results/j05/protocol-v1.json.

Если первый j05-judge успешен, инфраструктурная часть пройдена и можно переходить к анализу protocol.

## 19. Критерий технического успеха

Первый этап считается пройденным, если:

- [ ] pytest проходит;
- [ ] .env не tracked Git;
- [ ] j05-judge --provider jev_systemone проходит;
- [ ] создаётся cache;
- [ ] 10-episode smoke test проходит;
- [ ] 100-episode single seed проходит;
- [ ] five-seed protocol проходит;
- [ ] создаются labels-v1.json и protocol-v1.json.

## 20. Что именно исследует J05

J05 не должен заранее доказывать превосходство Jev.

Вопрос эксперимента:

> Может ли Jev System One, получая только наблюдаемый transition и не видя правильный event, выдавать reward signal, с которым RL обучается с приемлемым качеством и systems cost?

Контрольные условия:

    Native reward
    Rules reward

Экспериментальное условие:

    Jev System One reward

Основная ось анализа:

    judge quality
          ↓
    learning quality
          ↓
    systems cost

Не сводите эти три измерения к одному aggregate score.
