# J02 HARTH — локальный запуск на Windows

## Цель

Это пошаговый runbook для первого реального прогона J02 на локальной машине.

J02 использует реальный датасет HARTH и проверяет маленькую обучаемую MLP-модель на классификации окон акселерометра при Leave-One-Subject-Out (LOSO).

Важно:

- HARTH CSV не хранятся в Git;
- реальные CSV должны находиться локально;
- benchmark result должен быть получен фактическим запуском;
- перед публикацией результата нужно сохранить версию датасета, subjects, seeds и параметры обучения.

## 1. Подготовка

Нужны Windows, PowerShell, Python 3.10+, Git и локальная копия репозитория.

Ожидаемая структура:

~~~text
D:\datasets\harth\
  S006.csv
  S008.csv
  S009.csv
  ...
  S029.csv
~~~

Не объединяйте CSV субъектов в один файл.

## 2. Перейти на ветку J02 validation

Из каталога репозитория:

~~~powershell
cd D:\DeepSeek\jev-benchmark
git status
git fetch origin
git checkout feat/j02-harth-validation
git pull origin feat/j02-harth-validation
~~~

Если эта ветка уже используется для другой работы, лучше сделать отдельную рабочую копию репозитория.

## 3. Установить benchmark

~~~powershell
python -m pip install -e ".[dev]"
jev-bench --help
jev-bench list-tasks
pytest
~~~

Если pytest не проходит, не переходите к benchmark: сначала исправьте проблему окружения или кода.

## 4. Указать расположение HARTH

Например:

~~~powershell
$DATASET_ROOT = "D:\datasets\harth"
Get-ChildItem $DATASET_ROOT -Filter "*.csv" | Select-Object Name, Length
~~~

Для текущего HARTH ожидается 22 subject-файла.

## 5. Проверить датасет

Сначала ничего не обучаем:

~~~powershell
jev-bench harth-validate --dataset-root $DATASET_ROOT --output results/j02/harth-validation.json
~~~

Если здесь ошибка — не запускайте LOSO.

## 6. Проверить реальные окна

Например для S006:

~~~powershell
jev-bench harth-inspect --dataset-root $DATASET_ROOT --subject S006 --max-windows 1000
~~~

По умолчанию:

- window size = 128 samples;
- stride = 128 samples;
- 6 акселерометрических каналов;
- при 50 Hz одно окно = примерно 2.56 секунды.

## 7. Проверить manifest

~~~powershell
jev-bench harth-manifest --dataset-root $DATASET_ROOT
~~~

## 8. Сделать маленький smoke test

Не начинайте сразу с полного 22-subject запуска.

~~~powershell
jev-bench harth-loso --dataset-root $DATASET_ROOT --model tiny_mlp --hidden-units 8 --subjects S006 --max-train-windows-per-subject 100 --max-test-windows 500 --epochs 2 --seed 0 --output results/j02/j02-smoke-s006.json
~~~

Это НЕ научный результат. Это проверка полного pipeline:

~~~text
CSV
 ↓
windowing
 ↓
train/test split
 ↓
TinyMLP
 ↓
evaluation
 ↓
JSON result
~~~

Если smoke test падает, остановитесь здесь и разберите ошибку.

## 9. Один полный LOSO run

После успешного smoke test:

~~~powershell
jev-bench harth-loso --dataset-root $DATASET_ROOT --model tiny_mlp --hidden-units 8 --max-train-windows-per-subject 500 --max-test-windows 2000 --epochs 10 --lr 0.01 --batch-size 128 --seed 0 --output results/j02/j02-loso-h8-seed0.json
~~~

Benchmark делает отдельный тест на каждом subject:

~~~text
S006 → TEST, остальные → TRAIN
S008 → TEST, остальные → TRAIN
...
S029 → TEST, остальные → TRAIN
~~~

Так проверяется обобщение на человека, которого модель не видела при обучении.

## 10. Основной multi-seed запуск

Один seed недостаточен для устойчивого вывода. После успешного single-seed run:

~~~powershell
jev-bench harth-loso-multi-seed --dataset-root $DATASET_ROOT --model tiny_mlp --hidden-units 8 --seeds 0,1,2,3,4 --max-train-windows-per-subject 500 --max-test-windows 2000 --epochs 10 --lr 0.01 --batch-size 128 --output results/j02/j02-loso-multi-seed-h8.json
~~~

Отчёт позволяет отдельно видеть variation между seeds и variation между held-out subjects.

## 11. Sweep размера модели

После базового результата:

~~~powershell
jev-bench harth-loso-sweep --dataset-root $DATASET_ROOT --hidden-units 1,2,4,8,16,32,64 --max-train-windows-per-subject 500 --max-test-windows 2000 --epochs 10 --lr 0.01 --batch-size 128 --seed 0 --output results/j02/j02-loso-size-sweep.json
~~~

Цель sweep — исследовать trade-off между размером модели, latency и macro-F1, а не просто найти максимальную accuracy.

## 12. Какие файлы должны появиться

После основных запусков:

~~~text
results/
└── j02/
    ├── harth-validation.json
    ├── j02-smoke-s006.json
    ├── j02-loso-h8-seed0.json
    ├── j02-loso-multi-seed-h8.json
    └── j02-loso-size-sweep.json
~~~

HARTH CSV в Git не добавляйте.

## 13. Что является результатом

Не являются финальным benchmark result:

- pytest;
- synthetic test fixtures;
- harth-validate;
- harth-inspect;
- smoke test.

Настоящий benchmark run:

~~~text
harth-loso
~~~

Предпочтительный стабильный эксперимент:

~~~text
harth-loso-multi-seed
~~~

Исследование зависимости от размера модели:

~~~text
harth-loso-sweep
~~~

## 14. Минимальный маршрут первого запуска

Если нужно просто пройти правильную последовательность:

~~~powershell
cd D:\DeepSeek\jev-benchmark
git fetch origin
git checkout feat/j02-harth-validation
git pull origin feat/j02-harth-validation
python -m pip install -e ".[dev]"
pytest
$DATASET_ROOT = "D:\datasets\harth"
jev-bench harth-validate --dataset-root $DATASET_ROOT --output results/j02/harth-validation.json
jev-bench harth-inspect --dataset-root $DATASET_ROOT --subject S006 --max-windows 1000
jev-bench harth-loso --dataset-root $DATASET_ROOT --model tiny_mlp --hidden-units 8 --subjects S006 --max-train-windows-per-subject 100 --max-test-windows 500 --epochs 2 --seed 0 --output results/j02/j02-smoke-s006.json
~~~

На этом месте остановитесь.

## 15. Что прислать после smoke test

Не присылайте HARTH CSV.

Пришлите:

1. вывод harth-validate;
2. вывод harth-inspect;
3. вывод smoke-test команды;
4. содержимое results/j02/j02-smoke-s006.json.

По этому результату можно проверить pipeline и решить, запускать ли полный 22-subject multi-seed benchmark.

## 16. Что сейчас не нужно

Для J02 не нужны:

- OpenRouter;
- API keys;
- JEV API;
- GitHub Secrets;
- J05;
- загрузка HARTH в GitHub.

J02 — локальный эксперимент над реальными сенсорными данными.
