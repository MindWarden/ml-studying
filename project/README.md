# Telco Customer Churn — итоговый проект по курсу «Инженерия Искусственного Интеллекта»

Сервис прогноза оттока клиентов телеком-оператора на основе классических ML-моделей.

---

## 1. Паспорт проекта

- **Название проекта:** Telco Customer Churn — сервис прогноза оттока клиентов
- **Автор:** Власов Глеб Андреевич
- **Группа:** РКБО-02-23
- **Контакт:** glebvlasov.science@gmail.com

**Краткое описание:**
Проект решает задачу бинарной классификации: для клиента телеком-оператора по его профилю (тип контракта, услуги, платежи и т.д.) предсказывается вероятность ухода. Используется открытый датасет Telco Customer Churn (IBM, 7043 клиента). Сравниваются семь конфигураций: три базовые модели (LogisticRegression, RandomForest, GradientBoosting) и четыре улучшенные — те же модели после тюнинга `GridSearchCV` плюс нейросетевой бейзлайн MLP. Лучшая по CV ROC-AUC модель (tuned GradientBoosting) дополнена подбором порога классификации по F2 (recall 0.51 → 0.89) и обёрнута в FastAPI-сервис с эндпоинтами `/predict` и `/health`, упакована в Docker. Эксперименты логируются в MLflow.

---

## 2. Структура проекта

```
project/
├── README.md                  -- этот файл
├── report.md                  -- отчёт (постановка, данные, эксперименты, результаты)
├── self-checklist.md          -- чеклист самопроверки
├── requirements.txt           -- зависимости Python
├── Dockerfile                 -- сборка контейнера
├── .dockerignore
├── .gitignore
├── data/
│   └── raw/telco_churn.csv    -- открытый датасет
├── notebooks/
│   ├── 01_eda.ipynb           -- разведочный анализ данных
│   └── 02_models_comparison.ipynb -- сравнение моделей
├── src/
│   ├── data/                  -- загрузка и препроцессинг
│   │   ├── load.py
│   │   └── preprocess.py
│   ├── models/                -- обучение
│   │   └── train.py
│   └── service/               -- FastAPI сервис
│       ├── app.py
│       └── schemas.py
├── configs/
│   ├── config.yaml            -- параметры моделей, сетки тюнинга, β порога
│   └── .env.example           -- шаблон переменных окружения
├── tests/                     -- pytest-тесты
│   ├── test_data.py
│   └── test_service.py
├── mlruns/                    -- MLflow-трекинг экспериментов (не коммитится)
└── artifacts/                 -- сохранённая модель и метрики
    ├── model.pkl              (создаётся скриптом обучения)
    ├── threshold.json         (порог, подобранный по F2)
    ├── figures/               (графики для отчёта)
    └── metrics.json
```

---

## 3. Требования и установка

- **Python 3.13** (в Docker зафиксировано). Локально подойдёт 3.11+.
- Docker (для основного сценария запуска).

### Локально (опционально, без Docker)

```bash
cd project
python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash
# либо:  .venv\Scripts\activate   # Windows cmd
# либо:  source .venv/bin/activate  # Linux/macOS

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. Через Docker (рекомендуется)

```bash
cd project
docker build -t aie-churn .
docker run -p 8000:8000 aie-churn
```

Сервис поднимется на `http://localhost:8000`. Модель обучается автоматически на этапе `docker build` (~1–2 мин): так сериализованный Pipeline гарантированно совместим с версиями библиотек контейнера, а фиксированный seed делает результат детерминированным.

### 4.2. Локально без Docker

```bash
cd project

# 1) обучение: бейзлайны + GridSearchCV + MLP + подбор порога
#    (создаст artifacts/model.pkl, metrics.json, threshold.json и mlruns/)
python -m src.models.train

# 2) запуск сервиса
python -m uvicorn src.service.app:app --host 0.0.0.0 --port 8000

# (опционально) UI с экспериментами MLflow
python -m mlflow ui --backend-store-uri file:mlruns
```

### 4.3. Проверка работоспособности

```bash
# Health-check
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}

# Предсказание (пример "лояльного" клиента — длинный контракт, большой tenure)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 100.0, "TotalCharges": 6000.0
  }'
# {"churn_probability":0.049,"churn_class":"No","threshold":0.14}
```

Порог `0.14` подобран при обучении по F2 (см. `report.md` раздел 5.3) и читается сервисом из `artifacts/threshold.json`; переопределяется переменной окружения `CHURN_THRESHOLD`.

Также доступны:
- **Swagger UI:** http://localhost:8000/docs
- **OpenAPI JSON:** http://localhost:8000/openapi.json

### 4.4. Эндпоинты

| Метод | Путь        | Описание                                                            |
|-------|-------------|---------------------------------------------------------------------|
| GET   | `/`         | Информация о сервисе и доступных эндпоинтах                         |
| GET   | `/health`   | Health-check, возвращает статус и факт загрузки модели              |
| POST  | `/predict`  | Принимает JSON с признаками клиента, возвращает вероятность оттока  |
| GET   | `/docs`     | Swagger UI (автоматический)                                         |

---

## 5. Данные

- **Источник:** [Telco Customer Churn (IBM)](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) — открытый датасет.
- **Размер:** 7043 строки, 21 колонка.
- **Целевая переменная:** `Churn` (Yes/No → 1/0). Доля положительного класса — ~26.5%.
- **Файл:** `data/raw/telco_churn.csv` (~950 КБ, лежит в репозитории).
- **Очистка:** `TotalCharges` приводится к `float` (для клиентов с `tenure=0` там пробелы — заменяются на 0); `customerID` удаляется.

---

## 6. Тесты

```bash
cd project
python -m pytest tests/ -v
```

Покрывают (10 тестов):
- загрузку и очистку данных, отсутствие NaN, корректность колонок;
- препроцессор (ColumnTransformer — числовые + категориальные);
- стратифицированный train/test-split;
- сервис: `/health`, `/predict` (валидный и невалидный ввод), сравнение лояльного и рискового клиента (у второго вероятность ухода должна быть выше);
- согласованность порога: сервис использует именно тот порог, что подобран при обучении (`artifacts/threshold.json`).

---

## 7. Демонстрация на защите

1. Показ структуры репозитория (`src/`, `notebooks/`, `data/`, `artifacts/`, `Dockerfile`).
2. **Запуск через Docker:**
   ```bash
   docker build -t aie-churn .
   docker run -p 8000:8000 aie-churn
   ```
3. Демонстрация Swagger UI на `http://localhost:8000/docs`.
4. Два запроса к `/predict` для контраста:
   - «лояльный» клиент (Two year контракт, большой tenure) → вероятность ухода ~5% → «No»;
   - «рисковый» клиент (Month-to-month, tenure=1, Electronic check) → вероятность ухода ~81% → «Yes».
5. Открытие `notebooks/02_models_comparison.ipynb`: сравнительная таблица семи конфигураций, ROC-кривые, график подбора порога, confusion matrix до/после подбора порога. Опционально — `mlflow ui` с историей экспериментов.
6. `report.md`: разделы 5.3–5.4 «Подбор порога» и «Выбор финальной модели» (tuned GradientBoosting + порог 0.14, recall 0.51 → 0.89).

---

## 8. Ограничения и дальнейшая работа

**Текущие ограничения:**
- модель обучена на одном статичном датасете 2017 года — концепт-дрифт не отслеживается;
- нет аутентификации сервиса (по требованиям курса не нужна, но в проде потребуется);
- MLflow работает в локальном файловом режиме — для командной работы нужен общий tracking-сервер.

**Возможные улучшения:**
- калибровка вероятностей (Platt/isotonic) для честной интерпретации порога;
- эксперименты с CatBoost/XGBoost и SHAP-интерпретацией;
- мониторинг дрифта данных (PSI) и метрики Prometheus в сервисе;
- периодическое переобучение по расписанию с автоматическим сравнением с прод-моделью.

---

## 9. Оценка проекта

См. `self-checklist.md` для отметок по 10 критериям и `INFO/project-evaluation.md` для шкалы (целевая оценка — 5).
