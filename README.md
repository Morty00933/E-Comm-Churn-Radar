#  E-Comm Churn Radar

Предсказание оттока пользователей e-commerce на основе событий (просмотры, клики, покупки).  
Полный цикл: **сырые логи → фичи → обучение → метрики → интерпретация → дашборд и inference**.

>  Большие сырые CSV **не хранятся в репозитории** и добавлены в `.gitignore`.

---

##  Что умеет проект

- Агрегирует помесячные event-логи в пользовательские фичи (CLICKS, PURCHASES, вовлечённость и пр.).
- Обучает несколько моделей (LightGBM/XGBoost/LogReg) и выбирает лучшую.
- Считает **ROC-AUC, PR-AUC, F1**, подбирает оптимальный **threshold**.
- Строит объяснимость: **Feature Importances** + **SHAP**.
- Дашборд (Streamlit): кривые ROC/PR, важности, SHAP, поиск по `user_id`.
- Скрипт для онлайнового предсказания по пользователю.

---

##  Структура репозитория

```
├── configs/
│   └── config.yaml                  # Конфигурация путей/параметров
├── data/                            # Сырые/обработанные данные (в .gitignore)
│   ├── processed_train.csv          # (создаётся скриптами)
│   └── processed_test.csv           # (создаётся скриптами)
├── models/                          # Артефакты обучения (в .gitignore)
│   ├── model.pkl
│   ├── metrics.json
│   ├── feature_importances.csv
│   └── feature_columns.json
├── src/
│   ├── common/
│   │   └── utils.py
│   ├── data/
│   │   └── make_dataset.py          # Подготовка/семплинг данных (опц. demo)
│   ├── features/
│   │   └── build_features.py        # Агрегация event-логов в фичи
│   ├── models/
│   │   ├── train_model.py           # Обучение + сохранение артефактов
│   │   └── predict_model.py         # Предсказание по user_id / CSV
│   └── visualization/
│       └── dashboard.py             # Streamlit-дашборд
└── README.md
```

---

##  Установка

```powershell
python -m venv .venv
.\.venv\Scripts\ctivate
pip install -r requirements.txt
```

---

## Данные

- Локально положи помесячные CSV с событиями в `data/`.
- Имена могут быть любыми — пути задаются в `configs/config.yaml`.
- Если больших CSV нет, можно собрать **демо-набор** (см. ниже).

---

## Быстрый старт

### 1) (опционально) Сгенерировать демо-данные
```powershell
python -m src.data.make_dataset --config configs/config.yaml --demo
```

### 2) Обучение
```powershell
python -m src.models.train_model --config configs/config.yaml
```

### 3) Дашборд
```powershell
streamlit run src/visualization/dashboard.py
```

### 4) Предсказание по пользователю
```powershell
python -m src.models.predict_model --user-id 123456 --config configs/config.yaml
```

---

##  Метрики/артефакты

- `models/metrics.json` — итоговые метрики (ROC-AUC, PR-AUC, F1, threshold).
- `models/feature_importances.csv` — важность фич (LightGBM/XGB).
- `models/feature_columns.json` — **порядок признаков**, который использует дашборд/inference.
- `models/model.pkl` — пайплайн `preprocess + model`.

---

##  Интерпретация

- **Feature Importances** — глобальная важность признаков.
- **SHAP** — вклад признаков в вероятность ухода как глобально (summary), так и локально (по `user_id`).

---

##  .gitignore (важно)

В репозитории исключены:
- `data/*.csv`, `data/*.parquet`, `models/*.pkl` — тяжёлые артефакты,
- `.venv/`, `__pycache__/`, IDE-мусор,
- кэш Streamlit.  
Папки `data/` и `models/` оставлены через `.gitkeep`.
