# `README.md`

```markdown
# 🛒 Churn Prediction for E-Commerce

End-to-end ML-проект: предсказание оттока пользователей интернет-магазина на основе сырых event-логов (клики, покупки, просмотры).  
Система включает полный цикл: данные → фичи → модель → метрики → интерпретация → дашборд.

---

## 📌 Описание проекта

**Цель:** предсказать, какие пользователи интернет-магазина с высокой вероятностью перестанут возвращаться (churn).  
**Данные:** логи активности пользователей (клики, покупки, события в сессиях) за несколько месяцев (2019-Oct – 2020-Feb).  
**Выход:** интерактивный Streamlit-дашборд для анализа метрик модели и получения прогноза по любому `user_id`.

---

## 🚀 Функционал

- 📂 **Обработка сырых данных**: агрегация событий (click, purchase, session) в пользовательские признаки.
- 🏗 **Feature Engineering**:
  - активность (clicks, purchases),
  - вовлечённость (avg_session_time, days_since_last_visit),
  - ценность (avg_order_value, conversion_rate, high_value_user),
  - derived-features (order_value_per_time, engagement_score и т. д.).
- 📊 **Модели**:
  - LightGBM (основная),
  - XGBoost,
  - Logistic Regression (базовый бейзлайн).
- 🔍 **Метрики**:
  - ROC-AUC, PR-AUC, F1-score,
  - автоматический подбор оптимального порога.
- 📈 **Интерпретация**:
  - Feature Importances,
  - SHAP-графики (что влияет на отток).
- 🖥 **Dashboard (Streamlit)**:
  - ROC и PR кривые,
  - таблица важности признаков,
  - объяснения SHAP,
  - поиск по `user_id` → вероятность churn,
  - выгрузка списка пользователей с наибольшим риском оттока.

---

## 📂 Структура проекта

```

  ├── configs/
  │   └── config.yaml          # Конфигурация (пути к данным, параметры моделей)
  ├── data/
  │   ├── 2019-Oct.csv         # Сырые данные
  │   ├── 2019-Nov.csv
  │   ├── 2019-Dec.csv
  │   ├── 2020-Jan.csv
  │   └── 2020-Feb.csv
  ├── models/
  │   ├── model.pkl            # Сохранённая модель
  │   ├── metrics.json         # Метрики качества
  │   ├── feature\_importances.csv
  │   └── test\_predictions.csv
  ├── src/
  │   ├── features/
  │   │   └── build\_features.py
  │   ├── models/
  │   │   └── train\_model.py   # Основной скрипт обучения
  │   └── visualization/
  │       └── dashboard.py     # Streamlit дашборд
  └── README.md

````

---

## ⚙️ Установка и запуск

### 1. Клонировать репозиторий и перейти в папку
```bash
git clone https://github.com/username/churn-prediction.git
cd churn-prediction
````

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Обучить модель

```bash
python -m src.models.train_model --config configs/config.yaml
```

После обучения в папке `models/` появятся:

* `model.pkl` — сохранённая модель,
* `metrics.json` — метрики качества,
* `feature_importances.csv` — важность признаков,
* `test_predictions.csv` — предсказания на тесте.

### 4. Запустить дашборд

```bash
streamlit run src/visualization/dashboard.py
```

---

## 📊 Пример метрик

```json
{
  "roc_auc": 0.88,
  "pr_auc": 0.78,
  "f1": 0.84,
  "best_threshold": 0.85,
  "n_test": 391055,
  "labels_source": "synthetic_inactive_days>30"
}
```

* **ROC-AUC = 0.88** → модель хорошо различает churn / non-churn.
* **PR-AUC = 0.78** → высокое качество даже при дисбалансе классов.
* **F1 = 0.84** → практическая точность предсказаний.

---

## 🛠 Стек технологий

* **Python 3.10+**
* **pandas, numpy, scikit-learn** — обработка данных
* **LightGBM, XGBoost** — ML-модели
* **SHAP** — интерпретация
* **Streamlit, matplotlib, seaborn** — визуализация
* **PyYAML** — управление конфигурацией
