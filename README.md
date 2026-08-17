# Dry Bean Classification — ML Assignment 2

Multi-class classification of dry bean varieties using five machine-learning models, with an interactive Streamlit web app for live evaluation.

**Live app:** https://drybean-ml-classifier-sarasan-btappws67sq7wuamdgdfcl6.streamlit.app/
**Repository:** https://github.com/saravananarasan-bits-aiml/drybean-ml-classifier-sarasan

> **How to evaluate the app:** download `test_data.csv` from this GitHub repository, open the live app link above, and upload that file in the app. The app will then show the evaluation metrics and confusion matrix for the model you select from the dropdown.

---

## a. Problem statement

Given 16 numeric shape and geometry features extracted from images of dry beans, classify each bean into one of **7 varieties** (DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY). This is a supervised, multi-class classification problem. Five classification models are trained and compared on six evaluation metrics to determine which performs best on this dataset.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — Dry Bean Dataset (ID 602).
- **Instances:** 13,611
- **Features:** 16 (all numeric — area, perimeter, axis lengths, eccentricity, roundness, compactness, shape factors, etc.)
- **Target:** `Class` — 7 bean varieties.
- **Missing values:** 0.
- **Class distribution (imbalanced):**

  | Class | Count |
  |---|---|
  | DERMASON | 3,546 |
  | SIRA | 2,636 |
  | SEKER | 2,027 |
  | HOROZ | 1,928 |
  | CALI | 1,630 |
  | BARBUNYA | 1,322 |
  | BOMBAY | 522 |

- **Preprocessing:** 80/20 stratified train/test split (`random_state=42`); features standardized with `StandardScaler` (fit on training data only, to avoid data leakage). Because the classes are imbalanced (~7:1 between the largest and smallest), macro-averaged metrics and MCC are prioritized over raw accuracy.

## c. GitHub repository link

https://github.com/saravananarasan-bits-aiml/drybean-ml-classifier-sarasan

---

## d. Models used

Five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (k = 5)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble, 100 trees)

### Evaluation metrics (on the held-out test set)

Metrics use macro-averaging for precision/recall/F1 (each class weighted equally); AUC uses one-vs-rest, macro-averaged.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.9041 | 0.9822 | 0.9175 | 0.9147 | 0.9159 | 0.8841 |
| kNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9916 | 0.9112 | 0.9092 | 0.9091 | 0.8773 |
| Random Forest (Ensemble) | 0.9207 | 0.9926 | 0.9355 | 0.9313 | 0.9333 | 0.9041 |

### Observations

| ML Model | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression gave one of the best scores (accuracy 0.9207, MCC 0.9042). A simple straight-line model doing this well tells us the 7 bean types are mostly easy to separate — the shape measurements already carry enough signal, so a more complex model isn't really needed. |
| Decision Tree | The single decision tree was the weakest of the good models (accuracy 0.9041, MCC 0.8841). One tree tends to overfit and depends heavily on how the splits fall, so it doesn't hold up as well on new data as the forest or the linear model. |
| kNN | kNN did really well (accuracy 0.9166), but only because we scaled the features first. It works on distance, so without scaling the big-valued features would take over. Beans of the same type sit close together, so voting among the nearest neighbours works nicely here. |
| Naive Bayes | Naive Bayes had the lowest accuracy (0.8979) but still a high AUC (0.9916). It assumes all features are independent, which isn't true here — Area and ConvexArea clearly move together — and that hurts its final predictions. But it still ranks the classes well by probability, which is what AUC rewards. |
| Random Forest (Ensemble) | Random Forest tied for the top score (accuracy 0.9207) and clearly beat the single tree it's built from (up from 0.9041). Combining many different trees and taking their majority vote cancels out the instability of one tree, giving steadier, more accurate results. |
| **Overall winner for your dataset?** | Logistic Regression and Random Forest came out on top together, both at 0.9207 accuracy and ~0.904 MCC. Logistic Regression is the better pick when you want something fast and easy to explain; Random Forest is better when the data is messier or more complex. |

---

## Streamlit app features

The deployed app provides:

- **CSV upload** — upload test data for scoring.
- **Model selection dropdown** — choose any of the 5 trained models.
- **Evaluation metrics** — Accuracy, Precision, Recall, F1, MCC displayed for the selected model.
- **Confusion matrix** — a heatmap of actual vs predicted classes.

## Project structure

```
drybean-ml-classifier-sarasan/
├── app.py                 # Streamlit web application
├── requirements.txt       # dependencies
├── README.md              # this file
├── test_data.csv          # held-out test split (used by the app)
└── model/                 # saved artifacts
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── drybean_classifier.ipynb   # training + evaluation notebook
```

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL, upload `test_data.csv`, and select a model.

---

