# SINAYA 🌊
### Neuro-Fuzzy Regression for Water Quality Modeling

---

## Abstract

**SINAYA** is an inferential regression model designed for assessing ammonia levels in open water mariculture systems. The primary aim of this project is to present an unbiased, comprehensive analysis of a system relying on a TensorFlow-based Fuzzy Neural Network.

The model is trained to draw conclusions by utilizing datasets pertaining to three critical water quality parameters: **pH level**, **Temperature**, and **Dissolved Oxygen (DO)**. Employing a research design centered on correlational inference, this system leverages machine learning techniques to establish a connection between these input parameters and projected ammonia levels.

By evaluating various hyperparameters, neural network architectures, and dataset sizes, this project successfully identified the ideal configuration for its available computing resources, resulting in high prediction accuracy and low error rates — validated by the **Bureau of Fisheries and Aquatic Resources**.

---

## 🚀 Key Features

- **Hybrid Neuro-Fuzzy Architecture** — Integrates Gaussian fuzzy membership functions with a Feed-Forward Neural Network (FFNN).
- **Automated Data Preprocessing** — Built-in pipelines for median imputation, Z-score outlier removal (threshold = 2), and safe MinMax scaling.
- **Gaussian Fuzzification** — Uses fixed Gaussian membership functions with `σ = 0.05` and 4 evenly-spaced centers over `[0, 1]`, categorizing scaled inputs into fuzzy sets (`VeryLow`, `Low`, `Medium`, `High`).
- **Feature-Level Interpretability** — Utilizes SHAP (SHapley Additive exPlanations) to demystify how specific fuzzy ranges impact the final ammonia prediction.

---

## 🧠 Model Architecture

The model is structured across five conceptual layers:

| Layer | Description |
|---|---|
| **1. Input Layer** | Receives the raw, MinMax-scaled values of pH, DO, and Temperature (3 features). |
| **2. Fuzzy Layer** | Transforms each scaled input into 4 Gaussian membership values via `fuzzify_gaussian`, producing 15 total features (3 original + 12 fuzzy). |
| **3. Rule Layer** | Dense(64, `ReLU`) → Dense(32, `ReLU`) — learns non-linear feature interactions. |
| **4. Inference Layer** | Dense(16, `ReLU`) — distills higher-level rule representations. |
| **5. Output Layer** | Dense(1, `linear`) — outputs a MinMax-scaled ammonia prediction, inverse-transformed to the original unit. |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (`lr = 0.0015`) |
| Loss Function | Mean Squared Error (MSE) |
| Batch Size | 16 |
| Max Epochs | 800 |
| Early Stopping | patience = 100, restore best weights |
| Validation Split | 20% of training data |

## 📊 Model Performance

The pipeline was trained on **494 mariculture samples** with hold-out validation.

### Validation Metrics

| Metric | Value |
|---|---|
| Mean Squared Error (MSE) | `0.045588` |
| Root Mean Squared Error (RMSE) | `0.213514` |
| Mean Absolute Error (MAE) | `0.154955` |
| R-Squared (R²) | `0.733841` |

Based on the SHAP summary plot analysis:

- **Primary Drivers** — `DO_Medium` and `pH_Medium` are the most influential features, with the widest SHAP value spread and highest mean impact on ammonia predictions.
- **Inverse Relationships** — High values of `DO_High` (pink dots) consistently produce negative SHAP values, confirming that highly saturated dissolved oxygen strongly suppresses predicted ammonia.
- **Directional Asymmetry in pH** — `pH_Low` shows low feature values (blue) driving *positive* SHAP contributions, while `pH_High` produces scattered positive outliers, suggesting a non-linear relationship between pH extremes and ammonia levels.
- **Moderate Temperature Influence** — `Temp_Medium` and `Temp_High` cluster near zero with limited spread, indicating temperature contributes but is not a dominant driver in this dataset range.
- **Low Impact Variables** — `Temp_VeryLow`, `Temp_Low`, `DO_VeryLow`, and `pH_VeryLow` have the narrowest SHAP distributions, contributing the least to output variance.

---

## 🛠️ Tech Stack & Requirements

| Library | Version | Purpose |
|---|---|---|
| **TensorFlow / Keras** | ≥ 2.x | Building and training the Feed-Forward Neural Network |
| **Scikit-Learn** | ≥ 1.x | `MinMaxScaler`, `mean_squared_error`, `r2_score` |
| **SciPy** | ≥ 1.x | Z-score outlier detection (`stats.zscore`) |
| **Pandas** | ≥ 1.x | Data loading, deduplication, imputation |
| **NumPy** | ≥ 1.x | Matrix operations and fuzzy feature computation |
| **SHAP** | ≥ 0.4x | Feature explainability and beeswarm summary plots |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/sinaya.git
cd sinaya

# Install dependencies
pip install tensorflow scikit-learn scipy pandas numpy shap
```

---

## 📜 License

This project is intended for academic and research purposes. Please cite appropriately when using SINAYA in published work.

---

> *Validated by the Bureau of Fisheries and Aquatic Resources (BFAR), Philippines.*
