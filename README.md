# Welding Quality Prediction 

This repository contains the project we carried out as a team of four on the **prediction of welding quality** based on chemical composition and welding process parameters.  
We combine:

- classical **supervised learning**
- **semi-supervised learning (SSL)** to complete missing targets
- **clustering** of mechanical behaviours
- **classification** of the clusters
- prediction of a **global quality score** based on an L2 norm
- and **explainability (XAI)** methods

---

## 1. Dataset

We work with a welding database (~1600 rows) including:

- **Chemical composition** (C, Si, Mn, Ni, Cr, Mo, V, etc., in wt% and ppm)
- **Welding process parameters**: current, voltage, AC/DC, polarity, heat input, interpass temperature…
- **Post-weld heat treatments**: temperature and time
- **Mechanical properties** (targets):
  - Yield strength (MPa)
  - Ultimate tensile strength (MPa)
  - Elongation (%)
  - Reduction of Area (%)
  - Charpy impact toughness (J)

Mechanical properties are **partially missing**, which motivates the use of SSL.

---

## 2. Global Pipeline

### 2.1. Preprocessing & EDA (`Data Processing/`)

The notebook **`Welding_EDA.ipynb`** performs a complete exploratory data analysis and preprocessing of the raw dataset.

#### Main steps:

- Loading and cleaning (`welddb.data` or `welddb_cleaned.csv`)
- Manual assignment of **38 column names**
- Replacement of `"N"` with `NaN`
- Missing data analysis (overall missing rate: **30.7%**)
- Feature grouping into 6 families:
  - Chemical composition (wt%)
  - Chemical composition (ppm)
  - Welding parameters
  - Heat treatments
  - Mechanical properties
  - Microstructure

#### Statistical & visual analysis:

- Univariate statistics (mean, median, IQR, skewness, kurtosis)
- Histograms + KDE plots
- Boxplots to detect outliers
- Distribution of mechanical properties
- Categorical variable analysis (Type of weld, Weld ID)
- Correlation analysis between mechanical properties:
  - Yield ↔ UTS: **0.915**
  - UTS ↔ Elongation: **–0.739**
  - RA ↔ Charpy: **0.831**

#### PCA:

- PCA performed on chemical composition features
- **9 principal components** required to explain **95% of the variance**

Only **11 rows** are fully complete → a robust imputation strategy is necessary.

---

### 2.2. Classical Supervised Learning

Five separate supervised models are trained (one notebook per target):

- Yield strength (MPa)
- Ultimate tensile strength (MPa)
- Elongation (%)
- Reduction of Area (%)
- Charpy impact toughness (J)

All notebooks follow the same structure.

#### Example: Yield strength prediction (`Supervised_Learning_Yield_strength_MPa.ipynb`)

##### Steps:

1. Keep only rows where the target is known  
2. Remove all other mechanical targets  
3. Encode categorical variables and scale numerical features  
4. Apply **KNN imputation**  
5. Perform **PCA** (retain components explaining 95% variance)  
6. Compare models:
   - Random Forest  
   - XGBoost  
   - Gradient Boosting  
   - Ridge Regression  
   - Support Vector Regression  
7. Train **XGBoost without PCA** + hyperparameter tuning via GridSearchCV  
   - Best performance: **≈86% R²**

The exact same pipeline is applied to all mechanical targets.

---

### 2.3. Semi-Supervised Learning (SSL)

Notebook: **`SSL_Notebook.ipynb`**

Mechanical properties have **45–55% missing values**, making SSL relevant.

#### Simple SSL (baseline):

- Train a Random Forest on labeled data  
- Predict unlabeled data  
- Add “high-confidence” pseudo-labels  
- Iterate  

→ Result: little to no improvement, sometimes degraded performance.

#### Advanced SSL:

- Stronger models:
  - XGBoost  
  - LightGBM  
  - Random Forest  
  - Stacking models  
- Adaptive confidence threshold  
- Weighted pseudo-labels (based on model uncertainty)  
- Diversity sampling to avoid duplicate patterns  

##### Performance:

- **+2% R²** on average  
- **–4% MAE** on average  

The SSL-completed dataset is saved as: **`cleanwelddata_filled_ssl.csv`**


---

### 2.4. Clustering Mechanical Behaviour

Notebook: **`Clustring.ipynb`**

Clustering performed on:

- Yield strength  
- UTS  
- Elongation  
- Reduction of Area  

#### Steps:

1. Standardisation  
2. **K-means** clustering  
   - Elbow method → **k ≈ 4**
3. **Gaussian Mixture Model (GMM)**  
   - Best choice: **k = 3** (based on AIC/BIC)  
   - Outputs:
     - `cluster_gmm`
     - `cluster_proba`
4. PCA projection and cluster visualisation  
5. Interpretation of cluster behaviour:

- High strength / low ductility  
- Balanced mechanical behaviour  
- Lower strength / high ductility  

Dataset with cluster labels saved as: **`data_avec_clusters.csv`**


---

### 2.5. Cluster Classification

Notebook: **`prédiction_sur_les_clusters.ipynb`**

#### Objective:

Predict the mechanical-behaviour cluster **directly** from welding input features.

#### Models tested:

- RandomForest  
- XGBoost  
- Logistic Regression  
- SVM (RBF)  
- KNN  
- Gradient Boosting  
- AdaBoost  

**Best performance:** XGBoost and RandomForest.

---

### 2.6. Global Quality Score (L2 Norm)

Notebook: **`Quality_Score_prediction.ipynb`**

A synthetic global quality index is defined as:

$$
\text{quality score} = \sqrt{mean(UTS^2,\ A^2,\ RA^2)}
$$


#### Regression models:

- Random Forest Regressor (GridSearchCV)
- XGBoost Regressor (RandomizedSearchCV)

Both achieve good prediction performance and provide interpretable feature importance plots.

---

### 2.7. Explainability – XAI & SHAP

Notebook: **`XAI_SHAP.ipynb`**

Using **HistGradientBoostingClassifier** and SHAP values:

#### Methods:

- Global feature importance  
- SHAP summary plot  
- Class-wise SHAP explanations  
- Dependence plots  
- Force plots  
- Decision plots  
- SHAP interaction values  

#### Main findings:

- Chemical composition (Mo, Cr, Ni, Mn, P) strongly influences cluster assignment  
- SHAP clarifies how feature values push predictions toward specific clusters  
- Significant interactions include **Mo × Cr**

---

## Authors

Project carried out by:

- **Benkirane Mohamed**
- **Nawfal Benhamdane**
- **Badoules Fanny**
- **Boukhari Aya**

Students at **CentraleSupélec**.
