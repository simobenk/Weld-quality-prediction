# Welding Quality Prediction – Projet de groupe

Ce dépôt contient le projet que l’on a réalisé à 4 autour de la **prédiction de la qualité de soudures** à partir de leur composition chimique et des paramètres de procédé.  
On combine :

- apprentissage **supervisé classique** ;
- **semi-supervisé (SSL)** pour compléter les cibles manquantes ;
- **clustering** des comportements mécaniques ;
- **classification** des clusters ;
- la prédiction d’un **score global de qualité** basé sur une norme L2;
- et enfin l'application des méthodes d’explicabilité .

---

## 1. Données

On travaille à partir d’une base de données de soudures (~1600 lignes) comprenant :  

- **Composition chimique** (C, Si, Mn, Ni, Cr, Mo, V, etc., en % et en ppm)  
- **Paramètres de soudage** : courant, tension, AC/DC, polarité, énergie linéique, température inter-passe…  
- **Traitements thermiques post-soudage** : température et temps  
- **Propriétés mécaniques** (cibles) :
  - Yield strength (limite d’élasticité, MPa)
  - Ultimate tensile strength (résistance à la traction, MPa)
  - Elongation (%)
  - Reduction of Area (%)
  - Charpy impact toughness (J)

Les propriétés mécaniques sont **partiellement manquantes**, ce qui motive l’usage du semi-supervisé.

---

## 2. Pipeline global

### 2.1. Prétraitement & EDA (`Data Processing/`)

Les notebooks de **prétraitement** et d’**analyse exploratoire** (`Welding_EDA.ipynb`) réalisent une étude complète des données brutes de soudage avant toute modélisation.  
Ce travail inclut le chargement, le nettoyage, la structuration des variables, la gestion des valeurs manquantes et la création d’analyses descriptives avancées.

####  1. Chargement & harmonisation des données  
- Chargement des fichiers bruts (`welddb.data` ou `welddb_cleaned.csv` selon disponibilité).  
- Attribution manuelle des **38 noms de colonnes** couvrant :
  - composition chimique (weight% + ppm),
  - paramètres de soudage,
  - traitements thermiques,
  - propriétés mécaniques,
  - microstructure,
  - identifiants de soudure.  
- Remplacement des valeurs manquantes codées `"N"` par `NaN`.  

####  2. Analyse de complétude & structure du dataset  
- Dimensions finales du jeu de données : **1652 lignes × 38 colonnes**.  
- Typologie : **36 variables numériques**, **2 catégorielles**.  
- Taux de données manquantes global : **30,7 %**.  
- Détection des doublons (aucun détecté).  
- Calcul détaillé par variable : nombre de valeurs manquantes, pourcentage, skewness, kurtosis, coefficient de variation, etc.

Des visualisations dédiées permettent d’examiner :
- la **distribution des % de valeurs manquantes**,  
- une **heatmap des motifs de missingness**,  
- la répartition des manquants par groupes de features.

Une analyse MCAR/MAR/MNAR est également fournie pour comprendre l’origine des manquants (tests mécaniques partiels, mesures chimiques coûteuses, dépendances structurelles…).

####  3. Structuration du dataset en groupes logiques  
Les features sont regroupées en 6 familles pour guider l’analyse ultérieure :
- **Composition chimique (weight%)**  
- **Composition chimique (ppm)**  
- **Paramètres de procédé de soudage**  
- **Traitements thermiques post-soudage**  
- **Propriétés mécaniques (targets)**  
- **Microstructure** (souvent très incomplète dans ce dataset)

####  4. Analyse univariée & visualisation avancée  
Pour chaque famille de variables :
- statistiques détaillées (moyenne, médiane, IQR, skewness, kurtosis),  
- histograms + KDE pour analyser les distributions,  
- **boxplots** pour détecter les outliers,  
- visualisation des paramètres de procédé (courant, tension, heat input, interpass temperature…),  
- visualisation des propriétés mécaniques (Yield, UTS, RA, Elongation, Charpy…).  

Observations clés :
- présence d’outliers physiquement plausibles (aciers alliés, tests extrêmes),  
- distributions non gaussiennes pour plusieurs éléments d’alliage,  
- forte variabilité dans les tests mécaniques.

####  5. Analyse des variables catégorielles  
Deux variables qualitatives principales :
- **Type of weld** (10 procédés, dominé par MMA avec 69 %)  
- **Weld ID** (1490 identifiants uniques → quasi aucun réplica)

Graphiques + analyse d’équilibre des classes, avec discussion sur le **risque de biais** pour les procédés rares.

####  6. Analyse approfondie des propriétés mécaniques  
- Matrice de corrélation dédiée aux targets :  
  - Yield ↔ UTS : **r = 0.915** (forte dépendance)  
  - UTS ↔ Elongation : **r = –0.739**  
  - RA ↔ Charpy : **r = 0.831**  
- Interprétation métallurgique : compromis ductilité/résistance, dureté/ténacité, etc.

####  7. Analyse multivariée & PCA  
- Corrélation globale (features disponibles à >30 %).  
- Identification de 12 paires très corrélées (>0.8), dont :
  - plusieurs paires “ppm vs weight%” (corrélation parfaite),  
  - S ↔ P (0.947), Cr ↔ Nb (0.812), Current ↔ Heat Input (0.912).
- PCA sur la composition chimique (weight%) :  
  - **9 composantes nécessaires pour 95 % de variance**,  
  - PC1 fortement influencé par S, P, Cr, Mo, Mn,  
  - cartographie PCA montrant plusieurs profils métallurgiques distincts.

####  8. Synthèse EDA  
Le notebook conclut avec une synthèse automatique :
- fragmentation du dataset (seulement **11 lignes complètes** sans aucun NaN),  
- forte hétérogénéité entre familles de features,  
- importance d’imputations cohérentes (KNN, modèles supervisés, SSL…),  
- présence d’interdépendances fortes dans les propriétés mécaniques,  
- dataset très riche mais complexe, nécessitant un pipeline robuste de nettoyage + normalisation + sélection de features.

---


### 2.2. Supervised Learning « classique » – prédiction des propriétés mécaniques

Dans ce dossier, on a entraîné **cinq modèles supervisés**, chacun dans son propre notebook, pour prédire les propriétés mécaniques suivantes :

- Yield strength (MPa)  
- Ultimate tensile strength (MPa)  
- Elongation (%)  
- Reduction of Area (%)  
- Charpy impact toughness (J)

Les notebooks correspondants sont :

- `Supervised_Learning_Yield_strength_MPa.ipynb`  
- `Supervised_Learning_Ultimate_tensile_strength_MPa.ipynb`  
- `Supervised_Learning_Elongation_pct.ipynb`  
- `Supervised_Learning_Reduction_of_Area_pct.ipynb`  
- `Supervised_Learning_Charpy_impact_toughness_J.ipynb`

La **même méthodologie complète** est appliquée dans chacun de ces notebooks.  
Ci-dessous, on détaille l’exemple du Yield strength — mais **les quatre autres notebooks suivent exactement la même structure et les mêmes étapes**, simplement avec une cible différente.

---

#### Exemple détaillé : prédiction du *Yield strength (MPa)*

**Objectif :** prédire le Yield strength à partir de la composition chimique, des paramètres de soudage et des traitements thermiques.

#### Étapes :

1. **Sélection des cas complets**  
   - On conserve uniquement les lignes où la cible `Yield strength (MPa)` est renseignée.

2. **Sélection & préparation des features**  
   - Suppression des autres cibles mécaniques.  
   - Suppression des colonnes avec trop de valeurs manquantes.  
   - Encodage des variables catégorielles.  
   - Standardisation des variables numériques.  
   - Imputation des valeurs manquantes via **KNN Imputer**.

3. **Réduction de dimension par PCA**  
   - PCA appliquée sur les features scalées.  
   - Sélection des composantes expliquant **au moins 95 % de la variance**.

4. **Comparaison de modèles avec PCA**  
   - Modèles testés : Random Forest, XGBoost, Gradient Boosting, Ridge, SVR.  
   - Validation via **5-fold cross-validation** (R², MAE, RMSE).

5. **XGBoost sans PCA (full features)**  
   - Entraînement d’un XGBoost sur toutes les features.  
   - Optimisation des hyperparamètres avec **GridSearchCV**.  
   - Analyse d’importance des variables (composition, procédé, traitements).  
   - Résultat : meilleur modèle, ≈86 % de variance expliquée.

#### Conclusion  
Les modèles basés sur des méthodes boosting **sans PCA** capturent mieux les interactions non linéaires que ceux entraînés sur les composantes PCA, ce qui mène à de meilleures performances.

---

####  Application aux autres propriétés mécaniques

Les notebooks suivants répètent **la même procédure étape par étape** pour prédire chacune des autres cibles :

- *Ultimate tensile strength (MPa)*  
- *Elongation (%)*  
- *Reduction of Area (%)*  
- *Charpy impact toughness (J)*  

Dans chaque cas, seule la cible change, mais :

- le prétraitement,  
- la réduction de dimension,  
- les modèles entraînés,  
- les grilles d’hyperparamètres,  
- et les évaluations  

restent **identiques**, garantissant une cohérence méthodologique sur l’ensemble des propriétés mécaniques étudiées.

---

### 2.3. Semi-Supervised Learning (SSL)

Notebook : `SSL_Notebook.ipynb`  

Problème : les propriétés mécaniques (Yield, UTS, A, RA, Charpy) sont **partiellement étiquetées** (~45–55 % de lignes complètes). On veut exploiter aussi les lignes **sans labels**.  

#### 2.3.1. SSL simple (Self-Training basique)

Pour chaque cible :

1. On sépare :
   - **Labeled** : lignes où la cible est connue ;
   - **Unlabeled** : lignes où la cible est manquante ;
   - on split le labeled en **train/test** (80/20).  

2. Baseline : Random Forest uniquement sur les données étiquetées.  

3. SSL simple :
   - on entraîne un RF sur le train ;
   - on prédit sur l’unlabeled ;
   - on estime l’**incertitude** via la variance des arbres ;
   - on garde seulement les points les plus sûrs (seuil sur un percentile d’incertitude) comme **pseudo-labels**, qu’on rajoute dans le train ;
   - on boucle sur plusieurs itérations. 

Résultat :  
Pour la majorité des cibles, le SSL simple **n’améliore pas** les R² (voire les dégrade légèrement). Le seuil de confiance est trop permissif et introduit du bruit, et la baseline RF est déjà solide.

#### 2.3.2. SSL avancé

On met en place un **Self-Training avancé** :

- modèles de base plus puissants : **XGBoost, LightGBM, Random Forest, ensemble (Stacking)** ;  
- seuil de confiance **adaptatif** (strict au début, puis relâché au fil des itérations) ;  
- **pondération des pseudo-labels** en fonction de la confiance (poids 0.1 → 1.0) ;  
- stratégie de **diversity sampling** pour éviter de n’ajouter que des points quasi identiques.  

Résultat :  
Avec l’Advanced SSL (XGBoost comme modèle de base dans cette version) :

- les **5 cibles** montrent une amélioration de R² et de MAE ;
- l’amélioration moyenne en R² est positive (~+2 %), avec une baisse de l’erreur moyenne (~+4 % sur la MAE).  

Cela montre que, bien configuré, le SSL exploite réellement l’unlabeled pour affiner les modèles.

Les prédictions issues de ce SSL sont utilisées pour **compléter les colonnes mécaniques** et générer `cleanwelddata_filled_ssl.csv`.

---

### 2.4. Clustering des comportements mécaniques

Notebook : `Clustring.ipynb`  

Une fois les données complétées par SSL, on réalise un **clustering** sur les propriétés mécaniques afin de regrouper les soudures par comportement :

- Features de clustering :  
  `Yield strength`, `Ultimate tensile strength`, `Elongation`, `Reduction of Area`. 

Étapes :

1. **Standardisation** des cibles mécaniques.
2. Test de **K-means** :
   - on trace la courbe d’inertie (elbow method) pour choisir le nombre de clusters ;
   - un choix typique est autour de **k = 4** pour K-means.  
3. **Gaussian Mixture Model (GMM)** :
   - on ajuste des GMM pour différents nombres de clusters ;
   - on choisit **k = 3** en s’appuyant sur AIC/BIC ;  
   - on récupère pour chaque point :
     - le label `cluster_gmm`,
     - la probabilité d’appartenance `cluster_proba`.  

4. **Visualisation & interprétation physique**
   - projection PCA (2D) + GMM pour visualiser les clusters ;
   - statistiques par cluster (moyenne / std de Re, Rm, A, RA) ;  
   - on interprète physiquement :
     - un cluster **résistant mais peu ductile** (plus dur/fragile) ;
     - un cluster **équilibré** (bon compromis résistance/ductilité) ;
     - un cluster **moins résistant mais très ductile** (plus « mou » mais tenace). 

Ces labels sont sauvegardés dans `data_avec_clusters.csv` pour la suite.

---

### 2.5. Prédiction des clusters (classification)

Notebook : `prédiction_sur_les_clusters.ipynb`  

Objectif : une fois qu’on a défini des **clusters GMM**, on essaye de construire un modèle supervisé qui, à partir des features de soudage, **prédit directement le cluster** (donc le “type de comportement mécanique”).  

Pipeline :

1. Chargement de `data_avec_clusters.csv` + nettoyage de colonnes inutiles.  
2. Cible : `cluster_gmm`.  
3. On enlève de X :
   - les cibles mécaniques et `cluster_proba`/`cluster_gmm` lorsque nécessaire (pour éviter la fuite de labels),
   - on garde les features de procédé/composition. 

4. **Modèles testés** :
   - RandomForest
   - XGBoost
   - Logistic Regression
   - SVM (RBF)
   - KNN
   - Gradient Boosting
   - AdaBoost 

5. Évaluation :
   - split train/test (80/20, stratifié) ;
   - métriques : **accuracy**, classification report ;
   - **Stratified K-Fold (5 folds)** pour valider la stabilité des modèles.

Plusieurs modèles (en particulier XGBoost et RF) atteignent de **bonnes accuracies**, ce qui confirme que les clusters définis ont un signal suffisamment fort pour être prédits à partir des paramètres de soudage.

---

### 2.6. Score global de qualité (norme L2) & prédiction

Notebook : `Quality_Score_prediction.ipynb`  

Ici, on construit un **score global de qualité mécanique**, basé sur une **norme L2** des propriétés :

\[
\text{quality\_score} = \sqrt{\text{mean}\left(UTS^2,\ A^2,\ RA^2\right)}
\]

Concrètement :

1. On part de `cleanwelddata_filled_ssl.csv` (données complètes) ;  
2. On enlève `Charpy temperature` et `Charpy impact toughness` qui sont des paramètres expérimentaux fixés ;  
3. On crée `quality_score` comme la **racine de la moyenne des carrés** de :
   - `Ultimate tensile strength (MPa)`
   - `Elongation (%)`
   - `Reduction of Area (%)` 

Ensuite on pose un **problème supervisé** :

- Features X : toutes les colonnes sauf les cibles mécaniques et `quality_score` ;  
- Target y : `quality_score`, avec un split stratifié (binning) pour garder une distribution similaire train/test. 

Modèles :

1. **Random Forest Regressor**  
   - Tuning par **GridSearchCV** (n_estimators, max_depth, min_samples_split, min_samples_leaf).  
   - Visualisation prédits vs réels (train et test) et importance des variables. 

2. **XGBoost Regressor**  
   - Tuning par **RandomizedSearchCV**.
   - Évaluation R² / RMSE sur train et test + plot d’importance des variables. 

Ce notebook montre qu’on peut **prédire un score global de qualité** (norme L2 des propriétés mécaniques) directement à partir des paramètres de soudage et de la composition, ce qui est intéressant pour un indicateur synthétique.

---
### 2.7. Interprétabilité des modèles – XAI & SHAP

Notebook : `XAI_SHAP.ipynb` 

Dans ce notebook, on a appliqué des méthodes d’explicabilité (XAI) afin de comprendre **comment le modèle de classification des clusters** prend ses décisions.  
L’objectif est d’identifier quelles variables influencent le plus l’appartenance à chaque cluster mécanique.

####  Méthodes utilisées

- **HistGradientBoostingClassifier** comme modèle final.  
- **Feature Importance** (gain) pour obtenir une première hiérarchie des variables.  
- **SHAP (SHapley Additive exPlanations)** :
  - SHAP summary plot (vue globale de l’importance des variables),
  - SHAP summary par classe (analyse de l’impact des features cluster par cluster),
  - dependence plots (effet d’une variable + interaction avec une autre),
  - force plot (explication locale d’une prédiction),
  - decision plot (trajectoire des contributions SHAP),
  - interaction values (effets combinés entre variables).

####  Principaux enseignements

- Les variables liées à la **composition chimique** (ex : Mo, Cr, Ni, Mn, P) et certains paramètres de procédé ont un impact fort sur la classification.  
- Les SHAP values montrent **dans quel sens** une variable oriente la prédiction :
  - valeurs élevées → augmentent la probabilité d’appartenir à un certain cluster ;
  - valeurs faibles → l’atténuent.
- Les decision plots permettent de suivre la contribution cumulée des variables pour chaque soudure.
- Les SHAP interaction values révèlent quels couples de variables **interagissent réellement** (ex : Mo × Cr).
##  Auteurs

Projet réalisé par :

- **Nawfal Benhamdane**
- **Badoules Fanny**
- **Benkirane Mohamed**
- **Boukhari Aya**

Étudiants à l’École **CentraleSupélec**.

