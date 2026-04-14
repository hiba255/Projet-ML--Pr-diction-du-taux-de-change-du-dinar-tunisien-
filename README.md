#  Prévision du Taux de Change EUR/TND — Projet Machine Learning

> **Prédiction du taux Euro / Dinar Tunisien du lendemain à l'aide d'indicateurs macroéconomiques et d'algorithmes d'ensemble**

---

##  Auteurs & Contexte

| Champ | Détail |
|---|---|
| **Auteurs** | Ben Selma Hibe & Cherchir Aya |
| **Notebook** | `02_Modeling.ipynb` — Préprocessing, Feature Engineering & Modélisation |
| **Prérequis** | `01_EDA.ipynb` (Analyse Exploratoire des Données) |
| **Dataset d'entrée** | `macro_market_merged.csv` (produit par `01_EDA.ipynb`) |
| **Période couverte** | Janvier 2015 – Janvier 2025 (~10 ans de données journalières) |
| **Objectif** | Prédire le taux EUR/TND  |
| **Graine aléatoire** | `SEED = 42` (reproductibilité garantie) |

---

##  Problématique

Le Dinar Tunisien (TND) est une devise dont le taux de change est influencé à la fois par des facteurs **macroéconomiques locaux** (balance commerciale, taux d'intérêt, inflation) et par des **indicateurs de marché internationaux** (EUR/USD, cours du Brent, prix de l'or). Anticiper ces fluctuations présente un intérêt direct pour les acteurs économiques (importateurs, exportateurs, institutions financières).

Ce projet formule la prédiction comme un **problème de régression supervisée sur séries temporelles** : à partir des données disponibles jusqu'au jour J, prédire le taux EUR/TND du jour J+1.

---

##  Structure du Projet

```
.
├── data
    └──  Fichiers Excel des données collectées auprès de l’INS et de la BCT → données quotidiennes → fusionnées
├── drafts
    └── notbooks_daily_data
        └──draft2balance.ipynb
        └──draft2inflation.ipynb
        └──draft2pib
        └──draft2taux_interet.ipynb 
    └── notbooks_daily_donnees_initiaux
          └──Initial_EDA.ipynb
          └──Initial_Modeling.ipynb
          └──Initial_Modeling_draft.ipynb 
├── notebooks
    └──01_EDA.ipynb          
    └──02_Modeling.ipynb   # ← Notebook principal
├── presentation_EURTND.pptx                 
├── README.md           
└── requirements.txt                      
                              
```

---

##  Pipeline Complet

```
macro_market_merged.csv
        │
        ▼
┌─────────────────────────────────────────────┐
│  ÉTAPE 1 — Prétraitement des Données        │
│  ├─ Chargement & normalisation des colonnes │
│  │   (suppression des accents, lowercase)   │
│  ├─ Téléchargement EUR/TND via yfinance     │
│  │   (Yahoo Finance : EURTND=X)             │
│  ├─ Gestion des NaN :                       │
│  │   forward-fill ≤ 5 jours (week-ends)     │
│  ├─ Suppression des doublons d'index        │
│  └─ Suppression des lignes NaN résiduelles  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 2 — Feature Engineering                              │
│                                                             │
│  2.1 Lags (valeurs décalées)                                │
│      Colonnes : EUR/TND, EUR/USD, Brent, Or, EUR/GBP,       │
│                 US 10Y Yield, MSCI EM, Balance commerciale  │
│      Décalages : lag1, lag3, lag7, lag30                    │
│                                                             │
│  2.2 Statistiques glissantes sur EUR/TND                    │
│      Fenêtres : 7, 14, 30 jours                             │
│      → Moyenne mobile (MA) & Écart-type (STD)               │
│                                                             │
│  2.3 Indicateurs de momentum & vélocité                     │
│      → Variation absolue et relative jour-sur-jour          │
│                                                             │
│  2.4 Features calendaires                                   │
│      → Jour de la semaine, mois, trimestre                  │
│                                                             │
│  2.5 Proxy DXY                                              │
│      → dxy_proxy = 1 / EUR/USD                              │
│                                                             │
│  2.6 Sélection des features                                 │
│      → Top 20 par corrélation de Pearson avec EUR/TND       │
│                                                             │
│  2.7 Normalisation (StandardScaler)                         │
│      → Appliquée uniquement pour Ridge Regression           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  ÉTAPE 3 — Séparation Chronologique 80 / 20           │
│  ├─ Entraînement : 2015 → ~2023  (80% des données)    │
│  └─ Test         : ~2023 → 2025  (20% des données)    │
│                                                       │
│  ⚠️  Pas de mélange aléatoire (data leakage interdit) │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  ÉTAPE 4 — Baseline Naïve (référence)      │
│  └─ Prédiction : ŷ[t] = y[t-1]            │
│     Établit le plancher de performance     │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  ÉTAPE 5 — Entraînement & Comparaison des 4 Modèles        │
│  ├─ Ridge Regression  (features standardisées, RidgeCV)    │
│  ├─ Random Forest     (300 arbres, max_depth=10)            │
│  ├─ XGBoost           (500 estimateurs, lr=0.03)            │
│  └─ LightGBM          (500 estimateurs, lr=0.03)            │
└────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  ÉTAPE 6 — Évaluation & Analyse                     │
│  ├─ Tableau comparatif MAE / RMSE / MAPE            │
│  ├─ Graphiques Actual vs. Predicted (4 modèles)     │
│  └─ Distribution des résidus par modèle             │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  ÉTAPE 7 — Optimisation des Hyperparamètres              │
│  ├─ Méthode : RandomizedSearchCV (30 itérations)         │
│  ├─ Validation : TimeSeriesSplit (5 folds chronologiques)│
│  ├─ XGBoost  → grille sur 8 hyperparamètres              │
│  └─ LightGBM → grille sur 9 hyperparamètres              │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  ÉTAPE 8 — Importance des Features                       │
│  └─ Top 20 features pour RF, XGBoost (opt.), LGBM (opt.) │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 9 — Validation Walk-Forward (LightGBM optimisé)      │
│  ├─ Fenêtre minimale d'entraînement : 3 ans                 │
│  ├─ Pas d'avancement : 30 jours                             │
│  ├─ Horizon de prédiction : 30 jours                        │
│  └─ MAE glissante par fenêtre (graphique d'évolution)       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  ÉTAPE 10 — Prédiction du Lendemain                          │
│  ├─ Réentraînement sur l'intégralité du dataset              │
│  ├─ Prédiction ŷ[T+1]                                        │
│  └─ Intervalles de confiance empiriques : 68% et 90%         │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  ÉTAPE 11 — Bilan Final & Conclusions   │
│  └─ Tableau récapitulatif tous modèles  │
└─────────────────────────────────────────┘
```

---

##  Modèles Utilisés

### 1. Baseline Naïve
La prédiction la plus simple possible : le taux de demain est égal au taux d'aujourd'hui (`ŷ[t] = y[t-1]`). Elle sert de **borne inférieure de performance** : tout modèle ML doit impérativement faire mieux pour être considéré comme utile.

### 2. Ridge Regression
Régression linéaire pénalisée par la norme L2 (`‖β‖²`). L'alpha optimal est sélectionné automatiquement par validation croisée (`RidgeCV`, alphas testés : `[0.01, 0.1, 1, 10, 100, 1000]`). Ce modèle nécessite une **standardisation** préalable des features (StandardScaler) et offre une excellente **interprétabilité** via l'inspection des coefficients. Il constitue le modèle linéaire de référence.

### 3. Random Forest
Forêt de 300 arbres de décision entraînés en parallèle par **bagging** (Bootstrap Aggregating). Hyperparamètres : `max_depth=10`, `min_samples_leaf=5`, `max_features=0.5`. Robuste aux valeurs aberrantes et aux corrélations entre features, il capture les **non-linéarités** sans nécessiter de normalisation des données.

### 4. XGBoost
500 arbres entraînés **séquentiellement** par gradient boosting, chaque arbre corrigeant les erreurs du précédent. Taux d'apprentissage de 0.03 pour un apprentissage progressif. Régularisation L1 (`reg_alpha=0.1`) et L2 (`reg_lambda=1.0`) pour éviter le surapprentissage. Sous-échantillonnage des lignes (`subsample=0.8`) et des colonnes (`colsample_bytree=0.8`). Référence de l'état de l'art sur les données tabulaires structurées.

### 5. LightGBM
Architecture de gradient boosting similaire à XGBoost mais avec une stratégie de croissance des arbres **par feuille** (*leaf-wise*) plutôt que par niveau (*level-wise*). Cela le rend significativement plus rapide tout en atteignant des performances équivalentes voire supérieures, notamment sur les datasets à grande dimensionnalité. **Retenu comme modèle final** après comparaison systématique.

---

##  Feature Engineering — Détail

### Variables sources (colonnes de base)

| Variable | Description |
|---|---|
| `eurtnd` | Taux EUR/TND — variable **cible** |
| `eur_usd` | Taux de change Euro / Dollar américain |
| `brent_oil` | Prix du pétrole Brent (USD/baril) |
| `gold` | Prix de l'or (USD/once troy) |
| `eur_gbp` | Taux Euro / Livre sterling |
| `us_10y_yield` | Rendement des obligations du Trésor américain à 10 ans |
| `msci_em` | Indice MSCI Marchés Émergents |
| `balance_commerciale` | Balance commerciale tunisienne mensuelle |

### Features construites

| Catégorie | Features générées | Rôle |
|---|---|---|
| **Lags** | `*_lag1`, `*_lag3`, `*_lag7`, `*_lag30` pour chaque colonne source | Capturer l'autocorrélation et les dépendances temporelles |
| **Moyennes mobiles** | `eurtnd_ma7`, `eurtnd_ma14`, `eurtnd_ma30` | Représenter la tendance court et moyen terme |
| **Volatilité glissante** | `eurtnd_std7`, `eurtnd_std14`, `eurtnd_std30` | Mesurer l'incertitude et la turbulence du marché |
| **Momentum** | Variation absolue et relative J vs J-1 | Capturer l'accélération ou la décélération du taux |
| **Calendrier** | Jour de la semaine, numéro du mois, trimestre | Saisonnalité et effets de calendrier |
| **Proxy DXY** | `dxy_proxy = 1 / eur_usd` | Force relative du dollar américain |

> Toutes les features de lags et statistiques glissantes sont calculées avec un décalage de 1 jour (`shift(1)`) afin d'**éviter tout data leakage** : le modèle ne dispose jamais d'informations du jour J lors de la prédiction de J.

---

##  Métriques d'Évaluation

| Métrique | Formule | Interprétation |
|---|---|---|
| **MAE** | `mean(|y - ŷ|)` | Erreur absolue moyenne en TND — la plus intuitive et robuste aux outliers |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Pénalise davantage les grandes erreurs ; sensible aux pics d'erreur |
| **MAPE** | `mean(|y - ŷ| / y) × 100` | Erreur relative en % — indépendante de l'échelle du taux |
| **Δ MAE vs baseline** | `(1 - MAE_model / MAE_baseline) × 100` | Gain en % par rapport à la prédiction naïve |

La **MAE est la métrique principale** de sélection des modèles car elle est directement interprétable en dinars tunisiens et robuste aux valeurs extrêmes.

---

##  Résultats & Comparaison

Le tableau suivant présente les performances de tous les modèles, avant et après optimisation des hyperparamètres :

| Modèle | MAE (TND) | RMSE (TND) | MAPE (%) | Δ vs Baseline (%) |
|---|---|---|---|---|
| Baseline naïve | 0.009811 | 0.017180 | 0.293482|  0.00 |
| Ridge Regression | 0.001658 | 0.002368  |0.049618 |  83.10 |
| Random Forest | 0.008924 | 0.012363 | 0.266648 | 9.03|
| XGBoost |  0.005195|0.007172 |0.155133 | 47.05 |
| XGBoost (optimisé) | 0.004813 |0.006684|0.143652 | 50.94 |
| LightGBM | 0.005577 | 0.007510 | 0.166530  | 43.15 |
| LightGBM (optimisé) | 0.004929 | 0.006775 | 0.147136 | 49.76 |

> Meilleur modèle : Ridge
   MAE  : 0.00166 TND
   MAPE : 0.0496 %
   Amélioration vs baseline : 83.1%

> Les valeurs numériques exactes sont affichées dans la cellule 35 du notebook (`BILAN COMPLET — TOUS LES MODÈLES`). Les résultats dépendent de la date d'exécution car les données EUR/TND sont téléchargées en temps réel.

**Conclusion principale :** Les modèles ensemblistes (XGBoost, LightGBM) surpassent significativement la baseline naïve et la régression Ridge. LightGBM optimisé est systématiquement le plus performant et est sélectionné comme modèle de production.

---

##  Importance des Features

D'après l'analyse comparée des importances des trois modèles à base d'arbres (Random Forest, XGBoost optimisé, LightGBM optimisé) :

**Tier 1 — Très haute importance**
- `eurtnd_lag1`, `eurtnd_lag3`, `eurtnd_lag7` — La série EUR/TND présente une **forte autocorrélation** : le passé récent est le meilleur prédicteur du futur proche.
- `eurtnd_ma7`, `eurtnd_ma14` — Les moyennes mobiles capturent la **tendance court-terme** et lissent le bruit quotidien.

**Tier 2 — Haute importance**
- `eur_usd` et ses lags — Principal déterminant de la force de l'Euro face aux devises émergentes.
- `brent_oil` et ses lags — La Tunisie étant importatrice nette d'énergie, le prix du pétrole impacte directement la balance des paiements et la demande en devises étrangères.
- `gold` — Indicateur de l'appétit mondial pour le risque : une hausse de l'or signale généralement une fuite vers les valeurs refuge et affecte les devises émergentes.

**Tier 3 — Importance modérée**
- `us_10y_yield` — Proxy de l'aversion au risque global et des flux de capitaux vers les marchés émergents (effet de carry trade).
- `eur_gbp` — Indicateur indirect de la santé économique de la zone Euro.
- `msci_em` — Contexte des marchés émergents ; le TND suit parfois les tendances de cette classe d'actifs.
- `balance_commerciale` — Impact plus marqué à moyen terme qu'à court terme sur le taux de change.

---

##  Hyperparamètres Optimisés

### Grille de recherche XGBoost

| Hyperparamètre | Valeurs testées | Rôle |
|---|---|---|
| `n_estimators` | 200, 300, 500, 700 | Nombre d'arbres |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1 | Taux d'apprentissage (shrinkage) |
| `max_depth` | 3, 4, 5, 6 | Profondeur maximale de chaque arbre |
| `subsample` | 0.6, 0.7, 0.8, 0.9 | Fraction des observations par arbre |
| `colsample_bytree` | 0.6, 0.7, 0.8, 0.9 | Fraction des features par arbre |
| `min_child_weight` | 3, 5, 7, 10 | Poids minimum dans un noeud feuille |
| `reg_alpha` | 0.0, 0.05, 0.1, 0.5 | Régularisation L1 |
| `reg_lambda` | 0.5, 1.0, 2.0, 5.0 | Régularisation L2 |

### Grille de recherche LightGBM

| Hyperparamètre | Valeurs testées | Rôle |
|---|---|---|
| `n_estimators` | 200, 300, 500, 700 | Nombre d'arbres |
| `learning_rate` | 0.01, 0.03, 0.05, 0.1 | Taux d'apprentissage |
| `num_leaves` | 15, 31, 63, 127 | Nombre de feuilles par arbre (complexité) |
| `max_depth` | -1, 5, 8, 12 | Profondeur max (-1 = illimitée) |
| `min_child_samples` | 10, 20, 40 | Nb min d'observations par feuille |
| `subsample` | 0.6, 0.7, 0.8, 0.9 | Sous-échantillonnage des lignes |
| `colsample_bytree` | 0.6, 0.7, 0.8, 0.9 | Sous-échantillonnage des colonnes |
| `reg_alpha` | 0.0, 0.05, 0.1, 0.5 | Régularisation L1 |
| `reg_lambda` | 0.5, 1.0, 2.0, 5.0 | Régularisation L2 |

> **Méthode :** `RandomizedSearchCV` avec 30 tirages aléatoires. Validation croisée avec `TimeSeriesSplit(n_splits=5)`. Scoring : MAE négative (`neg_mean_absolute_error`).

---

##  Validation Walk-Forward

La validation walk-forward simule les **conditions réelles de déploiement** du modèle en production :

```
Itération 1 :
|←─────────────── Train (≥ 3 ans) ───────────────→|←─ Test 30j ─→|

Itération 2 (avance de 30 jours) :
|←──────────────── Train (≥ 3 ans + 30j) ─────────────────→|←─ Test 30j ─→|

Itération 3 :
|←─────────────────── Train (≥ 3 ans + 60j) ─────────────────────→|←─ Test 30j ─→|

... (répété jusqu'à la fin du dataset)
```

À chaque itération, le modèle LightGBM est **réentraîné de zéro** sur tout l'historique disponible, puis évalué sur les 30 jours suivants. Cette approche :
- Évite le biais d'optimisme lié à une évaluation statique unique
- Fournit une **distribution de MAE par fenêtre temporelle** permettant d'identifier les périodes de marché difficiles (crises, ruptures structurelles)
- Reflète fidèlement ce que l'on observerait en production réelle

---

##  Prédiction du Lendemain

Le modèle final (LightGBM réentraîné sur l'intégralité du dataset 2015–2025) produit :

- **ŷ[T+1]** — le taux EUR/TND prédit pour le lendemain
- **IC 68%** — intervalle de confiance basé sur les percentiles 16 et 84 des résidus du test
- **IC 90%** — intervalle de confiance basé sur les percentiles 5 et 95 des résidus du test

Les intervalles sont calculés de façon **empirique et non paramétrique** à partir de la distribution des erreurs observées sur l'ensemble de test. Cette approche est robuste à la non-normalité des résidus, fréquente sur les séries financières (queues épaisses, asymétrie).

---

##  Décisions Méthodologiques Clés

**Pourquoi pas de split aléatoire ?**
Un split aléatoire sur une série temporelle introduit du **data leakage** : des observations futures contaminent l'ensemble d'entraînement, ce qui gonfle artificiellement les performances et donne une vision irréaliste de la capacité prédictive. On utilise systématiquement une **séparation chronologique stricte** (les données de test sont toujours postérieures aux données d'entraînement).

**Pourquoi TimeSeriesSplit pour la validation croisée ?**
Le k-fold classique mélange les observations dans le temps, violant la causalité. `TimeSeriesSplit` garantit que chaque fold de validation est toujours **postérieur** au fold d'entraînement, ce qui donne une estimation non biaisée de la performance hors-échantillon.

**Pourquoi standardiser uniquement pour Ridge ?**
Ridge Regression est sensible à l'échelle des features car le terme de pénalisation L2 traite toutes les features de manière identique sans tenir compte de leur magnitude. Les modèles à base d'arbres (Random Forest, XGBoost, LightGBM) sont **invariants par rapport à la mise à l'échelle** des features et n'en ont pas besoin.

**Pourquoi un forward-fill limité à 5 jours ?**
Les marchés financiers sont fermés le week-end et certains jours fériés. Un forward-fill limité à 5 jours permet de propager la dernière valeur connue sur ces jours de fermeture sans risque de combler de véritables trous de données (une limite de 5 jours couvre un week-end + quelques jours fériés consécutifs au maximum).

---

##  Limites & Perspectives

### Limites actuelles

- **Régimes de crise :** Le modèle est entraîné sur une période incluant la pandémie COVID-19 (2020) et la forte dépréciation du TND (2022). Ces épisodes de rupture structurelle peuvent réduire la stabilité des prédictions sur des crises futures inédites.
- **Horizon court :** La performance se dégrade rapidement au-delà de T+1. Le modèle n'est pas conçu pour des prévisions multi-jours ou multi-semaines.
- **Données absentes :** Le modèle n'intègre pas les données de sentiment de marché (NLP sur flux d'actualités), les décisions de politique monétaire de la BCT ou de la BCE, ni les données politiques et géopolitiques.
- **Stationnarité implicite :** Aucun test formel de stationnarité (ADF, KPSS) n'est appliqué ; les lags et différences jouent implicitement ce rôle mais sans garantie.
- **Données en temps réel :** Les résultats numériques exacts varient selon la date d'exécution car les données EUR/TND sont téléchargées dynamiquement.

### Pistes d'amélioration

- Tester des architectures spécialisées en séries temporelles : ARIMAX, Prophet, N-BEATS, Temporal Fusion Transformer (TFT)
- Intégrer des features de **sentiment de marché** via l'analyse NLP de tweets financiers ou d'articles de presse économique
- Expérimenter le **stacking / blending** des quatre modèles pour réduire la variance de prédiction
- Ajouter une composante de **détection de changement de régime** (Hidden Markov Model, CUSUM)
- Déployer le modèle final via une **API REST** (FastAPI + MLflow) pour des prédictions automatisées en temps réel

---

##  Installation & Exécution

### Prérequis

- Python ≥ 3.9
- Jupyter Notebook ou JupyterLab
- Connexion Internet (pour le téléchargement des données via `yfinance`)

### Installation des dépendances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm yfinance
```

Ou via un fichier `requirements.txt` :

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
lightgbm>=3.3.0
yfinance>=0.2.0
```

```bash
pip install -r requirements.txt
```

### Ordre d'exécution

```bash
# Étape 1 — Analyse exploratoire (génère macro_market_merged.csv)
jupyter notebook 01_EDA.ipynb

# Étape 2 — Modélisation complète
jupyter notebook 02_Modeling.ipynb
```

>  **Important :** La cellule 1.1 télécharge le taux EUR/TND en temps réel via Yahoo Finance (`EURTND=X`). Une connexion Internet est indispensable. Les données sont récupérées du `2015-01-01` au `2025-01-02`.

### Temps d'exécution estimé

| Étape | Durée estimée |
|---|---|
| Chargement & Feature Engineering | ~30 secondes |
| Random Forest (300 arbres) | ~1–2 minutes |
| XGBoost (500 estimateurs) | ~1 minute |
| LightGBM (500 estimateurs) | ~30 secondes |
| RandomizedSearchCV XGBoost (30 iter.) | ~3–5 minutes |
| RandomizedSearchCV LightGBM (30 iter.) | ~2–4 minutes |
| Validation Walk-Forward | ~2–3 minutes |
| **Total estimé** | **~12–20 minutes** |

---

##  Licence

Ce projet a été développé dans un cadre académique. Les données sont issues d'APIs financières publiques (Yahoo Finance). Aucune utilisation commerciale n'est autorisée sans accord préalable des auteurs.
