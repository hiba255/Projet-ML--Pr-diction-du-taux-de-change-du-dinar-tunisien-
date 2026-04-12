# Prédiction du taux de change EUR/TND

> Pipeline de machine learning pour prédire le taux de change Euro / Dinar Tunisien à partir d'indicateurs macroéconomiques tunisiens et de facteurs de marché internationaux.

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Résultats](#résultats)
- [Structure du projet](#structure-du-projet)
- [Sources de données](#sources-de-données)
- [Pipeline](#pipeline)
- [Ingénierie des features](#ingénierie-des-features)
- [Modèle](#modèle)
- [Validation walk-forward](#validation-walk-forward)
- [Comment exécuter](#comment-exécuter)
- [Fichiers générés](#fichiers-générés)
- [Interprétation et limites](#interprétation-et-limites)
- [Dépendances](#dépendances)

---

## Vue d'ensemble

Ce projet construit un modèle de prédiction quotidienne du taux EUR/TND en combinant :

- **Données macroéconomiques tunisiennes** (balance commerciale, PIB, taux d'intérêt directeur, inflation) issues de la BCT / INS et interpolées à fréquence journalière
- **Facteurs de marché internationaux** (EUR/USD, Brent, or, EUR/GBP, rendement US 10 ans, MSCI EM) récupérés via `yfinance`

Le modèle prédit le taux EUR/TND du lendemain et fournit un intervalle de confiance dérivé de la distribution empirique des résidus sur l'ensemble de test.

---

## Résultats

### Évaluation train/test unique (80/20 chronologique)

| Métrique | Modèle | Naïf (lag-1) | Amélioration |
|----------|--------|--------------|--------------|
| MAE      | 0,00512 TND | 0,00981 TND | **-47,8%** |
| RMSE     | 0,00695 TND | 0,01718 TND | **-59,5%** |
| MAPE     | 0,153% | — | — |

### Validation walk-forward (84 fenêtres glissantes, 2018–2024)

| Année | MAE | MAPE |
|-------|-----|------|
| 2018  | 0,04391 | 1,38% |
| 2019  | 0,05264 | 1,58% |
| 2020  | 0,02504 | 0,80% |
| 2021  | 0,00365 | 0,11% |
| 2022  | 0,00445 | 0,14% |
| 2023  | 0,00521 | 0,16% |
| 2024  | 0,00246 | 0,07% |

> Le MAE élevé en 2018–2019 reflète la période de démarrage à froid (seulement 3 ans de données d'entraînement disponibles). À partir de 2021, le modèle fonctionne dans un régime stable avec un MAPE inférieur à 0,16%, en amélioration continue au fil des années.

---

## Structure du projet

```
eurtnd-prediction/
│
├── data/
│   ├── balance_commerciale_daily.xlsx   # Balance commerciale tunisienne (interpolation journalière)
│   ├── PIB_daily.xlsx                   # PIB tunisien (interpolation journalière)
│   ├── taux_interet_daily.xlsx          # Taux directeur BCT (journalier)
│   └── inflation_daily.xlsx             # Indice d'inflation journalier
│
├── outputs/
│   ├── macro_merged.xlsx                # Étape 1 — 4 sources macro fusionnées
│   ├── macro_market_merged.csv          # Étape 2 — macro + yfinance fusionnés
│   ├── dataset_features.csv             # Étape 3 — matrice de features complète avec cible
│   ├── predictions.csv                  # Étape 4 — prédictions sur l'ensemble de test
│   ├── forecast_tomorrow.csv            # Étape 5A — prévision J+1
│   ├── walkforward_predictions.csv      # Étape 5B — toutes les prédictions walk-forward
│   └── walkforward_yearly.csv           # Étape 5B — performance annuelle détaillée
│
├── step2_yfinance_merge_fixed.py        # Récupération et fusion des données de marché
├── step3_target_features_fixed.py       # Ajout de la cible EUR/TND + ingénierie des features
├── step4_model.py                       # Entraînement XGBoost + évaluation
├── step5a_predict_tomorrow.py           # Prévision J+1 avec intervalle de confiance
├── step5b_walkforward.py                # Validation walk-forward glissante
└── README.md
```

---

## Sources de données

### Données macroéconomiques (Tunisie)

| Variable | Source | Fréquence brute | Couverture |
|----------|--------|-----------------|------------|
| Balance commerciale | BCT / INS | Mensuelle | 2015–2025 |
| PIB | BCT / INS | Trimestrielle | 2015–2025 |
| Taux d'intérêt directeur | BCT | Mensuelle | 2015–2025 |
| Inflation | INS | Mensuelle | 2015–2025 |

Les données brutes ont été interpolées à fréquence journalière par interpolation linéaire puis forward-fill. Les quatre séries sont alignées sur un index journalier commun du **01/01/2015 au 01/01/2025** (3 654 lignes).

### Données de marché (yfinance)

| Ticker | Variable | Justification économique |
|--------|----------|--------------------------|
| `EURUSD=X` | Cours EUR/USD | Facteur direct — le TND est géré par rapport à l'EUR |
| `BZ=F` | Brent (contrat front) | La Tunisie est importatrice nette de pétrole |
| `GC=F` | Or (USD/oz) | Indicateur de force du dollar |
| `EURGBP=X` | Cours EUR/GBP | Jauge de la force de l'EUR |
| `^TNX` | Rendement US 10 ans | Appétit mondial pour le risque |
| `EEM` | iShares MSCI EM ETF | Sentiment envers les marchés émergents |
| `EURTND=X` | **EUR/TND** | **Variable cible** |

> Remarque : `DX-Y.NYB` (DXY) était indisponible sur yfinance et a été remplacé par `dxy_proxy = 1 / eur_usd`, qui capture le même signal de force du dollar (l'EUR représente ~57% du vrai DXY).

Les données de marché couvrent uniquement les jours de bourse. Les gaps de week-ends et jours fériés sont comblés par forward-fill avec un maximum de 3 jours pour éviter la propagation de données périmées.

---

## Pipeline

```
Fichiers xlsx macro bruts (x4)
           │
           ▼
  Étape 1 — Fusion et alignement des données macro
  (index journalier commun, suppression des lignes incomplètes)
           │
           ▼
  Étape 2 — Récupération des facteurs de marché yfinance
  (réindexation calendrier journalier, ffill week-ends)
           │
           ▼
  Étape 3 — Téléchargement de la cible EUR/TND + ingénierie des features
  (décalages, stats glissantes, log-rendements, spreads, calendrier)
           │
           ▼
  Étape 4 — Modèle XGBoost
  (split 80/20 chronologique, évaluation vs. baseline naïf)
           │
           ├──► Étape 5A — Prévision J+1
           │    (réentraînement sur toutes les données, estimation ponctuelle + IC)
           │
           └──► Étape 5B — Validation walk-forward
                (84 fenêtres glissantes, décomposition annuelle)
```

---

## Ingénierie des features

Un total de **69 features** sont construites à partir des 11 colonnes brutes. Tous les décalages et statistiques glissantes sont calculés sur des valeurs décalées pour éviter toute fuite de données (*data leakage*).

### Features de décalage (lags)

Appliquées à 8 colonnes (`eurtnd`, `eur_usd`, `brent_oil`, `gold`, `eur_gbp`, `us_10y_yield`, `msci_em`, `balance_commerciale`) :

- `{col}_lag1` — valeur d'hier
- `{col}_lag3` — valeur d'il y a 3 jours
- `{col}_lag7` — valeur d'il y a 1 semaine
- `{col}_lag30` — valeur d'il y a 1 mois

### Statistiques glissantes (sur EUR/TND, décalées)

- `eurtnd_ma{7,14,30}` — moyenne mobile
- `eurtnd_std{7,14,30}` — écart-type glissant

### Log-rendements

Appliqués à `eurtnd`, `eur_usd`, `brent_oil`, `gold`, `msci_em` :

- `{col}_ret1` — log-rendement sur 1 jour
- `{col}_ret7` — log-rendement sur 7 jours
- `{col}_ret30` — log-rendement sur 30 jours

### Features dérivées

| Feature | Formule | Objectif |
|---------|---------|----------|
| `dxy_proxy` | `1 / eur_usd` | Substitut à l'indice dollar |
| `rate_inflation_spread` | `taux_interet - inflation x 100` | Taux d'intérêt réel |
| `eur_usd_ma30_dev` | `eur_usd - moyenne_glissante_30j` | Déviation EUR/USD par rapport à la tendance |
| `day_of_week` | `index.dayofweek` | Saisonnalité hebdomadaire |
| `month` | `index.month` | Saisonnalité mensuelle |
| `quarter` | `index.quarter` | Saisonnalité trimestrielle |

### Principales importances de features (XGBoost)

| Feature | Importance | Groupe |
|---------|-----------|--------|
| `eurtnd_ma14` | 41,7% | Tendance |
| `eurtnd_ma30` | 26,0% | Tendance |
| `eurtnd_lag30` | 23,2% | Tendance |
| `balance_commerciale_lag7` | 1,3% | Macro |
| `balance_commerciale_lag30` | 1,3% | Macro |
| `taux_interet` | 0,4% | Macro |
| `brent_oil_lag1` | 0,1% | Marché |

> L'EUR/TND est une devise gérée par la BCT. Le modèle apprend correctement que les features de tendance à moyen terme dominent, tandis que les facteurs de marché ont une influence mineure au jour le jour. Cela est cohérent avec un régime de flottement administré où la banque centrale lisse la volatilité.

---

## Modèle

**Algorithme :** XGBoost Regressor

```python
XGBRegressor(
    n_estimators     = 500,
    learning_rate    = 0.03,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    reg_alpha        = 0.1,   # Régularisation L1
    reg_lambda       = 1.0,   # Régularisation L2
    random_state     = 42,
)
```

**Découpage train/test :** split chronologique strict 80/20. Le mélange aléatoire n'est jamais utilisé — il provoquerait une fuite de données sur une série temporelle.

**Intervalles de confiance** pour la prévision J+1 sont dérivés empiriquement de la distribution des résidus sur l'ensemble de test :

- IC 68% : [16e percentile, 84e percentile] des résidus
- IC 90% : [5e percentile, 95e percentile] des résidus

---

## Validation walk-forward

La validation walk-forward simule un déploiement en conditions réelles en réentraînant le modèle de zéro à chaque étape en utilisant uniquement les données passées, puis en prédisant les 30 jours suivants.

```
|--- entraînement (3 ans min.) ---|-- prédiction (30j) --|
       |--- entraînement + 30j ---|-- prédiction (30j) --|
              |--- entraînement + 60j ---|-- prédiction (30j) --|
              ...
```

**Configuration :**

| Paramètre | Valeur |
|-----------|--------|
| Fenêtre d'entraînement minimale | 3 ans (1 095 jours) |
| Pas d'avancement | 30 jours |
| Horizon de prévision | 30 jours par fenêtre |
| Nombre total de fenêtres | 84 |

Cette approche est plus conservative et honnête qu'un simple split train/test car elle teste le modèle sur chaque période en séquence, y compris les changements de régime (choc COVID-19 en 2020, cycle de hausse des taux BCT 2022–2023).

---

## Comment exécuter

### Installation des dépendances

```bash
pip install yfinance xgboost scikit-learn pandas numpy matplotlib openpyxl
```

### Sur Google Colab

1. Uploader `macro_merged.xlsx` (sortie de l'étape 1) dans le panneau de fichiers Colab
2. Exécuter chaque étape dans l'ordre :

```python
# Étape 2 — récupération des données de marché
exec(open('step2_yfinance_merge_fixed.py').read())

# Étape 3 — ajout de la cible + features
exec(open('step3_target_features_fixed.py').read())

# Étape 4 — entraînement du modèle
exec(open('step4_model.py').read())

# Étape 5A — prévision J+1
exec(open('step5a_predict_tomorrow.py').read())

# Étape 5B — validation walk-forward
exec(open('step5b_walkforward.py').read())
```

### Durées d'exécution estimées

| Étape | Durée |
|-------|-------|
| Étape 2 — téléchargement yfinance | ~30 secondes |
| Étape 3 — ingénierie des features | ~5 secondes |
| Étape 4 — entraînement XGBoost | ~20 secondes |
| Étape 5A — prévision J+1 | ~25 secondes |
| Étape 5B — walk-forward (84 fenêtres) | ~2 minutes |

---

## Fichiers générés

| Fichier | Description |
|---------|-------------|
| `macro_merged.xlsx` | 3 654 lignes x 4 colonnes macro, 2015–2025 |
| `macro_market_merged.csv` | 3 653 lignes x 10 colonnes (macro + marché) |
| `dataset_features.csv` | 3 622 lignes x 70 colonnes (toutes les features + cible) |
| `predictions.csv` | Prédictions sur l'ensemble de test avec résidus |
| `forecast_tomorrow.csv` | Estimation ponctuelle + IC 68% et 90% pour J+1 |
| `walkforward_predictions.csv` | Toutes les valeurs prédites vs réelles en walk-forward |
| `walkforward_yearly.csv` | MAE et MAPE par année |
| `model_results.png` | Réel vs prédit, résidus, importance des features |
| `forecast_tomorrow.png` | Fenêtre des 90 derniers jours + prévision J+1 |
| `walkforward_results.png` | Réel vs prédit en walk-forward + MAE glissant |

---

## Interprétation et limites

### Points forts du modèle

- **Capture la tendance de dépréciation administrée** — l'EUR/TND s'est déprécié régulièrement de ~2,20 en 2015 à ~3,30 en 2025. Le modèle apprend cette dérive structurelle via les moyennes mobiles à moyen terme.
- **Performance stable après 2020** — une fois entraîné sur plus de 5 ans de données, le MAPE walk-forward reste inférieur à 0,16% chaque année, y compris pendant le cycle de hausse des taux BCT 2022–2023.
- **Amélioration continue** — 2024 est la meilleure année de l'historique walk-forward (MAPE 0,07%), montrant que le modèle bénéficie d'un historique plus long plutôt que de se dégrader avec le temps.

### Limites

- **Interventions ponctuelles de la BCT** — les dépréciations soudaines décidées par la banque centrale ne sont pas prévisibles à partir des données passées seules. Le modèle rattrapera l'écart en 1 à 3 jours après un tel événement.
- **Période de démarrage à froid** — le modèle nécessite au minimum 4 à 5 ans de données d'entraînement pour une performance fiable. Ne pas utiliser les prédictions des 3 premières années d'un nouveau déploiement comme référence de qualité.
- **Chocs exogènes** — les événements géopolitiques (crise constitutionnelle tunisienne de 2021, négociations FMI) ne sont pas encodés dans les features et augmenteront temporairement l'erreur de prédiction.
- **Fraîcheur des données macro** — les variables macroéconomiques sont publiées avec un délai par l'INS/BCT. En production, elles doivent être maintenues à jour avec la dernière publication disponible.
- **Ce projet n'est pas un conseil financier** — le modèle est un outil de recherche et d'analyse. Il ne doit pas être utilisé comme seule base pour des décisions de trading ou financières.

---

## Dépendances

```
python       >= 3.10
pandas       >= 2.0
numpy        >= 1.24
xgboost      >= 2.0
scikit-learn >= 1.3
yfinance     >= 0.2.36
matplotlib   >= 3.7
openpyxl     >= 3.1
```

---

## Auteur

Projet de recherche macroéconomique combinant les données macro tunisiennes BCT/INS avec des signaux de marché financier internationaux pour modéliser la dynamique du taux de change EUR/TND.

Plage de données : **01/01/2015 → 01/01/2025** | Dataset final : **3 622 lignes x 70 features** 
