# MLOps Lab 01 — Prédiction du churn

## Présentation

Ce projet est un laboratoire MLOps de bout en bout autour de la prédiction du
départ de clients (« churn »). Il couvre la génération et la préparation des
données, l'entraînement et l'évaluation d'un modèle, son enregistrement et son
activation, le service de prédiction ainsi qu'une surveillance simple du drift.

Le jeu de données est synthétique et reproductible (graine par défaut : `42`).
Chaque ligne décrit un client avec les variables `tenure_months`,
`num_complaints`, `avg_session_minutes`, `plan_type` et `region`; `churn` est la
cible binaire (`1` : départ, `0` : client conservé).

## Contenu du dossier

| Chemin | Rôle |
| --- | --- |
| `src/generate_data.py` | Génère le jeu brut synthétique (`data/raw.csv`). |
| `src/prepare_data.py` | Nettoie et contrôle les données, puis écrit `data/processed.csv`. |
| `src/train.py` | Crée les partitions train/validation/test, entraîne un pipeline de régression logistique, choisit le seuil sur la validation et enregistre un candidat. |
| `src/evaluate.py` | Évalue le candidat sur le test indépendant et écrit les métriques. |
| `src/register.py` | Applique le seuil de qualité F1 et, si le candidat est accepté, l'active dans le registre. |
| `src/api.py` | API FastAPI avec endpoints `/health` et `/predict`; les prédictions sont journalisées en JSON Lines. |
| `src/monitor_drift.py` | Compare les moyennes des variables numériques observées dans les logs aux statistiques d'entraînement. |
| `src/rollback.py` | Réactive un ancien modèle accepté et restaure ses métadonnées et statistiques. |
| `src/pipeline_utils.py` | Chemins, noms des variables, lecture/écriture JSON et calcul SHA-256 partagés. |
| `data/raw.csv` | Données synthétiques brutes. |
| `data/processed.csv` | Données nettoyées utilisées pour l'entraînement. |
| `data/splits/` | Partitions `train.csv`, `validation.csv` et `test.csv`, créées par l'entraînement. |
| `models/` | Candidat, alias du modèle actif (`model.joblib`) et modèles acceptés versionnés. |
| `registry/` | Métadonnées du candidat, historique des modèles, modèle courant et statistiques du scaler. |
| `reports/metrics.json` | Dernier rapport d'évaluation sur le jeu de test. |
| `logs/predictions.log` | Journal des prédictions de l'API, une entrée JSON par ligne. |
| `venv_mlops/` | Environnement virtuel Python local, non nécessaire au code source et à ne pas versionner. |

## Exécution

Depuis ce dossier, activez l'environnement fourni dans PowerShell :

```powershell
.\venv_mlops\Scripts\Activate.ps1
```

Le projet utilise Python avec `numpy`, `pandas`, `scikit-learn`, `joblib`,
`fastapi`, `pydantic` et `uvicorn`. Si l'environnement virtuel n'est pas
disponible, créez-en un et installez ces dépendances avec `pip`.

Exécutez les étapes dans l'ordre :

```powershell
python src\generate_data.py
python src\prepare_data.py
python src\train.py --version v3
python src\evaluate.py
python src\register.py --gate-f1 0.70
```

L'entraînement utilise une séparation stratifiée approximative de 60 % pour
l'entraînement, 15 % pour la validation et 25 % pour le test. Le seuil de
décision est choisi sur la validation; le test reste réservé à l'évaluation.
L'enregistrement accepte le candidat si son F1 de test atteint le seuil indiqué
et dépasse le F1 de la baseline.

Pour démarrer l'API depuis la racine du dossier :

```powershell
python -m uvicorn src.api:app --reload
```

Une fois lancée, l'API est disponible sur `http://127.0.0.1:8000`. La
documentation interactive FastAPI se trouve à `/docs`. `/health` indique si
un modèle courant est configuré. `/predict` reçoit les variables du client et
renvoie la classe prédite, sa probabilité, la latence et la version du modèle.
Les requêtes sont ajoutées à `logs/predictions.log`.

Après avoir collecté des prédictions, lancez le contrôle de drift :

```powershell
python src\monitor_drift.py
```

Il examine jusqu'aux 200 dernières requêtes et signale un drift si le score Z
d'une moyenne dépasse `2.5`. Pour revenir au modèle accepté précédent :

```powershell
python src\rollback.py
```

`rollback.py --target <nom-du-fichier-modèle>` permet de sélectionner
explicitement un modèle accepté figurant dans l'historique du registre.

## État des artefacts présents

Le registre contient deux versions acceptées (`v1` et `v2`); `v2` est le modèle
courant. Le dernier rapport enregistré porte sur 300 observations de test :
F1 `0.7081`, précision `0.5736`, rappel `0.9250`, pour une baseline F1 de
`0.6957`; le gate était fixé à `0.70`. Ces chiffres décrivent les artefacts
présents et seront remplacés lors d'une nouvelle exécution du pipeline.

À noter : l'entraînement et l'évaluation utilisent le seuil appris sur la
validation (actuellement `0.31` dans les métadonnées), tandis que l'endpoint
`/predict` applique actuellement un seuil fixe de `0.5`.