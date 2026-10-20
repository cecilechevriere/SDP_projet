# Projet SDP – Apprentissage de modèles MR-Sort

## Jalon 1 – Programmation mathématique ([jalon1_MRSort/](jalon1_MRSort/))
Apprentissage exact d'un MR-Sort (frontières `b`, poids `w`, seuil `λ`) par un MIP résolu avec Gurobi, en maximisant la marge de séparation.
- `mrsort_gurobi.py` : modèle, classification et protocole de test.
- `run_dataset0.py` : apprentissage sur Dataset0, 100 % des 60 objets restitués.
- Résultats : qualité moyenne en test de 0,95 (p = 2), 0,91 (p = 3) et 0,88 (p = 4).

## Jalon 2 – Heuristiques : lincs et inv-2-MR-Sort ([jalon2_MRSort/](jalon2_MRSort/))
1. **Un seul MR-Sort** appris avec l'heuristique de lincs (`mrsort_lincs.py`).
2. **Deux segments de clientèle** (`two_mrsort.py`) : heuristique alternée, sur le principe de k-means. On répète « apprendre M1 et M2 avec lincs » puis « réaffecter chaque produit au modèle qui l'explique le mieux », avec 5 redémarrages, un critère d'arrêt par patience et un choix final par parcimonie.
- Résultats sur le supermarket (6 critères, 3 classes, 24 fichiers) : 0,67 de bien classés en test avec un seul modèle, contre **0,92** avec l'heuristique à deux segments.
- Détails dans [jalon2_MRSort/README.md](jalon2_MRSort/README.md) et le notebook `jalon2.ipynb`.

## Installation
```bash
python3.11 -m venv .venv
.venv/bin/pip install gurobipy lincs numpy pandas matplotlib
```
