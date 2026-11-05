# Projet SDP – Jalon 3 : apprentissage MR-Sort face à un agent malicieux

A observe des objets classés par un vrai modèle MR-Sort `M_A` et les transmet à B. Pour chaque donnée de A, un agent
malicieux ajoute une fausse donnée. B reçoit le mélange, sans savoir quelles données sont fausses, et cherche à
retrouver `M_A`.

## Fichiers

| fichier | rôle |
|---|---|
| `brouillage.py` | stratégies de l'agent malicieux, choix du modèle par B, une expérience complète |
| `run_brouillage.py` | toutes les expériences → `results/brouillage.csv` (log dans `results/log_brouillage.txt`) |
| `jalon3.ipynb` | notebook de synthèse : tableaux, graphiques, conclusions (généré par `make_notebook.py`) |

Le code réutilise les briques du jalon 2 (`../jalon2_MRSort/mrsort_lincs.py` et `two_mrsort.py`).

```bash
cd jalon3_MRSort
../.venv/bin/python run_brouillage.py 3        # 3 instances par attaque (5 pour l'agent basique), ~20 min sur 4 cœurs
../.venv/bin/python make_notebook.py
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace jalon3.ipynb
```

## Protocole

- `M_A` aléatoire (6 critères, 3 classes). A envoie 300 objets uniformes sur `[0,1]^6`, classés par `M_A`. L'agent
  malicieux en ajoute 300, et B reçoit les 600 mélangés.
- Stratégies de B :
  - **un seul MR-Sort** appris avec lincs sur tout ce qu'il reçoit ;
  - **deux MR-Sort** avec l'heuristique inv-2-MR-Sort du jalon 2 ; B garde celui qui classe le mieux 20 **objets de
    confiance** ;
  - pour référence, un MR-Sort appris sur les **20 objets de confiance seuls**.
- **Pourquoi des objets de confiance** : avec autant de vraies que de fausses données, les deux segments jouent des rôles
  symétriques. Sans information en plus, B retrouve deux modèles mais ne peut pas savoir lequel vient de A. On suppose donc
  que B connaît 20 objets envoyés à coup sûr par A.
- **Mesure** : la fidélité, c'est-à-dire la proportion de 5000 objets de test classés comme par `M_A` (la « qualité »
  du jalon 1). Le plafond est un MR-Sort appris sur les seules données de A. La pureté est la part de vraies données
  dans le segment retenu par B.
- Pour une même instance, `M_A` et les données de A sont identiques d'une attaque à l'autre.

## Stratégies de brouillage

| attaque | principe |
|---|---|
| `modele_aleatoire` | **agent basique (partie 1)** : MR-Sort tiré au hasard, objets uniformes classés par ce modèle |
| `etiquettes_aleatoires` | objets uniformes, classe tirée au hasard |
| `frontiere_d` | modèle `M_E` = `M_A` avec les frontières relevées de `d` ; seulement des objets où `M_E` et `M_A` ne sont pas d'accord (juste au-dessus des frontières de `M_A`, avec la classe inférieure) |
| `copies_contradictoires` | chaque objet de A est recopié (bruit de 0,01) avec une classe voisine |

## Résultats (fidélité à `M_A`, moyenne sur 3 instances, 5 pour l'agent basique)

| attaque | un MR-Sort | deux MR-Sort + confiance | 20 objets de confiance seuls | pureté du segment retenu |
|:---|---:|---:|---:|---:|
| aucune | 0,972 | – | 0,659 | – |
| modele_aleatoire (partie 1) | 0,631 | **0,909** | 0,643 | 0,80 |
| etiquettes_aleatoires | **0,816** | 0,554 | 0,659 | 0,56 |
| frontiere_0.05 | **0,806** | 0,708 | 0,659 | 0,54 |
| frontiere_0.10 | 0,685 | 0,668 | 0,659 | 0,69 |
| frontiere_0.20 | 0,577 | **0,730** | 0,659 | 0,94 |
| copies_contradictoires | 0,536 | 0,536 | **0,659** | 0,57 |

Plafond (MR-Sort appris sur les seules données de A) : 0,97.

### Partie 1 : agent basique
Avec un seul MR-Sort, B est très dégradé (0,63). L'heuristique à deux modèles du jalon 2 sépare bien les données de A de
celles de l'agent malicieux. Avec les objets de confiance pour choisir le bon modèle, B remonte à **0,91** (plafond 0,97).
Le brouillage par un MR-Sort aléatoire est donc peu efficace contre un B qui s'attend à deux sources.

### Partie 2 : quelle stratégie de brouillage ?
- **Aucune stratégie de B ne gagne partout.** Contre des étiquettes aléatoires, un seul MR-Sort résiste (0,82), mais
  l'heuristique à deux modèles s'effondre (0,55). Le bruit ne suit aucun MR-Sort, donc la partition trouvée n'a pas de
  sens. B devrait d'abord tester s'il y a vraiment deux modèles.
- **Attaque près de la frontière** : avec un petit décalage (0,05 à 0,10), les fausses données sont presque impossibles à
  distinguer des vraies (pureté 0,54 à 0,69). Le modèle de B tombe vers 0,67 à 0,81. Avec un décalage plus grand (0,20),
  un seul MR-Sort souffre davantage (0,58), mais le brouillage redevient séparable (pureté 0,94) et l'heuristique remonte
  en partie (0,73, entre 0,52 et 0,89 selon l'instance).
- **Copies contradictoires : la meilleure attaque trouvée.** Chaque vraie donnée a un double presque identique avec une
  classe voisine. Les deux stratégies de B tombent à **0,54**, moins bien que s'il n'utilisait que ses 20 objets de
  confiance (0,66). Rien dans les performances ne distingue la vraie donnée de sa copie.

### Limites et pistes
- Il n'y a que 3 à 5 instances par attaque. La variance est visible, en particulier pour `frontiere_0.20`.
- L'attaque près de la frontière suppose que l'agent connaît `M_A`. Il peut l'estimer à partir des données de A, qu'il
  intercepte. Les copies contradictoires supposent qu'il voit les données de A.
- Pistes pour B :
  - choisir entre un modèle, deux modèles et les objets de confiance seuls, selon la précision obtenue sur les objets de
    confiance ;
  - repérer les paires d'objets presque identiques de classes différentes, puis les écarter ou les départager avec les
    objets de confiance.
