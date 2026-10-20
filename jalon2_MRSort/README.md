# Projet SDP – Jalon 2 : MR-Sort avec lincs, puis inv-2-MR-Sort

## Installation

lincs 2.0.2 n'a pas de wheel reconnu par le pip d'Anaconda (Python 3.9), d'où un environnement Python 3.11 dédié :

```bash
cd SDP_projet
python3.11 -m venv .venv          # Python 3.10 à 3.13
.venv/bin/pip install -r requirements.txt
cd jalon2_MRSort
unzip <chemin>/Datasets-Step2.zip   # datasets supermarket (non versionnés)
```

## Fichiers

| fichier | rôle |
|---|---|
| `mrsort_lincs.py` | partie 1 : chargement des données, apprentissage d'**un** MR-Sort avec lincs, classification numpy |
| `two_mrsort.py` | partie 2 : heuristique inv-2-MR-Sort, générateur de données à 2 segments, mesures |
| `run_synthetic.py` | validation sur données synthétiques (segments et modèles connus) → `results/synthetic.csv` |
| `run_supermarket.py` | expériences sur les datasets supermarket → `results/supermarket_*.csv`, modèles dans `results/models/` |
| `jalon2.ipynb` | notebook de démonstration et de synthèse (généré par `make_notebook.py`) |

## Données

`Datasets-Step2` : 72 fichiers au format lincs. `type_a` / `type_b` × `crXX_caYY` (XX critères dans [0, 1], YY catégories)
× `nNNNN` objets × `repAA-BB` (proportion des deux segments) × `noiseNN` (% d'étiquettes bruitées).
Les objets sont identiques entre `type_a` et `type_b`, seules les catégories changent. Le segment de chaque objet
n'est pas donné. Le `problem.yml` fourni est dans un ancien format (il manque `kind: classification-problem`), donc on
construit le problème lincs nous-mêmes.

## Partie 1 : un MR-Sort appris avec lincs

`learn_mrsort` utilise l'heuristique *weights-profiles-breed* de lincs. Une population de 9 modèles alterne trois étapes :
- optimisation des poids par programmation linéaire (GLOP), à profils fixés ;
- amélioration locale des profils ;
- réinitialisation des modèles les moins bons.

L'apprentissage s'arrête quand le learning set est entièrement restitué, après 20 itérations sans progrès, ou à la
limite de temps. lincs fixe le seuil à 1 avec des poids non normalisés. On normalise (`w / W`, `λ = 1 / W`) pour
retrouver les notations du jalon 1.

Détail technique : notre `classify` (numpy) reproduit **exactement** `lincs.classify_alternatives`.
Cela a été vérifié sur plusieurs jeux et plusieurs graines. Il a fallu comparer en float32, comme lincs : ses profils
sont posés exactement sur des valeurs d'objets, et ses sommes de poids du type 0,999999 doivent perdre.

## Partie 2 : heuristique pour inv-2-MR-Sort

On cherche une partition (S1, S2) du learning set et deux MR-Sort (M1, M2) qui maximisent le nombre d'objets bien
classés par le modèle de leur segment. Les deux problèmes sont couplés : il faut les modèles pour savoir où placer
les objets, et les segments pour apprendre les modèles. On les résout donc **en alternance**, comme dans *k-means*,
où chaque « centre » serait ici un modèle MR-Sort :

1. **Initialisation**, en alternant selon les redémarrages :
   - `random` : partition aléatoire 50/50 ;
   - `residual` : un MR-Sort global est appris sur tout le learning set, et ses erreurs forment S2. Le modèle global
     capte surtout le segment majoritaire, ce qui est utile quand les segments sont déséquilibrés (75/25).
2. **Apprentissage** de M1 sur S1 et de M2 sur S2 avec lincs.
3. **Réaffectation** de chaque objet au modèle qui l'explique le mieux, selon un score lexicographique :
   bien classé > erreur d'une seule catégorie > ... ; à égalité, plus grande **marge** signée
   `min(s_{c-1}(x) − λ, λ − s_c(x))`. Un objet ne bouge que si son score s'améliore strictement. Chaque segment garde
   au moins 10 % des objets, sinon un modèle pourrait tout absorber.
4. **Itérations** : on répète les étapes 2 et 3 jusqu'à 6 itérations sans progrès, ou jusqu'à ce que tout soit
   expliqué. On ne s'arrête *pas* dès que la partition est stable : lincs étant stochastique, réapprendre sur la même
   partition améliore encore souvent la solution (constaté expérimentalement).
5. **Choix final** parmi toutes les itérations des 5 redémarrages :
   - on retient les solutions à moins de 0,5 % du meilleur nombre d'objets expliqués ;
   - parmi elles, on prend celle qui maximise le nombre d'objets bien classés par M1 **plus** le nombre d'objets bien
     classés par M2.

   La justification est la suivante : plusieurs partitions, dont certaines sont fausses, expliquent 100 % des données.
   Les vrais modèles, eux, sont d'accord sur une grande partie de l'espace, donc chacun explique aussi beaucoup d'objets
   de l'autre segment. Ce critère de parcimonie écarte les paires de modèles sur-spécialisés.

### Variantes testées et écartées (données synthétiques, 4 instances)

| variante | effet |
|---|---|
| arrêt dès que la partition est stable | arrêt prématuré : fidélité des modèles ≈ 0,80 au lieu de ≈ 0,92 |
| objets ambigus laissés dans leur segment (`stay`) | légèrement moins bon que l'affectation par la marge |
| objets ambigus partagés entre les deux learning sets (`shared`) | moins d'objets expliqués, pas de gain de fidélité |
| consolidation finale (réapprendre chaque modèle sur tout ce qu'il explique) | ne corrige pas une mauvaise partition |
| choix final sur le seul nombre d'objets expliqués | retient parfois une partition fausse qui explique 100 % des données |

### Formulation exacte (pour mémoire)

Le MIP du jalon 1 s'étend en ajoutant des binaires `z(x)` (segment de x) et `c_k(x)` (x bien classé par M_k).
Les contraintes de classement du modèle k ne sont actives que si `c_k(x) = 1` (big-M), avec `c_k(x) ≤ [z(x) = k]`.
L'objectif est `max Σ_x Σ_k c_k(x)`. Le nombre de binaires (2 × n × (p−1) × m pour les δ, plus 3m) rend ce programme
intraitable pour m = 300 à 1200, d'où l'heuristique, qui s'appuie sur lincs.

## Résultats

### Datasets supermarket `cr06_ca03` (6 critères, 3 catégories ; 24 fichiers ; découpage 80/20)

Moyennes sur les tailles 300, 600 et 1200 :

| type, segments, bruit (%) | test 1 MR-Sort | test témoin | **test 2 MR-Sort** | train 1 MR-Sort | train 2 MR-Sort |
|:---|---:|---:|---:|---:|---:|
| type_a, 50-50, 0  | 0,696 | 0,826 | **0,926** | 0,763 | 0,989 |
| type_a, 50-50, 10 | 0,586 | 0,729 | **0,861** | 0,692 | 0,927 |
| type_a, 75-25, 0  | 0,818 | 0,893 | **0,958** | 0,845 | 0,985 |
| type_a, 75-25, 10 | 0,707 | 0,792 | **0,856** | 0,727 | 0,936 |
| type_b, 50-50, 0  | 0,543 | 0,754 | **0,968** | 0,664 | 0,995 |
| type_b, 50-50, 10 | 0,492 | 0,686 | **0,924** | 0,626 | 0,938 |
| type_b, 75-25, 0  | 0,819 | 0,833 | **0,962** | 0,810 | 0,996 |
| type_b, 75-25, 10 | 0,683 | 0,706 | **0,878** | 0,736 | 0,953 |
| **moyenne**       | 0,668 | 0,777 | **0,917** | 0,733 | 0,965 |

- Un seul MR-Sort n'explique qu'environ 73 % du learning set, même sans bruit : les données ne sont pas compatibles
  avec un modèle unique, ce qui confirme l'hypothèse des deux segments.
- Avec l'heuristique, 96,5 % du learning set est expliqué. Sans bruit, on atteint ~99 % ; avec 10 % de bruit, ~93-95 %,
  ce qui est cohérent avec le taux de bruit.
- En test, l'heuristique dépasse nettement le témoin (2 modèles appris sur une partition aléatoire) : 0,917 contre
  0,777. Le gain ne vient donc pas seulement du fait d'avoir « deux chances » de bien classer.
- Limite de la mesure en test : le segment d'un nouvel objet n'est pas observé, donc on compte l'objet comme bien classé
  si l'un des deux modèles le classe bien (c'est aussi le cas pour le témoin).
- Tailles des segments trouvés (`taille_seg_min`) : 0,38 en moyenne sur les fichiers 75-25 (de 0,21 à 0,49), loin
  des 25 % attendus. Elles ne reflètent pas les vraies proportions : beaucoup d'objets sont bien classés par les deux
  modèles (les deux vrais modèles sont d'accord sur une grande partie de l'espace), et ces objets sont répartis selon
  la marge, sans lien avec leur vrai segment. Seuls les objets expliqués par un seul modèle renseignent sur le segment.
- Temps : environ 3 min par fichier avec l'heuristique (5 redémarrages), contre ~5 s pour un seul modèle.

### Validation synthétique (segments connus ; 300 objets, 6 critères, 3 classes ; 5 instances par configuration)

| | 50/50, sans bruit | 50/50, 10 % de bruit | 75/25, sans bruit | 75/25, 10 % de bruit |
|:---|---:|---:|---:|---:|
| learning set expliqué, 1 MR-Sort | 0,795 | 0,725 | 0,868 | 0,783 |
| learning set expliqué, 2 MR-Sort | 0,997 | 0,951 | 0,994 | 0,955 |
| segments retrouvés (objets discriminants) | 0,869 | 0,621 | 0,824 | 0,806 |
| fidélité M1 / M2 : heuristique | 0,77 / 0,85 | 0,53 / 0,68 | 0,78 / 0,74 | 0,82 / 0,66 |
| fidélité M1 / M2 : 1 seul MR-Sort | 0,69 / 0,70 | 0,67 / 0,69 | 0,91 / 0,50 | 0,82 / 0,54 |
| fidélité M1 / M2 : plafond (vraie partition) | 0,93 / 0,95 | 0,87 / 0,91 | 0,95 / 0,92 | 0,88 / 0,84 |

La fidélité est la proportion de 5000 objets de test classés de la même façon par le modèle appris et par le vrai modèle
(c'est la « qualité » du jalon 1). Le plafond correspond aux modèles appris par lincs sur la **vraie** partition.

Analyse honnête :
- **Sans bruit**, l'heuristique retrouve en moyenne les segments et les modèles mieux qu'un seul MR-Sort, mais reste
  sous le plafond. La variance est forte : selon les instances, la fidélité va de ~0,65 à ~0,95.
- En **75/25**, un seul MR-Sort reproduit déjà bien le segment majoritaire (0,91), mais il rate complètement le segment
  minoritaire (0,50). L'heuristique trouve les deux segments, au prix d'un modèle majoritaire un peu moins fidèle.
- **Avec 10 % de bruit**, la qualité baisse nettement. Deux modèles flexibles peuvent « absorber » le bruit, et le
  critère de sélection (nombre d'objets expliqués à 0,5 % près) devient lui-même bruité.
- Un diagnostic des candidats (`result["candidates"]`) le confirme. Une même instance relancée peut donner une bien
  meilleure solution, car lincs est stochastique avec arrêt sur limite de temps. La principale piste d'amélioration est
  donc la **recherche** : plus de redémarrages, des apprentissages lincs plus longs.
- Autres pistes : adapter la tolérance de sélection au taux d'erreur, et utiliser une validation croisée pour choisir
  entre les candidats.
