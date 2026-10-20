# Projet SDP – Jalon 1 : apprentissage d'un modèle MR-Sort par programmation mathématique (Gurobi)

## Modèle

Données : `n` critères, `p` catégories ordonnées `C^1 < ... < C^p`, learning set d'objets `x` avec leur classe.
Inconnues : frontières `b^h` (`h = 1..p-1`, avec `b^{h-1}_i <= b^h_i`), poids `w_i >= 0` (`sum w_i = 1`), seuil `lambda in [0,1]`.
Règle : `x` est dans `C^h` ssi `sum_{i : x_i >= b^{h-1}_i} w_i >= lambda` et `sum_{i : x_i >= b^h_i} w_i < lambda`.

Programme (MIP) :
- Variables binaires `delta^h_i(x) = 1 <=> x_i >= b^h_i`, imposées par deux contraintes big-M
  (`x_i >= b^h_i - M(1-delta)` et `x_i <= b^h_i - eps + M delta`).
- Variables continues `w^h_i(x) = delta^h_i(x) * w_i`, linéarisées par
  `w_i >= w^h_i(x) >= 0` et `delta >= w^h_i(x) >= delta + w_i - 1`.
- Pour un objet de classe `c` : `sum_i w^{c-1}_i(x) >= lambda + alpha` (s'il y a une frontière inférieure) et
  `sum_i w^{c}_i(x) <= lambda - alpha - eps_w` (s'il y a une frontière supérieure).
- Objectif : `max alpha` (marge de séparation, `alpha in [-1,1]`). `alpha > 0` : le learning set est parfaitement séparé.
- Seules les frontières utiles à la classe de l'objet (au plus 2) sont modélisées.
- Option `compact=True` : version allégée de même optimum (vérifiée sur des instances aléatoires), utile avec une licence Gurobi limitée en taille.
- `eps` (inégalité stricte sur les notes) et `eps_w` (inégalité stricte sur la somme des poids) sont distincts.

## Fichiers
- `mrsort_gurobi.py` : implémentation (`learn_mr_sort`, `classify`, génération aléatoire, protocole de test).
- `mrsort_gurobi.ipynb` : notebook (vérification, protocole de test sur 20 essais pour p = 2, 3, 4, apprentissage sur Dataset0).
- `run_dataset0.py` : apprentissage sur `dataset0.txt` et affichage des paramètres.
- `dataset0.txt` : données fournies.

## Résultats

Protocole de test (n = 4, 40 objets d'apprentissage, 1000 objets de test, 20 essais, learning set sans bruit) : qualité moyenne
0,95 (p = 2), 0,91 (p = 3), 0,88 (p = 4).

## Paramètres appris sur Dataset0 (60 objets, 5 critères, 3 catégories ; eps = 1, eps_w = 1e-3)

| | crit1 | crit2 | crit3 | crit4 | crit5 |
|---|---|---|---|---|---|
| frontière b^1 (cat. 1 / 2) | 38 | 35 | 30 | 34 | 36 |
| frontière b^2 (cat. 2 / 3) | 50 | 53 | 37 | 63 | 50 |
| poids w | 0,1429 (1/7) | 0,2857 (2/7) | 0,1429 (1/7) | 0,1429 (1/7) | 0,2857 (2/7) |

- seuil `lambda = 0,6434` (une coalition gagne si elle pèse au moins 5/7 ≈ 0,714, la coalition de poids 4/7 ≈ 0,571 perd) ;
- marge `alpha = 0,0709` ;
- 0 erreur sur les 60 objets (le learning set est entièrement restitué).

Le modèle optimal n'est pas forcément unique : une autre exécution ou une autre version de Gurobi peut renvoyer
d'autres valeurs de `b`, `w`, `lambda` avec la même marge.
