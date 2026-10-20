"""Génère jalon2.ipynb (exécuter ensuite : jupyter nbconvert --to notebook --execute --inplace jalon2.ipynb)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Projet SDP – Jalon 2 : MR-Sort avec lincs, puis inv-2-MR-Sort\n"
       "1. Apprendre **un** modèle MR-Sort sur le dataset supermarket avec l'heuristique de la librairie `lincs`.\n"
       "2. Clientèle à **deux segments** : apprendre simultanément une partition du learning set et un MR-Sort par segment "
       "(heuristique proposée, qui utilise lincs comme brique d'apprentissage).\n\n"
       "Code : `mrsort_lincs.py` (enveloppe lincs), `two_mrsort.py` (heuristique), `run_synthetic.py` et "
       "`run_supermarket.py` (expériences)."),
    code("import numpy as np, pandas as pd, matplotlib.pyplot as plt, time\n"
         "from mrsort_lincs import load_dataset, learn_mrsort, classify, describe\n"
         "from two_mrsort import learn_two_mrsort, oracle_accuracy, random_halves\n"
         "D = 'Datasets-Step2/type_a/cr06_ca03/'\n"
         "X, y, cats = load_dataset(D + 'n0300_rep50-50_noise00.csv', D + 'problem.yml')\n"
         "n, p = X.shape[1], len(cats)\n"
         "print(X.shape, 'catégories :', cats, '| effectifs :', np.bincount(y))"),
    md("## Partie 1 – Un modèle MR-Sort appris avec lincs\n"
       "Heuristique *weights-profiles-breed* : une population de modèles alterne optimisation des poids par PL "
       "(profils fixés), amélioration locale des profils, et réinitialisation des moins bons modèles."),
    code("t = time.time()\n"
         "single = learn_mrsort(X, y, p, seed=0, max_seconds=5)\n"
         "print(f'temps {time.time()-t:.1f} s | lambda = {single[\"lam\"]:.3f} | '\n"
         "      f'objets bien classés : {np.mean(classify(X, single) == y):.3f}')\n"
         "describe(single)"),
    md("Un seul MR-Sort ne restitue qu'environ 80 % du learning set, alors que ces données sont **sans bruit** : "
       "elles ne sont pas compatibles avec un modèle unique. C'est l'hypothèse des deux segments de clientèle."),
    md("## Partie 2 – Heuristique pour inv-2-MR-Sort\n"
       "Analogue de *k-means* où chaque « centre » est un modèle MR-Sort :\n"
       "1. partition initiale (aléatoire, ou « residual » : les objets mal classés par un MR-Sort global) ;\n"
       "2. apprentissage de M1 sur S1 et M2 sur S2 avec lincs ;\n"
       "3. réaffectation de chaque objet au modèle qui l'explique le mieux "
       "(bien classé > erreur d'une catégorie > ... ; à égalité, la plus grande marge) ;\n"
       "4. retour en 2 jusqu'à ce qu'il n'y ait plus de progrès (patience), plusieurs redémarrages ;\n"
       "5. choix final : le plus d'objets expliqués, puis à quasi-égalité les modèles les plus « larges » "
       "(maximiser le nombre d'objets bien classés par M1 + par M2)."),
    code("t = time.time()\n"
         "two = learn_two_mrsort(X, y, p, seed=0, learn_kw=dict(max_seconds=1), verbose=True)\n"
         "print(f'\\n{time.time()-t:.0f} s | expliqués : {two[\"n_ok\"]}/{len(y)} | segments : {np.bincount(two[\"seg\"])} '\n"
         "      f'| solution retenue : redémarrage {two[\"restart\"]} ({two[\"init\"]}), itération {two[\"iteration\"]}')"),
    code("fig, ax = plt.subplots(figsize=(7, 3.5))\n"
         "for r, h in enumerate(two['history']):\n"
         "    ax.plot(np.array(h) / len(y), marker='o', ms=3, label=f'redémarrage {r}')\n"
         "ax.axhline(np.mean(classify(X, single) == y), color='grey', ls='--', label='1 seul MR-Sort')\n"
         "ax.set_xlabel('itération'); ax.set_ylabel('part du learning set expliquée'); ax.legend(fontsize=8)\n"
         "plt.tight_layout(); plt.show()"),
    code("for k, m in enumerate(two['models']):\n"
         "    print(f'Segment {k+1} ({np.sum(two[\"seg\"] == k)} objets) : lambda = {m[\"lam\"]:.3f}')\n"
         "    display(describe(m))"),
    md("## Validation sur données synthétiques (segments connus)\n"
       "`run_synthetic.py` : 2 modèles ground truth, 300 objets, 6 critères, 3 classes ; 5 instances par configuration. "
       "*fid* = proportion de 5000 objets de test classés pareil par le modèle appris et le vrai modèle (qualité du jalon 1). "
       "*plafond* = modèles appris par lincs sur la **vraie** partition. *segments_retrouves* est mesuré sur les objets où "
       "les deux vrais modèles divergent (pour les autres, le segment n'est pas identifiable)."),
    code("syn = pd.read_csv('results/synthetic.csv')\n"
         "syn.groupby(['share', 'noise']).mean(numeric_only=True).drop(columns='instance').round(3).T"),
    md("## Résultats sur les datasets supermarket (6 critères, 3 catégories)\n"
       "Découpage 80/20. *test_2_modeles* = proportion d'objets de test bien classés par le modèle de leur segment "
       "(segment le plus favorable, puisque le segment d'un nouvel objet n'est pas observé). Pour vérifier que le gain "
       "ne vient pas seulement du fait d'avoir « deux chances », on compare au *témoin* : 2 modèles appris sur une "
       "partition aléatoire."),
    code("sup = pd.read_csv('results/supermarket_cr06_ca03.csv')\n"
         "cols = ['train_1_modele', 'test_1_modele', 'train_2_modeles', 'test_2_modeles', 'train_temoin', 'test_temoin']\n"
         "sup.groupby(['type', 'rep', 'bruit'])[cols].mean().round(3)"),
    code("g = sup.groupby(['type', 'rep', 'bruit'])[['test_1_modele', 'test_temoin', 'test_2_modeles']].mean()\n"
         "ax = g.plot.bar(figsize=(9, 3.5), ylim=(0.5, 1.0), rot=30)\n"
         "ax.set_ylabel('précision en test'); ax.legend(['1 MR-Sort', 'témoin (partition aléatoire)', 'heuristique 2 MR-Sort'], fontsize=8)\n"
         "plt.tight_layout(); plt.show()"),
]
nbf.write(nb, "jalon2.ipynb")
