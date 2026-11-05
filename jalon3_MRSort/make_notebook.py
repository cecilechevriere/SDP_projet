"""Génère jalon3.ipynb (exécuter ensuite : jupyter nbconvert --to notebook --execute --inplace jalon3.ipynb)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# Projet SDP – Jalon 3 : apprentissage MR-Sort face à un agent malicieux\n"
       "A transmet à B des objets classés par un vrai modèle $M_A$. Pour chaque donnée de A, un agent malicieux ajoute "
       "une fausse donnée. B reçoit le mélange et cherche à retrouver $M_A$.\n\n"
       "Code : `brouillage.py` (attaques, stratégie de B, expérience) et `run_brouillage.py` (expériences). "
       "Les briques d'apprentissage viennent du jalon 2 (`mrsort_lincs.py`, `two_mrsort.py`)."),
    code("import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from brouillage import ATTAQUES, attaque_frontiere, random_model, classify\n"
         "res = pd.read_csv('results/brouillage.csv')\n"
         "cols = ['fid_plafond', 'fid_un_modele', 'fid_confiance_seule', 'fid_deux_modeles', "
         "'fid_deux_modeles_meilleur', 'purete_segment_A']"),
    md("## Protocole\n"
       "- $M_A$ aléatoire (6 critères, 3 classes), A envoie 300 objets uniformes sur $[0,1]^6$ classés par $M_A$ ; "
       "l'agent malicieux en ajoute 300.\n"
       "- B compare : **un seul MR-Sort** appris avec lincs sur tout ce qu'il reçoit ; **deux MR-Sort** appris avec "
       "l'heuristique inv-2-MR-Sort du jalon 2, dont il garde celui qui explique le mieux 20 **objets de confiance** "
       "(sûrement envoyés par A) ; et, pour référence, un MR-Sort appris sur ces 20 objets seuls.\n"
       "- *fid* = proportion de 5000 objets de test classés comme par $M_A$ (qualité du jalon 1). "
       "*plafond* = MR-Sort appris sur les seules données de A. *meilleur* = le meilleur des deux modèles "
       "(choix idéal). *pureté* = part de vraies données dans le segment retenu par B.\n"
       "- Pourquoi des objets de confiance : avec autant de vraies que de fausses données, les deux segments jouent des "
       "rôles symétriques. Sans information en plus, B retrouve deux modèles mais ne peut pas savoir lequel vient de A."),
    md("## Partie 1 – Agent basique (MR-Sort aléatoire)"),
    code("res[res.attaque.isin(['aucune', 'modele_aleatoire'])].groupby('attaque', sort=False)[cols].mean().round(3)"),
    md("## Partie 2 – Stratégies de brouillage\n"
       "- `etiquettes_aleatoires` : objets uniformes, classe au hasard.\n"
       "- `frontiere_d` : l'agent prend $M_E$ = $M_A$ avec les frontières relevées de $d$, et n'envoie que des objets où "
       "$M_E$ et $M_A$ ne sont pas d'accord (juste au-dessus des frontières de $M_A$, avec la classe inférieure).\n"
       "- `copies_contradictoires` : chaque objet de A est recopié (à un léger bruit près) avec une classe voisine."),
    code("ordre = ['aucune', 'modele_aleatoire', 'etiquettes_aleatoires', 'frontiere_0.05', 'frontiere_0.10', "
         "'frontiere_0.20', 'copies_contradictoires']\n"
         "g = res.groupby('attaque')[cols].mean().reindex(ordre)\n"
         "g.round(3)"),
    code("ax = g[['fid_un_modele', 'fid_deux_modeles', 'fid_confiance_seule']].plot.bar(figsize=(11, 4), rot=20)\n"
         "ax.axhline(g['fid_plafond'].mean(), color='k', ls='--', lw=1, label='plafond (données de A seules)')\n"
         "ax.set_ylabel('fidélité à $M_A$'); ax.set_ylim(0.3, 1)\n"
         "ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)\n"
         "ax.set_title('Qualité du modèle de B selon la stratégie de brouillage'); plt.tight_layout()"),
    md("### Pourquoi l'attaque près de la frontière est efficace\n"
       "Sur un exemple à 2 critères : les fausses données (rouge) sont toutes dans la bande située juste au-dessus des "
       "frontières de $M_A$. Ailleurs, $M_E$ et $M_A$ donnent la même classe : rien ne permet de séparer les deux "
       "sources."),
    code("rng = np.random.default_rng(3)\n"
         "M = {'b': np.array([[0.35, 0.35], [0.65, 0.65]]), 'w': np.array([0.5, 0.5]), 'lam': 0.75}\n"
         "XA = rng.uniform(0, 1, (300, 2)); yA = classify(XA, M)\n"
         "XE, yE = attaque_frontiere(XA, yA, M, 3, rng, decalage=0.1)\n"
         "fig, ax = plt.subplots(figsize=(4.5, 4.5))\n"
         "ax.scatter(*XA.T, c=yA, cmap='viridis', s=12, label='données de A (couleur = classe)')\n"
         "ax.scatter(*XE.T, c='red', marker='x', s=14, label='brouillage')\n"
         "for bh in M['b']:\n"
         "    ax.plot([bh[0], bh[0], 1], [1, bh[1], bh[1]], 'k--', lw=1)\n"
         "ax.set_xlabel('critère 1'); ax.set_ylabel('critère 2'); ax.legend(fontsize=7, loc='lower left')\n"
         "plt.tight_layout()"),
    md("## Conclusions\n"
       "- **Agent basique (partie 1)** : un seul MR-Sort est très dégradé (fidélité 0,63), alors que l'heuristique à "
       "deux modèles du jalon 2, avec 20 objets de confiance pour choisir le bon modèle, remonte à 0,91 "
       "(plafond 0,97). Le brouillage d'un MR-Sort aléatoire se sépare bien des vraies données.\n"
       "- **Aucune stratégie de B ne gagne partout.** Contre des étiquettes aléatoires, un seul MR-Sort résiste "
       "(0,82) mais l'heuristique à deux modèles s'effondre (0,55) : le bruit ne suit aucun MR-Sort, la partition "
       "trouvée n'a pas de sens. B devrait donc d'abord tester s'il y a vraiment deux modèles.\n"
       "- **Attaque près de la frontière** : avec un petit décalage (0,05 à 0,10), les fausses données sont presque "
       "indiscernables des vraies (pureté du segment retenu 0,54 à 0,69) et les deux stratégies de B tombent vers "
       "0,67 à 0,81. Avec un décalage plus grand (0,20), un seul modèle souffre davantage (0,58), mais le brouillage "
       "redevient séparable (pureté 0,94) et l'heuristique remonte en partie (0,73).\n"
       "- **Copies contradictoires (la meilleure attaque trouvée)** : chaque vraie donnée a un double presque "
       "identique avec une classe voisine. Les deux stratégies de B tombent à 0,54, **moins bien que s'il "
       "n'utilisait que ses 20 objets de confiance (0,66)**. Rien dans les performances ne distingue la vraie "
       "donnée de sa copie.\n"
       "- Limites : 3 à 5 instances par attaque (variance visible, en particulier pour `frontiere_0.20`) ; "
       "l'attaque près de la frontière suppose que l'agent connaît $M_A$ (il peut l'estimer à partir des données de "
       "A, qu'il intercepte) ; les copies contradictoires supposent qu'il voit les données de A.\n"
       "- Pistes pour B : choisir entre un modèle, deux modèles et les objets de confiance seuls selon la précision "
       "sur les objets de confiance ; détecter les paires d'objets presque identiques de classes différentes et "
       "les écarter."),
]
nbf.write(nb, "jalon3.ipynb")
