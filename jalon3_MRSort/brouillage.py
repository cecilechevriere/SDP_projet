"""
Jalon 3 : apprentissage MR-Sort avec des données brouillées par un agent malicieux.

A observe des objets classés par un vrai modèle M_A et les transmet à B. Pour chaque donnée de A, l'agent malicieux
ajoute une fausse donnée. B reçoit le mélange (sans savoir quelles données sont fausses) et cherche à retrouver M_A.

Conventions des jalons 1 et 2 : X (m, n) dans [0, 1]^n, classes 0 .. p-1, modèle = dict {"b", "w", "lam"}.
"""
import os
import sys
import zlib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jalon2_MRSort"))
from mrsort_lincs import learn_mrsort, classify, margin          # noqa: E402
from two_mrsort import learn_two_mrsort, random_model             # noqa: E402


# ---------------------------------------------------------------- stratégies de l'agent malicieux

def attaque_modele_aleatoire(X_A, y_A, M_A, p, rng):
    """Agent basique : il tire un MR-Sort au hasard et envoie des objets uniformes classés par ce modèle."""
    M_E = random_model(X_A.shape[1], p, rng)
    X = rng.uniform(0, 1, X_A.shape)
    return X, classify(X, M_E)


def attaque_etiquettes_aleatoires(X_A, y_A, M_A, p, rng):
    """Objets uniformes avec une classe tirée au hasard (aucun modèle derrière)."""
    X = rng.uniform(0, 1, X_A.shape)
    return X, rng.integers(0, p, len(X))


def attaque_frontiere(X_A, y_A, M_A, p, rng, decalage=0.1):
    """Attaque ciblée : l'agent prend un modèle M_E très proche de M_A (mêmes poids, même seuil, frontières
    relevées de `decalage` sur chaque critère) et n'envoie que des objets où M_E et M_A sont en désaccord,
    c'est-à-dire juste au-dessus des frontières de M_A. Ces objets reçoivent la classe inférieure.
    Les deux modèles sont d'accord partout ailleurs : B ne peut pas les séparer, et ses frontières sont tirées vers le haut."""
    M_E = dict(M_A, b=np.clip(np.asarray(M_A["b"]) + decalage, 0, 1))
    m, n = X_A.shape
    garde = []
    for _ in range(200):                                   # tirage par rejet dans la zone de désaccord
        X = rng.uniform(0, 1, (5 * m, n))
        garde.append(X[classify(X, M_E) != classify(X, M_A)])
        if sum(len(g) for g in garde) >= m:
            break
    X = np.concatenate(garde)[:m]
    if len(X) < m:                                         # zone de désaccord trop petite : on complète au hasard
        X = np.concatenate([X, rng.uniform(0, 1, (m - len(X), n))])
    return X, classify(X, M_E)


def attaque_copies_contradictoires(X_A, y_A, M_A, p, rng, bruit=0.01):
    """L'agent recopie chaque objet de A (à un léger bruit près) avec une classe voisine différente.
    B reçoit deux versions presque identiques de chaque objet : rien dans les performances ne les distingue."""
    X = np.clip(X_A + rng.normal(0, bruit, X_A.shape), 0, 1)
    sens = np.where(y_A == 0, 1, np.where(y_A == p - 1, -1, rng.choice([-1, 1], len(y_A))))
    return X, y_A + sens


ATTAQUES = {
    "modele_aleatoire": attaque_modele_aleatoire,
    "etiquettes_aleatoires": attaque_etiquettes_aleatoires,
    "frontiere_0.05": lambda *a: attaque_frontiere(*a, decalage=0.05),
    "frontiere_0.10": lambda *a: attaque_frontiere(*a, decalage=0.10),
    "frontiere_0.20": lambda *a: attaque_frontiere(*a, decalage=0.20),
    "copies_contradictoires": attaque_copies_contradictoires,
}


# ---------------------------------------------------------------- stratégie de B

def choisir_modele(models, X_conf, y_conf):
    """Les deux segments jouent des rôles symétriques (autant de vraies que de fausses données) : sans information
    supplémentaire, B ne peut pas savoir lequel vient de A. Hypothèse : B dispose de quelques objets de confiance
    (sûrement envoyés par A). Il garde le modèle qui les classe le mieux (à égalité : la plus grande marge moyenne)."""
    cles = [(np.mean(classify(X_conf, m) == y_conf), np.mean(np.clip(margin(X_conf, y_conf, m), -1, 1)))
            for m in models]
    return int(max(range(len(models)), key=lambda k: cles[k]))


# ---------------------------------------------------------------- une expérience

def experience(attaque, instance, n=6, p=3, m=300, n_conf=20, m_test=5000, learn_kw=None, deux_modeles=True):
    """Génère M_A et les données de A (identiques pour toutes les attaques d'une même instance), applique l'attaque,
    puis compare les modèles appris par B. Fidélité = proportion d'objets de test classés comme par M_A."""
    learn_kw = learn_kw or dict(max_seconds=1)
    rng = np.random.default_rng(1000 + instance)
    M_A = random_model(n, p, rng)
    X_A = rng.uniform(0, 1, (m, n))
    y_A = classify(X_A, M_A)
    Xt = rng.uniform(0, 1, (m_test, n))
    yt = classify(Xt, M_A)
    conf = rng.choice(m, n_conf, replace=False)                    # objets de confiance de B

    rng_e = np.random.default_rng(zlib.crc32(attaque.encode()) + instance)
    if attaque == "aucune":
        X_E, y_E = np.empty((0, n)), np.empty(0, dtype=int)
    else:
        X_E, y_E = ATTAQUES[attaque](X_A, y_A, M_A, p, rng_e)
    X = np.concatenate([X_A, X_E])
    y = np.concatenate([y_A, y_E]).astype(int)
    faux = np.r_[np.zeros(m, bool), np.ones(len(y_E), bool)]
    perm = rng.permutation(len(y))                                 # B reçoit les données mélangées
    X, y, faux = X[perm], y[perm], faux[perm]

    fid = lambda model: float(np.mean(classify(Xt, model) == yt))
    res = {"attaque": attaque, "instance": instance,
           "accord_A_brouillage": float(np.mean(classify(X_E, M_A) == y_E)) if len(y_E) else 1.0,
           "fid_plafond": fid(learn_mrsort(X_A, y_A, p, seed=instance, max_seconds=5)),
           "fid_un_modele": fid(learn_mrsort(X, y, p, seed=instance, max_seconds=5)),
           "fid_confiance_seule": fid(learn_mrsort(X_A[conf], y_A[conf], p, seed=instance, max_seconds=5))}
    if deux_modeles and len(y_E):
        two = learn_two_mrsort(X, y, p, seed=instance, learn_kw=learn_kw)
        k = choisir_modele(two["models"], X_A[conf], y_A[conf])
        res.update({
            "fid_deux_modeles": fid(two["models"][k]),
            "fid_deux_modeles_meilleur": max(fid(mk) for mk in two["models"]),
            "purete_segment_A": float(np.mean(~faux[two["seg"] == k])),   # part de vraies données dans le segment retenu
            "train_deux_modeles": two["n_ok"] / len(y),
        })
    return res
