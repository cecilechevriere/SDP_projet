"""
Jalon 2 - partie 2 : inv-2-MR-Sort.
On cherche simultanément une partition du learning set en 2 segments S1, S2 et un modèle MR-Sort par segment,
de façon à maximiser le nombre d'objets correctement classés par le modèle de leur segment.

Heuristique proposée : "k-modèles" (analogue de k-means, où chaque "centre" est un modèle MR-Sort) :
  1. initialiser une partition (S1, S2) ;
  2. apprendre M1 sur S1 et M2 sur S2 avec lincs ;
  3. réaffecter chaque objet au modèle qui l'explique le mieux (bien classé > presque bien classé > ...) ;
  4. recommencer 2-3 jusqu'à stabilité de la partition ; garder la meilleure solution vue.
Plusieurs redémarrages (initialisations différentes) limitent le risque de rester dans un optimum local.
"""
import numpy as np
from mrsort_lincs import learn_mrsort, classify, margin


# ---------------------------------------------------------------- score d'explication

def explain_score(X, y, model):
    """Score de "qualité d'explication" de chaque objet par un modèle, à maximiser (ordre lexicographique) :
       1) l'objet est-il bien classé ? (poids 100)
       2) sinon, de combien de catégories le modèle se trompe-t-il ? (poids 3)
       3) marge signée (dans [-1, 1]) : à quel point l'objet est loin / proche de la frontière de décision.
    Retourne (score, bien_classé)."""
    pred = classify(X, model)
    ok = pred == y
    marg = np.clip(margin(X, y, model), -1.0, 1.0)
    return 100.0 * ok - 3.0 * np.abs(pred - y) + marg, ok


# ---------------------------------------------------------------- réaffectation

def reassign(X, y, models, seg, min_size, ambiguous="margin"):
    """Étape 3 : chaque objet va au segment dont le modèle l'explique le mieux.
    ambiguous = "margin" : objet bien classé par les deux modèles -> modèle de plus grande marge ;
    ambiguous = "stay"   : il reste dans son segment actuel ;
    ambiguous = "shared" : comme "margin" pour le segment, mais l'objet sera utilisé pour apprendre les DEUX modèles
                           (cf. learn_two_mrsort ; il est compatible avec les deux, autant en profiter).
    Un objet ne change de segment que si le score s'améliore strictement (pas d'oscillation sur les égalités).
    Chaque segment garde au moins min_size objets (sinon un modèle peut tout absorber et l'autre disparaître)."""
    scores, oks = zip(*(explain_score(X, y, m) for m in models))
    S = np.stack(scores)                         # (2, m)
    OK = np.stack(oks)
    idx = np.arange(len(y))
    other = 1 - seg
    gain = S[other, idx] - S[seg, idx]
    move = gain > 1e-9
    if ambiguous == "stay":
        move &= ~(OK[0] & OK[1])
    new = np.where(move, other, seg)

    for k in (0, 1):                             # contrainte de taille minimale
        deficit = min_size - np.sum(new == k)
        if deficit > 0:
            cand = np.where(new != k)[0]
            cand = cand[np.argsort(-(S[k, cand] - S[1 - k, cand]))][:deficit]
            new[cand] = k
    ok = OK[new, idx]
    both = OK[0] & OK[1]
    return new, ok, both


# ---------------------------------------------------------------- initialisation

def init_partition(X, y, p, kind, rng, min_size, learn_kw):
    """'random'   : partition aléatoire 50/50.
       'residual' : un MR-Sort appris sur tout le learning set ; les objets qu'il classe mal forment S2
                    (complété par les objets de plus faible marge si S2 est trop petit). Idée : le modèle global
                    capte surtout le segment majoritaire, ses erreurs viennent surtout de l'autre segment."""
    m = len(y)
    if kind == "random":
        return rng.permutation(np.arange(m) % 2)
    model = learn_mrsort(X, y, p, seed=int(rng.integers(1 << 30)), **learn_kw)
    score, ok = explain_score(X, y, model)
    seg = (~ok).astype(int)
    if seg.sum() < min_size:
        seg[np.argsort(score)[:min_size]] = 1
    return seg


# ---------------------------------------------------------------- heuristique complète

def learn_two_mrsort(X, y, p, n_restarts=5, max_iter=30, patience=6, min_frac=0.1, ambiguous="margin",
                     inits=("random", "residual"), tol=0.005, seed=0, learn_kw=None, verbose=False):
    """
    Apprend 2 modèles MR-Sort et une partition du learning set.
    inits : types d'initialisation utilisés à tour de rôle sur les n_restarts redémarrages.
    Arrêt d'un redémarrage : tout le learning set est expliqué, ou `patience` itérations sans amélioration,
    ou max_iter itérations. On ne s'arrête PAS dès que la partition est stable : lincs étant stochastique,
    un nouvel apprentissage sur la même partition peut encore améliorer la solution.
    Choix de la solution finale parmi toutes les itérations de tous les redémarrages :
      1) critère principal n_ok = nb d'objets bien classés par le modèle de leur segment ;
      2) parmi les solutions à moins de tol * m objets de la meilleure, on prend celle qui maximise
         total_ok = nb d'objets bien classés par M1 + nb d'objets bien classés par M2.
         Plusieurs partitions (parfois fausses) expliquent tout le learning set ; les vrais modèles, eux, sont
         d'accord sur une grande partie de l'espace, donc chacun explique aussi beaucoup d'objets de l'autre segment.
         On préfère ainsi des modèles "larges" plutôt que deux modèles sur-spécialisés (parcimonie).
    learn_kw : paramètres passés à learn_mrsort (ex. max_seconds).
    Retourne un dict : models (liste de 2 modèles), seg (segment de chaque objet), n_ok, total_ok,
    history (n_ok à chaque itération de chaque redémarrage), candidates (toutes les solutions visitées).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    learn_kw = learn_kw or {}
    rng = np.random.default_rng(seed)
    min_size = max(int(min_frac * len(y)), p)
    candidates, history = [], []

    for r in range(n_restarts):
        kind = inits[r % len(inits)]
        seg = init_partition(X, y, p, kind, rng, min_size, learn_kw)
        both = np.zeros(len(y), dtype=bool)        # objets expliqués par les deux modèles (mode "shared")
        h, best_run, stall = [], -1, 0
        for it in range(max_iter):
            train = [(seg == k) | both for k in (0, 1)]
            models = [learn_mrsort(X[train[k]], y[train[k]], p, seed=int(rng.integers(1 << 30)), **learn_kw)
                      for k in (0, 1)]
            new, ok, both = reassign(X, y, models, seg, min_size, ambiguous)
            total_ok = int(sum(np.sum(classify(X, mk) == y) for mk in models))
            if ambiguous != "shared":
                both[:] = False
            n_ok = int(ok.sum())
            h.append(n_ok)
            if verbose:
                print(f"  redémarrage {r} ({kind}) it {it} : {n_ok}/{len(y)} expliqués, total_ok {total_ok}, "
                      f"tailles {np.bincount(new, minlength=2)}, {int(np.sum(new != seg))} objets déplacés")
            candidates.append({"models": models, "seg": new.copy(), "n_ok": n_ok, "total_ok": total_ok,
                               "restart": r, "iteration": it, "init": kind})
            stall = 0 if n_ok > best_run else stall + 1
            best_run = max(best_run, n_ok)
            seg = new
            if n_ok == len(y) or stall >= patience:
                break
        history.append(h)
    n_best = max(c["n_ok"] for c in candidates)
    eligible = [c for c in candidates if c["n_ok"] >= n_best - tol * len(y)]
    best = max(eligible, key=lambda c: (c["total_ok"], c["n_ok"]))
    best = dict(best, history=history, candidates=candidates)
    return best


# ---------------------------------------------------------------- évaluation

def oracle_accuracy(X, y, models):
    """Proportion d'objets bien classés par au moins un des modèles (= par le modèle de leur meilleur segment)."""
    return float(np.mean(np.any([classify(X, m) == y for m in models], axis=0)))


def random_halves(X, y, p, seed=0, learn_kw=None):
    """Témoin : 2 modèles appris sur une partition aléatoire, sans réaffectation.
    Permet de distinguer le gain dû à la segmentation du simple fait d'avoir "deux chances" de bien classer."""
    rng = np.random.default_rng(seed)
    seg = rng.permutation(np.arange(len(y)) % 2)
    return [learn_mrsort(X[seg == k], y[seg == k], p, seed=int(rng.integers(1 << 30)), **(learn_kw or {}))
            for k in (0, 1)]


# ---------------------------------------------------------------- données synthétiques (segments connus)

def random_model(n, p, rng):
    """Modèle MR-Sort aléatoire sur [0, 1]^n (même principe qu'au jalon 1)."""
    w = rng.dirichlet(np.ones(n))
    lam = rng.uniform(0.5, 0.8)
    b = np.sort(rng.uniform(0.15, 0.85, (p - 1, n)), axis=0)
    return {"b": b, "w": w, "lam": lam}


def make_two_segment_data(n, p, m, share=0.5, noise=0.0, rng=None, models=None):
    """Objets uniformes sur [0,1]^n ; une part `share` est classée par le modèle 0, le reste par le modèle 1.
    noise : proportion d'étiquettes remplacées par une autre classe au hasard.
    Retourne X, y, segment réel, modèles réels."""
    rng = rng or np.random.default_rng()
    models = models or [random_model(n, p, rng), random_model(n, p, rng)]
    X = rng.uniform(0, 1, (m, n))
    seg = (rng.uniform(size=m) >= share).astype(int)
    y = np.where(seg == 0, classify(X, models[0]), classify(X, models[1]))
    if noise > 0:
        flip = rng.uniform(size=m) < noise
        y[flip] = (y[flip] + rng.integers(1, p, flip.sum())) % p
    return X, y, seg, models


def segment_recovery(seg_true, seg_pred):
    """Proportion d'objets dont le segment est retrouvé (à permutation des segments près)."""
    a = np.mean(seg_true == seg_pred)
    return max(a, 1 - a)


def matched_models(seg_true, seg_pred, learned):
    """Ordonne les modèles appris pour qu'ils correspondent aux segments réels."""
    return learned if np.mean(seg_true == seg_pred) >= 0.5 else learned[::-1]
