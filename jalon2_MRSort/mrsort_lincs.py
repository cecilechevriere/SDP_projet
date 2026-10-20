"""
Jalon 2 - partie 1 : apprentissage d'un modèle MR-Sort avec la librairie lincs.

Conventions (identiques au jalon 1) :
- X : tableau (m, n) des performances, y : classes 0 .. p-1 (0 = la pire) ;
- un modèle est un dict {"b": (p-1, n), "w": (n,), "lam": float} avec sum(w) = 1 ;
- x est dans la classe c s'il "bat" exactement c frontières, une frontière b^h étant battue
  si la coalition des critères où x_i >= b^h_i pèse au moins lam.
"""
import io
import time
import numpy as np
import pandas as pd
import lincs.classification as lc


# ---------------------------------------------------------------- données

def load_dataset(csv_path, yml_path=None):
    """Charge un fichier CSV au format lincs. Retourne (X, y, noms des catégories).
    Les catégories sont recodées 0 .. p-1 dans l'ordre du problem.yml (pire -> meilleure)."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["name", "category"]).to_numpy(dtype=float)
    if yml_path is not None:
        cats = [l.split("name:")[1].strip() for l in open(yml_path) if "- name:" in l and "Criterion" not in l]
    else:
        cats = sorted(df["category"].astype(str).unique())
    code = {c: k for k, c in enumerate(cats)}
    y = df["category"].astype(str).map(code).to_numpy(dtype=int)
    return X, y, cats


def make_problem(n, p, vmin=0.0, vmax=1.0):
    """Problème lincs : n critères réels croissants sur [vmin, vmax], p catégories ordonnées."""
    inc = lc.Criterion.PreferenceDirection.increasing
    criteria = [lc.Criterion(f"c{i + 1}", lc.Criterion.RealValues(inc, vmin, vmax)) for i in range(n)]
    categories = [lc.Category(f"cat{h}") for h in range(p)]
    return lc.Problem(criteria, categories)


def to_alternatives(problem, X, y=None):
    """Convertit (X, y) numpy en objet lincs Alternatives."""
    alts = []
    for k, x in enumerate(X):
        perf = [lc.Performance(lc.Performance.Real(float(v))) for v in x]
        alts.append(lc.Alternative(f"a{k}", perf, None if y is None else int(y[k])))
    return lc.Alternatives(problem, alts)


# ---------------------------------------------------------------- apprentissage

def learn_mrsort(X, y, p, seed=0, max_seconds=5.0, max_iter_no_progress=20, models_count=9, vmin=0.0, vmax=1.0):
    """
    Apprend un MR-Sort avec l'heuristique "weights-profiles-breed" de lincs :
    une population de `models_count` modèles alterne
      - optimisation des poids par programmation linéaire (GLOP) à profils fixés,
      - amélioration locale des profils (heuristique d'accuracy),
      - réinitialisation des modèles les moins bons (breeding).
    Arrêt : learning set entièrement restitué, ou plus de progrès, ou limite de temps.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    problem = make_problem(X.shape[1], p, vmin, vmax)
    learning_set = to_alternatives(problem, X, y)

    L = lc.LearnMrsortByWeightsProfilesBreed
    pls = lc.PreprocessedLearningSet(problem, learning_set)
    models = L.ModelsBeingLearned(pls, models_count, int(seed))
    init = lc.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(pls, models)
    weights = lc.OptimizeWeightsUsingGlop(pls, models)
    profiles = lc.ImproveProfilesWithAccuracyHeuristicOnCpu(pls, models)
    breed = lc.ReinitializeLeastAccurate(models, init, weights, models_count // 2)
    stop = lc.TerminateWhenAny([
        lc.TerminateAtAccuracy(models, len(y)),
        lc.TerminateAfterIterationsWithoutProgress(models, max_iter_no_progress),
        lc.TerminateAfterSeconds(max_seconds),
    ])
    lincs_model = L(pls, models, init, weights, profiles, breed, stop).perform()
    return from_lincs(lincs_model, X.shape[1], p)


def from_lincs(lincs_model, n, p):
    """Extrait (b, w, lam) d'un modèle lincs.
    lincs fixe le seuil à 1 (coalition suffisante <=> somme des poids >= 1) avec des poids non normalisés :
    on normalise w -> w / W et lam = 1 / W, ce qui définit les mêmes coalitions gagnantes.
    Les poids bruts sont gardés ("w_raw") pour reproduire à l'identique les arrondis float32 de lincs."""
    b = np.full((p - 1, n), np.inf)          # profil "None" = jamais atteint
    for i, av in enumerate(lincs_model.accepted_values):
        for h, t in enumerate(av.real_thresholds.thresholds):
            if t is not None:
                b[h, i] = t
    w_raw = np.array(lincs_model.sufficient_coalitions[0].weights.criterion_weights, dtype=float)
    W = w_raw.sum()
    if W <= 0:   # cas dégénéré (poids tous nuls) : aucune frontière n'est jamais battue
        return {"b": b, "w": np.full(n, 1.0 / n), "lam": np.inf, "w_raw": w_raw}
    return {"b": b, "w": w_raw / W, "lam": 1.0 / W, "w_raw": w_raw}


# ---------------------------------------------------------------- classification

def coalitions(X, model):
    """s[h, x] = poids de la coalition des critères où x atteint la frontière h. Forme (p-1, m).
    Comparaisons en float32 comme dans lincs (ses profils sont posés exactement sur des valeurs d'objets)."""
    X = np.asarray(X, dtype=np.float32)
    b = np.asarray(model["b"], dtype=np.float32)
    return np.stack([(X >= bh) @ model["w"] for bh in b])


def classify(X, model, tol=1e-9):
    """Classe de chaque objet : nombre de frontières battues.
    Modèle issu de lincs : comme lincs, somme séquentielle en float32 des poids bruts, comparée à 1
    (les poids GLOP sont du type 0.333332986, une somme à 1e-7 près de 1 peut gagner ou perdre).
    Autre modèle (ex. ground truth) : somme normalisée >= lam, tol absorbant l'arrondi."""
    X = np.asarray(X, dtype=np.float32)
    b = np.asarray(model["b"], dtype=np.float32)
    if "w_raw" in model:
        w32 = np.asarray(model["w_raw"], dtype=np.float32)
        gagne = []
        for bh in b:
            A = X >= bh
            s = np.zeros(len(X), dtype=np.float32)
            for i in range(len(w32)):
                s = s + np.where(A[:, i], w32[i], np.float32(0))
            gagne.append(s >= np.float32(1))
        return np.sum(gagne, axis=0).astype(int)
    return (coalitions(X, model) / model["lam"] >= 1 - tol).sum(axis=0).astype(int)


def margin(X, y, model):
    """Marge signée de chaque objet par rapport à sa vraie classe c :
    min(s_{c-1}(x) - lam, lam - s_c(x)) (termes absents pour c = 0 ou c = p-1).
    > 0 : objet bien classé "avec de la marge" ; < 0 : il faudrait modifier le modèle d'autant pour le corriger."""
    s = coalitions(X, model)
    y = np.asarray(y, dtype=int)
    m = len(y)
    pm1 = s.shape[0]
    low = np.full(m, np.inf)
    up = np.full(m, np.inf)
    idx = np.arange(m)
    has_low = y >= 1
    has_up = y <= pm1 - 1
    low[has_low] = s[y[has_low] - 1, idx[has_low]] - model["lam"]
    up[has_up] = model["lam"] - s[y[has_up], idx[has_up]]
    return np.minimum(low, up)


def accuracy(X, y, model):
    return float(np.mean(classify(X, model) == np.asarray(y)))


def describe(model, criteria=None):
    """Tableau lisible des paramètres d'un modèle."""
    n = len(model["w"])
    criteria = criteria or [f"c{i + 1}" for i in range(n)]
    rows = {f"b^{h + 1}": model["b"][h] for h in range(len(model["b"]))}
    rows["w"] = model["w"]
    df = pd.DataFrame(rows, index=criteria).T.round(3)
    return df


if __name__ == "__main__":
    # Vérification : notre classify (numpy) reproduit exactement lincs.classify_alternatives
    d = "Datasets-Step2/type_a/cr06_ca03/"
    X, y, cats = load_dataset(d + "n0300_rep50-50_noise00.csv", d + "problem.yml")
    p = len(cats)
    t = time.time()
    problem = make_problem(X.shape[1], p)
    alts = to_alternatives(problem, X, y)
    L = lc.LearnMrsortByWeightsProfilesBreed
    pls = lc.PreprocessedLearningSet(problem, alts)
    for seed in range(5):
        mbl = L.ModelsBeingLearned(pls, 9, seed)
        ini = lc.InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(pls, mbl)
        wo = lc.OptimizeWeightsUsingGlop(pls, mbl)
        lm = L(pls, mbl, ini, wo, lc.ImproveProfilesWithAccuracyHeuristicOnCpu(pls, mbl),
               lc.ReinitializeLeastAccurate(mbl, ini, wo, 4), lc.TerminateAfterSeconds(2)).perform()
        alts_pred = to_alternatives(problem, X)
        lc.classify_alternatives(problem, lm, alts_pred)
        y_lincs = np.array([a.category_index for a in alts_pred.alternatives])
        y_np = classify(X, from_lincs(lm, X.shape[1], p))
        print(f"seed {seed} : numpy == lincs : {np.array_equal(y_lincs, y_np)} | accuracy : {np.mean(y_np == y):.3f}")
