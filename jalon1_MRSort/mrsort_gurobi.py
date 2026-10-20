import numpy as np
import gurobipy as gp
from gurobipy import GRB


def classify(X, b, w, lam):
    """
    Classe les objets avec un modèle MR-Sort à p catégories.
    b : tableau (p-1, n) des frontières ; classes numérotées 0 .. p-1.
    Un objet est dans la classe c s'il bat (coalition >= lam) exactement c frontières.
    Tolérances 1e-6 alignées sur celles de Gurobi.
    """
    X = np.asarray(X, dtype=float)
    b = np.atleast_2d(b)
    gagne = np.stack([(X >= bh - 1e-6) @ w >= lam - 1e-6 for bh in b])  # (p-1, m)
    return gagne.sum(axis=0).astype(int)


def learn_mr_sort(X, y, p=None, eps=1e-3, eps_w=1e-3, vmax=20.0, time_limit=60, verbose=False, compact=False):
    """
    Apprend un modèle MR-Sort à p catégories (questions vii et viii), objectif : max alpha.
    X : (m, n) performances ; y : (m,) classes dans {0, ..., p-1} (0 = la pire).
    eps   : écart pour l'inégalité stricte x_i < b_i (échelle des notes ; 1 si notes entières).
    eps_w : écart pour l'inégalité stricte sur la somme des poids (échelle [0,1]).
    compact=True : version allégée (moins de contraintes), de même optimum. Pour une frontière
    que l'objet doit atteindre, seules les contraintes "delta=1 => x>=b" et "w_i(x) <= delta*w_i"
    sont utiles ; pour une frontière qu'il ne doit pas atteindre, seules "delta=0 => x<b" et
    "w_i(x) >= delta + w_i - 1".
    Retourne (b, w, lam, alpha, nb_erreurs) avec b de forme (p-1, n).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    m, n = X.shape
    if p is None:
        p = int(y.max()) + 1
    M = vmax + eps

    mdl = gp.Model("MRSort")
    mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.TimeLimit = time_limit

    # Frontière h (h = 0..p-2) sépare la classe h de la classe h+1
    b     = mdl.addVars(p - 1, n, lb=0.0, ub=vmax, name="b")
    w     = mdl.addVars(n, lb=0.0, ub=1.0, name="w")
    lam   = mdl.addVar(lb=0.0, ub=1.0, name="lambda")
    alpha = mdl.addVar(lb=-1.0, ub=1.0, name="alpha")

    mdl.addConstr(w.sum() == 1)
    # Frontières ordonnées : b^h <= b^(h+1) sur chaque critère
    for h in range(p - 2):
        for i in range(n):
            mdl.addConstr(b[h, i] <= b[h + 1, i])

    def coalition(x, h, atteint):
        """Somme des poids des critères où l'objet x atteint la frontière h.
        atteint=True : l'objet doit atteindre la frontière ; False : il ne doit pas l'atteindre."""
        termes = []
        for i in range(n):
            d  = mdl.addVar(vtype=GRB.BINARY)          # delta_i(x) pour la frontière h
            wx = mdl.addVar(lb=0.0, ub=1.0)            # w_i(x) = delta * w_i
            # Question v : définition de delta
            if not compact or atteint:
                mdl.addConstr(b[h, i] - M * (1 - d) <= float(X[x, i]))     # delta=1 => x >= b
            if not compact or not atteint:
                mdl.addConstr(b[h, i] - eps + M * d >= float(X[x, i]))     # delta=0 => x < b
            # Question vi : wx = delta * w_i
            if not compact or atteint:
                mdl.addConstr(wx <= w[i])
                mdl.addConstr(wx <= d)
            if not compact or not atteint:
                mdl.addConstr(wx >= d + w[i] - 1)
            termes.append(wx)
        return gp.quicksum(termes)

    for x in range(m):
        c = y[x]
        if c >= 1:        # doit atteindre la frontière inférieure b^c (indice c-1)
            mdl.addConstr(coalition(x, c - 1, True) >= lam + alpha)
        if c <= p - 2:    # ne doit pas atteindre la frontière supérieure (indice c)
            mdl.addConstr(coalition(x, c, False) <= lam - alpha - eps_w)

    mdl.setObjective(alpha, GRB.MAXIMIZE)
    mdl.optimize()

    b_v = np.array([[b[h, i].X for i in range(n)] for h in range(p - 1)])
    w_v = np.array([w[i].X for i in range(n)])
    n_err = int(np.sum(classify(X, b_v, w_v, lam.X) != y))
    return b_v, w_v, lam.X, alpha.X, n_err


def random_model(n, p, rng, vmax=20.0):
    """Modèle MR-Sort aléatoire : poids, lambda, p-1 frontières ordonnées."""
    w = rng.dirichlet(np.ones(n))
    lam = rng.uniform(0.5, 0.8)
    b = np.sort(rng.uniform(0.15 * vmax, 0.85 * vmax, (p - 1, n)), axis=0)
    return b, w, lam


def experiment(n=4, p=2, m_learn=40, m_test=1000, seed=0, **kw):
    """Protocole du jalon 1 : ground truth -> learning set -> apprentissage -> test."""
    rng = np.random.default_rng(seed)
    b0, w0, lam0 = random_model(n, p, rng)
    X_l = rng.uniform(0, 20, (m_learn, n))
    y_l = classify(X_l, b0, w0, lam0)
    b, w, lam, alpha, err = learn_mr_sort(X_l, y_l, p=p, **kw)
    X_t = rng.uniform(0, 20, (m_test, n))
    quality = np.mean(classify(X_t, b0, w0, lam0) == classify(X_t, b, w, lam))
    return quality, err


if __name__ == "__main__":
    # 1) Exemple de la question i (p = 2)
    X = [[11, 12, 13, 12], [13, 12, 15, 10], [11, 13, 11, 13], [12, 12, 16, 12]]
    y = classify(X, np.full((1, 4), 12.0), np.array([0.25, 0.15, 0.2, 0.4]), 0.66)
    print("Classes (Alice, Bob, Charlie, David) :", y)
    b, w, lam, alpha, err = learn_mr_sort(X, y, p=2)
    print("b =", b.round(2).tolist(), "w =", w.round(3), "lambda =", round(lam, 3),
          "alpha =", round(alpha, 3), "erreurs =", err)

    # 2) Test sur plusieurs modèles aléatoires, pour plusieurs valeurs de p
    for p in [2, 3, 4]:
        qs = [experiment(p=p, seed=k, time_limit=30)[0] for k in range(5)]
        print(f"p = {p} : qualité moyenne {np.mean(qs):.3f}")
