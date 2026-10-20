import numpy as np
from mrsort_gurobi import learn_mr_sort, classify

lignes = [l.split() for l in open("dataset0.txt") if l.strip()]
data = np.array([[float(v) for v in l[1:]] for l in lignes[1:]])   # saute l'en-tête et le nom de l'objet
X = data[:, :-1]
y = data[:, -1].astype(int) - 1          # classes 1..3 -> 0..2
p = int(y.max()) + 1
print("n objets =", X.shape[0], "| n critères =", X.shape[1], "| p =", p,
      "| effectifs =", np.bincount(y))

# eps = 1 car les notes sont entières ; vmax = 100 (échelle des notes)
b, w, lam, alpha, err = learn_mr_sort(X, y, p=p, eps=1.0, eps_w=1e-3, vmax=100.0, time_limit=250, verbose=False, compact=True)
print("\nFrontières b (lignes = frontière entre classes h et h+1) :\n", b.round(2))
print("Poids w :", w.round(4), "| somme =", w.sum().round(4))
print("lambda =", round(lam, 4), "| alpha =", round(alpha, 4))
print("Erreurs sur le learning set :", err, "/", len(y))
print("Précision :", 1 - err / len(y))
