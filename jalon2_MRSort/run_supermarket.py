"""
Application aux datasets "supermarket" (Datasets-Step2).
Pour chaque fichier : découpage apprentissage / test (80 % / 20 %), puis
  - 1 MR-Sort appris avec lincs (partie 1 du jalon) ;
  - 2 MR-Sort appris avec l'heuristique inv-2-MR-Sort (partie 2) ;
  - témoin : 2 MR-Sort appris sur une partition aléatoire (sans réaffectation).
Usage : python run_supermarket.py [motif]   ex. "cr06_ca03" (défaut), "type_a/cr08", "" (tous les fichiers)
Résultats : results/supermarket_<motif>.csv et les modèles appris en JSON dans results/models/.
"""
import sys
import json
import time
import glob
import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from mrsort_lincs import load_dataset, learn_mrsort, classify
from two_mrsort import learn_two_mrsort, oracle_accuracy, random_halves

ROOT = "Datasets-Step2"


def model_json(m):
    return {"b": np.where(np.isinf(m["b"]), None, m["b"]).tolist(), "w": m["w"].tolist(), "lam": float(m["lam"])}


def one_file(path):
    rel = os.path.relpath(path, ROOT)
    typ, pb, fname = rel.split(os.sep)
    X, y, cats = load_dataset(path, os.path.join(ROOT, typ, pb, "problem.yml"))
    p = len(cats)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]

    t = time.time()
    single = learn_mrsort(X[tr], y[tr], p, seed=0, max_seconds=5)
    t1 = time.time() - t
    t = time.time()
    two = learn_two_mrsort(X[tr], y[tr], p, seed=0, learn_kw=dict(max_seconds=1))
    t2 = time.time() - t
    halves = random_halves(X[tr], y[tr], p, seed=0, learn_kw=dict(max_seconds=1))

    os.makedirs("results/models", exist_ok=True)
    name = f"{typ}_{pb}_{fname[:-4]}"
    json.dump({"un_modele": model_json(single), "deux_modeles": [model_json(m) for m in two["models"]],
               "tailles_segments": np.bincount(two["seg"], minlength=2).tolist()},
              open(f"results/models/{name}.json", "w"), indent=1)

    parts = fname[:-4].split("_")                # n0300_rep50-50_noise00
    return {
        "type": typ, "probleme": pb, "m": int(parts[0][1:]), "rep": parts[1][3:], "bruit": int(parts[2][5:]),
        "n": X.shape[1], "p": p,
        "train_1_modele": float(np.mean(classify(X[tr], single) == y[tr])),
        "test_1_modele": float(np.mean(classify(X[te], single) == y[te])),
        "train_2_modeles": two["n_ok"] / len(tr),
        "test_2_modeles": oracle_accuracy(X[te], y[te], two["models"]),
        "train_temoin": oracle_accuracy(X[tr], y[tr], halves),
        "test_temoin": oracle_accuracy(X[te], y[te], halves),
        "taille_seg_min": float(np.bincount(two["seg"], minlength=2).min() / len(tr)),
        "temps_1_modele_s": t1, "temps_2_modeles_s": t2,
    }


if __name__ == "__main__":
    motif = sys.argv[1] if len(sys.argv) > 1 else "cr06_ca03"
    files = sorted(f for f in glob.glob(f"{ROOT}/*/*/*.csv") if motif in f)
    print(len(files), "fichiers")
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f, r in zip(files, ex.map(one_file, files)):
            rows.append(r)
            print(os.path.relpath(f, ROOT), {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/supermarket_{motif.replace('/', '_') or 'tous'}.csv", index=False)
    cols = ["train_1_modele", "test_1_modele", "train_2_modeles", "test_2_modeles", "train_temoin", "test_temoin"]
    print(df.groupby(["type", "rep", "bruit"])[cols].mean().round(3))
