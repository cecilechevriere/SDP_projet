"""
Validation de l'heuristique inv-2-MR-Sort sur des données synthétiques dont on connaît les segments et les modèles
(même protocole qu'au jalon 1, étendu à 2 segments) :
  - 2 modèles MR-Sort "ground truth" aléatoires, une part `share` des objets suit le modèle 1, le reste le modèle 2 ;
  - on apprend (a) un seul MR-Sort, (b) 2 MR-Sort avec l'heuristique, (c) 2 MR-Sort sur la VRAIE partition (plafond) ;
  - on mesure la fidélité de chaque modèle appris à son modèle réel sur 5000 objets de test.
Usage : python run_synthetic.py [nb_instances_par_config]   (résultats dans results/synthetic.csv)
"""
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from mrsort_lincs import learn_mrsort, classify
from two_mrsort import (learn_two_mrsort, make_two_segment_data, segment_recovery, matched_models)

N, P, M = 6, 3, 300          # comme le dataset supermarket principal (6 critères, 3 classes, 300 objets)


def one_run(args):
    share, noise, s = args
    rng = np.random.default_rng(1000 + s)
    X, y, seg, gt = make_two_segment_data(N, P, M, share, noise, rng)
    Xt = rng.uniform(0, 1, (5000, N))
    yt = [classify(Xt, g) for g in gt]
    disc = classify(X, gt[0]) != classify(X, gt[1])          # objets où les 2 vrais modèles divergent

    t = time.time()
    single = learn_mrsort(X, y, P, seed=s, max_seconds=5)
    two = learn_two_mrsort(X, y, P, seed=s, learn_kw=dict(max_seconds=1))
    duration = time.time() - t
    mm = matched_models(seg[disc], two["seg"][disc], two["models"])
    ceiling = [learn_mrsort(X[seg == k], y[seg == k], P, seed=s, max_seconds=1) for k in (0, 1)]

    return {
        "share": share, "noise": noise, "instance": s,
        "accord_vrais_modeles": float(np.mean(yt[0] == yt[1])),
        "train_1_modele": float(np.mean(classify(X, single) == y)),
        "train_2_modeles": two["n_ok"] / M,
        "segments_retrouves": segment_recovery(seg[disc], two["seg"][disc]),
        "fid_M1_heuristique": float(np.mean(classify(Xt, mm[0]) == yt[0])),
        "fid_M2_heuristique": float(np.mean(classify(Xt, mm[1]) == yt[1])),
        "fid_M1_1_modele": float(np.mean(classify(Xt, single) == yt[0])),
        "fid_M2_1_modele": float(np.mean(classify(Xt, single) == yt[1])),
        "fid_M1_plafond": float(np.mean(classify(Xt, ceiling[0]) == yt[0])),
        "fid_M2_plafond": float(np.mean(classify(Xt, ceiling[1]) == yt[1])),
        "temps_s": duration,
    }


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    configs = [(share, noise, s) for share in (0.5, 0.75) for noise in (0.0, 0.1) for s in range(k)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        rows = []
        for r in ex.map(one_run, configs):
            rows.append(r)
            print({a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/synthetic.csv", index=False)
    print(df.groupby(["share", "noise"]).mean(numeric_only=True).drop(columns="instance").round(3).T)
