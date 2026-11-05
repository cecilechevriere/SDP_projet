"""
Expériences du jalon 3 : chaque attaque est appliquée aux mêmes données de A (instances 0 .. k-1).
Usage : python run_brouillage.py [nb_instances]   (résultats dans results/brouillage.csv)
"""
import os
import sys
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from brouillage import experience, ATTAQUES


def one_run(args):
    attaque, s = args
    t = time.time()
    r = experience(attaque, s)
    r["temps_s"] = time.time() - t
    return r


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    configs = [("aucune", s) for s in range(k)]
    configs += [("modele_aleatoire", s) for s in range(k + 2)]     # partie 1 : agent basique, 2 instances de plus
    configs += [(a, s) for a in ATTAQUES if a != "modele_aleatoire" for s in range(k)]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for r in ex.map(one_run, configs):
            rows.append(r)
            print({a: round(b, 3) if isinstance(b, float) else b for a, b in r.items()}, flush=True)
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/brouillage.csv", index=False)
    print(df.groupby("attaque", sort=False).mean(numeric_only=True).drop(columns="instance").round(3))
