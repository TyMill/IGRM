"""Reproducible v1.6 counterexample-and-power experiment.

No machine learning training and no use of hidden risk values in certification.
The population truth is used only to evaluate false certifications and oracle gap.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.overlap_v16 import (certifier, closed_form_three_state, margin, shared_allocation,
                                  simultaneous_radius)


def run(out: Path, repeats: int = 300, seed: int = 20261112) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    q3 = np.ones(3) / 3
    r3 = np.full(3, .08)
    family3 = ((0, 1), (1, 2))
    design = shared_allocation(q3, family3, [2 * (.15 - .08) / 3] * 2)
    cf = closed_form_three_state()
    assert abs(cf["shared_total"] - design.objective_real) < 1e-4
    assert all(simultaneous_radius(q3, np.asarray(design.n_integer), a, 2, .05) <= cf["margin"] / 2 for a in family3)

    # Repeated synthetic worlds. The true risks only appear in the generator and
    # in the *offline* evaluation column.
    q4 = np.ones(4) / 4
    r_worlds = {"mixed_safe_unsafe": np.array([.16, .09, .09, .24]),
                "boundary_safe": np.array([.14, .14, .14, .17]),
                "all_unsafe": np.array([.22, .18, .19, .26])}
    family4 = ((0, 1), (1, 2), (2, 3), (0, 1, 2))
    n_levels = (200, 1000, 4000)
    rows = []
    for world_name, r in r_worlds.items():
        oracle = max((sum(q4[list(a)]) for a in family4 if margin(q4, r, a, .15) >= 0), default=0.0)
        for n_state in n_levels:
            for repeat in range(repeats):
                errors = rng.binomial(n_state, r)
                n = np.full(4, n_state)
                for method in ("shared-hoeffding", "cp-box"):
                    a = certifier(q4, errors, n, family4, tau=.15, gamma=.1, method=method)
                    true_cov = float(np.sum(q4[list(a)])) if a else 0.0
                    true_risk = float(np.average(r[list(a)], weights=q4[list(a)])) if a else None
                    rows.append({"world": world_name, "repeat": repeat, "n_per_state": n_state,
                                 "method": method, "selected_mask": "-" if a is None else ",".join(map(str,a)),
                                 "certified_cov": true_cov, "population_risk": true_risk if true_risk is not None else "",
                                 "unsafe_certification": int(bool(a) and (true_risk > .15 + 1e-12)),
                                 "oracle_cov": oracle})
    file_name = out / "overlap_multipolicy_v16.csv"
    with file_name.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    summary = []
    for world in r_worlds:
        for n in n_levels:
            for method in ("shared-hoeffding", "cp-box"):
                subset = [row for row in rows if row["world"] == world and row["n_per_state"]==n and row["method"]==method]
                summary.append({"world":world,"n_per_state":n,"method":method,
                                "mean_certified_coverage":round(float(np.mean([x["certified_cov"] for x in subset])),6),
                                "certification_rate":round(float(np.mean([x["selected_mask"]!="-" for x in subset])),6),
                                "false_certifications":sum(x["unsafe_certification"] for x in subset),
                                "oracle_coverage":subset[0]["oracle_cov"],"replicates":repeats})
    payload={"design_closed_form":cf,"design_numeric":{"n_real":design.n_real,"n_integer":design.n_integer,"objective":design.objective_real},
             "seed":seed,"repeats":repeats,"worlds":{key:val.tolist() for key,val in r_worlds.items()},
             "family":[list(a) for a in family4], "n_levels":n_levels, "summary":summary,
             "comparison_scope":"known q, fixed state quotas, synthetic independent binomial outcomes only"}
    (out/"overlap_summary_v16.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    with (out/"overlap_summary_v16.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print(json.dumps({"rows":len(rows),"design":payload["design_numeric"],"summary":summary},indent=2))
    return payload


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",default="data")
    parser.add_argument("--repeats",type=int,default=300)
    parser.add_argument("--seed",type=int,default=20261112)
    args=parser.parse_args()
    run(Path(args.output),args.repeats,args.seed)
