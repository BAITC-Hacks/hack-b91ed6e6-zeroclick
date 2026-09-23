from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
RUNTIME = ROOT / "runtime"
WEB_DATA = ROOT / "web" / "graph_data.json"


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [
        {k: clean_value(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def main():
    required = [
        DATA / "edges.parquet",
        OUT / "nodes_roles.csv",
        OUT / "top_nodes.csv",
        OUT / "clusters.csv",
        RUNTIME / "nodes_metrics.parquet",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Не хватает файлов для интерфейса:\n"
            + "\n".join(str(p) for p in missing)
        )

    edges = pd.read_parquet(DATA / "edges.parquet")[
        ["src", "dst", "sum_kzt", "n_tx", "depth"]
    ].copy()

    roles = pd.read_csv(OUT / "nodes_roles.csv")
    metrics = pd.read_parquet(RUNTIME / "nodes_metrics.parquet")
    roles = roles.merge(metrics, on="gid", how="left", validate="one_to_one")

    top = pd.read_csv(OUT / "top_nodes.csv")
    clusters = pd.read_csv(OUT / "clusters.csv")

    payload = {
        "meta": {
            "n_nodes": int(len(roles)),
            "n_edges": int(len(edges)),
            "n_clusters": int(roles["cluster_id"].nunique()),
            "max_priority": float(roles["priority_score"].max()),
        },
        "roles": records(roles),
        "edges": records(edges),
        "top_nodes": records(top),
        "clusters": records(clusters),
    }

    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Web data: {WEB_DATA}")
    print(
        f"{payload['meta']['n_nodes']} узлов, "
        f"{payload['meta']['n_edges']} рёбер, "
        f"{payload['meta']['n_clusters']} кластеров"
    )


if __name__ == "__main__":
    main()
