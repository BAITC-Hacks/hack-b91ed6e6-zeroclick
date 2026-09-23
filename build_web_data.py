from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"
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
    result = []
    for row in df.to_dict(orient="records"):
        result.append({k: clean_value(v) for k, v in row.items()})
    return result


def main():
    edges_path = DATA / "edges.parquet"
    roles_path = OUT / "nodes_roles.csv"
    top_path = OUT / "top_nodes.csv"
    clusters_path = OUT / "clusters.csv"

    required = [edges_path, roles_path, top_path, clusters_path]
    missing = [p for p in required if not p.exists()]
    if missing:
        names = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Не хватает файлов для сайта. Сначала запусти starter.py:\n" + names
        )

    edges = pd.read_parquet(edges_path)[["src", "dst", "sum_kzt", "n_tx", "depth"]].copy()
    roles = pd.read_csv(roles_path)
    top = pd.read_csv(top_path)
    clusters = pd.read_csv(clusters_path)

    # Оставляем только то, что реально нужно интерфейсу.
    role_cols = [
        "gid", "role", "role_score", "cluster_id", "priority_score", "evidence",
        "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank", "pass_through",
        "depth", "is_seed", "truncated_by_depth"
    ]
    roles = roles[[c for c in role_cols if c in roles.columns]].copy()

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
