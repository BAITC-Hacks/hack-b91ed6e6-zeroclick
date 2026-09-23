import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"

ROLES = {
    "consolidator",
    "transit",
    "distributor",
    "terminal",
    "coordinator",
    "peripheral",
}


def main():
    nodes = pd.read_parquet(DATA / "nodes.parquet")
    roles = pd.read_csv(OUT / "nodes_roles.csv")
    clusters = pd.read_csv(OUT / "clusters.csv")
    top = pd.read_csv(OUT / "top_nodes.csv")

    required_roles = [
        "gid", "role", "role_score", "cluster_id",
        "priority_score", "evidence",
    ]
    required_clusters = [
        "cluster_id", "n_nodes", "n_seed",
        "sum_kzt_internal", "top_gids", "hypothesis",
    ]
    required_top = ["rank", "gid", "role", "priority_score", "why"]

    assert list(roles.columns) == required_roles, (
        f"nodes_roles.csv: схема должна быть ровно {required_roles}, получено {list(roles.columns)}"
    )
    assert list(clusters.columns) == required_clusters, (
        f"clusters.csv: схема должна быть ровно {required_clusters}, получено {list(clusters.columns)}"
    )
    assert list(top.columns) == required_top, (
        f"top_nodes.csv: схема должна быть ровно {required_top}, получено {list(top.columns)}"
    )

    assert len(roles) == len(nodes), f"nodes_roles.csv: ожидалось {len(nodes)} строк, получено {len(roles)}"
    assert set(roles["gid"]) == set(nodes["gid"]), "nodes_roles.csv: набор gid не совпадает с nodes.parquet"

    assert roles[required_roles].notna().all().all(), "nodes_roles.csv: есть пропуски"
    assert roles["role"].isin(ROLES).all(), "nodes_roles.csv: обнаружена неизвестная роль"
    assert roles["role_score"].between(0, 1).all(), "role_score должен быть 0..1"
    assert roles["priority_score"].between(0, 1).all(), "priority_score должен быть 0..1"
    assert roles["evidence"].astype(str).str.strip().ne("").all(), "evidence не должен быть пустым"
    assert roles["evidence"].astype(str).str.len().le(200).all(), "evidence длиннее 200 символов"
    assert roles["cluster_id"].notna().all(), "есть узлы без cluster_id"

    assert clusters[required_clusters].notna().all().all(), "clusters.csv: есть пропуски"
    assert int(clusters["n_nodes"].sum()) == len(nodes), "clusters.csv: сумма n_nodes не равна числу узлов"

    assert len(top) >= 20, "top_nodes.csv должен содержать не менее 20 строк"
    assert top[required_top].notna().all().all(), "top_nodes.csv: есть пропуски"
    assert top["priority_score"].is_monotonic_decreasing, "top_nodes.csv не отсортирован по priority_score"
    assert top["why"].astype(str).str.strip().ne("").all(), "top_nodes.csv: why пустой"

    # Ловушка depth=4 проверяется через внутренние метрики.
    metrics_path = ROOT / "runtime" / "nodes_metrics.parquet"
    if metrics_path.exists():
        metrics = pd.read_parquet(metrics_path)[["gid", "depth", "out_deg"]]
        check = roles.merge(metrics, on="gid", how="left")
        bad = check[
            (check["depth"] == 4)
            & (check["out_deg"] == 0)
            & (check["role"] == "terminal")
        ]
        assert bad.empty, "depth=4 с out_deg=0 ошибочно помечен terminal"

    print("VALIDATION OK")
    print(f"  nodes_roles.csv : {len(roles)} узлов")
    print(f"  clusters.csv    : {len(clusters)} кластеров")
    print(f"  top_nodes.csv   : {len(top)} строк")


if __name__ == "__main__":
    main()
