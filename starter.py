#!/usr/bin/env python3
"""
HackAlem AI — кейс «Граф денег».

Пайплайн:
  1. загружает и проверяет три parquet-файла;
  2. строит направленный взвешенный граф;
  3. считает базовые метрики узлов;
  4. присваивает объяснимые роли и role_score;
  5. считает priority_score;
  6. выполняет Louvain-кластеризацию;
  7. создаёт nodes_roles.csv, clusters.csv и top_nodes.csv.

По умолчанию:
    data/ — входные parquet
    out/  — итоговые CSV

Запуск:
    py starter.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from clustering import assign_cluster_ids, build_clusters_output, validate_clusters

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]


# ---------------------------------------------------------------- загрузка

def load(data_dir: Path):
    edges = pd.read_parquet(data_dir / "edges.parquet")
    nodes = pd.read_parquet(data_dir / "nodes.parquet")
    tx = pd.read_parquet(data_dir / "transactions.parquet")
    tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def sanity_check(edges, nodes, tx):
    """Проверки, которые стоит пройти до того, как строить модель."""
    print("=" * 64)
    print("ПРОВЕРКА ДАННЫХ")
    print("=" * 64)
    print(f"  узлов в nodes.parquet : {len(nodes):>6}")
    print(f"  рёбер                 : {len(edges):>6}")
    print(f"  транзакций            : {len(tx):>6}")
    print(f"  seed-клиентов         : {int(nodes.is_seed.sum()):>6}")
    print(f"  оборот, KZT           : {edges.sum_kzt.sum():>14,.0f}")
    print(f"  период                : {tx.date.min().date()} — {tx.date.max().date()}")

    # транзакции должны складываться в рёбра
    agg = tx.groupby(["src", "dst"]).agg(
        s=("sum_kzt", "sum"),
        c=("sum_kzt", "size"),
    ).reset_index()
    m = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    assert (m._merge == "both").all(), "edges и transactions не сходятся по парам"
    assert np.allclose(m["sum_kzt"], m["s"]), "edges.sum_kzt != сумма transactions"
    assert (m["n_tx"].astype(int) == m["c"].astype(int)).all(), "edges.n_tx != число transactions"
    print("  edges == transactions : OK")

    # узлы без единого ребра
    in_edges = set(edges.src) | set(edges.dst)
    orphans = set(nodes.gid) - in_edges
    print(f"\n  ВНИМАНИЕ: {len(orphans)} узлов нет ни в одном ребре "
          f"(из них seed: {len(orphans & set(nodes[nodes.is_seed].gid))})")
    print("  → они всё равно должны попасть в nodes_roles.csv")
    print("=" * 64, "\n")
    return orphans


# ---------------------------------------------------------------- граф

def build_graph(edges) -> nx.DiGraph:
    """Направленный граф. sum_kzt — вес ребра, n_tx — количество переводов."""
    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def basic_features(G: nx.DiGraph, nodes: pd.DataFrame) -> pd.DataFrame:
    """Базовые метрики. Это старт, а не финиш — добавляйте свои."""
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_kzt = dict(G.in_degree(weight="sum_kzt"))
    out_kzt = dict(G.out_degree(weight="sum_kzt"))
    in_tx = dict(G.in_degree(weight="n_tx"))
    out_tx = dict(G.out_degree(weight="n_tx"))
    pr = nx.pagerank(G, weight="sum_kzt")

    df = nodes[["gid", "depth", "is_seed"]].copy()
    df["in_deg"] = df.gid.map(in_deg).fillna(0).astype(int)
    df["out_deg"] = df.gid.map(out_deg).fillna(0).astype(int)
    df["in_kzt"] = df.gid.map(in_kzt).fillna(0.0)
    df["out_kzt"] = df.gid.map(out_kzt).fillna(0.0)
    df["in_tx"] = df.gid.map(in_tx).fillna(0).astype(int)
    df["out_tx"] = df.gid.map(out_tx).fillna(0).astype(int)
    df["pagerank"] = df.gid.map(pr).fillna(0.0)

    # доля полученного, которая ушла дальше. Около 1.0 — деньги не задерживаются.
    df["pass_through"] = np.where(df.in_kzt > 0, df.out_kzt / df.in_kzt.replace(0, np.nan), np.nan)

    # ЛОВУШКА КЕЙСА: узел на 4-м колене без исходящих может быть не «стоком»,
    # а просто местом, где закончился обход. Разберитесь с этим.
    df["truncated_by_depth"] = (df.depth == 4) & (df.out_deg == 0)
    return df

def assign_roles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Пороги считаем от самих данных
    in_deg_high = max(3, df["in_deg"].quantile(0.90))
    out_deg_high = max(10, df["out_deg"].quantile(0.90))
    pagerank_high = df["pagerank"].quantile(0.95)

    roles = []
    scores = []
    evidence = []

    for _, r in df.iterrows():

        # 1. COORDINATOR
        # Очень важный узел по PageRank + есть и входящие, и исходящие связи
        if (
            r["pagerank"] >= pagerank_high
            and r["in_deg"] > 0
            and r["out_deg"] > 0
        ):
            role = "coordinator"
            score = 0.95
            reason = (
                f"PageRank={r['pagerank']:.6f} >= q95={pagerank_high:.6f}; "
                f"in={r['in_deg']}, out={r['out_deg']}"
            )

        # 2. DISTRIBUTOR
        # Отправляет деньги большому числу разных клиентов
        elif r["out_deg"] >= out_deg_high:
            role = "distributor"
            score = min(1.0, 0.7 + r["out_deg"] / max(out_deg_high, 1) * 0.1)
            reason = (
                f"out_deg={r['out_deg']} >= {out_deg_high:.0f}; "
                f"out_kzt={r['out_kzt']:.0f} KZT"
            )

        # 3. CONSOLIDATOR
        # Получает деньги от большого числа разных клиентов
        elif r["in_deg"] >= in_deg_high:
            role = "consolidator"
            score = min(1.0, 0.7 + r["in_deg"] / max(in_deg_high, 1) * 0.1)
            reason = (
                f"in_deg={r['in_deg']} >= {in_deg_high:.0f}; "
                f"in_kzt={r['in_kzt']:.0f} KZT"
            )

        # 4. TRANSIT
        # Получил деньги и примерно столько же отправил дальше
        # seed специально не используем, т.к. у seed входящие данные неполные
        elif (
            not r["is_seed"]
            and r["in_deg"] > 0
            and r["out_deg"] > 0
            and pd.notna(r["pass_through"])
            and 0.8 <= r["pass_through"] <= 1.2
        ):
            role = "transit"

            # Чем ближе pass_through к 1, тем увереннее
            score = max(0.6, 1 - abs(r["pass_through"] - 1))

            reason = (
                f"Получил {r['in_kzt']:.0f} KZT, "
                f"отправил {r['out_kzt']:.0f} KZT, "
                f"pass={r['pass_through']:.2f}"
            )

        # 5. TERMINAL
        # Нет исходящих, но НЕ на глубине 4
        elif (
            r["out_deg"] == 0
            and r["in_deg"] > 0
            and not r["truncated_by_depth"]
        ):
            role = "terminal"
            score = 0.80
            reason = (
                f"Получил {r['in_kzt']:.0f} KZT, "
                f"исходящих связей=0, depth={r['depth']}"
            )

        # 6. PERIPHERAL
        else:
            role = "peripheral"
            score = 0.50

            if r["truncated_by_depth"]:
                reason = (
                    f"Узел depth=4, исходящих=0; "
                    f"конец графа, terminal не подтвержден"
                )
            else:
                reason = (
                    f"Входов={r['in_deg']}, выходов={r['out_deg']}, "
                    f"PageRank={r['pagerank']:.6f}"
                )

        roles.append(role)
        scores.append(round(float(score), 3))
        evidence.append(reason[:200])

    df["role"] = roles
    df["role_score"] = scores
    df["evidence"] = evidence

    return df

def calculate_priority(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Нормализуем показатели в диапазон 0..1
    df["pr_norm"] = df["pagerank"] / max(df["pagerank"].max(), 1e-9)

    df["in_deg_norm"] = (
        df["in_deg"] / max(df["in_deg"].max(), 1)
    )

    df["out_deg_norm"] = (
        df["out_deg"] / max(df["out_deg"].max(), 1)
    )

    total_money = df["in_kzt"] + df["out_kzt"]

    df["money_norm"] = (
        total_money / max(total_money.max(), 1)
    )

    # Базовый score
    df["priority_score"] = (
        0.35 * df["pr_norm"] +
        0.25 * df["in_deg_norm"] +
        0.20 * df["out_deg_norm"] +
        0.20 * df["money_norm"]
    )

    # Бонус за важную роль
    role_bonus = {
        "coordinator": 0.15,
        "consolidator": 0.10,
        "distributor": 0.08,
        "transit": 0.05,
        "terminal": 0.02,
        "peripheral": 0.00
    }

    df["priority_score"] += df["role"].map(role_bonus).fillna(0)

    # Ограничиваем 0..1
    df["priority_score"] = df["priority_score"].clip(0, 1)

    df["priority_score"] = df["priority_score"].round(3)

    return df




# ---------------------------------------------------------------- выгрузки

def write_outputs(df: pd.DataFrame, clusters_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. nodes_roles.csv
    roles = df[
        [
            "gid",
            "role",
            "role_score",
            "cluster_id",
            "priority_score",
            "evidence",
            "in_deg",
            "out_deg",
            "in_kzt",
            "out_kzt",
            "pagerank",
            "pass_through",
            "depth",
            "is_seed",
            "truncated_by_depth"
        ]
    ].copy()

    roles.to_csv(out_dir / "nodes_roles.csv", index=False)

    # 2. clusters.csv — норм
    clusters_df.to_csv(out_dir / "clusters.csv", index=False)

    # 3. top_nodes.csv — норм
    top = df.sort_values(
        by="priority_score",
        ascending=False
    ).head(20).copy()

    top["rank"] = range(1, len(top) + 1)
    top["why"] = top.apply(
        lambda r: (
            f"priority={float(r['priority_score']):.3f}; "
            f"role={r['role']}; "
            f"in={int(r['in_deg'])}; out={int(r['out_deg'])}; "
            f"PageRank={float(r['pagerank']):.6f}. "
            f"{r['evidence']}"
        )[:200],
        axis=1,
    )

    top = top[
        [
            "rank",
            "gid",
            "role",
            "priority_score",
            "why"
        ]
    ]

    top.to_csv(out_dir / "top_nodes.csv", index=False)
    
    print(f"Выгрузки записаны в {out_dir}/")


# ---------------------------------------------------------------- подсказки

def hints(G: nx.DiGraph, df: pd.DataFrame):
    """Куда смотреть дальше. Ответов здесь нет — только направления."""
    print("\nС ЧЕГО НАЧАТЬ")
    print("-" * 64)
    print(f"  узлов, получающих от 3+ разных плательщиков : {(df.in_deg >= 3).sum()}")
    print(f"  узлов, рассылающих на 10+ получателей       : {(df.out_deg >= 10).sum()}")
    print(f"  узлов и с входом, и с выходом               : {((df.in_deg > 0) & (df.out_deg > 0)).sum()}")
    print(f"  узлов, обрезанных 4-м коленом               : {df.truncated_by_depth.sum()}  <- разберитесь")
    print(f"  слабосвязных компонент                      : {nx.number_weakly_connected_components(G)}")
    print("""
  Вопросы, на которые стоит ответить метриками:
    * чем «деньги пришли и остались» отличается от «пришли и ушли дальше»?
    * что важнее для роли — количество плательщиков или сумма?
    * узел собирает средства от нескольких SEED — это случайность или структура?
    * если убрать узел, сеть распадётся или переживёт?

  Полезное в networkx: pagerank, hits, betweenness_centrality,
  community.louvain_communities, simple_cycles, all_simple_paths.
  Не забудьте: граф НАПРАВЛЕННЫЙ и ВЗВЕШЕННЫЙ.
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", help="папка с parquet-файлами")
    ap.add_argument("--out", default="./out", help="куда писать выгрузки")
    a = ap.parse_args()

    edges, nodes, tx = load(Path(a.data))
    sanity_check(edges, nodes, tx)

    G = build_graph(edges)

    df = basic_features(G, nodes)
    df = assign_roles(df)
    df = calculate_priority(df)
    df = assign_cluster_ids(df, G)

    clusters_df = build_clusters_output(df, G)
    validate_clusters(df, nodes, clusters_df)

    write_outputs(df, clusters_df, Path(a.out))

    hints(G, df)


if __name__ == "__main__":
    main()
