#!/usr/bin/env python3
"""
clustering.py — модуль кластеризации графа (зона ответственности: Человек 3).

Обязанности этого модуля:
  * разбить граф на кластеры (community detection);
  * посчитать сводные метрики по каждому кластеру;
  * сохранить clusters.csv в требуемой ТЗ схеме.

Этот модуль НЕ делает и не должен делать:
  * не присваивает role / role_score / evidence — это core (Человек 1);
  * не считает priority_score — это core (Человек 1);
  * не меняет структуру графа и не переопределяет basic_features().

Зависимость от core минимальна и однонаправленна: этому модулю нужен
только DiGraph G (из build_graph) и DataFrame df с колонками
gid, is_seed, pagerank (из basic_features). Никаких role/priority полей
модуль не читает и не создаёт — так изменения в role-логике другого
участника не могут незаметно сломать кластеризацию.

Алгоритм: Louvain community detection (networkx.community.louvain_communities).
Библиотека (networkx) уже есть в зависимостях проекта, новых пакетов не
добавляется. seed зафиксирован — результат детерминирован между запусками
на одном и том же графе.
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

CLUSTER_SEED = 42
TOP_N_GIDS = 5


# ---------------------------------------------------------------- кластеризация

def cluster_graph(G: nx.DiGraph, seed: int = CLUSTER_SEED) -> dict:
    """
    Разбивает граф на кластеры алгоритмом Louvain.

    Louvain определён для неориентированных графов, поэтому работаем на
    неориентированной проекции G.to_undirected(); вес ребра (sum_kzt)
    сохраняется и учитывается при оптимизации модулярности.

    Возвращает {gid: cluster_id} только для узлов, у которых есть хотя бы
    одно ребро в G (т.е. которые попали в edges.parquet). Узлы без единого
    ребра ("orphans" по терминологии sanity_check в starter.py) сюда не
    попадают — их кластер_id назначается на следующем шаге, в assign_cluster_ids.

    Детерминированность: при одинаковых G и seed networkx.community.
    louvain_communities возвращает одинаковое разбиение между запусками.
    """
    if G.number_of_nodes() == 0:
        return {}

    UG = G.to_undirected()
    communities = nx.community.louvain_communities(UG, weight="sum_kzt", seed=seed)

    cluster_map: dict = {}
    for cluster_id, community in enumerate(communities):
        for gid in community:
            cluster_map[gid] = cluster_id
    return cluster_map


def assign_cluster_ids(df: pd.DataFrame, G: nx.DiGraph, seed: int = CLUSTER_SEED) -> pd.DataFrame:
    """
    Добавляет колонку cluster_id к df (одна строка на каждый gid из nodes.parquet).

    Требование ТЗ: каждый существующий gid из nodes.parquet должен получить
    cluster_id — включая узлы без единого ребра (orphans). Такие узлы
    получают собственный уникальный cluster_id, идущий следом за
    Louvain-кластерами, а не проваливаются в один общий "мусорный" кластер
    (это исказило бы n_nodes/sum_kzt_internal их соседей по индексу).
    """
    df = df.copy()
    cluster_map = cluster_graph(G, seed=seed)

    next_cluster_id = (max(cluster_map.values()) + 1) if cluster_map else 0
    missing = [gid for gid in df["gid"] if gid not in cluster_map]
    for gid in missing:
        cluster_map[gid] = next_cluster_id
        next_cluster_id += 1

    df["cluster_id"] = df["gid"].map(cluster_map).astype(int)

    if missing:
        logger.info("%d узлов без рёбер получили отдельный cluster_id", len(missing))

    return df


# ---------------------------------------------------------------- clusters.csv

def _cluster_hypothesis(n_nodes: int, n_seed: int, internal_edge_ratio: float | None) -> str:
    """
    Короткая человекочитаемая СТРУКТУРНАЯ гипотеза о кластере, основанная
    только на реально посчитанных метриках этого же кластера.

    Намеренно НЕ используются role/priority_score другого участника и не
    делается никаких утверждений о личностях или незаконной деятельности —
    только статистико-структурное описание (размер, доля seed, доля
    внутренних связей).
    """
    parts = []

    seed_share = (n_seed / n_nodes) if n_nodes else 0.0
    if n_seed == 0:
        parts.append("seed-узлов в кластере нет")
    elif seed_share >= 0.5:
        parts.append(f"высокая доля seed-узлов ({seed_share:.0%})")

    if n_nodes <= 2:
        parts.append("минимальный кластер (2 и менее узлов)")
    elif n_nodes >= 20:
        parts.append(f"крупный кластер ({n_nodes} узлов)")

    if internal_edge_ratio is not None:
        if internal_edge_ratio >= 0.7:
            parts.append(f"большинство связей узлов внутренние ({internal_edge_ratio:.0%} рёбер кластера)")
        elif internal_edge_ratio <= 0.2:
            parts.append("узлы слабо связаны между собой, кластер скорее формальный")

    if not parts:
        parts.append("выраженных структурных особенностей не обнаружено")

    return "; ".join(parts).capitalize()


def build_clusters_output(df: pd.DataFrame, G: nx.DiGraph, top_n: int = TOP_N_GIDS) -> pd.DataFrame:
    """
    Строит итоговую таблицу clusters.csv по df с колонкой cluster_id.

    top_gids: топ-N узлов кластера по pagerank. pagerank уже считается
    core-модулем в basic_features() по всему графу (nx.pagerank, weight=
    sum_kzt) — это объективная, воспроизводимая графовая метрика значимости
    узла, не зависящая от role/priority_score. Явно НЕ используется
    priority_score, т.к. он включает role_bonus — субъективную надстройку
    другого модуля.

    sum_kzt_internal: сумма sum_kzt всех рёбер G, у которых И src, И dst
    принадлежат одному и тому же cluster_id. Считается за один проход по
    рёбрам графа — O(E), без квадратичной сложности по узлам кластера.
    """
    if "cluster_id" not in df.columns:
        raise ValueError("df должен содержать cluster_id — сначала вызовите assign_cluster_ids()")
    if "pagerank" not in df.columns:
        raise ValueError("df должен содержать pagerank — колонка из core.basic_features()")

    gid_to_cluster = dict(zip(df["gid"], df["cluster_id"]))

    internal_sum: dict = {}
    internal_edges: dict = {}
    touching_edges: dict = {}

    for u, v, data in G.edges(data=True):
        cu = gid_to_cluster.get(u)
        cv = gid_to_cluster.get(v)
        w = data.get("sum_kzt", 0.0)

        if cu is not None:
            touching_edges[cu] = touching_edges.get(cu, 0) + 1
        if cv is not None and cv != cu:
            touching_edges[cv] = touching_edges.get(cv, 0) + 1
        if cu is not None and cu == cv:
            internal_sum[cu] = internal_sum.get(cu, 0.0) + w
            internal_edges[cu] = internal_edges.get(cu, 0) + 1

    rows = []
    for cluster_id, group in df.groupby("cluster_id"):
        n_nodes = len(group)
        n_seed = int(group["is_seed"].sum())

        top_gids = (
            group.sort_values("pagerank", ascending=False)
            .head(top_n)["gid"]
            .astype(str)
            .tolist()
        )

        touching = touching_edges.get(cluster_id, 0)
        internal = internal_edges.get(cluster_id, 0)
        internal_ratio = (internal / touching) if touching else None

        rows.append({
            "cluster_id": int(cluster_id),
            "n_nodes": n_nodes,
            "n_seed": n_seed,
            "sum_kzt_internal": round(internal_sum.get(cluster_id, 0.0), 2),
            "top_gids": ",".join(top_gids),
            "hypothesis": _cluster_hypothesis(n_nodes, n_seed, internal_ratio),
        })

    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def save_clusters_csv(clusters_df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "clusters.csv"
    clusters_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------- проверки качества

def validate_clusters(df: pd.DataFrame, nodes: pd.DataFrame, clusters_df: pd.DataFrame) -> None:
    """
    Проверки из чек-листа ТЗ. Бросает AssertionError с понятным сообщением,
    если что-то не так — удобно вызывать сразу после build_clusters_output().
    """
    assert set(df["gid"]) == set(nodes["gid"]), \
        "Набор gid в результате не совпадает с nodes.parquet"
    assert df["cluster_id"].notna().all(), \
        "Есть узлы без cluster_id"
    required_cols = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
    assert not clusters_df[required_cols].isna().any().any(), \
        "В clusters.csv есть пропуски в обязательных колонках"
    assert clusters_df["n_nodes"].sum() == len(df), \
        "Сумма n_nodes по кластерам не равна числу узлов"
    logger.info("Проверки качества кластеризации пройдены (%d кластеров, %d узлов)",
                len(clusters_df), len(df))