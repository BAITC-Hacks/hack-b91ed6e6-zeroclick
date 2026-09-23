from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"

st.set_page_config(page_title="Граф денег", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #eef2f8; border-right: 1px solid #dbe3ef; }
    h1 { color: #172033; letter-spacing: -1px; }
    h2, h3 { color: #24324a; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px; }
    .node-card { background: white; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; box-shadow: 0 3px 14px rgba(31, 48, 81, .06); }
    .legend { background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; line-height: 1.8; }
</style>
""", unsafe_allow_html=True)
st.title("Граф денег")
st.caption("Поиск узлов, направленные переводы и приоритеты для AML-аналитика")


@st.cache_data
def load_data():
    edges = pd.read_parquet(DATA / "edges.parquet")
    roles = pd.read_csv(OUT / "nodes_roles.csv")
    top_path = OUT / "top_nodes.csv"
    top = pd.read_csv(top_path) if top_path.exists() else pd.DataFrame()
    clusters_path = OUT / "clusters.csv"
    clusters = pd.read_csv(clusters_path) if clusters_path.exists() else pd.DataFrame()
    return edges, roles, top, clusters


edges, roles, top, clusters = load_data()
roles["gid"] = roles["gid"].astype(int)
edges["src"] = edges["src"].astype(int)
edges["dst"] = edges["dst"].astype(int)

role_colors = {
    "coordinator": "#950606",
    "consolidator": "#ff7f0e",
    "distributor": "#2ca02c",
    "transit": "#1f77b4",
    "terminal": "#9467bd",
    "peripheral": "#9e9e9e",
}


with st.sidebar:
    st.header("Фильтры")
    query = st.text_input("Поиск по gid", placeholder="например, 123456")
    role_values = ["Все"] + sorted(roles["role"].dropna().unique().tolist())
    selected_role = st.selectbox("Роль", role_values)
    cluster_values = ["Все"] + sorted(roles["cluster_id"].dropna().astype(int).unique().tolist())
    selected_cluster = st.selectbox("Кластер", cluster_values)
    max_nodes = st.slider("Максимум узлов на графе", 20, 250, 100, 10)

filtered = roles.copy()
if selected_role != "Все":
    filtered = filtered[filtered["role"] == selected_role]
if selected_cluster != "Все":
    filtered = filtered[filtered["cluster_id"] == int(selected_cluster)]
if query.strip():
    filtered = filtered[filtered["gid"].astype(str).str.contains(query.strip(), regex=False)]

if filtered.empty:
    st.warning("По заданным фильтрам узлы не найдены.")
    st.stop()

selected_gid = int(filtered.sort_values("priority_score", ascending=False).iloc[0]["gid"])
if query.strip().isdigit() and int(query) in set(roles["gid"]):
    selected_gid = int(query)

# Строим подграф: выбранный узел, его соседи и самые приоритетные узлы.
seed_nodes = set(filtered.sort_values("priority_score", ascending=False).head(max_nodes)["gid"])
neighbors = set(edges.loc[edges.src.isin(seed_nodes) | edges.dst.isin(seed_nodes), "src"])
neighbors |= set(edges.loc[edges.src.isin(seed_nodes) | edges.dst.isin(seed_nodes), "dst"])
node_ids = list((seed_nodes | neighbors) & set(roles.gid))[:max_nodes]
if selected_gid not in node_ids:
    node_ids = [selected_gid] + node_ids[:-1]

sub_edges = edges[edges.src.isin(node_ids) & edges.dst.isin(node_ids)].copy()
graph = nx.Graph()
graph.add_nodes_from(node_ids)
graph.add_edges_from(zip(sub_edges.src, sub_edges.dst))
pos = nx.spring_layout(graph, seed=42, k=1.1 / max(len(node_ids) ** 0.5, 1), iterations=50)
role_by_gid = roles.set_index("gid")

edge_x, edge_y = [], []
for row in sub_edges.itertuples():
    x0, y0 = pos[row.src]
    x1, y1 = pos[row.dst]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x, node_y, colors, sizes, texts, hover = [], [], [], [], [], []
for gid in node_ids:
    x, y = pos[gid]
    r = role_by_gid.loc[gid]
    node_x.append(x)
    node_y.append(y)
    colors.append(role_colors.get(r.get("role", "peripheral"), "#9e9e9e"))
    sizes.append(18 + 35 * float(r.get("priority_score", 0)))
    texts.append(str(gid))
    hover.append(
        f"gid: {gid}<br>роль: {r.get('role', '')}<br>"
        f"приоритет: {float(r.get('priority_score', 0)):.3f}<br>"
        f"кластер: {r.get('cluster_id', '')}<br>{r.get('evidence', '')}"
    )

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.7, color="#b0b0b0"), hoverinfo="none"))
fig.add_trace(go.Scatter(
    x=node_x, y=node_y, mode="markers+text", text=texts, textposition="top center",
    marker=dict(size=sizes, color=colors, line=dict(width=1, color="#333")),
    hovertext=hover, hoverinfo="text", customdata=node_ids,
))
fig.update_layout(
    height=720, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    plot_bgcolor="white", paper_bgcolor="white",
)
fig.update_layout(annotations=[
    dict(
        x=pos[row.dst][0], y=pos[row.dst][1], ax=pos[row.src][0], ay=pos[row.src][1],
        xref="x", yref="y", axref="x", ayref="y", showarrow=True,
        arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#9aa5b5",
        text="", opacity=0.75,
    )
    for row in sub_edges.itertuples()
])

left, right = st.columns([3, 1])
with left:
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("Цвет узла соответствует роли. Размер узла отражает priority_score.")
with right:
    st.markdown('<div class="node-card">', unsafe_allow_html=True)
    st.subheader("Карточка узла")
    r = role_by_gid.loc[selected_gid]
    st.metric("gid", selected_gid)
    st.write(f"**Роль:** {r.get('role', '')}")
    st.write(f"**Role score:** {float(r.get('role_score', 0)):.3f}")
    st.write(f"**Priority score:** {float(r.get('priority_score', 0)):.3f}")
    st.write(f"**Кластер:** {int(r.get('cluster_id', -1))}")
    st.info(str(r.get("evidence", "")))
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader("TOP узлов")
if not top.empty:
    st.dataframe(top.head(20), use_container_width=True, hide_index=True)
else:
    st.info("Файл out/top_nodes.csv ещё не создан.")

st.subheader("Легенда ролей")
legend = " ".join(
    f'<span style="color:{color};font-weight:600">●</span> {role}'
    for role, color in role_colors.items()
)
st.markdown(f'<div class="legend">{legend}</div>', unsafe_allow_html=True)
