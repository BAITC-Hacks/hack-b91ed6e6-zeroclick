
const ROLE_COLORS = {
  coordinator: "#dc2626",
  consolidator: "#f97316",
  distributor: "#16a34a",
  transit: "#2563eb",
  terminal: "#9333ea",
  peripheral: "#64748b"
};

const state = {
  data: null,
  roleMap: new Map(),
  edgesByNode: new Map(),
  selectedGid: null,
  visibleNodes: [],
  visibleEdges: [],
  positions: new Map(),
  transform: { scale: 1, x: 0, y: 0 },
  dragging: false,
  dragStart: { x: 0, y: 0 },
  hoverGid: null
};

const $ = (id) => document.getElementById(id);
const canvas = $("graphCanvas");
const ctx = canvas.getContext("2d");

function money(v) {
  const n = Number(v || 0);
  return `${Math.round(n).toLocaleString("ru-RU")} ₸`;
}

function roleColor(role) {
  return ROLE_COLORS[role] || "#64748b";
}

function clusterColor(clusterId) {
  const hue = (Number(clusterId) * 47) % 360;
  return `hsl(${hue} 68% 48%)`;
}

function getNodeColor(node) {
  return $("colorMode").value === "cluster"
    ? clusterColor(node.cluster_id)
    : roleColor(node.role);
}

function setMetrics(meta) {
  $("metrics").innerHTML = [
    ["Узлов", meta.n_nodes],
    ["Рёбер", meta.n_edges],
    ["Кластеров", meta.n_clusters],
    ["Макс. priority", Number(meta.max_priority).toFixed(3)]
  ].map(([label, value]) => `
    <div class="metric card">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
    </div>
  `).join("");
}

function prepareIndexes() {
  for (const n of state.data.roles) {
    state.roleMap.set(Number(n.gid), n);
  }

  for (const e of state.data.edges) {
    const s = Number(e.src), d = Number(e.dst);
    if (!state.edgesByNode.has(s)) state.edgesByNode.set(s, []);
    if (!state.edgesByNode.has(d)) state.edgesByNode.set(d, []);
    state.edgesByNode.get(s).push(e);
    state.edgesByNode.get(d).push(e);
  }
}

function populateFilters() {
  const roles = [...new Set(state.data.roles.map(n => n.role))].sort();
  $("roleFilter").innerHTML =
    `<option value="Все">Все</option>` +
    roles.map(r => `<option value="${r}">${r}</option>`).join("");

  const clusters = [...new Set(state.data.roles.map(n => Number(n.cluster_id)))].sort((a,b)=>a-b);
  const options = `<option value="Все">Все</option>` +
    clusters.map(c => `<option value="${c}">${c}</option>`).join("");
  $("clusterFilter").innerHTML = options;
  $("clusterDetailSelect").innerHTML = clusters.map(c => `<option value="${c}">${c}</option>`).join("");
}

function defaultGid() {
  return Number([...state.data.roles].sort((a,b)=>Number(b.priority_score)-Number(a.priority_score))[0].gid);
}

function chooseVisibleGraph() {
  const role = $("roleFilter").value;
  const cluster = $("clusterFilter").value;
  const maxNodes = Number($("maxNodes").value);

  let filtered = state.data.roles.filter(n => {
    if (role !== "Все" && n.role !== role) return false;
    if (cluster !== "Все" && Number(n.cluster_id) !== Number(cluster)) return false;
    return true;
  });

  filtered.sort((a,b)=>Number(b.priority_score)-Number(a.priority_score));

  const chosen = new Set();
  if (state.selectedGid != null) chosen.add(Number(state.selectedGid));

  // Сначала соседи выбранного узла.
  const around = state.edgesByNode.get(Number(state.selectedGid)) || [];
  around
    .sort((a,b)=>Number(b.sum_kzt)-Number(a.sum_kzt))
    .forEach(e => {
      if (chosen.size < maxNodes) chosen.add(Number(e.src));
      if (chosen.size < maxNodes) chosen.add(Number(e.dst));
    });

  // Затем наиболее приоритетные узлы текущего фильтра.
  for (const n of filtered) {
    if (chosen.size >= maxNodes) break;
    chosen.add(Number(n.gid));
  }

  state.visibleNodes = [...chosen]
    .map(gid => state.roleMap.get(gid))
    .filter(Boolean);

  const visible = new Set(state.visibleNodes.map(n => Number(n.gid)));
  state.visibleEdges = state.data.edges.filter(
    e => visible.has(Number(e.src)) && visible.has(Number(e.dst))
  );

  computeDepthLayout();
  resetView(false);
  draw();

  $("graphStatus").textContent =
    `${state.visibleNodes.length} узлов · ${state.visibleEdges.length} связей`;
}

function computeDepthLayout() {
  state.positions.clear();

  // Колонки по depth дают понятное направление "по коленам".
  const groups = new Map();
  for (const n of state.visibleNodes) {
    const depth = Number(n.depth ?? 0);
    if (!groups.has(depth)) groups.set(depth, []);
    groups.get(depth).push(n);
  }

  const depths = [...groups.keys()].sort((a,b)=>a-b);
  const xGap = 260;

  depths.forEach((depth, colIndex) => {
    const arr = groups.get(depth);
    arr.sort((a,b) => {
      const c = Number(a.cluster_id) - Number(b.cluster_id);
      if (c !== 0) return c;
      return Number(b.priority_score) - Number(a.priority_score);
    });

    const yGap = Math.max(34, Math.min(84, 620 / Math.max(arr.length, 1)));
    const startY = -((arr.length - 1) * yGap) / 2;

    arr.forEach((n, i) => {
      // Небольшой deterministic jitter, чтобы линии не накладывались идеально.
      const jitter = ((Number(n.gid) % 7) - 3) * 3;
      state.positions.set(Number(n.gid), {
        x: colIndex * xGap + jitter,
        y: startY + i * yGap
      });
    });
  });
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function worldToScreen(p) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: rect.width / 2 + (p.x * state.transform.scale + state.transform.x),
    y: rect.height / 2 + (p.y * state.transform.scale + state.transform.y)
  };
}

function screenToWorld(x, y) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (x - rect.width / 2 - state.transform.x) / state.transform.scale,
    y: (y - rect.height / 2 - state.transform.y) / state.transform.scale
  };
}

function nodeRadius(n) {
  return 7 + 10 * Number(n.priority_score || 0) + (Number(n.gid) === Number(state.selectedGid) ? 4 : 0);
}

function drawArrow(from, to, color="#cbd5e1") {
  const a = worldToScreen(from);
  const b = worldToScreen(to);

  const dx = b.x - a.x, dy = b.y - a.y;
  const len = Math.hypot(dx, dy);
  if (len < 1) return;

  const ux = dx / len, uy = dy / len;
  const startPad = 10;
  const endPad = 14;

  const x1 = a.x + ux * startPad;
  const y1 = a.y + uy * startPad;
  const x2 = b.x - ux * endPad;
  const y2 = b.y - uy * endPad;

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.1;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();

  const size = 6;
  const angle = Math.atan2(y2-y1, x2-x1);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - size*Math.cos(angle-Math.PI/6), y2 - size*Math.sin(angle-Math.PI/6));
  ctx.lineTo(x2 - size*Math.cos(angle+Math.PI/6), y2 - size*Math.sin(angle+Math.PI/6));
  ctx.closePath();
  ctx.fill();
}

function draw() {
  if (!state.data) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  for (const e of state.visibleEdges) {
    const a = state.positions.get(Number(e.src));
    const b = state.positions.get(Number(e.dst));
    if (a && b) drawArrow(a, b);
  }

  for (const n of state.visibleNodes) {
    const p = state.positions.get(Number(n.gid));
    if (!p) continue;
    const s = worldToScreen(p);
    const r = nodeRadius(n);

    ctx.beginPath();
    ctx.arc(s.x, s.y, r, 0, Math.PI*2);
    ctx.fillStyle = getNodeColor(n);
    ctx.fill();

    ctx.lineWidth = Number(n.gid) === Number(state.selectedGid) ? 3 : 1.5;
    ctx.strokeStyle = Number(n.gid) === Number(state.selectedGid) ? "#0f172a" : "#ffffff";
    ctx.stroke();

    const showLabel =
      Number(n.gid) === Number(state.selectedGid) ||
      Number(n.priority_score) >= 0.55 ||
      state.visibleNodes.length <= 60;

    if (showLabel) {
      ctx.fillStyle = "#0f172a";
      ctx.font = "11px system-ui";
      ctx.textAlign = "center";
      ctx.fillText(String(n.gid), s.x, s.y - r - 5);
    }
  }
}

function findNodeAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;

  let hit = null;
  let best = Infinity;

  for (const n of state.visibleNodes) {
    const p = state.positions.get(Number(n.gid));
    if (!p) continue;
    const s = worldToScreen(p);
    const dist = Math.hypot(x - s.x, y - s.y);
    const r = nodeRadius(n) + 5;
    if (dist <= r && dist < best) {
      hit = n;
      best = dist;
    }
  }
  return hit;
}

function selectNode(gid, rebuild=true) {
  const node = state.roleMap.get(Number(gid));
  if (!node) {
    alert("GID не найден");
    return;
  }

  state.selectedGid = Number(gid);
  $("gidInput").value = String(gid);
  renderNodeCard(node);

  if (rebuild) chooseVisibleGraph();
  else draw();
}

function renderNodeCard(n) {
  const color = roleColor(n.role);
  $("nodeCard").classList.remove("empty");
  $("nodeCard").innerHTML = `
    <div class="gid">GID ${n.gid}</div>
    <div class="role-pill" style="background:${color}">${n.role}</div>
    <div class="kv"><span>Role score</span><b>${Number(n.role_score||0).toFixed(3)}</b></div>
    <div class="kv"><span>Priority</span><b>${Number(n.priority_score||0).toFixed(3)}</b></div>
    <div class="kv"><span>Кластер</span><b>${n.cluster_id}</b></div>
    <div class="kv"><span>Depth</span><b>${n.depth}</b></div>
    <div class="kv"><span>Входящих связей</span><b>${n.in_deg}</b></div>
    <div class="kv"><span>Исходящих связей</span><b>${n.out_deg}</b></div>
    <div class="kv"><span>Получено</span><b>${money(n.in_kzt)}</b></div>
    <div class="kv"><span>Отправлено</span><b>${money(n.out_kzt)}</b></div>
    <div class="evidence">${escapeHtml(n.evidence || "")}</div>
  `;
}

function renderLegend() {
  const mode = $("colorMode").value;
  if (mode === "role") {
    $("legend").innerHTML = Object.entries(ROLE_COLORS).map(([role,color]) =>
      `<div class="legend-item"><span class="dot" style="background:${color}"></span>${role}</div>`
    ).join("");
  } else {
    const clusters = [...new Set(state.visibleNodes.map(n=>Number(n.cluster_id)))].sort((a,b)=>a-b);
    $("legend").innerHTML = clusters.slice(0,24).map(c =>
      `<div class="legend-item"><span class="dot" style="background:${clusterColor(c)}"></span>cluster ${c}</div>`
    ).join("") + (clusters.length > 24 ? `<div class="legend-item">+ ещё ${clusters.length-24}</div>` : "");
  }
}

function renderTopTable() {
  $("topTableBody").innerHTML = state.data.top_nodes.map(r => {
    const color = roleColor(r.role);
    return `
      <tr class="clickable" data-gid="${r.gid}">
        <td>${r.rank}</td>
        <td><strong>${r.gid}</strong></td>
        <td><span class="table-pill" style="background:${color}">${r.role}</span></td>
        <td class="score">${Number(r.priority_score).toFixed(3)}</td>
        <td>${escapeHtml(r.why || "")}</td>
      </tr>
    `;
  }).join("");

  document.querySelectorAll("#topTableBody tr[data-gid]").forEach(tr => {
    tr.addEventListener("click", () => {
      showTab("graph");
      selectNode(Number(tr.dataset.gid));
    });
  });
}

function renderClustersTable() {
  $("clustersTableBody").innerHTML = state.data.clusters.map(c => `
    <tr class="clickable" data-cluster="${c.cluster_id}">
      <td><strong>${c.cluster_id}</strong></td>
      <td>${c.n_nodes}</td>
      <td>${c.n_seed}</td>
      <td>${money(c.sum_kzt_internal)}</td>
      <td>${escapeHtml(c.top_gids || "")}</td>
      <td>${escapeHtml(c.hypothesis || "")}</td>
    </tr>
  `).join("");

  document.querySelectorAll("#clustersTableBody tr[data-cluster]").forEach(tr => {
    tr.addEventListener("click", () => {
      $("clusterDetailSelect").value = tr.dataset.cluster;
      renderClusterDetail(Number(tr.dataset.cluster));
    });
  });
}

function renderClusterDetail(clusterId) {
  const nodes = state.data.roles
    .filter(n => Number(n.cluster_id) === Number(clusterId))
    .sort((a,b)=>Number(b.priority_score)-Number(a.priority_score));

  const cluster = state.data.clusters.find(c => Number(c.cluster_id) === Number(clusterId));
  $("clusterSummary").innerHTML = cluster ? `
    <div class="node-card">
      <div class="kv"><span>Узлов</span><b>${cluster.n_nodes}</b></div>
      <div class="kv"><span>Seed</span><b>${cluster.n_seed}</b></div>
      <div class="kv"><span>Внутренний оборот</span><b>${money(cluster.sum_kzt_internal)}</b></div>
      <div class="evidence">${escapeHtml(cluster.hypothesis || "")}</div>
    </div>
  ` : "";

  $("clusterNodes").innerHTML = nodes.slice(0,12).map(n => `
    <div class="mini-node clickable" data-gid="${n.gid}">
      <strong>${n.gid} · ${n.role}</strong>
      <span>priority ${Number(n.priority_score).toFixed(3)} · in ${n.in_deg} · out ${n.out_deg}</span>
    </div>
  `).join("");

  document.querySelectorAll("#clusterNodes [data-gid]").forEach(el => {
    el.addEventListener("click", () => {
      showTab("graph");
      selectNode(Number(el.dataset.gid));
    });
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;");
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "graph") {
    setTimeout(resizeCanvas, 0);
  }
}

function resetView(redraw=true) {
  state.transform = { scale: 1, x: 0, y: 0 };
  if (redraw) draw();
}

function bindEvents() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  });

  $("searchBtn").addEventListener("click", () => {
    const gid = Number($("gidInput").value.trim());
    if (Number.isFinite(gid)) selectNode(gid);
  });

  $("gidInput").addEventListener("keydown", e => {
    if (e.key === "Enter") $("searchBtn").click();
  });

  $("roleFilter").addEventListener("change", chooseVisibleGraph);
  $("clusterFilter").addEventListener("change", chooseVisibleGraph);

  $("colorMode").addEventListener("change", () => {
    renderLegend();
    draw();
  });

  $("maxNodes").addEventListener("input", () => {
    $("maxNodesValue").textContent = $("maxNodes").value;
  });
  $("maxNodes").addEventListener("change", chooseVisibleGraph);

  $("clusterDetailSelect").addEventListener("change", e => {
    renderClusterDetail(Number(e.target.value));
  });

  $("resetViewBtn").addEventListener("click", () => resetView());

  window.addEventListener("resize", resizeCanvas);

  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    state.transform.scale = Math.min(4, Math.max(.35, state.transform.scale * factor));
    draw();
  }, { passive:false });

  canvas.addEventListener("mousedown", e => {
    state.dragging = true;
    state.dragStart = { x: e.clientX - state.transform.x, y: e.clientY - state.transform.y };
  });

  window.addEventListener("mouseup", () => state.dragging = false);

  window.addEventListener("mousemove", e => {
    if (state.dragging) {
      state.transform.x = e.clientX - state.dragStart.x;
      state.transform.y = e.clientY - state.dragStart.y;
      draw();
    }
  });

  canvas.addEventListener("click", e => {
    const node = findNodeAt(e.clientX, e.clientY);
    if (node) selectNode(Number(node.gid), false);
  });

  canvas.addEventListener("mousemove", e => {
    if (state.dragging) return;
    const node = findNodeAt(e.clientX, e.clientY);
    const tooltip = $("tooltip");
    if (!node) {
      tooltip.classList.add("hidden");
      canvas.style.cursor = "grab";
      return;
    }

    canvas.style.cursor = "pointer";
    const rect = canvas.getBoundingClientRect();
    tooltip.style.left = `${e.clientX - rect.left + 14}px`;
    tooltip.style.top = `${e.clientY - rect.top + 14}px`;
    tooltip.innerHTML = `
      <strong>GID ${node.gid}</strong><br>
      ${node.role} · priority ${Number(node.priority_score).toFixed(3)}<br>
      cluster ${node.cluster_id} · depth ${node.depth}<br>
      ${escapeHtml(node.evidence || "")}
    `;
    tooltip.classList.remove("hidden");
  });
}


async function uploadOne(inputId, targetName) {
  const input = $(inputId);
  const file = input.files?.[0];
  if (!file) return false;

  const res = await fetch(`/api/upload?name=${encodeURIComponent(targetName)}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file
  });

  const payload = await res.json();
  if (!res.ok || !payload.ok) {
    throw new Error(payload.error || `Не удалось загрузить ${targetName}`);
  }
  return true;
}

async function refreshPipelineStatus() {
  const res = await fetch("/api/status", { cache: "no-store" });
  const status = await res.json();

  const readyCount = Object.values(status.files).filter(Boolean).length;
  const badge = $("pipelineBadge");

  if (status.has_results) {
    badge.textContent = "Результаты готовы";
    badge.className = "pipeline-badge ok";
  } else if (status.ready) {
    badge.textContent = "3/3 файла готовы";
    badge.className = "pipeline-badge running";
  } else {
    badge.textContent = `${readyCount}/3 файла`;
    badge.className = "pipeline-badge";
  }

  return status;
}

async function uploadAndAnalyze() {
  const btn = $("uploadAnalyzeBtn");
  const badge = $("pipelineBadge");
  const msg = $("pipelineMessage");

  btn.disabled = true;
  badge.textContent = "Загрузка...";
  badge.className = "pipeline-badge running";
  msg.textContent = "";

  try {
    await uploadOne("edgesFile", "edges.parquet");
    await uploadOne("nodesFile", "nodes.parquet");
    await uploadOne("transactionsFile", "transactions.parquet");

    const status = await refreshPipelineStatus();
    if (!status.ready) {
      throw new Error("Нужно загрузить edges.parquet, nodes.parquet и transactions.parquet.");
    }

    badge.textContent = "Считаем метрики...";
    badge.className = "pipeline-badge running";

    const res = await fetch("/api/analyze", { method: "POST" });
    const payload = await res.json();

    if (!res.ok || !payload.ok) {
      const details = (payload.logs || [])
        .map(x => `${x.command}\n${x.stderr || x.stdout || ""}`)
        .join("\n\n");
      throw new Error((payload.error || "Ошибка анализа") + "\n" + details);
    }

    badge.textContent = payload.under_5_minutes
      ? `Готово за ${payload.seconds} сек`
      : `Готово за ${payload.seconds} сек (>5 мин)`;
    badge.className = payload.under_5_minutes
      ? "pipeline-badge ok"
      : "pipeline-badge error";

    msg.textContent = "CSV созданы и проверены. Обновляю граф...";
    setTimeout(() => window.location.reload(), 500);
  } catch (err) {
    badge.textContent = "Ошибка";
    badge.className = "pipeline-badge error";
    msg.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
}

async function init() {
  $("uploadAnalyzeBtn").addEventListener("click", uploadAndAnalyze);

  const status = await refreshPipelineStatus();

  try {
    const res = await fetch("graph_data.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    state.data = await res.json();

    setMetrics(state.data.meta);
    prepareIndexes();
    populateFilters();
    renderTopTable();
    renderClustersTable();

    state.selectedGid = defaultGid();
    $("gidInput").value = String(state.selectedGid);
    renderNodeCard(state.roleMap.get(state.selectedGid));

    bindEvents();
    renderLegend();
    renderClusterDetail(Number($("clusterDetailSelect").value));
    chooseVisibleGraph();
    resizeCanvas();
  } catch (err) {
    $("pipelineMessage").textContent =
      status.ready
        ? "Данные загружены. Нажми «Загрузить и пересчитать», чтобы построить результаты."
        : "Загрузи три parquet-файла, чтобы построить граф.";

    $("metrics").innerHTML = "";
    $("graphStatus").textContent = "Нет рассчитанных данных";
  }
}
init();
