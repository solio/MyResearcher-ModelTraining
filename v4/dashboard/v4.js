/* V4 research report — vanilla JS + local ECharts. No network calls. */
(function () {
  "use strict";

  if (new URLSearchParams(location.search).get("embed") === "1") {
    document.body.classList.add("embed");
  }

  const C = {
    ink: "#0b0b0b",
    ink2: "#52514e",
    muted: "#898781",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    surface: "#fcfcfb",
    card: "#ffffff",
    blue: "#2a78d6",
    blueLight: "#cde2fb",
    teal: "#1baf7a",
    green: "#008300",
    red: "#d03b3b",
    orange: "#eb6834",
    yellow: "#eda100",
    violet: "#4a3aa7",
    gray: "#898781",
    ref: "#898781",
    ref2: "#a8a69f",
    ref3: "#b8b6b0",
    ref4: "#c8c6c0",
    up: "#c0392b",
    down: "#157347"
  };

  const S = {
    currentFilter: "all",
    ledgerFilter: "all",
    stock: null,
    charts: {},
    rendered: { observation: false, training: false, prediction: false }
  };

  let DATA = null;
  let FONT = "system-ui, -apple-system, 'PingFang SC', sans-serif";

  const $ = (sel) => document.querySelector(sel);
  const escapeHtml = (s) =>
    String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtInt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));
  const fmtPct = (n, d = 1) => (n == null ? "—" : `${Number(n).toFixed(d)}%`);
  const fmtPp = (n, d = 2) => (n == null ? "—" : `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(Number(n)).toFixed(d)}pp`);
  const fmtNum = (n, d = 2) => (n == null ? "—" : Number(n).toFixed(d));
  const ppClass = (n) => (n == null ? "num-zero" : n > 0 ? "num-pos" : n < 0 ? "num-neg" : "num-zero");
  const tag = (text, cls = "neutral") => `<span class="tag ${cls}">${escapeHtml(text)}</span>`;
  const badge = (text, cls = "neutral") => `<span class="badge ${cls}">${escapeHtml(text)}</span>`;

  function statusBadge(status) {
    const map = {
      limited: ["有限", "warn"],
      mixed: ["混合", "warn"],
      not_established: ["未成立", "bad"],
      retained_filter: ["保留为分诊", "good"],
      retained_guardrail: ["保留为护栏", "good"],
      falsified_as_independent_alpha: ["证伪 · 独立 alpha", "bad"],
      candidate_weak: ["弱候选", "warn"],
      forward_pending: ["待前瞻", "info"],
      forward_matured_failed: ["前瞻已失败", "bad"],
      computed_in_v4: ["v4 已计算", "neutral"],
      analysis_only: ["分析中", "info"]
    };
    const [label, cls] = map[status] || [status || "—", "neutral"];
    return badge(label, cls);
  }

  function epistemicBadge(ep) {
    const map = {
      reference: ["K3 参考", "neutral"],
      reference_recomputed: ["K3 事件复算", "neutral"],
      reference_timing_shift: ["换执行时点", "neutral"],
      reference_threshold_filter: ["因果阈值过滤", "neutral"],
      retrospective_v4: ["v4 回顾", "info"],
      retrospective_control: ["价格状态对照", "warn"],
      forward_single: ["单例前瞻", "bad"]
    };
    const [label, cls] = map[ep] || [ep || "—", "neutral"];
    return badge(label, cls);
  }

  const bucketLabel = (b) => ({
    candidate: "候选",
    deep_not_fear: "深回撤，未恐慌",
    fear_not_deep: "恐慌，未深跌",
    near: "接近门槛",
    deep_near_fear: "深回撤 + 接近极端"
  }[b] || b || "—");

  function bucketBadge(b) {
    const cls = b === "candidate" ? "bad" : b === "deep_near_fear" ? "warn" : b === "fear_not_deep" ? "warn" : "info";
    return badge(bucketLabel(b), cls);
  }

  function formatEffect(effect) {
    if (effect == null) return "—";
    if (typeof effect === "string") return escapeHtml(effect);
    if (typeof effect !== "object") return escapeHtml(String(effect));
    const labels = {
      A0_pp: "A0", A5_pp: "A5", paired_pp: "配对差", stocks_improved: "改善股",
      action_hard: "动作难例", no_action_share: "NO_ACTION", watch: "WATCH",
      B_pp: "B", H_pp: "H", HC_pp: "HC", dual_pass: "双条件",
      has_emotion_agreement: "有无情绪一致", content_agreement: "内容一致", k3_composite: "K3 复核",
      declaration: "宣言", agitated: "激动", emphasis: "强调", calm: "平静",
      "top_1.2pct_author_post_share": "头部作者帖量", singleton_author_share: "单帖作者",
      weight_scheme_p: "换权 p", raw_pp: "原始", raw_t: "原始 t", matched_pp: "匹配后",
      matched_t: "匹配 t", A_minus_B_pp: "A−B", A_pp: "A 段", B_pp: "B 段",
      clean_low_concentration_pp: "低集中度", pp: "均值", t: "t",
      v391_pp: "v3.9.1", v391_t: "v3.9.1 t", tmp392_pp: "深水展示", tmp392_stocks: "展示股数",
      v39_val_pp: "v3.9 验证", v391_val_pp: "v3.9.1 验证", z_selection_pp: "z 选择",
      z_selection_t: "z 选择 t", better_than_constant: "优于恒定", better_than_linear: "优于线性",
      BY: "BY 修正", gross_pp: "毛收益", same_close_pp: "同日收盘",
      regime_condition: "regime", n_events: "事件", n_clusters: "簇",
      mean_pp: "事件等权", t_event: "事件 t", cluster_mean_pp: "簇等权", cluster_t: "簇 t",
      stock_equal_mean_pp: "股等权", stock_equal_t: "股 t", n_stocks: "股票数",
      n_stocks_equal: "股数", ci_low_pp: "CI 低", ci_high_pp: "CI 高",
      boot_mean_pp: "Bootstrap", training_cutoff: "训练截止", n_days: "天数",
      corr_sd_fwd5: "相关", n_pairs: "匹配对"
    };
    const parts = [];
    for (const [k, v] of Object.entries(effect)) {
      if (v == null || k === "note") continue;
      const key = labels[k] || k;
      if (typeof v === "number") {
        const isPp = /_pp$/.test(k);
        const isT = /_t$|^t$/.test(k);
        parts.push(`${escapeHtml(key)} ${isPp ? fmtPp(v) : isT ? `t=${fmtNum(v)}` : fmtNum(v, 1)}`);
      } else if (typeof v === "object") {
        continue;
      } else {
        parts.push(`${escapeHtml(key)} ${escapeHtml(String(v))}`);
      }
    }
    return parts.length ? parts.join(" · ") : "—";
  }

  function chartBase() {
    return {
      animationDuration: 350,
      textStyle: { fontFamily: FONT, color: C.ink2, fontSize: 11 },
      tooltip: {
        backgroundColor: "rgba(255,255,255,0.98)",
        borderColor: "rgba(11,11,11,0.12)",
        borderWidth: 1,
        padding: [8, 10],
        textStyle: { color: C.ink, fontSize: 12, fontFamily: FONT },
        extraCssText: "box-shadow:0 8px 24px rgba(11,11,11,.08);border-radius:8px;"
      }
    };
  }

  function axisX(data, extra = {}) {
    return Object.assign(
      {
        type: "category",
        data,
        axisLine: { lineStyle: { color: C.axis } },
        axisTick: { show: false },
        axisLabel: { color: C.muted, fontSize: 10.5, hideOverlap: true },
        splitLine: { show: false }
      },
      extra
    );
  }

  function axisY(extra = {}) {
    return Object.assign(
      {
        type: "value",
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: C.muted, fontSize: 10.5 },
        splitLine: { lineStyle: { color: C.grid, type: "solid", width: 1 } }
      },
      extra
    );
  }

  function chart(id) {
    const el = document.getElementById(id);
    if (!el || !window.echarts) return null;
    if (S.charts[id]) S.charts[id].dispose();
    const instance = window.echarts.init(el, null, { renderer: "canvas" });
    S.charts[id] = instance;
    return instance;
  }

  function resizeCharts() {
    requestAnimationFrame(() => {
      Object.values(S.charts).forEach((c) => {
        try { c.resize(); } catch (e) { /* ignore */ }
      });
    });
  }

  function showError(message) {
    $("#status-strip").innerHTML = `<div class="status-card alert"><div class="label">构建产物不可用</div><div class="value">无法读取 v4 数据</div><div class="note">${escapeHtml(message)}</div></div>`;
    const main = document.querySelector("main.wrap");
    main.insertAdjacentHTML(
      "beforeend",
      `<div class="build-error"><h2>先生成数据，再打开页面</h2><p>${escapeHtml(message)}</p><p>在 worktree 根目录运行：</p><pre><code>python3 v4/build_v4.py</code></pre><p>然后重新加载本页。页面只读取本地 <code>static/v4-data.json</code>，不访问外部服务。</p></div>`
    );
  }

  function setTabs() {
    const buttons = [...document.querySelectorAll(".tab-button")];
    const panels = {
      observation: $("#tab-observation"),
      training: $("#tab-training"),
      prediction: $("#tab-prediction"),
      notes: $("#tab-notes")
    };
    function activate(name) {
      if (!panels[name]) name = "observation";
      buttons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
      Object.entries(panels).forEach(([key, panel]) => { panel.hidden = key !== name; });
      const strip = document.getElementById("status-strip");
      if (strip) strip.style.display = name === "prediction" ? "none" : "";
      if (name === "training") ensureTraining();
      if (name === "prediction") ensurePrediction();
      setTimeout(resizeCharts, 30);
      if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    }
    buttons.forEach((b) => b.addEventListener("click", () => activate(b.dataset.tab)));
    window.addEventListener("hashchange", () => {
      const name = String(location.hash || "").replace("#", "");
      if (["observation", "training", "prediction", "notes"].includes(name)) activate(name);
    });
    const initial = String(location.hash || "").replace("#", "") || "observation";
    activate(initial);
    window.addEventListener("resize", resizeCharts);
    return activate;
  }

  /* ------------------------------------------------------------------ */
  /* Observation                                                        */
  /* ------------------------------------------------------------------ */

  function renderHeader() {
    const a = DATA.as_of, r = DATA.observation.regime;
    $("#asof-info").textContent = a.info_day || "—";
    $("#asof-market").textContent = a.market_as_of || "—";
    const pill = $("#regime-pill");
    const label = { risk_off: "risk-off", risk_on: "risk-on", mixed: "mixed", unknown: "unknown" }[r.label] || "—";
    pill.className = `pill ${r.label || "unknown"}`;
    pill.textContent = label;
    const p = DATA.prediction.headline;
    const h4 = p.h4;
    const cards = [
      {
        cls: p.candidate_count > 0 ? "caution" : "alert",
        label: "当前可执行候选",
        value: `${p.candidate_count} / 20`,
        note: p.candidate_count > 0 ? "有候选，但仍不可部署；需人工复核" : "S∩D 无合格组合；不是买入建议"
      },
      {
        cls: "alert",
        label: "H-4 · 第一笔预声明前瞻",
        value: h4 ? fmtPp(h4.gross_pp) : "—",
        note: h4 ? `${h4.name} ${h4.signal_date} → ${h4.entry_date} 开盘 → ${h4.exit_date} 收盘 · 流程已跑通` : "尚无结算"
      },
      {
        cls: "caution",
        label: "市场 regime",
        value: label,
        note: `CSI 300 ${fmtNum(r.close, 2)} · MA200 ${fmtNum(r.ma200, 2)} · 20 日 ${fmtPct(r.ret20_pp, 2)} · 60 日回撤 ${fmtPct(r.dd60_pp, 2)}`
      }
    ];
    $("#status-strip").innerHTML = cards.map((c) => `<div class="status-card ${c.cls}"><div class="label">${escapeHtml(c.label)}</div><div class="value">${escapeHtml(c.value)}</div><div class="note">${escapeHtml(c.note)}</div></div>`).join("");
  }

  function renderObservationKpis() {
    const m = DATA.meta, q = DATA.observation.data_quality, cur = DATA.observation.current;
    const sd = cur.filter((x) => x.s_plus && x.d_plus).length;
    const fear = cur.filter((x) => x.s_plus).length;
    const deep = cur.filter((x) => x.d_plus).length;
    const kpis = [
      ["帖子总量", fmtInt(m.posts_total), `信息日 ${DATA.as_of.info_day}`, "info"],
      ["股票数", fmtInt(m.stocks), `${m.stocks_with_posts} 只有帖子`, "info"],
      ["作者数", fmtInt(m.authors), "按 author_id 去重", "info"],
      ["正文可用", fmtPct(m.content_nonempty_pct), `${fmtInt(m.content_nonempty)} 条非空`, "caution"],
      ["污染日", fmtInt(q.polluted_days), "作者集中度>60%，已剔除", "caution"],
      ["当前 S / D / S∩D", `${fear} / ${deep} / ${sd}`, "20 只股票的截面", sd ? "alert" : "ok"]
    ];
    $("#obs-kpis").innerHTML = kpis.map(([k, v, d, cls]) => `<div class="kpi ${cls}"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v)}</div><div class="d">${escapeHtml(d)}</div></div>`).join("");
    const warnings = [...new Set([...(DATA.as_of.warnings || []), ...(q.warnings || [])])];
    $("#obs-warning").innerHTML = warnings.length
      ? `<div class="callout warn" style="margin-bottom:14px"><b>数据边界：</b>${warnings.map(escapeHtml).join("；")}</div>`
      : `<div class="callout good" style="margin-bottom:14px"><b>数据边界：</b>没有已知异常；仍然只是 20 只股票的局部样本。</div>`;
  }

  function renderRegime() {
    const r = DATA.observation.regime;
    $("#regime-tag").textContent = (r.label || "unknown").toUpperCase();
    $("#regime-tag").className = `tag ${r.label === "risk_off" ? "bad" : r.label === "risk_on" ? "good" : "neutral"}`;
    const rows = r.series || [];
    const chart = chartX("regime-chart");
    if (!chart) return;
    chart.setOption(Object.assign(chartBase(), {
      grid: { left: 54, right: 70, top: 22, bottom: 32 },
      tooltip: Object.assign(chartBase().tooltip, {
        trigger: "axis",
        formatter: (ps) => {
          const p = ps[0];
          const row = rows.find((x) => x.d === p.axisValue);
          if (!row) return "";
          return `<b>${escapeHtml(row.d)}</b><br>收盘 ${fmtNum(row.close, 2)}<br>MA200 ${fmtNum(row.ma200, 2)}<br>60 日回撤 ${fmtPct(row.dd60_pp, 2)}<br>状态 ${escapeHtml(row.label)}`;
        }
      }),
      xAxis: axisX(rows.map((x) => x.d)),
      yAxis: axisY({ scale: true, axisLabel: { color: C.muted, fontSize: 10.5, formatter: (v) => fmtInt(v) } }),
      series: [
        {
          name: "CSI 300",
          type: "line",
          data: rows.map((x) => x.close),
          showSymbol: false,
          lineStyle: { width: 2, color: C.blue },
          areaStyle: { color: C.blue, opacity: 0.07 },
          endLabel: { show: true, formatter: (p) => fmtNum(p.value, 0), color: C.ink2, fontSize: 11 }
        },
        {
          name: "MA200",
          type: "line",
          data: rows.map((x) => x.ma200),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.5, color: C.gray, type: "dashed" }
        }
      ]
    }));
    const last = rows[rows.length - 1] || {};
    $("#regime-note").textContent = `定义：${r.definition || ""} 当前 ${r.label}；信号日 ${r.signal_day}，行情日 ${r.as_of}。公开报道的 2026-10 弱势与本地 baostock 冻结行情一致；本图不做实时更新。`;
  }

  function chartX(id) {
    return chart(id);
  }

  function renderStateMap() {
    const cur = DATA.observation.current.filter((x) => x.z != null);
    $("#map-asof").textContent = DATA.observation.current[0]?.as_of || DATA.as_of.info_day;
    const chart = chartX("state-map");
    if (!chart) return;
    const colorOf = (x) => (x.s_plus && x.d_plus ? C.red : x.s_plus ? C.yellow : x.d_plus ? C.blue : C.gray);
    const data = cur.map((x) => ({
      name: x.name,
      value: [x.dd20_pct, x.z],
      symbol: x.data_status === "ok" ? "circle" : "emptyCircle",
      symbolSize: 8 + Math.min(18, Math.sqrt(Math.max(1, x.n_posts)) * 1.4),
      itemStyle: { color: x.data_status === "ok" ? colorOf(x) : "transparent", borderColor: colorOf(x), borderWidth: 2 },
      label: { show: Boolean(x.s_plus || x.d_plus), formatter: () => x.name, color: C.ink2, fontSize: 10, position: "top" },
      raw: x
    }));
    const maxPosts = Math.max(...cur.map((x) => x.n_posts), 1);
    chart.setOption(Object.assign(chartBase(), {
      grid: { left: 44, right: 22, top: 18, bottom: 34 },
      tooltip: Object.assign(chartBase().tooltip, {
        formatter: (p) => {
          const x = p.data.raw;
          return `<b>${escapeHtml(x.name)} ${escapeHtml(x.code)}</b><br>回撤 ${fmtPct(x.dd20_pct, 2)} · z ${fmtNum(x.z, 2)}<br>密度 ${fmtPct(x.density_pct)} · 帖数 ${fmtInt(x.n_posts)}<br>5 日 ${fmtPct(x.p5_pct, 2)} · ${escapeHtml(x.reason)}`;
        }
      }),
      xAxis: axisX([], {
        type: "value",
        name: "20 日回撤 %",
        nameLocation: "middle",
        nameGap: 24,
        nameTextStyle: { color: C.muted, fontSize: 10.5 },
        min: -50,
        max: 3,
        axisLabel: { color: C.muted, fontSize: 10.5, formatter: (v) => `${v}%` },
        splitLine: { lineStyle: { color: C.grid, type: "solid" } }
      }),
      yAxis: axisY({
        name: "情绪 z",
        nameTextStyle: { color: C.muted, fontSize: 10.5 },
        min: -4,
        max: 6,
        axisLabel: { color: C.muted, fontSize: 10.5 }
      }),
      series: [
        {
          type: "scatter",
          data,
          markArea: {
            silent: true,
            itemStyle: { opacity: 0.45 },
            data: [
              [{ xAxis: -50, yAxis: 2, itemStyle: { color: "#fdecec" } }, { xAxis: -20, yAxis: 6 }],
              [{ xAxis: -50, yAxis: -4, itemStyle: { color: "#eef4ff" } }, { xAxis: -20, yAxis: 2 }],
              [{ xAxis: -20, yAxis: 2, itemStyle: { color: "#fff8e8" } }, { xAxis: 3, yAxis: 6 }],
              [{ xAxis: -20, yAxis: -4, itemStyle: { color: "#f7f7f4" } }, { xAxis: 3, yAxis: 2 }]
            ],
            label: { show: true, color: C.muted, fontSize: 10, position: "insideTopLeft", formatter: (p) => p.name || "" }
          },
          markLine: {
            silent: true,
            symbol: ["none", "none"],
            lineStyle: { color: C.axis, type: "dashed", width: 1 },
            label: { color: C.muted, fontSize: 10, formatter: (p) => p.name },
            data: [
              { xAxis: -20, name: "深回撤 -20%" },
              { yAxis: 2, name: "z=2" }
            ]
          }
        }
      ]
    }));
    const withSignal = cur.filter((x) => x.s_plus || x.d_plus).length;
    $("#map-asof").textContent = `${DATA.as_of.info_day} · ${cur.length} 只有 z`;
    $("#state-map-card .chart-note").textContent = `${cur.length} 只股票有可用 z；${withSignal} 只落在 S 或 D 区域。点大小≈帖数（最大 ${fmtInt(maxPosts)}）。空心=薄日/污染/预热。`;
  }

  function renderCurrentTable() {
    const all = [...DATA.observation.current].sort((a, b) => (a.dd20_pct == null ? 1 : b.dd20_pct == null ? -1 : a.dd20_pct - b.dd20_pct));
    const filters = [
      ["all", "全部"],
      ["s", "情绪极端"],
      ["d", "深回撤"],
      ["near", "接近门槛"]
    ];
    $("#current-filter").innerHTML = filters.map(([k, label]) => `<button class="chip" type="button" data-f="${k}" aria-pressed="${S.currentFilter === k}">${label}</button>`).join("");
    $("#current-filter").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      S.currentFilter = b.dataset.f;
      renderCurrentTable();
    }));
    let rows = all;
    if (S.currentFilter === "s") rows = all.filter((x) => x.s_plus);
    if (S.currentFilter === "d") rows = all.filter((x) => x.d_plus);
    if (S.currentFilter === "near") rows = all.filter((x) => x.z != null && x.z >= 1.5 || x.dd20_pct != null && x.dd20_pct <= -18);
    const sd = all.filter((x) => x.s_plus && x.d_plus).length;
    $("#current-count").textContent = `${all.length} 只 · ${sd} 只 S∩D`;
    if (!rows.length) {
      $("#current-table").innerHTML = `<div class="empty">这个筛选下没有股票。</div>`;
      return;
    }
    $("#current-table").innerHTML = `<div class="table-wrap"><table><thead><tr><th class="l">股票</th><th>z</th><th>20日回撤</th><th>5日收益</th><th>帖数</th><th>密度</th><th class="l">状态</th></tr></thead><tbody>${rows.map((x) => {
      const tags = [x.s_plus ? badge("S", "warn") : "", x.d_plus ? badge("D", "info") : "", x.data_status !== "ok" ? badge({ no_posts: "无帖", thin: "薄日", polluted: "污染", warmup: "预热" }[x.data_status] || x.data_status, "neutral") : ""].join(" ");
      return `<tr><td class="l"><b>${escapeHtml(x.name)}</b><span class="sub">${escapeHtml(x.code)} · ${escapeHtml(x.sector)}</span></td><td>${fmtNum(x.z, 2)}</td><td class="${ppClass(x.dd20_pct)}">${fmtPct(x.dd20_pct, 2)}</td><td class="${ppClass(x.p5_pct)}">${fmtPct(x.p5_pct, 2)}</td><td>${fmtInt(x.n_posts)}</td><td>${fmtPct(x.density_pct)}</td><td class="l">${tags} <span class="sub">${escapeHtml(x.reason)}</span></td></tr>`;
    }).join("")}</tbody></table><div class="table-note">z 使用此前至少 60 个有效观测；S = z≥2；D = 距 20 日高点 ≤-20%。当前没有 S∩D 候选。</div></div>`;
  }

  function renderStockDetail() {
    const select = $("#stock-select");
    const stocks = [...DATA.observation.current].sort((a, b) => a.name.localeCompare(b.name, "zh"));
    if (!stocks.length) return;
    if (!S.stock || !DATA.observation.series.by_stock[S.stock]) S.stock = stocks.find((x) => x.d_plus)?.code || stocks[0].code;
    select.innerHTML = stocks.map((x) => `<option value="${escapeHtml(x.code)}" ${x.code === S.stock ? "selected" : ""}>${escapeHtml(x.name)} ${escapeHtml(x.code)}</option>`).join("");
    select.onchange = () => {
      S.stock = select.value;
      renderStockDetail();
    };
    const rows = DATA.observation.series.by_stock[S.stock] || [];
    const meta = DATA.observation.current.find((x) => x.code === S.stock);
    const chart = chartX("stock-chart");
    if (!chart) return;
    const dates = rows.map((r) => r[0]);
    const close = rows.map((r) => r[1]);
    const density = rows.map((r) => r[4]);
    const z = rows.map((r) => r[5]);
    const posts = rows.map((r) => r[3]);
    const h4 = DATA.prediction.headline.h4;
    const ledger = DATA.prediction.forward_ledger.filter((e) => e.stock === S.stock && e.entry_date);
    const eventData = ledger.map((e) => {
      const i = dates.indexOf(e.signal_date);
      return i >= 0 ? { value: [dates[i], close[i]], name: e.event_id, entry: e.entry_date, gross: e.gross_pp, status: e.status, isForward: e.is_forward } : null;
    }).filter(Boolean);
    chart.setOption(Object.assign(chartBase(), {
      grid: [
        { left: 54, right: 22, top: 28, height: 78 },
        { left: 54, right: 22, top: 135, height: 62 },
        { left: 54, right: 22, top: 226, height: 56 },
        { left: 54, right: 22, top: 312, height: 52 }
      ],
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      tooltip: Object.assign(chartBase().tooltip, {
        trigger: "axis",
        formatter: (ps) => {
          if (!ps.length) return "";
          const i = ps[0].dataIndex;
          const r = rows[i];
          if (!r) return "";
          return `<b>${escapeHtml(r[0])}</b><br>收盘 ${fmtNum(r[1], 2)} · 20日回撤 ${fmtPct(r[2], 2)}<br>看跌密度 ${fmtPct(r[4])} · z ${fmtNum(r[5], 2)}<br>帖数 ${fmtInt(r[3])} · 5日 ${fmtPct(r[6], 2)}`;
        }
      }),
      xAxis: [0, 1, 2, 3].map((i) => axisX(dates, {
        gridIndex: i,
        axisLabel: { show: i === 3, color: C.muted, fontSize: 10.5, hideOverlap: true },
        axisPointer: { label: { show: i === 3 } }
      })),
      yAxis: [
        axisY({ gridIndex: 0, scale: true, name: "价格", nameTextStyle: { color: C.muted, fontSize: 10 }, axisLabel: { color: C.muted, fontSize: 10, formatter: (v) => fmtNum(v, 0) } }),
        axisY({ gridIndex: 1, scale: true, name: "看跌%", nameTextStyle: { color: C.muted, fontSize: 10 }, axisLabel: { color: C.muted, fontSize: 10, formatter: (v) => `${v}%` }, splitNumber: 2 }),
        axisY({ gridIndex: 2, scale: true, name: "z", nameTextStyle: { color: C.muted, fontSize: 10 }, axisLabel: { color: C.muted, fontSize: 10 }, splitNumber: 2 }),
        axisY({ gridIndex: 3, scale: true, name: "帖数", nameTextStyle: { color: C.muted, fontSize: 10 }, axisLabel: { color: C.muted, fontSize: 10, formatter: (v) => fmtInt(v) }, splitNumber: 2 })
      ],
      dataZoom: [{ type: "inside", xAxisIndex: [0, 1, 2, 3], start: 0, end: 100 }],
      series: [
        {
          name: "收盘", type: "line", xAxisIndex: 0, yAxisIndex: 0, data: close, showSymbol: false,
          lineStyle: { width: 2, color: C.blue },
          markPoint: { symbol: "circle", symbolSize: 9, itemStyle: { color: C.orange, borderColor: C.surface, borderWidth: 2 }, label: { show: false }, data: eventData }
        },
        { name: "看跌密度", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: density, showSymbol: false, connectNulls: false, lineStyle: { width: 2, color: C.orange } },
        {
          name: "z", type: "line", xAxisIndex: 2, yAxisIndex: 2, data: z, showSymbol: false, connectNulls: false,
          lineStyle: { width: 2, color: C.teal },
          markLine: { silent: true, symbol: ["none", "none"], lineStyle: { color: C.axis, type: "dashed", width: 1 }, label: { show: false }, data: [{ yAxis: 2 }] }
        },
        { name: "帖数", type: "bar", xAxisIndex: 3, yAxisIndex: 3, data: posts, barMaxWidth: 10, itemStyle: { color: C.blueLight, borderRadius: [2, 2, 0, 0] } }
      ]
    }));
    const status = meta ? meta.reason : "无当前状态";
    $("#stock-tag").textContent = meta ? `${meta.s_plus ? "S" : "—"} / ${meta.d_plus ? "D" : "—"} · z ${fmtNum(meta.z, 2)}` : "—";
    $("#stock-tag").className = `tag ${meta && meta.s_plus && meta.d_plus ? "bad" : meta && (meta.s_plus || meta.d_plus) ? "warn" : "neutral"}`;
    $("#stock-note").innerHTML = meta
      ? `${escapeHtml(meta.name)}：${escapeHtml(status)}；20 日回撤 ${fmtPct(meta.dd20_pct, 2)}，5 日 ${fmtPct(meta.p5_pct, 2)}，看跌密度 ${fmtPct(meta.density_pct)}，${fmtInt(meta.n_posts)} 帖。橙点是事件入场（含已结算/待结算）；价格、密度、z、帖数分四个独立坐标，避免双轴误读。${h4 && S.stock === h4.stock ? " 本股含 H-4 前瞻案例。" : ""}`
      : "该股当前没有可用状态。";
  }

  function renderBreadth() {
    const b = DATA.observation.breadth;
    const rows = b.series || [];
    const tag = $("#breadth-tag");
    const peak = Math.max(0, ...rows.map((x) => x.n_sd || 0));
    tag.textContent = `S∩D 峰值 ${peak} 只 · 相关 ${fmtNum(b.summary.corr_sd_fwd5, 3)}`;
    const chart = chartX("breadth-chart");
    if (!chart) return;
    chart.setOption(Object.assign(chartBase(), {
      grid: [
        { left: 46, right: 22, top: 28, height: 92 },
        { left: 46, right: 22, top: 160, height: 82 }
      ],
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      tooltip: Object.assign(chartBase().tooltip, {
        trigger: "axis",
        formatter: (ps) => {
          if (!ps.length) return "";
          const i = ps[0].dataIndex;
          const r = rows[i];
          if (!r) return "";
          return `<b>${escapeHtml(r.d)}</b><br>S ${r.n_s} 只 · S∩D ${r.n_sd} 只<br>CSI 300 未来 5 日 ${fmtPct(r.fwd5_pp, 2)}`;
        }
      }),
      legend: { top: 0, right: 0, itemWidth: 8, itemHeight: 8, textStyle: { color: C.ink2, fontSize: 10 } },
      xAxis: [0, 1].map((i) => axisX(rows.map((r) => r.d), { gridIndex: i, axisLabel: { show: i === 1, color: C.muted, fontSize: 10, hideOverlap: true } })),
      yAxis: [
        axisY({ gridIndex: 0, name: "股票数", nameTextStyle: { color: C.muted, fontSize: 10 }, minInterval: 1, axisLabel: { color: C.muted, fontSize: 10 } }),
        axisY({ gridIndex: 1, name: "未来5日%", nameTextStyle: { color: C.muted, fontSize: 10 }, axisLabel: { color: C.muted, fontSize: 10, formatter: (v) => `${v}%` }, splitNumber: 2 })
      ],
      series: [
        { name: "S", type: "bar", xAxisIndex: 0, yAxisIndex: 0, data: rows.map((r) => r.n_s), barMaxWidth: 6, itemStyle: { color: C.blueLight, borderRadius: [2, 2, 0, 0] } },
        { name: "S∩D", type: "bar", xAxisIndex: 0, yAxisIndex: 0, data: rows.map((r) => r.n_sd), barMaxWidth: 6, itemStyle: { color: C.red, borderRadius: [2, 2, 0, 0] } },
        { name: "CSI300 fwd5", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: rows.map((r) => r.fwd5_pp), showSymbol: false, connectNulls: false, lineStyle: { width: 2, color: C.blue } }
      ]
    }));
    $("#breadth-note").textContent = `每根柱=当日有多少只股票处于 S（浅蓝）或 S∩D（红）；下=CSI 300 从该日收盘起未来 5 日收益。${b.summary.note} 相关 ${fmtNum(b.summary.corr_sd_fwd5, 3)}，n=${b.summary.n_days}。`;
  }

  function renderAuthor() {
    const a = DATA.observation.author;
    const bins = a.concentration_bins || [];
    const total = bins.reduce((s, x) => s + x.days, 0) || 1;
    const max = Math.max(1, ...bins.map((x) => x.days));
    $("#author-body").innerHTML = `
      <div class="grid two" style="gap:10px;margin-bottom:12px">
        <div class="kpi info" style="padding:10px"><div class="k">头部 1% 作者帖量占比</div><div class="v">${fmtPct(a.top1pct_post_share_pct)}</div><div class="d">当前 613,985 帖快照</div></div>
        <div class="kpi caution" style="padding:10px"><div class="k">只发 1 帖的作者</div><div class="v">${fmtInt(a.singleton_authors)}</div><div class="d">共 ${fmtInt(a.total_authors)} 位作者</div></div>
      </div>
      <div style="display:grid;gap:7px">${bins.map((b) => `<div><div style="display:flex;justify-content:space-between;font-size:11.5px;color:var(--ink-2)"><span>单日 top1 作者占比 ${escapeHtml(b.bin)}</span><span class="mono">${fmtInt(b.days)} 天 · ${fmtPct((b.days / total) * 100)}</span></div><div style="height:7px;background:var(--neutral-bg);border-radius:999px;margin-top:3px"><div style="height:7px;width:${Math.max(2, (b.days / max) * 100)}%;background:${b.bin === "≥60%" ? C.red : b.bin === "40–60%" ? C.orange : C.blue};border-radius:999px"></div></div></div>`).join("")}</div>
      <div class="chart-note">${escapeHtml(a.note || "")} 单日 top1 作者占比 >60% 的日子已从关系统计剔除；K3 的 1.2%/29.5% 来自 377,163 帖快照，当前重算为 ${fmtPct(a.top1pct_post_share_pct)}。</div>`;
  }

  function renderHealth() {
    const rows = DATA.observation.stocks;
    const polluted = rows.reduce((s, x) => s + (x.polluted_days || 0), 0);
    $("#health-tag").textContent = `${rows.length} 只 · ${polluted} 个污染日已剔除`;
    $("#health-table").innerHTML = `<div class="table-wrap"><table><thead><tr><th class="l">股票</th><th>总帖</th><th>覆盖</th><th>最长断档</th><th>薄日</th><th>污染日</th><th>全期日均</th><th>近20日</th><th>衰减</th><th class="l">提示</th></tr></thead><tbody>${rows.map((x) => `<tr><td class="l"><b>${escapeHtml(x.name)}</b><span class="sub">${escapeHtml(x.code)} · ${escapeHtml(x.sector)}</span></td><td>${fmtInt(x.total_posts)}</td><td class="${x.coverage_pct < 90 ? "num-neg" : ""}">${fmtPct(x.coverage_pct)}</td><td>${fmtInt(x.longest_gap)}</td><td>${fmtInt(x.thin_days)}</td><td>${fmtInt(x.polluted_days)}</td><td>${fmtNum(x.full_avg, 1)}</td><td>${fmtNum(x.recent20_avg, 1)}</td><td class="${ppClass(x.decay_pct)}">${fmtPct(x.decay_pct, 2)}</td><td class="l">${(x.warnings || []).map((w) => badge(w, /骤降/.test(w) ? "bad" : /污染/.test(w) ? "warn" : "neutral")).join(" ") || '<span class="muted">—</span>'}</td></tr>`).join("")}</tbody></table><div class="table-note">覆盖率 = 有帖信息日 ÷ 首末区间交易日。断档以交易日计。薄日 = 当日 &lt;5 帖。污染日 = 单日 top1 作者占比 &gt;60%。K3 健康表基于 377,163 帖；当前 613,985 帖，部分历史断档可能已被回填。</div></div>`;
  }

  function renderObservation() {
    if (S.rendered.observation) return;
    S.rendered.observation = true;
    renderObservationKpis();
    renderRegime();
    renderStateMap();
    renderCurrentTable();
    renderStockDetail();
    renderBreadth();
    renderAuthor();
    renderHealth();
  }

  /* ------------------------------------------------------------------ */
  /* Training                                                            */
  /* ------------------------------------------------------------------ */

  function renderTrainingAlert() {
    const p = DATA.training.primary;
    const w = DATA.training.waterfall;
    const k3 = w.find((x) => x.id === "K3_H1_PUBLISHED");
    const causal = w.find((x) => x.id === "V4_CAUSAL_DEEP");
    const matched = w.find((x) => x.id === "V4_MATCHED_INCREMENT");
    const nextOpen = w.find((x) => x.id === "K3_SAME_EVENTS_NEXT_OPEN");
    const causalFilter = w.find((x) => x.id === "K3_EVENTS_CAUSAL_FILTER");
    $("#training-alert").innerHTML = `<div class="callout ${causal && causal.mean_pp != null && causal.mean_pp < 0 ? "bad" : "warn"}" style="margin-bottom:14px">
      <b>一句话：</b>K3 发布值 ${fmtPp(k3?.mean_pp)}（t=${fmtNum(k3?.t, 2)}，n=${k3?.n_events}）→ 同批事件次日开盘 ${fmtPp(nextOpen?.mean_pp)}（t=${fmtNum(nextOpen?.t, 2)}，这一步没有衰减）→ 因果 z 过滤后 ${fmtPp(causalFilter?.mean_pp)}（t=${fmtNum(causalFilter?.t, 2)}，n=${causalFilter?.n_events}）→ v4 重选事件集 ${fmtPp(causal?.mean_pp)}（t=${fmtNum(causal?.t, 2)}，n=${causal?.n_events}，${causal?.n_clusters} 簇）→ 同股匹配 ${matched?.n_events} 对，${matched?.n_events < 3 ? "样本不足，不报 t/CI" : `均值 ${fmtPp(matched?.mean_pp)}`}。执行时点不是主要衰减源；全样本阈值/事件选择才是。
    </div>`;
    $("#waterfall-tag").textContent = `${p.n_events} 事件 · ${p.n_clusters} 簇 · ${p.n_stocks} 股票`;
  }

  function renderWaterfall() {
    const rows = DATA.training.waterfall;
    const labels = rows.map((x) => x.label);
    const values = rows.map((x) => x.mean_pp);
    const colorOf = (ep) => ep === "reference" || ep.startsWith("reference") ? C.ref2 : ep === "retrospective_v4" ? C.blue : ep === "retrospective_control" ? C.orange : C.red;
    const chart = chartX("waterfall-chart");
    if (!chart) return;
    chart.setOption(Object.assign(chartBase(), {
      grid: { left: 200, right: 42, top: 18, bottom: 42 },
      tooltip: Object.assign(chartBase().tooltip, {
        formatter: (p) => {
          const r = rows[p.dataIndex];
          return `<b>${escapeHtml(r.label)}</b><br>${escapeHtml(r.change_axis)}<br>n=${r.n_events} / 簇 ${r.n_clusters ?? "—"} / 股 ${r.n_stocks}<br>均值 ${fmtPp(r.mean_pp)} · t ${fmtNum(r.t, 2)}${r.ci_low_pp != null ? `<br>95% CI [${fmtPp(r.ci_low_pp)}, ${fmtPp(r.ci_high_pp)}]` : ""}<br><span style="color:${C.muted}">${escapeHtml(r.note)}</span>`;
        }
      }),
      xAxis: axisX([], {
        type: "value",
        name: "5 日超额（pp）",
        nameLocation: "middle",
        nameGap: 22,
        nameTextStyle: { color: C.muted, fontSize: 10.5 },
        axisLabel: { color: C.muted, fontSize: 10.5, formatter: (v) => `${v > 0 ? "+" : ""}${v}` }
      }),
      yAxis: axisX(labels, {
        type: "category",
        inverse: true,
        axisLabel: { color: C.ink2, fontSize: 10.5, width: 180, overflow: "truncate" },
        splitLine: { show: false }
      }),
      series: [{
        type: "bar",
        data: rows.map((r, i) => ({
          value: r.mean_pp,
          itemStyle: { color: colorOf(r.epistemic), borderRadius: [0, 3, 3, 0] },
          label: { show: true, position: "insideRight", color: r.mean_pp != null && r.mean_pp < 0 ? "#fff" : C.ink, fontSize: 10.5, formatter: () => `${r.mean_pp == null ? "—" : fmtPp(r.mean_pp)} · n=${r.n_events}` }
        })),
        barMaxWidth: 18,
        markLine: { silent: true, symbol: ["none", "none"], lineStyle: { color: C.axis, width: 1 }, label: { show: false }, data: [{ xAxis: 0 }] }
      }]
    }));
    $("#waterfall-table").innerHTML = `<div class="table-wrap"><table class="waterfall-table"><thead><tr><th class="l">步骤</th><th class="l">只改了什么</th><th>事件 / 簇 / 股</th><th>均值</th><th>t / CI</th><th class="l">读法</th></tr></thead><tbody>${rows.map((r) => `<tr><td class="l"><b>${escapeHtml(r.label)}</b><span class="sub">${escapeHtml({reference:"K3 参考",reference_recomputed:"K3 事件复算",reference_timing_shift:"换执行时点",reference_threshold_filter:"因果阈值过滤",retrospective_v4:"v4 回顾",retrospective_control:"价格状态对照",forward_single:"单例前瞻"}[r.epistemic] || r.epistemic)}</span></td><td class="l">${escapeHtml(r.change_axis)}</td><td>${r.n_events} / ${r.n_clusters ?? "—"} / ${r.n_stocks}</td><td class="${ppClass(r.mean_pp)}">${fmtPp(r.mean_pp)}</td><td>${r.t == null ? "—" : `t=${fmtNum(r.t, 2)}`}${r.ci_low_pp != null ? `<span class="sub">[${fmtPp(r.ci_low_pp)}, ${fmtPp(r.ci_high_pp)}]</span>` : ""}</td><td class="l">${escapeHtml(r.note)}</td></tr>`).join("")}</tbody></table><div class="table-note">步骤 0–3 是同一批 K3 事件逐层复算；步骤 4–5 是 v4 的新事件集/对照。它们不能被当成连续净值。H-4 是单例，不能与均值并列比较。</div></div>`;
  }

  function renderQuadrants() {
    const q = DATA.training.quadrants;
    const cells = q.cells;
    const order = ["S+D+", "S+D-", "S-D+", "S-D-"];
    const cls = { "S+D+": "sd", "S+D-": "s", "S-D+": "d", "S-D-": "none" };
    $("#quadrant-body").innerHTML = `<div class="quadrants">${order.map((k) => { const c = cells[k]; return `<div class="quadrant ${cls[k]}"><div class="q-label">${escapeHtml(c.label)}</div><div class="q-value ${ppClass(c.mean_pp)}">${fmtPct(c.mean_pp, 2)}</div><div class="q-meta">${fmtInt(c.n_days)} 个股票日 · ${c.n_stocks} 只股票</div></div>`; }).join("")}</div>
      <div style="margin-top:10px">${Object.entries(q.contrasts || {}).map(([name, v]) => `<div class="contrast-row"><span>${escapeHtml(name)}</span><span class="val ${ppClass(v.mean_pp)}">${fmtPp(v.mean_pp)}</span><span class="muted">${v.t == null ? "t=—" : `t=${fmtNum(v.t, 2)}`} · ${v.n_stocks} 股</span></div>`).join("")}</div>
      <div class="chart-note">${escapeHtml(q.note)} S+D+ 相对 S-D+ 的差为 ${fmtPp(q.contrasts?.["S+D+ vs S-D+"]?.mean_pp)}（t=${fmtNum(q.contrasts?.["S+D+ vs S-D+"]?.t, 2)}）；单元格是日度池化，股票构成不同；对比是逐股先平均，不能把两个单元格均值直接相减。没有看到情绪在深回撤内的可辨识正增量。</div>`;
  }

  function renderMatched() {
    const m = DATA.training.matched_controls;
    const s = m.summary;
    $("#matched-tag").textContent = `${s.n_pairs} 对 · ${s.n_stocks} 股`;
    const warning = s.n_pairs < 3
      ? `<div class="callout warn" style="margin-bottom:10px"><b>样本不足：</b>只有 ${s.n_pairs} 对匹配，不报 t/CI。这里的 ${fmtPp(s.mean_diff_pp)} 不能当作情绪增量的证据；它只说明这个对照在这个样本上跑不起来。</div>`
      : `<div class="callout ${s.t != null && s.t > 2 ? "good" : "warn"}" style="margin-bottom:10px"><b>匹配增量：</b>均值 ${fmtPp(s.mean_diff_pp)}，t=${fmtNum(s.t, 2)}，95% CI [${fmtPp(s.ci_low_pp)}, ${fmtPp(s.ci_high_pp)}]，${s.n_pairs} 对 / ${s.n_stocks} 股。CI 按股票重采样。</div>`;
    const rows = m.pairs || [];
    $("#matched-body").innerHTML = warning + (rows.length ? `<div class="table-wrap"><table><thead><tr><th class="l">股票</th><th>信号日</th><th>对照日</th><th>信号收益</th><th>对照收益</th><th>差</th><th>dd 信号/对照</th></tr></thead><tbody>${rows.map((p) => `<tr><td class="l">${escapeHtml(p.stock)}</td><td>${escapeHtml(p.signal_date)}</td><td>${escapeHtml(p.control_date)}</td><td class="${ppClass(p.signal_return_pp)}">${fmtPct(p.signal_return_pp, 2)}</td><td class="${ppClass(p.control_return_pp)}">${fmtPct(p.control_return_pp, 2)}</td><td class="${ppClass(p.diff_pp)}">${fmtPp(p.diff_pp)}</td><td>${fmtPct(p.dd_signal_pp, 2)} / ${fmtPct(p.dd_control_pp, 2)}</td></tr>`).join("")}</tbody></table><div class="table-note">${escapeHtml(s.note)}</div></div>` : `<div class="empty">没有满足约束的匹配对。</div>`) + `<div class="chart-note">匹配不是随机对照；它只排除“同股、相近回撤、相近过去 5 日”的解释。即使 n≥3，也只能当作一个窄条件对照。</div>`;
  }

  function renderSpec() {
    const spec = DATA.training.specification_curve;
    const pts = spec.points.filter((p) => p.t_cross != null && p.mean_pp != null);
    const primary = spec.primary;
    const best = [...pts].sort((a, b) => b.t_cross - a.t_cross)[0];
    const colorOf = (h) => (h === 1 ? C.blue : h === 3 ? C.teal : C.orange);
    $("#spec-tag").textContent = `${spec.n_tests} 个规格 · 主规格 ${primary?.id || "—"}`;
    const chart = chartX("spec-chart");
    if (!chart) return;
    chart.setOption(Object.assign(chartBase(), {
      grid: { left: 46, right: 24, top: 32, bottom: 36 },
      legend: { top: 0, right: 0, itemWidth: 8, itemHeight: 8, textStyle: { color: C.ink2, fontSize: 10 }, data: ["1 日", "3 日", "5 日"] },
      tooltip: Object.assign(chartBase().tooltip, {
        formatter: (p) => {
          const r = p.data.raw;
          return `<b>${escapeHtml(r.id)}</b><br>均值 ${fmtPp(r.mean_pp)} · 跨股 t ${fmtNum(r.t_cross, 2)}<br>事件 ${r.n_events} · 簇 ${r.n_clusters} · 股 ${r.n_stocks}<br>事件 t ${fmtNum(r.event_t, 2)} · 簇 t ${fmtNum(r.cluster_t, 2)}`;
        }
      }),
      xAxis: axisX([], {
        type: "value",
        name: "逐股等权超额（pp）",
        nameLocation: "middle",
        nameGap: 24,
        nameTextStyle: { color: C.muted, fontSize: 10.5 },
        axisLabel: { color: C.muted, fontSize: 10.5, formatter: (v) => `${v > 0 ? "+" : ""}${v}` }
      }),
      yAxis: axisY({
        name: "跨股 t",
        nameTextStyle: { color: C.muted, fontSize: 10.5 },
        axisLabel: { color: C.muted, fontSize: 10.5 }
      }),
      series: [1, 3, 5].map((h) => ({
        name: `${h} 日`,
        type: "scatter",
        data: pts.filter((p) => p.horizon === h).map((p) => ({
          value: [p.mean_pp, p.t_cross],
          symbolSize: 8 + Math.min(12, Math.sqrt(Math.max(1, p.n_events)) * 2.2),
          itemStyle: { color: colorOf(h), opacity: p.is_primary ? 1 : 0.78, borderColor: p.is_primary ? C.ink : "transparent", borderWidth: p.is_primary ? 2 : 0 },
          raw: p
        })),
        markLine: h === 5 ? {
          silent: true,
          symbol: ["none", "none"],
          lineStyle: { width: 1, color: C.axis, type: "dashed" },
          label: { color: C.muted, fontSize: 10, formatter: (p) => p.name },
          data: [{ yAxis: 3, name: "t=3 提醒线" }, { yAxis: 0, name: "t=0" }, { xAxis: 0 }]
        } : undefined
      }))
    }));
    $("#spec-note").innerHTML = `主规格 <b>${escapeHtml(primary?.id || "—")}</b>：逐股等权 ${fmtPp(primary?.mean_pp)}（t=${fmtNum(primary?.t_cross, 2)}），逐事件 ${fmtPp(primary?.event_mean_pp)}（t=${fmtNum(primary?.event_t, 2)}），${primary?.n_events} 事件 / ${primary?.n_clusters} 簇。最高 t 的角落是 <b>${escapeHtml(best?.id || "—")}</b>：${fmtPp(best?.mean_pp)}（t=${fmtNum(best?.t_cross, 2)}，仅 ${best?.n_events} 事件），仍低于 t=3。48 个规格全部显示；不挑最好看的那个。`;
  }

  function renderWalk() {
    const w = DATA.training.walk_forward;
    const rows = w.series || [];
    $("#walk-tag").textContent = `${w.summary.n_events} 事件 · 净 ${fmtPp(w.summary.mean_net_pp)}`;
    const chart = chartX("walk-chart");
    if (!chart) return;
    chart.setOption(Object.assign(chartBase(), {
      grid: { left: 52, right: 24, top: 24, bottom: 36 },
      tooltip: Object.assign(chartBase().tooltip, {
        trigger: "axis",
        formatter: (ps) => {
          const p = ps[0];
          const r = rows[p.dataIndex];
          if (!r) return "";
          return `<b>${escapeHtml(r.d)} · ${escapeHtml(r.stock)}</b><br>本次毛超额 ${fmtPp(r.gross_pp)}<br>累计毛 ${fmtPp(r.cum_gross_pp)}<br>累计净 ${fmtPp(r.cum_net_pp)}`;
        }
      }),
      xAxis: axisX(rows.map((r) => r.d), { axisLabel: { color: C.muted, fontSize: 10.5, hideOverlap: true } }),
      yAxis: axisY({ name: "累计净超额（pp）", nameTextStyle: { color: C.muted, fontSize: 10.5 }, scale: true, axisLabel: { color: C.muted, fontSize: 10.5, formatter: (v) => `${v > 0 ? "+" : ""}${v}` } }),
      series: [{
        type: "line",
        data: rows.map((r) => r.cum_net_pp),
        showSymbol: true,
        symbolSize: 7,
        lineStyle: { width: 2, color: C.teal },
        itemStyle: { color: C.teal, borderColor: C.surface, borderWidth: 2 },
        areaStyle: { color: C.teal, opacity: 0.08 },
        markLine: { silent: true, symbol: ["none", "none"], lineStyle: { color: C.axis, width: 1 }, label: { show: false }, data: [{ yAxis: 0 }] }
      }]
    }));
    $("#walk-note").textContent = `${w.note} 汇总：${w.summary.n_events} 事件 / ${w.summary.n_clusters} 簇，净均值 ${fmtPp(w.summary.mean_net_pp)}，t=${fmtNum(w.summary.t, 2)}。`;
  }

  function renderRegimeCondition() {
    const rows = DATA.training.regime_condition || [];
    const max = Math.max(1, ...rows.map((x) => Math.abs(x.mean_pp || 0)));
    const cls = { risk_on: "good", risk_off: "bad", mixed: "warn", unknown: "neutral" };
    $("#regime-cond-body").innerHTML = rows.map((x) => `<div style="margin-bottom:10px"><div style="display:flex;justify-content:space-between;font-size:12px;gap:8px"><span>${badge(x.regime, cls[x.regime] || "neutral")} <span class="muted">${x.n_events} 事件 · ${x.n_stocks} 股</span></span><span class="${ppClass(x.mean_pp)}">${fmtPp(x.mean_pp)} <span class="muted">${x.t == null ? "t=—" : `t=${fmtNum(x.t, 2)}`}</span></span></div><div class="meter ${x.mean_pp != null && x.mean_pp < 0 ? "bad" : "warn"}" style="margin-top:4px"><span style="width:${Math.max(2, Math.abs((x.mean_pp || 0) / max) * 100)}%"></span></div></div>`).join("") + `<div class="chart-note">当前只有 risk_off（${rows.find((x) => x.regime === "risk_off")?.n_events || 0}）与 mixed（${rows.find((x) => x.regime === "mixed")?.n_events || 0}）有事件，risk_on 为 0。不能据此声称 regime 依赖；只能说当前样本无法分层支持。</div>`;
  }

  function renderHypotheses() {
    const rows = DATA.training.hypotheses || [];
    $("#hypothesis-list").innerHTML = rows.map((h) => `<div class="hypothesis"><div class="id">${escapeHtml(h.id)}</div><div><div class="stmt">${escapeHtml(h.statement)}</div><div class="meta">${escapeHtml(h.source)} · ${escapeHtml(h.sample)}</div></div><div class="meta"><b>效应：</b>${formatEffect(h.effect)}</div><div class="verdict"><b>判决：</b>${escapeHtml(h.verdict)}</div><div>${statusBadge(h.status)}<div class="tiny muted" style="margin-top:4px;max-width:200px;white-space:normal" title="${escapeHtml(h.evidence)}">${escapeHtml(String(h.evidence).slice(0, 58))}${String(h.evidence).length > 58 ? "…" : ""}</div></div></div>`).join("");
  }

  function renderLineage() {
    $("#lineage-body").innerHTML = (DATA.training.lineage || []).map((l) => `<article class="lineage-card"><h4>${escapeHtml(l.family)}</h4><div class="period">${escapeHtml(l.period)}</div><div class="row"><b>层</b>${l.layers.map(escapeHtml).join(" · ")}</div><div class="row"><b>证明了什么</b>${escapeHtml(l.proved)}</div><div class="row"><b>失败/未证明</b>${escapeHtml(l.failed)}</div><div class="row"><b>当前状态</b>${escapeHtml(l.status)}</div><div class="row tiny muted">${escapeHtml(l.evidence)}</div></article>`).join("");
  }

  function renderEvidence() {
    $("#evidence-body").innerHTML = (DATA.training.external_evidence || []).map((e) => `<article class="evidence-card"><div class="cite">${escapeHtml(e.citation)}</div><div class="finding"><b>发现：</b>${escapeHtml(e.finding)}</div><div class="implication"><b>对我们的意义：</b>${escapeHtml(e.implication)}</div><div class="foot"><span>${escapeHtml(e.kind)}</span><a href="${escapeHtml(e.url)}" target="_blank" rel="noopener noreferrer">来源 ↗</a></div><div class="tiny muted" style="margin-top:6px">${escapeHtml(e.caveat)}</div></article>`).join("");
  }

  function ensureTraining() {
    if (S.rendered.training) { resizeCharts(); return; }
    S.rendered.training = true;
    renderTrainingAlert();
    renderWaterfall();
    renderQuadrants();
    renderMatched();
    renderSpec();
    renderWalk();
    renderRegimeCondition();
    renderHypotheses();
    renderLineage();
    renderEvidence();
    setTimeout(resizeCharts, 50);
  }

  /* ------------------------------------------------------------------ */
  /* Prediction                                                          */
  /* ------------------------------------------------------------------ */

  function renderPrediction() {
    if (S.rendered.prediction) { resizeCharts(); return; }
    S.rendered.prediction = true;
    const p = DATA.prediction.headline;
    const h4 = p.h4;
    const r = DATA.observation.regime;
    $("#prediction-hero").innerHTML = `
      <div class="hero-card ${p.candidate_count ? "caution" : "alert"}" style="border-left:4px solid ${p.candidate_count ? C.yellow : C.red}"><div class="k">可执行候选</div><div class="value">${p.candidate_count} / 20</div><div class="note">${escapeHtml(p.statement)}</div></div>
      <div class="hero-card" style="border-left:4px solid ${C.red}"><div class="k">H-4 · 第一笔预声明前瞻</div><div class="value ${ppClass(h4?.gross_pp)}">${h4 ? fmtPp(h4.gross_pp) : "—"}</div><div class="note">${h4 ? `${escapeHtml(h4.name)} ${escapeHtml(h4.signal_date)} → ${escapeHtml(h4.entry_date)} 开盘 ${fmtNum(h4.entry_price, 2)} → ${escapeHtml(h4.exit_date)} 收盘 ${fmtNum(h4.exit_price, 2)} · 毛收益，未含费用 · 单例失败` : "尚无结算"}</div></div>
      <div class="hero-card" style="border-left:4px solid ${C.blue}"><div class="k">S∩D 条件分布</div><div class="value">${fmtPp(DATA.prediction.event_distribution.primary_excess.mean_pp)}</div><div class="note">n=${DATA.prediction.event_distribution.primary_excess.n} · 中位 ${fmtPp(DATA.prediction.event_distribution.primary_excess.median_pp)} · 胜率 ${fmtPct(DATA.prediction.event_distribution.primary_excess.win_rate_pct)} · 市场 ${escapeHtml(r.label || "unknown")} · 红正绿负；不是点预测</div></div>`;
    renderCandidates();
    renderWatchlist();
    renderProgress();
    renderDistribution();
    renderLedger();
    renderLadder();
    renderExecution();
    renderKill();
    setTimeout(resizeCharts, 50);
  }

  function renderCandidates() {
    const cs = DATA.prediction.candidates || [];
    $("#candidate-tag").textContent = cs.length ? `${cs.length} 个待复核` : "0 个";
    if (!cs.length) {
      const cur = DATA.observation.current;
      const fear = cur.filter((x) => x.s_plus).map((x) => `${x.name} z=${fmtNum(x.z, 2)}`);
      const deep = cur.filter((x) => x.d_plus).map((x) => `${x.name} dd=${fmtPct(x.dd20_pct, 2)}`);
      $("#candidate-body").innerHTML = `<div class="candidate-empty"><div class="mark">—</div><div><h4>当前没有 S∩D 合格候选</h4><p>已检查 ${cur.length} 只股票。只有情绪极端：${escapeHtml(fear.join("、") || "无")}；只有深回撤：${escapeHtml(deep.join("、") || "无")}。没有任何一只同时满足 z≥2 与 dd≤-20%，所以不生成“预测候选”。</p><div style="margin-top:8px">${badge("信息日 " + DATA.as_of.info_day, "info")} ${badge("行情至 " + DATA.as_of.market_as_of, "neutral")} ${badge("无可执行", "bad")}</div></div></div>`;
      return;
    }
    $("#candidate-body").innerHTML = `<div class="grid two">${cs.map((x) => `<div class="card" style="box-shadow:none"><div class="card-head"><div><h3>${escapeHtml(x.name)} ${escapeHtml(x.code)}</h3><p>${escapeHtml(x.sector)} · ${escapeHtml(x.reason)}</p></div>${badge("待人工复核", "warn")}</div><div class="exec-grid"><div class="exec-item"><div class="k">z</div><div class="v">${fmtNum(x.z, 2)}</div></div><div class="exec-item"><div class="k">20 日回撤</div><div class="v ${ppClass(x.dd20_pct)}">${fmtPct(x.dd20_pct, 2)}</div></div><div class="exec-item"><div class="k">入场</div><div class="v">${escapeHtml(x.entry_day || "—")} 开盘</div></div><div class="exec-item"><div class="k">当前标记</div><div class="v">${fmtPct(x.mark_return_pct, 2)}</div></div></div></div>`).join("")}</div>`;
  }

  function renderWatchlist() {
    const rows = DATA.prediction.watchlist || [];
    $("#watch-body").innerHTML = rows.length ? `<div class="table-wrap"><table><thead><tr><th class="l">股票</th><th class="l">类型</th><th>z</th><th>dd</th><th>密度</th><th>帖数</th><th class="l">缺哪道闸</th></tr></thead><tbody>${rows.map((x) => `<tr><td class="l"><b>${escapeHtml(x.name)}</b><span class="sub">${escapeHtml(x.code)} · ${escapeHtml(x.sector)}</span></td><td class="l">${bucketBadge(x.bucket)}</td><td>${fmtNum(x.z, 2)}</td><td class="${ppClass(x.dd20_pct)}">${fmtPct(x.dd20_pct, 2)}</td><td>${fmtPct(x.density_pct)}</td><td>${fmtInt(x.n_posts)}</td><td class="l">${escapeHtml(x.missing_gate)}</td></tr>`).join("")}</tbody></table><div class="table-note">观察名单不是候选；它只是解释“为什么当前没有候选”。深回撤但 z<2 的股票是价格状态，不是情绪信号。</div></div>` : `<div class="empty">没有接近门槛的股票。</div>`;
  }

  function renderProgress() {
    const rows = DATA.prediction.hypotheses || [];
    $("#progress-body").innerHTML = rows.map((h) => {
      const f = h.forward || {};
      const n = f.n || 0, req = f.required || 20;
      const pct = Math.min(100, (n / req) * 100);
      const statusCls = h.status === "forward_matured_failed" ? "bad" : pct >= 100 ? "good" : "warn";
      return `<div class="item"><div class="top"><b>${escapeHtml(h.label)}</b>${statusBadge(h.status)}</div><div class="detail"><span>前瞻 ${n}/${req}</span><span>${escapeHtml(f.status || "")}</span>${f.last_pp != null ? `<span>最近 ${fmtPct(f.last_pp, 2)}</span>` : ""}</div><div class="meter ${statusCls}"><span style="width:${Math.max(2, pct)}%"></span></div><div class="detail" style="margin-top:6px"><span>样本内：${formatEffect(h.in_sample)}</span></div><div class="tiny muted" style="margin-top:3px">放行：${escapeHtml(h.release)} · 失败：${escapeHtml(h.kill)}</div></div>`;
    }).join("");
  }

  function renderDistribution() {
    const d = DATA.prediction.event_distribution;
    const primary = d.primary_excess || d.primary;
    $("#dist-tag").textContent = `n=${primary.n} · 均值 ${fmtPp(primary.mean_pp)}`;
    const bins = d.excess_histogram || [];
    const chart = chartX("dist-chart");
    if (!chart) return;
    chart.setOption(Object.assign(chartBase(), {
      grid: { left: 48, right: 24, top: 28, bottom: 36 },
      tooltip: Object.assign(chartBase().tooltip, {
        formatter: (p) => {
          const r = bins[p.dataIndex];
          return `<b>${r.bin_pp}% ~ ${r.bin_pp + 2}%</b><br>${r.count} 个事件`;
        }
      }),
      xAxis: axisX(bins.map((r) => (r.bin_pp > 0 ? `+${r.bin_pp}` : String(r.bin_pp))), {
        axisLabel: { color: C.muted, fontSize: 10, interval: 3 },
        name: "相对同股全期均值的 5 日超额（pp，2pp 一档）",
        nameLocation: "middle",
        nameGap: 24,
        nameTextStyle: { color: C.muted, fontSize: 10.5 }
      }),
      yAxis: axisY({ name: "事件数", nameTextStyle: { color: C.muted, fontSize: 10.5 }, minInterval: 1, axisLabel: { color: C.muted, fontSize: 10.5 } }),
      series: [{
        type: "bar",
        data: bins.map((r) => ({ value: r.count, itemStyle: { color: r.bin_pp < 0 ? C.down : C.up, borderRadius: [2, 2, 0, 0] } })),
        barCategoryGap: "18%",
        markLine: { silent: true, symbol: ["none", "none"], lineStyle: { color: C.ink2, width: 1, type: "dashed" }, label: { show: true, color: C.ink2, fontSize: 10, formatter: () => `均值 ${fmtPp(primary.mean_pp)}` }, data: [{ xAxis: String(Math.round(primary.mean_pp / 2) * 2) }] }
      }]
    }));
    const matched = d.matched_increment;
    $("#dist-note").innerHTML = `S∩D 因果事件 ${primary.n} 个：均值 ${fmtPp(primary.mean_pp)}、中位 ${fmtPp(primary.median_pp)}、胜率 ${fmtPct(primary.win_rate_pct)}、p10 ${fmtPp(primary.p10_pp)}、p90 ${fmtPp(primary.p90_pp)}。匹配增量 n=${matched.n}（${matched.status === "insufficient_sample" ? "样本不足" : `均值 ${fmtPp(matched.mean_pp)}`}），不足以读。H-4 是前瞻单例，单独列在台账，不进入这条历史分布。柱色只是正负（红正/绿负），不是显著性。`;
  }

  function renderLedger() {
    const rows = DATA.prediction.forward_ledger || [];
    const filters = [
      ["all", "全部"],
      ["forward", "前瞻"],
      ["settled", "已结算"],
      ["retro", "样本内"]
    ];
    $("#ledger-filter").innerHTML = filters.map(([k, l]) => `<button class="chip" type="button" data-lf="${k}" aria-pressed="${S.ledgerFilter === k}">${l}</button>`).join("");
    $("#ledger-filter").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { S.ledgerFilter = b.dataset.lf; renderLedger(); }));
    let view = rows;
    if (S.ledgerFilter === "forward") view = rows.filter((r) => r.is_forward);
    if (S.ledgerFilter === "settled") view = rows.filter((r) => r.gross_pp != null);
    if (S.ledgerFilter === "retro") view = rows.filter((r) => !r.is_forward);
    const statusLabel = (r) => {
      if (r.status === "pending_entry") return badge("待入场", "warn");
      if (r.status === "pending_exit") return badge("待到期", "warn");
      if (r.status === "matured" || r.status === "complete") return badge("已结算", r.is_forward ? "forward" : "good");
      return badge(r.status || "—", "neutral");
    };
    $("#ledger-table").innerHTML = view.length ? `<div class="table-wrap"><table><thead><tr><th class="l">信号日 / 股票</th><th class="l">状态</th><th>z</th><th>dd</th><th class="l">入场 → 出场</th><th>毛收益</th><th>相对基线</th><th>净超额(20bp)</th></tr></thead><tbody>${view.map((r) => `<tr><td class="l"><b>${escapeHtml(r.signal_date)}</b><span class="sub">${escapeHtml(r.name)} ${escapeHtml(r.stock)}</span></td><td class="l">${statusLabel(r)}${r.is_forward ? badge("前瞻", "forward") : ""}</td><td>${fmtNum(r.z, 2)}</td><td class="${ppClass(r.dd20_pct)}">${fmtPct(r.dd20_pct, 2)}</td><td class="l">${escapeHtml(r.entry_date || "—")} → ${escapeHtml(r.exit_date || "—")}</td><td class="${ppClass(r.gross_pp)}">${r.gross_pp == null ? "—" : fmtPp(r.gross_pp)}</td><td class="${ppClass(r.excess_pp)}">${r.excess_pp == null ? "—" : fmtPp(r.excess_pp)}</td><td class="${ppClass(r.net_pp)}">${r.net_pp == null ? "—" : fmtPp(r.net_pp)}</td></tr>`).join("")}</tbody></table><div class="table-note">共 ${rows.length} 行；H-4 是唯一预声明前瞻。pending/missing 不填 0；毛收益为绝对收益；净超额 = 相对基线 − 双边 20bp。历史事件全部来自 v4 主规则（z≥2、dd≤-20%、F、次日开盘）。这里不展示全部 S 事件，只展示 S∩D。</div></div>` : `<div class="empty">这个筛选下没有事件。</div>`;
  }

  function renderLadder() {
    const rows = DATA.prediction.decision_ladder || [];
    $("#ladder-body").innerHTML = rows.map((r) => `<div class="rung ${r.state}"><div class="lvl">${escapeHtml(r.level)}</div><div class="name">${escapeHtml(r.label)}</div><div class="state">${badge(r.current || r.state, r.state === "open" ? "good" : r.state === "partial" ? "warn" : "neutral")}</div><div class="criteria">${escapeHtml(r.criteria)}</div></div>`).join("");
  }

  function renderExecution() {
    const e = DATA.prediction.execution || {};
    const items = [
      ["入场", e.entry], ["5 日出场", e.exit_h5], ["T+1", e.t_plus_1 ? "是（A 股）" : "—"],
      ["费用假设", `${fmtInt(e.cost_bps_roundtrip)} bp 双边`],
      ["涨跌停", Object.entries(e.price_limits_pct || {}).reduce((acc, [, v]) => ({ ...acc, [v]: (acc[v] || 0) + 1 }), {}) && Object.entries(Object.entries(e.price_limits_pct || {}).reduce((acc, [, v]) => { acc[v] = (acc[v] || 0) + 1; return acc; }, {})).map(([k, v]) => `${k}%×${v}`).join(" · ")],
      ["未建模", (e.unmodeled || []).join("、")]
    ];
    $("#execution-body").innerHTML = items.map(([k, v]) => `<div class="exec-item"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(v || "—")}</div></div>`).join("") + `<div class="callout warn" style="grid-column:1/-1;margin-top:6px">${escapeHtml(e.note || "")}</div>`;
  }

  function renderKill() {
    $("#kill-list").innerHTML = (DATA.prediction.kill_criteria || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
    $("#change-list").innerHTML = (DATA.prediction.change_my_mind || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
  }

  function ensurePrediction() {
    renderPrediction();
  }

  /* ------------------------------------------------------------------ */
  /* Boot                                                                */
  /* ------------------------------------------------------------------ */

  async function boot() {
    FONT = getComputedStyle(document.body).fontFamily || FONT;
    try {
      const response = await fetch(window.V4_CONFIG.dataUrl, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      DATA = await response.json();
    } catch (error) {
      showError(`无法读取 ${window.V4_CONFIG.dataUrl}：${error.message}`);
      return;
    }
    renderHeader();
    renderObservation();
    setTabs();
    resizeCharts();
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
