/* ==========================================================================
   Chronicle · Role D client runtime
   - synchronises the hidden video clock with the server playhead
   - draws object overlays inside the letterboxed video content area
   - owns Vis.js graph instances, the transport, drawer and query hooks
   ========================================================================== */

(function () {
  "use strict";

  const CHANNEL_MARKERS = {
    GEOM: "#38bdf8",
    TRACK: "#a78bfa",
    STATE: "#fbbf24",
    VLM: "#e879f9",
    GRAPH: "#34d399",
    QUERY: "#f472b6",
    SYSTEM: "#94a3b8",
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtTime(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    const rest = value - minutes * 60;
    return (minutes > 0 ? String(minutes).padStart(2, "0") + ":" : "") + rest.toFixed(2).padStart(5, "0");
  }

  function hexAlpha(hex, alpha) {
    let value = String(hex || "#94a3b8").replace("#", "");
    if (value.length === 3) value = value.split("").map((c) => c + c).join("");
    if (value.length !== 6) return "rgba(148,163,184," + alpha + ")";
    const r = parseInt(value.slice(0, 2), 16);
    const g = parseInt(value.slice(2, 4), 16);
    const b = parseInt(value.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function text(id, value) {
    const node = byId(id);
    if (node) node.textContent = value;
  }

  function safeFloat(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  /* ---------------------------------------------------------------------- */
  /* Vis.js graph                                                           */
  /* ---------------------------------------------------------------------- */

  class ChronicleGraph {
    constructor(containerId) {
      this.container = byId(containerId);
      this.network = null;
      this.nodes = null;
      this.edges = null;
      this.nodeSig = "";
      this.edgeSig = "";
      this.loaded = false;
      this.fitted = false;
      this.stabilizing = false;
      this.selected = null;

      if (!this.container) return;
      if (typeof vis === "undefined") {
        this.container.innerHTML =
          '<div class="graph-empty">vis-network failed to load · check /chronicle/vis-network.min.js</div>';
        return;
      }

      this.nodes = new vis.DataSet([]);
      this.edges = new vis.DataSet([]);
      this.network = new vis.Network(
        this.container,
        { nodes: this.nodes, edges: this.edges },
        {
          autoResize: true,
          height: "100%",
          width: "100%",
          nodes: {
            borderWidth: 2,
            borderWidthSelected: 3,
            shadow: false,
            chosen: true,
            font: {
              color: "#c5d7e8",
              size: 12,
              face: "ui-monospace, Cascadia Code, monospace",
              strokeWidth: 3,
              strokeColor: "#060b12",
            },
          },
          edges: {
            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
            smooth: { enabled: true, type: "continuous", roundness: 0.22 },
            selectionWidth: 2,
            hoverWidth: 1.6,
            font: {
              color: "#9db0c4",
              size: 9,
              face: "ui-monospace, Cascadia Code, monospace",
              strokeWidth: 3,
              strokeColor: "#060b12",
              align: "middle",
            },
          },
          layout: { improvedLayout: true, randomSeed: 11, clusterThreshold: 110 },
          physics: {
            enabled: true,
            solver: "forceAtlas2Based",
            forceAtlas2Based: {
              gravitationalConstant: -58,
              centralGravity: 0.012,
              springLength: 118,
              springConstant: 0.075,
              damping: 0.48,
              avoidOverlap: 0.45,
            },
            stabilization: {
              enabled: true,
              iterations: 260,
              updateInterval: 25,
              fit: true,
            },
            maxVelocity: 24,
            minVelocity: 0.15,
          },
          interaction: {
            hover: true,
            hoverConnectedEdges: true,
            tooltipDelay: 160,
            zoomView: true,
            dragView: true,
            dragNodes: true,
            multiselect: false,
            selectConnectedEdges: true,
          },
        },
      );

      this.network.on("stabilizationIterationsDone", () => {
        if (!this.loaded) return;
        this.stabilizing = false;
        this.network.setOptions({ physics: { enabled: false } });
        if (!this.fitted) {
          this.fitted = true;
          this.network.fit({ animation: { duration: 420, easingFunction: "easeInOutQuad" } });
        }
      });

      this.network.on("click", (params) => {
        if (params.nodes && params.nodes.length) {
          const id = String(params.nodes[0]);
          this.selected = id;
          if (window.Chronicle) Chronicle.node(id);
        }
      });
    }

    stabilize(iterations) {
      if (!this.network || this.stabilizing) return;
      this.stabilizing = true;
      this.network.setOptions({ physics: { enabled: true } });
      try {
        this.network.stabilize(Math.max(60, iterations || 140));
      } catch (error) {
        this.stabilizing = false;
      }
    }

    update(graph) {
      if (!this.network || !graph) return;
      const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
      const edges = Array.isArray(graph.edges) ? graph.edges : [];

      const nextNodeSig = nodes.map((node) => String(node.id)).sort().join("|");
      const nextEdgeSig = edges.map((edge) => String(edge.id) + ":" + edge.from + ">" + edge.to).sort().join("|");
      const structural = nextNodeSig !== this.nodeSig;
      const edgeStructural = nextEdgeSig !== this.edgeSig;

      const existingNodes = this.nodes.getIds();
      const currentNodeIds = new Set(nodes.map((node) => String(node.id)));
      const deadNodes = existingNodes.filter((id) => !currentNodeIds.has(String(id)));
      if (deadNodes.length) this.nodes.remove(deadNodes);

      const selection = this.network.getSelectedNodes();
      this.selected = selection.length ? String(selection[0]) : null;
      if (nodes.length) this.nodes.update(nodes);

      const existingEdges = this.edges.getIds();
      const currentEdgeIds = new Set(edges.map((edge) => String(edge.id)));
      const deadEdges = existingEdges.filter((id) => !currentEdgeIds.has(String(id)));
      if (deadEdges.length) this.edges.remove(deadEdges);
      if (edges.length) this.edges.update(edges);

      if (structural && !this.loaded && nodes.length) {
        this.loaded = true;
        this.stabilize(280);
      } else if (structural && this.loaded && nodes.length) {
        this.stabilize(150);
      } else if (!this.loaded && nodes.length) {
        this.loaded = true;
        this.stabilize(240);
      } else if (!structural && edgeStructural && nodes.length && !this.loaded) {
        this.loaded = true;
        this.stabilize(200);
      }

      if (this.selected) {
        try {
          this.network.selectNodes([this.selected], false);
        } catch (error) {
          /* node may have disappeared */
        }
      }

      this.nodeSig = nextNodeSig;
      this.edgeSig = nextEdgeSig;
    }

    setSelected(nodeId) {
      this.selected = nodeId ? String(nodeId) : null;
      if (!this.network) return;
      if (this.selected) {
        try {
          this.network.selectNodes([this.selected], false);
        } catch (error) {
          /* ignore */
        }
      } else {
        this.network.unselectAll();
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Chronicle controller                                                   */
  /* ---------------------------------------------------------------------- */

  const Chronicle = {
    booted: false,
    attempts: 0,
    video: null,
    overlay: null,
    graph: null,
    playhead: 0,
    duration: 0,
    playing: false,
    rate: 1,
    graphMode: "live",
    dragging: false,
    dragTimer: null,
    lastVideoUrl: null,
    clockTimer: null,

    boot() {
      if (this.booted) return true;
      this.attempts += 1;

      const shell = document.querySelector(".video-shell");
      this.video = document.querySelector(".chron-video video") || document.querySelector(".video-shell video");
      this.overlay = byId("chron-overlay");

      if (!shell || !this.video || !this.overlay) {
        if (this.attempts < 80) setTimeout(() => this.boot(), 100);
        return false;
      }

      this.video.muted = true;
      this.video.playsInline = true;
      this.video.loop = true;
      this.video.preload = "auto";
      this.video.setAttribute("muted", "");
      this.video.setAttribute("playsinline", "");
      this.video.style.pointerEvents = "none";

      if (byId("chron-graph")) {
        this.graph = new ChronicleGraph("chron-graph");
      }

      const scrub = byId("chron-scrub");
      if (scrub) {
        scrub.addEventListener("pointerdown", () => {
          this.dragging = true;
          if (this.dragTimer) clearTimeout(this.dragTimer);
        });
        scrub.addEventListener("input", () => this.preview(safeFloat(scrub.value, 0)));
        scrub.addEventListener("change", () => {
          const value = safeFloat(scrub.value, 0);
          this.commit(value);
        });
        scrub.addEventListener("pointerup", () => {
          /* change fires after pointerup; keep the local preview briefly */
          if (this.dragTimer) clearTimeout(this.dragTimer);
          this.dragTimer = setTimeout(() => {
            this.dragging = false;
          }, 450);
        });
      }

      document.addEventListener("keydown", (event) => this.onKey(event));
      this.clockTimer = setInterval(() => this.tickClock(), 1000);
      this.tickClock();

      this.booted = true;
      return true;
    },

    tickClock() {
      const now = new Date();
      text(
        "chron-clock",
        now.toLocaleTimeString("en-GB", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        }),
      );
      text(
        "chron-clock-zone",
        now.toLocaleDateString("en-GB", { day: "2-digit", month: "short" }).toUpperCase() + " · LOCAL",
      );
    },

    onKey(event) {
      const target = event.target;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        this.togglePlay();
      } else if (event.code === "ArrowRight") {
        event.preventDefault();
        this.seek(Math.min(this.duration, this.playhead + (event.shiftKey ? 5 : 1)), false);
      } else if (event.code === "ArrowLeft") {
        event.preventDefault();
        this.seek(Math.max(0, this.playhead - (event.shiftKey ? 5 : 1)), false);
      } else if (event.key === "r" || event.key === "R") {
        this.restart();
      } else if (event.key === "g" || event.key === "G") {
        this.setGraphMode(this.graphMode === "live" ? "full" : "live");
      } else if (event.key === "Escape") {
        this.closeDrawer();
      }
    },

    sync(data) {
      if (!data) return;
      if (!this.booted && !this.boot()) {
        return;
      }
      this.playhead = safeFloat(data.t, this.playhead);
      this.duration = safeFloat(data.duration, this.duration);
      this.playing = Boolean(data.playing);
      this.rate = safeFloat(data.rate, 1);

      this.syncVideo(data);
      this.renderBoxes(data.boxes || [], data);
      this.renderHud(data.hud || {}, data);
      this.setTransport(data);
      this.setStats(data.stats || {}, data);

      if (this.graph && data.graph) {
        this.graph.update(data.graph);
      }
    },

    syncVideo(data) {
      const video = this.video;
      if (!video) return;

      if (data.video_url && this.lastVideoUrl === null) {
        this.lastVideoUrl = data.video_url;
      } else if (data.video_url && data.video_url !== this.lastVideoUrl) {
        this.lastVideoUrl = data.video_url;
        try {
          video.src = data.video_url;
          video.load();
        } catch (error) {
          /* ignore */
        }
      }

      try {
        if (video.playbackRate !== this.rate) video.playbackRate = this.rate;
      } catch (error) {
        /* ignore */
      }

      const shouldLoop = String(data.mode || "demo").toLowerCase() === "demo";
      if (video.loop !== shouldLoop) video.loop = shouldLoop;

      const target = safeFloat(data.t, 0);
      if (video.readyState >= 1) {
        const drift = Math.abs(video.currentTime - target);
        const looped = target + 0.45 < video.currentTime;
        if (drift > 0.38 || looped) {
          try {
            video.currentTime = target;
          } catch (error) {
            /* seeking may fail before metadata */
          }
        }
      }

      if (this.playing) {
        if (video.paused) {
          const promise = video.play();
          if (promise && promise.catch) promise.catch(() => {});
        }
      } else if (!video.paused) {
        video.pause();
      }
    },

    renderBoxes(boxes, data) {
      const overlay = this.overlay;
      const video = this.video;
      if (!overlay || !video || !boxes.length) {
        if (overlay) overlay.innerHTML = "";
        return;
      }

      const overlayRect = overlay.getBoundingClientRect();
      const videoRect = video.getBoundingClientRect();
      if (!overlayRect.width || !overlayRect.height) return;

      const vw = video.videoWidth || safeFloat(data && data.width, 1280);
      const vh = video.videoHeight || safeFloat(data && data.height, 720);
      const scale = Math.min(videoRect.width / vw, videoRect.height / vh);
      const contentW = vw * scale;
      const contentH = vh * scale;
      const offsetX = videoRect.left - overlayRect.left + (videoRect.width - contentW) / 2;
      const offsetY = videoRect.top - overlayRect.top + (videoRect.height - contentH) / 2;

      const html = [];
      boxes.forEach((box) => {
        const bbox = Array.isArray(box.bbox) ? box.bbox : null;
        if (!bbox || bbox.length < 4) return;
        const x1 = Math.max(0, Math.min(1, safeFloat(bbox[0], 0)));
        const y1 = Math.max(0, Math.min(1, safeFloat(bbox[1], 0)));
        const x2 = Math.max(0, Math.min(1, safeFloat(bbox[2], 0)));
        const y2 = Math.max(0, Math.min(1, safeFloat(bbox[3], 0)));
        const width = (x2 - x1) * contentW;
        const height = (y2 - y1) * contentH;
        if (width < 4 || height < 4) return;

        const color = box.color || "#38bdf8";
        const state = String(box.state || "NEW").replace(/_/g, " ");
        const label = String(box.label || box.id || "").toUpperCase();
        const held = box.held ? " held" : "";
        html.push(
          '<div class="obj-box' +
            held +
            '" data-node-id="' +
            escapeHtml(box.id) +
            '" onclick="Chronicle.node(this.dataset.nodeId)" style="' +
            "left:" +
            (offsetX + x1 * contentW).toFixed(2) +
            "px;top:" +
            (offsetY + y1 * contentH).toFixed(2) +
            "px;width:" +
            width.toFixed(2) +
            "px;height:" +
            height.toFixed(2) +
            "px;--box-color:" +
            escapeHtml(color) +
            ";background:linear-gradient(180deg," +
            hexAlpha(color, 0.16) +
            "," +
            hexAlpha(color, 0.015) +
            ');">' +
            '<span class="obj-tag">' +
            escapeHtml(label) +
            ' <span class="obj-state">' +
            escapeHtml(state) +
            "</span></span></div>",
        );
      });
      overlay.innerHTML = html.join("");
    },

    renderHud(hud, data) {
      const camera = hud.camera || {};
      const cameraState = String(camera.state || "—").toUpperCase();
      text("hud-camera", "CAM · " + cameraState);
      text("hud-zone", "ZONE · " + String(hud.zone || "UNKNOWN").toUpperCase());
      text("hud-tracks", "TRACKS · " + String(hud.tracks || "0") + "/" + String(hud.total || "0"));
      text("hud-latency", "PIPE · " + (hud.latency == null ? "—" : Math.round(hud.latency) + "ms"));
      text("hud-fps", (data.fps ? String(Math.round(data.fps)) : "—") + " FPS");
      text("hud-mode", String(data.source_label || data.mode || "LIVE").toUpperCase());
      text("hud-time", "TC " + fmtTime(data.t));

      const shell = document.querySelector(".video-shell");
      if (shell) {
        shell.dataset.camera = cameraState;
      }

      const caption = byId("hud-caption");
      if (!caption) return;
      if (hud.caption) {
        caption.classList.remove("hidden");
        const tag = byId("hud-caption-tag");
        const body = byId("hud-caption-text");
        if (tag) {
          tag.innerHTML =
            "VLM · " +
            escapeHtml(hud.caption.action || "INTERACTION") +
            " · " +
            escapeHtml(hud.caption.entity || "") +
            " · " +
            escapeHtml(fmtTime(hud.caption.t)) +
            "s";
        }
        if (body) body.textContent = hud.caption.text || "";
      } else {
        caption.classList.add("hidden");
      }
    },

    setTransport(data) {
      const t = safeFloat(data.t, 0);
      const duration = Math.max(0.001, safeFloat(data.duration, 1));
      const percent = Math.max(0, Math.min(100, (t / duration) * 100));

      if (!this.dragging) {
        const fill = byId("chron-fill");
        const head = byId("chron-head");
        const scrub = byId("chron-scrub");
        if (fill) fill.style.width = percent.toFixed(2) + "%";
        if (head) head.style.left = percent.toFixed(2) + "%";
        if (scrub) scrub.value = String(t);
      }

      const time = byId("chron-time");
      if (time) {
        time.innerHTML =
          fmtTime(t) + ' <span class="total">/ ' + fmtTime(duration) + "</span>";
      }

      const play = byId("chron-play-icon");
      if (play) play.textContent = this.playing ? "❚❚" : "▶";
      const playButton = byId("chron-play");
      if (playButton) playButton.title = this.playing ? "Pause (Space)" : "Play (Space)";

      const rate = byId("chron-rate");
      if (rate) rate.textContent = this.rate.toFixed(1) + "×";

      text("chron-mode-badge", String(data.source_label || data.mode || "DEMO").toUpperCase());
      text("chron-meta", fmtTime(duration) + " · " + data.fps + " FPS · " + data.width + "×" + data.height);
      text("chron-episode", data.title || "Chronicle Episode");

      const badge = byId("chron-mode");
      if (badge) {
        const live = String(data.mode || "").toLowerCase() === "live";
        badge.classList.toggle("live", live);
        badge.classList.toggle("demo", !live);
      }
    },

    setStats(stats, data) {
      text("stat-objects", String(stats.entities == null ? "—" : stats.entities));
      text("stat-active", String(stats.active_entities == null ? "—" : stats.active_entities));
      text("stat-edges", String(stats.edges == null ? "—" : stats.edges));
      text("stat-vlm", String(stats.vlm_calls == null ? "—" : stats.vlm_calls));
      text("stat-fps", stats.fps == null ? "—" : String(Math.round(stats.fps)));
      text("stat-latency", stats.latency_ms == null ? "—" : String(Math.round(stats.latency_ms)));

      const graphCount = byId("graph-count");
      if (graphCount && data.graph) {
        graphCount.textContent =
          String((data.graph.nodes || []).length) + "N / " + String((data.graph.edges || []).length) + "E";
      }

      text("query-provider", data.query_provider || "local-graphrag");
      text("source-events", stats.live_events == null ? "0" : String(stats.live_events));
      text("header-rate", stats.event_rate == null ? "0.0" : Number(stats.event_rate).toFixed(1));
    },

    preview(t) {
      const duration = Math.max(0.001, this.duration);
      const percent = Math.max(0, Math.min(100, (t / duration) * 100));
      const fill = byId("chron-fill");
      const head = byId("chron-head");
      const time = byId("chron-time");
      if (fill) fill.style.width = percent.toFixed(2) + "%";
      if (head) head.style.left = percent.toFixed(2) + "%";
      if (time) {
        time.innerHTML = fmtTime(t) + ' <span class="total">/ ' + fmtTime(duration) + "</span>";
      }
    },

    commit(t) {
      this.dragging = false;
      this.playhead = safeFloat(t, 0);
      this.preview(this.playhead);
      if (window.emitEvent) emitEvent("chron-seek", this.playhead, 0);
      if (this.dragTimer) clearTimeout(this.dragTimer);
      this.dragTimer = setTimeout(() => {
        this.dragging = false;
      }, 250);
    },

    seek(t, pause) {
      const target = Math.max(0, Math.min(this.duration || 9999, safeFloat(t, 0)));
      this.playhead = target;
      this.preview(target);
      if (window.emitEvent) emitEvent("chron-seek", target, pause ? 1 : 0);
    },

    togglePlay() {
      this.playing = !this.playing;
      const play = byId("chron-play-icon");
      if (play) play.textContent = this.playing ? "❚❚" : "▶";
      if (window.emitEvent) emitEvent("chron-toggle");
    },

    restart() {
      this.playhead = 0;
      this.preview(0);
      if (window.emitEvent) emitEvent("chron-restart");
    },

    cycleRate() {
      if (window.emitEvent) emitEvent("chron-rate");
    },

    setGraphMode(mode) {
      const next = mode === "full" ? "full" : "live";
      this.graphMode = next;
      document.querySelectorAll(".mode-btn[data-mode]").forEach((button) => {
        button.classList.toggle("active", button.dataset.mode === next);
      });
      if (window.emitEvent) emitEvent("chron-graph-mode", next);
    },

    node(nodeId) {
      if (!nodeId) return;
      if (this.graph) this.graph.setSelected(nodeId);
      if (window.emitEvent) emitEvent("chron-node", String(nodeId));
    },

    openDrawer() {
      const drawer = byId("chron-inspector");
      const backdrop = byId("chron-backdrop");
      if (drawer) drawer.classList.add("open");
      if (backdrop) backdrop.classList.add("open");
    },

    closeDrawer() {
      const drawer = byId("chron-inspector");
      const backdrop = byId("chron-backdrop");
      if (drawer) drawer.classList.remove("open");
      if (backdrop) backdrop.classList.remove("open");
      if (this.graph) this.graph.setSelected(null);
    },

    setQueryResult(html) {
      const result = byId("chron-query-result");
      if (result) result.innerHTML = html;
    },

    ask(textValue) {
      const input = document.querySelector(".chron-query-input input");
      if (input && !textValue) textValue = input.value;
      if (!textValue) return;
      if (input) input.value = "";
      if (window.emitEvent) emitEvent("chron-query", String(textValue));
    },

    scrollTelemetry() {
      const log = document.querySelector(".chron-log");
      if (!log) return;
      const scroll = log.querySelector(".q-scrollarea__container");
      if (scroll) scroll.scrollTop = scroll.scrollHeight;
    },
  };

  window.Chronicle = Chronicle;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => setTimeout(() => Chronicle.boot(), 40));
  } else {
    setTimeout(() => Chronicle.boot(), 40);
  }

  // Absolutely no console spam from expected autoplay policies.
  window.addEventListener("unhandledrejection", (event) => {
    const reason = event && event.reason;
    if (reason && String(reason).toLowerCase().includes("play")) event.preventDefault();
  });
})();
