/* ==========================================================================
   Chronicle · Role D client runtime — zine edition
   - synchronises the hidden video clock with the server playhead
   - draws object overlays inside the letterboxed video content area
   - owns the Vis.js memory graph, chat engine, entity list, event strips
   ========================================================================== */

(function () {
  "use strict";

  /* ---------------- zine palette ---------------- */

  var STATE_ZINE = {
    NEW: "#a8a29e",
    VISIBLE_STATIC: "#37c585",
    INTERACTED: "#ffd34d",
    VISIBLE_MOVING: "#56b8d8",
    HELD: "#b08bd9",
    CONTAINED: "#56d0c0",
    OCCLUDED: "#b7ae98",
    OUT_OF_VIEW: "#ff9eb5",
    LOST: "#ff6b4a",
    ANCHOR: "#26221a",
    AGENT: "#ff6b4a",
  };

  var EV_COLORS = {
    state: "#56b8d8",
    interaction: "#ff6b4a",
    edge: "#37c585",
    zone: "#b08bd9",
  };

  var ACTION_REL = {
    PICKED_UP: 1,
    PICK_UP: 1,
    PUT_DOWN: 1,
    PLACED_IN: 1,
    INSERT_INTO: 1,
    PLACED_ON: 1,
    TOUCHED: 1,
    OPENED: 1,
    CLOSED: 1,
    USED: 1,
    MOVED: 1,
  };

  var THINKING_LINES = [
    "flipping through my memory…",
    "rewatching the good parts…",
    "cross-checking my notes…",
    "squinting at frame buffers…",
  ];

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
    let value = String(hex || "#ffd34d").replace("#", "");
    if (value.length === 3) value = value.split("").map((c) => c + c).join("");
    if (value.length !== 6) return "rgba(255,211,77," + alpha + ")";
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

  function zineColor(state) {
    return STATE_ZINE[String(state || "").toUpperCase()] || "#ffd34d";
  }

  /* ---------------------------------------------------------------------- */
  /* Vis.js graph — zine theme                                              */
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
              color: "#26221a",
              size: 12,
              face: "'DM Sans', 'Bricolage Grotesque', sans-serif",
              strokeWidth: 4,
              strokeColor: "#fffcf4",
            },
          },
          edges: {
            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
            smooth: { enabled: true, type: "continuous", roundness: 0.22 },
            selectionWidth: 2,
            hoverWidth: 1.6,
            font: {
              color: "#7c7361",
              size: 10,
              face: "'IBM Plex Mono', monospace",
              strokeWidth: 4,
              strokeColor: "#fdf8ea",
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

    /* Restyle a payload node for the paper look */
    restyleNode(node) {
      const state = String(node.state || "NEW").toUpperCase();
      const kind = String(node.kind || "entity");
      const dot = zineColor(state);
      if (kind === "entity") {
        node.color = {
          background: dot,
          border: "#26221a",
          highlight: { background: dot, border: "#26221a" },
          hover: { background: dot, border: "#26221a" },
        };
        node.borderWidth = node.dim ? 1.5 : 2.5;
        node.opacity = node.dim ? 0.35 : 1;
        node.font = {
          color: "#26221a",
          size: 12,
          face: "'DM Sans', 'Bricolage Grotesque', sans-serif",
          strokeWidth: 4,
          strokeColor: "#fffcf4",
        };
      } else if (kind === "zone") {
        node.color = {
          background: "rgba(255,211,77,0.18)",
          border: "rgba(38,34,26,0.9)",
          highlight: { background: "rgba(255,211,77,0.6)", border: "#26221a" },
          hover: { background: "rgba(255,211,77,0.4)", border: "#26221a" },
        };
        node.borderWidth = 2;
        node.shapeProperties = { borderDashes: node.dim ? [4, 3] : false };
        node.font = {
          color: node.active ? "#26221a" : "#7c7361",
          size: 14,
          face: "'IBM Plex Mono', monospace",
          strokeWidth: 4,
          strokeColor: "#fdf8ea",
          vadjust: -30,
        };
      } else if (kind === "anchor") {
        node.color = {
          background: "rgba(124,115,97,0.16)",
          border: "rgba(38,34,26,0.55)",
          highlight: { background: "rgba(124,115,97,0.4)", border: "#26221a" },
        };
        node.font = {
          color: "#7c7361",
          size: 11,
          face: "'IBM Plex Mono', monospace",
          strokeWidth: 4,
          strokeColor: "#fdf8ea",
        };
      } else if (kind === "user") {
        node.color = {
          background: "#ff6b4a",
          border: "#26221a",
          highlight: { background: "#ff6b4a", border: "#26221a" },
        };
        node.font = {
          color: "#26221a",
          size: 12,
          face: "'DM Sans', sans-serif",
          strokeWidth: 4,
          strokeColor: "#fffcf4",
        };
      }
      return node;
    }

    restyleEdge(edge) {
      const relation = String(edge.relation || "").toUpperCase();
      const isAction = !!ACTION_REL[relation];
      const color = isAction ? "#ffd34d" : relation === "INSIDE" || relation === "CONTAINED_IN" ? "#56d0c0" : "#37c585";
      edge.color = {
        color: hexAlpha(color, edge.dim ? 0.35 : 0.95),
        highlight: color,
        hover: color,
      };
      edge.dashes = edge.dashes !== false;
      edge.font = {
        color: "#7c7361",
        size: 10,
        face: "'IBM Plex Mono', monospace",
        strokeWidth: 4,
        strokeColor: "#fdf8ea",
        align: "middle",
      };
      return edge;
    }

    update(graph) {
      if (!this.network || !graph) return;
      const nodes = Array.isArray(graph.nodes) ? graph.nodes.map((node) => this.restyleNode(Object.assign({}, node))) : [];
      const edges = Array.isArray(graph.edges) ? graph.edges.map((edge) => this.restyleEdge(Object.assign({}, edge))) : [];

      const nextNodeSig = nodes.map((node) => String(node.id)).sort().join("|");
      const nextEdgeSig = edges.map((edge) => String(edge.id) + ":" + edge.from + ">" + edge.to).sort().join("|");
      const structural = nextNodeSig !== this.nodeSig;

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
    /* client state */
    selected: null,
    entities: [],
    markers: [],
    entFilter: "",
    chatBusy: false,
    chatId: 0,
    lastHtml: {},   /* render caches */
    suggestions: [],

    boot() {
      if (this.booted) return true;
      this.attempts += 1;

      const shell = byId("chron-shell");
      this.video = document.querySelector("#chron-shell video") || document.querySelector(".chron-video");
      this.overlay = byId("chron-overlay");

      if (!shell || !this.overlay || (this.video === null && document.querySelector("#stage-hint") === null)) {
        if (this.attempts < 80) setTimeout(() => this.boot(), 100);
        return false;
      }

      if (this.video) {
        this.video = this.video;
        this.video.muted = true;
        this.video.playsInline = true;
        this.video.loop = true;
        this.video.preload = "auto";
        this.video.setAttribute("muted", "");
        this.video.setAttribute("playsinline", "");
        this.video.style.pointerEvents = "none";
      }

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
        this.closeInspector();
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
      if (data.entities) {
        this.entities = data.entities;
        this.renderEntities();
        this.updateChatStatus(data);
      }
      if (data.markers) {
        this.markers = data.markers;
        this.renderEvents();
        this.renderActions();
      }
      if (data.suggestions) {
        this.setSuggestions(data.suggestions);
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
        const hint = byId("stage-hint");
        if (hint) hint.remove();
      } else if (!video.paused) {
        video.pause();
      }
    },

    renderBoxes(boxes, data) {
      const overlay = this.overlay;
      const video = this.video;
      if (!overlay) return;
      if (!boxes.length) {
        if (this.lastHtml.boxes) {
          this.lastHtml.boxes = "";
          overlay.innerHTML = "";
        }
        return;
      }
      if (!video) return;

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

        const color = zineColor(box.state) || "#ffd34d";
        const state = String(box.state || "NEW").replace(/_/g, " ").toLowerCase();
        const label = String(box.label || box.id || "");
        const held = box.held ? " held" : "";
        html.push(
          '<div class="obj-box' + held + '" data-node-id="' +
            escapeHtml(box.id) +
            '" onclick="Chronicle.node(this.dataset.nodeId)" style="' +
            "left:" + (offsetX + x1 * contentW).toFixed(2) + "px;top:" +
            (offsetY + y1 * contentH).toFixed(2) + "px;width:" + width.toFixed(2) +
            "px;height:" + height.toFixed(2) + "px;--obj-color:" + escapeHtml(color) + ";" +
            "background:" + hexAlpha(color, 0.14) + ';">' +
            '<span class="obj-tag">' + escapeHtml(label) +
            ' <span class="obj-state">' + escapeHtml(state) + "</span></span></div>"
        );
      });
      const next = html.join("");
      if (this.lastHtml.boxes === next) return;
      this.lastHtml.boxes = next;
      overlay.innerHTML = next;
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

      const shell = byId("chron-shell");
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
            escapeHtml(String(hud.caption.action || "INTERACTION").replace(/_/g, " ").toUpperCase()) +
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
        if (scrub && String(scrub.max) !== String(data.duration) && data.duration) {
          scrub.max = String(data.duration);
        }
        if (scrub && document.activeElement !== scrub) scrub.value = String(t);
      }

      const time = byId("chron-time");
      if (time) {
        time.innerHTML = fmtTime(t) + ' <span class="total">/ ' + fmtTime(duration) + "</span>";
      }

      const play = byId("chron-play-icon");
      if (play) play.textContent = this.playing ? "❚❚" : "▶";
      const playButton = byId("chron-play");
      if (playButton) playButton.title = this.playing ? "Pause (Space)" : "Play (Space)";
      if (playButton) playButton.style.background = this.playing ? "" : "var(--mustard)";

      const rate = byId("chron-rate");
      if (rate) rate.textContent = this.rate.toFixed(1) + "×";

      const labels = document.querySelector(".tl-labels");
      if (labels && duration > 0.01 && labels.dataset.dur !== String(data.duration)) {
        labels.dataset.dur = String(data.duration);
        labels.innerHTML = "\u003cspan>" + [0, 1, 2, 3, 4, 5]
          .map((index) => ((duration * index) / 5).toFixed(0) + "s")
          .join("</span>\u003cspan>") + "</span>";
      }
      text("t-loop", String(data.source_label || ""));
      text("transport-label", this.playing ? "SYNCED PLAYBACK · LOOP" : "PAUSED · HOLDING MY BREATH");

      const badge = byId("chron-mode");
      const badgeText = byId("chron-mode-badge");
      if (badge) {
        const live = String(data.mode || "").toLowerCase() === "live";
        badge.classList.toggle("is-live", live);
        badge.classList.toggle("is-demo", !live);
        badge.classList.toggle("is-empty", !live && !data.duration);
      }
      if (badgeText) {
        badgeText.textContent = String(data.source_label || data.mode || "DEMO").toUpperCase();
      }
      text("foot-source", String(data.source_label || data.mode || "demo"));
    },

    renderMarkers() {
      /* timeline diamonds inside the scrub track */
      const track = document.querySelector(".tl-track");
      if (!track) return;
      const duration = Math.max(0.001, this.duration);
      const html = [];
      this.markers.forEach((marker) => {
        const t = safeFloat(marker.t, 0);
        const left = Math.max(0, Math.min(100, (t / duration) * 100));
        const kind = String(marker.kind || "state");
        html.push(
          '<span class="tl-marker" style="left:' + left.toFixed(3) +
          "%;--marker:" + escapeHtml(EV_COLORS[kind] || "#ffd34d") +
          ';" title="' + escapeHtml(String(marker.label || kind)) + " @ " + t.toFixed(2) + 's"></span>'
        );
      });
      const next = html.join("");
      if (this.lastHtml.markers === next) return;
      this.lastHtml.markers = next;
      track.querySelectorAll(".tl-marker").forEach((node) => node.remove());
      track.insertAdjacentHTML("beforeend", next);
    },

    setStats(stats, data) {
      text("stat-objects", String(stats.entities == null ? "—" : stats.entities));
      text("stat-active", String(stats.active_entities == null ? "—" : stats.active_entities));
      text("stat-edges", String(stats.edges == null ? "—" : stats.edges));
      text("stat-vlm", String(stats.vlm_calls == null ? "—" : stats.vlm_calls));
      text("stat-fps", stats.fps == null ? "—" : String(Math.round(stats.fps)));
      text("stat-latency", stats.latency_ms == null ? "—" : String(Math.round(stats.latency_ms)) + " ms");

      text("metric-frames", String(stats.frames == null ? "—" : stats.frames));
      text("metric-fps", stats.fps == null ? "—" : String(Math.round(stats.fps)));
      text("metric-entities", String(stats.entities == null ? "—" : stats.entities));
      text("metric-edges", String(stats.edges == null ? "—" : stats.edges));
      text("metric-actions", String(stats.vlm_calls == null ? "—" : stats.vlm_calls));

      const graphCount = byId("graph-count");
      if (graphCount && data.graph) {
        graphCount.textContent =
          String((data.graph.nodes || []).length) + "N / " + String((data.graph.edges || []).length) + "E";
      }

      text("query-provider", data.query_provider || "local-graphrag");
      text("chron-meta", fmtTime(data.duration) + " · " + (data.fps == null ? "—" : Math.round(data.fps)) + " FPS");
      this.renderMarkers();
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
      this.selected = String(nodeId);
      if (this.graph) this.graph.setSelected(this.selected);
      this.renderEntities();
      if (window.emitEvent) emitEvent("chron-node", this.selected);
    },

    /* ---------------- inspector dossier ---------------- */

    closeInspector() {
      this.selected = null;
      if (this.graph) this.graph.setSelected(null);
      this.renderEntities();
      const host = byId("chron-inspector-content");
      if (host) {
        host.innerHTML =
          '<div class="inspector-head">' +
          '<div class="inspector-head-main">' +
          '<div class="inspector-eyebrow">Entity dossier</div>' +
          '<div class="inspector-title">no file open</div>' +
          '<div class="inspector-sub">click any object in the list or the graph</div>' +
          "</div>" +
          '<button class="inspector-close" onclick="Chronicle.closeInspector()" aria-label="Close dossier">×</button>' +
          "</div>" +
          '<div class="inspector-body">' +
          '<div class="empty-strip"><span class="empty-note" style="text-align:center;padding:0 14px">' +
          "pick an entity — its crop, state history, clips and relations open here." +
          "</span></div></div>";
      }
    },

    /* ---------------- entity list ---------------- */

    filterEntities(value) {
      this.entFilter = String(value || "").trim().toLowerCase();
      this.renderEntities();
    },

    renderEntities() {
      const host = byId("chron-ent-list");
      if (!host) return;
      const query = this.entFilter;
      const seen = [];
      for (const entity of this.entities) {
        if (!query) { seen.push(entity); continue; }
        const hay = (
          String(entity.label || "") + " " + String(entity.zone || "") + " " + String(entity.state || "")
        ).toLowerCase();
        if (hay.indexOf(query) !== -1) seen.push(entity);
      }
      const bits = [];
      seen.forEach((entity) => {
        const isActive = !!entity.seen;
        const confPct = Math.round(safeFloat(entity.conf, 0) * 100);
        const dataSel = this.selected === String(entity.id);
        bits.push(
          '<button class="ent-row' + (dataSel ? " selected" : "") + (isActive ? "" : " unseen") + '"' +
            ' data-node-id="' + escapeHtml(entity.id) + '"' +
            ' onclick="Chronicle.node(this.dataset.nodeId)"' +
            ' style="--ent-color:' + escapeHtml(entity.color || "#ffd34d") + '">' +
            '<span class="ent-dot"></span>' +
            '<span class="ent-main">' +
              '<span class="ent-name">' + escapeHtml(entity.label) +
                (isActive && confPct > 0 && confPct < 70 ? '<span class="chipz chipz-blush ent-tag">low conf</span>' : "") +
                (!isActive ? '<span class="chipz chipz-secondary ent-tag">not seen</span>' : "") +
              "</span>" +
              '<span class="ent-sub">' +
                escapeHtml(isActive ? (entity.zone || "—") + " · " + String(entity.state || "").replace(/_/g, " ").toLowerCase() : "—") +
              "</span>" +
            "</span>" +
            (isActive
              ? '<span class="ent-ask" onclick="event.stopPropagation();Chronicle.chatSend(\'where is the ' +
                escapeHtml(String(entity.label).replace(/'/g, "").toLowerCase()) +
                '?\')">ask</span>'
              : "") +
          "</button>"
        );
      });
      const next = bits.join("");
      if (this.lastHtml.ents === next) return;
      this.lastHtml.ents = next;
      host.innerHTML =
        next ||
        '<div class="empty-strip"><span class="empty-note">' +
        (query ? "no match — try “cup” or “table”" : "nothing yet — feed me video") +
        "</span></div>";
      const count = byId("ent-count");
      if (count) {
        const seenCount = this.entities.filter((entity) => entity.seen).length;
        count.textContent = seenCount + " / " + this.entities.length + " objects";
      }
    },

    /* ---------------- event strips ---------------- */

    renderEvents() {
      const host = byId("chron-events");
      if (!host) return;
      const items = this.markers.slice(-24);
      if (!items.length) {
        const empty = '<div class="empty-strip" style="width:100%"><span class="empty-note">events land here once I start noticing things…</span></div>';
        if (this.lastHtml.events !== empty) {
          this.lastHtml.events = empty;
          host.innerHTML = empty;
        }
        return;
      }
      const cleanId = (value) => String(value).replace(/[^A-Za-z0-9_:\-]/g, "");
      const bits = [];
      items.forEach((marker, index) => {
        const kind = String(marker.kind || "state");
        const label = String(marker.label || kind).replace(/_/g, " ").toLowerCase();
        const target = marker.entity_id ? cleanId(marker.entity_id) : "";
        const click = target
          ? "Chronicle.node('" + target + "')"
          : "Chronicle.seek(" + safeFloat(marker.t, 0).toFixed(3) + ", true)";
        bits.push(
          '<button class="ev-item" title="' + escapeHtml(label) + ' · ' + escapeHtml(fmtTime(marker.t)) +
            '" onclick="' + click + '">' +
            '<span class="ev-t">' + escapeHtml(fmtTime(marker.t)) + "</span>" +
            '<span class="ev-diamond" style="--ev-color:' + escapeHtml(EV_COLORS[kind] || "#ffd34d") + '"></span>' +
            '<span class="ev-text">' + escapeHtml(label) + "</span>" +
          "</button>"
        );
        if (index < items.length - 1) bits.push('<span class="ev-link"></span>');
      });
      const next = bits.join("");
      if (this.lastHtml.events === next) return;
      this.lastHtml.events = next;
      host.innerHTML = next;
      const scroller = host.parentElement;
      if (scroller && !this.dragging) scroller.scrollLeft = scroller.scrollWidth;
      const count = byId("ev-count");
      if (count) count.textContent = this.markers.length + " events";
    },

    renderActions() {
      const host = byId("chron-actions");
      if (!host) return;
      const items = this.markers.filter((marker) => marker.kind === "interaction").slice(-7).reverse();
      if (!items.length) {
        const empty = '<div class="empty-strip"><span class="empty-note">no drama yet — it\u2019s early</span></div>';
        if (this.lastHtml.actions !== empty) {
          this.lastHtml.actions = empty;
          host.innerHTML = empty;
        }
        return;
      }
      const cleanId = (value) => String(value).replace(/[^A-Za-z0-9_:\-]/g, "");
      const bits = [];
      items.forEach((marker) => {
        const targetId = marker.entity_id ? cleanId(marker.entity_id) : "";
        const click = targetId
          ? "Chronicle.node('" + targetId + "')"
          : "Chronicle.seek(" + safeFloat(marker.t, 0).toFixed(3) + ", true)";
        bits.push(
          '<div class="action-row" onclick="' + click + '" style="cursor:pointer">' +
            '<span class="act-ico" style="--act-color:#ff6b4a">' +
              '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true">' +
                '<path d="M1.5 8 C 4 3.5, 12 3.5, 14.5 8 C 12 12.5, 4 12.5, 1.5 8 Z" stroke="currentColor" stroke-width="1.8"/>' +
                '<circle cx="8" cy="8" r="2.2" fill="currentColor"/>' +
              "</svg>" +
            "</span>" +
            '<span class="act-text">' +
              escapeHtml(String(marker.label || "action").replace(/_/g, " ").toLowerCase()) +
            "</span>" +
            '<span class="act-time">' + escapeHtml(fmtTime(marker.t)) + "</span>" +
          "</div>"
        );
      });
      const next = bits.join("");
      if (this.lastHtml.actions === next) return;
      this.lastHtml.actions = next;
      host.innerHTML = next;
    },

    /* ---------------- chat ---------------- */

    setSuggestions(list) {
      const host = byId("chron-suggestions");
      if (!host || !Array.isArray(list) || !list.length) return;
      const next = list
        .map(
          (query, index) =>
            '<button class="chipz sug' + (index === 0 ? " first" : "") + '" data-query="' +
            escapeHtml(query) + '" onclick="Chronicle.chatSend(this.dataset.query)">' +
            escapeHtml(query) + "</button>"
        )
        .join("");
      if (this.lastHtml.sugs === next) return;
      this.lastHtml.sugs = next;
      this.suggestions = list;
      host.innerHTML = next;
    },

    updateChatStatus(data) {
      const count = (this.entities || []).filter((entity) => entity.seen).length;
      text("chat-status", count > 0 ? count + " objects remembered" : "watching for objects…");
    },

    chatSendFromInput() {
      const input = byId("chron-chat-input");
      if (!input) return;
      const value = String(input.value || "").trim();
      if (!value) return;
      input.value = "";
      this.chatBus(value);
    },

    chatSend(value) {
      const textValue = String(value || "").trim();
      if (!textValue) return;
      this.chatBus(textValue);
    },

    chatBus(textValue) {
      /* client → server; server answers via Chronicle.chatQuestion/Answer */
      if (window.emitEvent) emitEvent("chron-query", String(textValue));
    },

    /* server calls right before computing the answer */
    chatQuestion(question) {
      const log = byId("chron-chat-log");
      if (!log) return;
      this.chatId += 1;
      const row = document.createElement("div");
      row.className = "msg-row me popin";
      row.setAttribute("data-chat-id", String(this.chatId));
      row.innerHTML =
        '<div>' +
          '<div class="bub">' + escapeHtml(question) + "</div>" +
          '<div class="msg-meta right">you · ' + escapeHtml(fmtTime(this.playhead)) + "</div>" +
        "</div>";
      const thinking = document.createElement("div");
      thinking.className = "think-row";
      thinking.id = "chron-thinking";
      thinking.innerHTML =
        '<span class="ava wiggle">' +
        '<svg viewBox="0 0 48 48" fill="none" aria-hidden="true" width="22" height="22">' +
        '<path d="M4 24 C 12 10, 36 10, 44 24 C 36 38, 12 38, 4 24 Z" fill="#fffcf4" stroke="#26221a" stroke-width="3"/>' +
        '<circle cx="24" cy="24" r="8.5" fill="#ffd34d" stroke="#26221a" stroke-width="3"/>' +
        '<circle cx="24" cy="24" r="3.4" fill="#26221a"/></svg>' +
        "</span>" +
        '<span class="chipz think-chip">' + THINKING_LINES[this.chatId % THINKING_LINES.length] + "</span>";
      log.appendChild(row);
      log.appendChild(thinking);
      log.scrollTop = log.scrollHeight;
    },

    /* server pushes the computed answer here */
    chatAnswer(data) {
      this.chatBusy = false;
      if (this.thinkingTimer) {
        clearTimeout(this.thinkingTimer);
        this.thinkingTimer = null;
      }
      const log = byId("chron-chat-log");
      if (!log) return;
      const thinking = byId("chron-thinking");
      if (thinking) thinking.remove();

      const route = String(data.route || "none");
      const grounded = route !== "none";
      const answerText = String(data.answer || "no answer was produced");
      const meta = String(data.provider || "local-graphrag") + " · " + String(data.model || "chronicle-local") +
        (data.duration_ms != null ? " · " + Math.round(Number(data.duration_ms)) + "ms" : "") +
        (data.confidence != null ? " · conf " + Number(data.confidence).toFixed(2) : "");

      const row = document.createElement("div");
      row.className = "msg-row popin";
      row.innerHTML =
        '<span class="ava">' +
        '<svg viewBox="0 0 48 48" fill="none" aria-hidden="true" width="22" height="22">' +
        '<path d="M4 24 C 12 10, 36 10, 44 24 C 36 38, 12 38, 4 24 Z" fill="#fffcf4" stroke="#26221a" stroke-width="3"/>' +
        '<circle cx="24" cy="24" r="8.5" fill="#ffd34d" stroke="#26221a" stroke-width="3"/>' +
        '<circle cx="24" cy="24" r="3.4" fill="#26221a"/></svg>' +
        "</span>" +
        '<div class="chat-reply-body" style="max-width:86%">' +
          '<div class="bub' + (grounded ? "" : " bub-fail") + '">' + escapeHtml(answerText) + "</div>" +
        "</div>";

      const host = row.querySelector(".chat-reply-body");
      const badges = document.createElement("div");
      badges.className = "badges-row";
      badges.innerHTML =
        grounded
          ? '<span class="chipz chipz-mint" style="font-size:.6rem;padding:.1rem .5rem">✓ grounded</span>'
          : '<span class="chipz chipz-blush" style="font-size:.6rem;padding:.1rem .5rem">not in memory</span>';
      (data.citations || []).slice(0, 4).forEach((citation) => {
        const t = safeFloat(citation.t, 0);
        const chip = document.createElement("button");
        chip.className = "chipz chipz-mustard cite-chip";
        chip.title = "jump to " + fmtTime(t);
        chip.innerHTML =
          '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="6.2" stroke="#26221a" stroke-width="1.8"/>' +
          '<path d="M8 4.8 V8 L10.2 9.4" stroke="#26221a" stroke-width="1.8" stroke-linecap="round"/></svg>' +
          fmtTime(t) + " · " + escapeHtml(String(citation.label || citation.kind || "evidence"));
        chip.onclick = () => {
          this.seek(t, true);
          if (data.matched_entity) Chronicle.node(String(data.matched_entity));
        };
        badges.appendChild(chip);
      });
      if (data.llm_error) {
        const warn = document.createElement("span");
        warn.className = "chipz chipz-secondary";
        warn.style.fontSize = ".56rem";
        warn.textContent = "LLM fallback";
        badges.appendChild(warn);
      }
      const metaLine = document.createElement("div");
      metaLine.className = "msg-meta";
      metaLine.textContent = "chronicle · " + meta;
      host.appendChild(badges);
      host.appendChild(metaLine);

      log.appendChild(row);
      log.scrollTop = log.scrollHeight;

      if (grounded && data.matched_entity) {
        Chronicle.node(String(data.matched_entity));
      } else if (grounded && (data.citations || []).length && this.duration > 0) {
        const first = data.citations[0];
        if (first && first.t != null) this.preview(safeFloat(first.t, 0));
      }
      const status = byId("chat-status");
      if (status) status.textContent = grounded ? "answered from memory" : String(data.answer).slice(0, 28) + "…";

      setTimeout(() => {
        this.chatBusy = false;
      }, 250);
    },

    scrollTelemetry() {
      const logEl = document.querySelector(".chron-log");
      if (!logEl) return;
      const scroll = logEl.querySelector(".q-scrollarea__container");
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
