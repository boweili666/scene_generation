    /* ===== Toast ===== */
    function toast(type, title, msg){
      const root = document.getElementById("toasts");
      const el = document.createElement("div");
      el.className = "toast";
      const ico = document.createElement("div");
      ico.className = `ico ${type}`;
      ico.textContent = type === "ok" ? "✓" : type === "warn" ? "!" : type === "err" ? "×" : "i";
      const txt = document.createElement("div");
      txt.className = "txt";
      const b = document.createElement("b");
      b.textContent = title;
      const s = document.createElement("span");
      s.textContent = msg;
      txt.appendChild(b); txt.appendChild(s);
      el.appendChild(ico); el.appendChild(txt);
      root.appendChild(el);
      setTimeout(() => { el.style.opacity = "0"; el.style.transform = "translateY(-6px)"; }, 2600);
      setTimeout(() => { el.remove(); }, 3000);
    }

    /* ===== Status Pills ===== */
    function setPill(which, state, text){
      const dot = document.getElementById(which === "model" ? "dotModel" : which === "graph" ? "dotGraph" : "dotSim");
      const label = document.getElementById(which === "model" ? "statusModel" : which === "graph" ? "statusGraph" : "statusSim");
      dot.className = "dot " + (state || "");
      label.textContent = text || "Idle";
    }

    function toggleDrawer(){
      const d = document.getElementById("drawer");
      const chev = document.getElementById("chev");
      const open = d.classList.toggle("open");
      chev.textContent = open ? "▴" : "▾";
    }

    function updateInputMeta(){
      const text = document.getElementById("sceneInput").value.trim();
      const chars = text.length;
      const approxTokens = Math.ceil(chars / 3.6) || 0;
      document.getElementById("metaChars").textContent = chars;
      document.getElementById("metaTok").textContent = approxTokens;

      const imageCount = document.getElementById("imageInput").files.length;
      const classCount = document.getElementById("classDirPicker").files.length;
      document.getElementById("uploadStatus").textContent = imageCount ? `${imageCount} image selected` : "Using text";
      document.getElementById("classStatus").textContent = classCount ? `${classCount} files selected` : "No folder selected";
    }

    function clearReferenceImagePreview() {
      if (imagePreviewObjectUrl) {
        URL.revokeObjectURL(imagePreviewObjectUrl);
        imagePreviewObjectUrl = null;
      }
      const wrap = document.getElementById("imagePreview");
      const img = document.getElementById("imagePreviewImg");
      document.getElementById("imagePreviewName").textContent = "No image selected";
      document.getElementById("imagePreviewMeta").textContent = "Choose an image to preview it here.";
      img.removeAttribute("src");
      wrap.classList.remove("is-visible");
    }

    function updateReferenceImagePreview() {
      const fileInput = document.getElementById("imageInput");
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        clearReferenceImagePreview();
        return;
      }

      if (imagePreviewObjectUrl) {
        URL.revokeObjectURL(imagePreviewObjectUrl);
      }
      imagePreviewObjectUrl = URL.createObjectURL(file);

      const wrap = document.getElementById("imagePreview");
      const img = document.getElementById("imagePreviewImg");
      const nameEl = document.getElementById("imagePreviewName");
      const metaEl = document.getElementById("imagePreviewMeta");

      nameEl.textContent = file.name;
      metaEl.textContent = "Loading preview...";
      img.onload = () => {
        const kb = Math.max(1, Math.round(file.size / 1024));
        metaEl.textContent = `${img.naturalWidth}×${img.naturalHeight} • ${kb} KB`;
      };
      img.onerror = () => {
        metaEl.textContent = "Preview failed to load in browser.";
      };
      img.src = imagePreviewObjectUrl;
      wrap.classList.add("is-visible");
    }

    function setFeedback(lines){
      const box = document.getElementById("feedbackBox");
      box.textContent = (lines && lines.length) ? lines.map(s => "• " + s).join("\n") : "-";
    }

    function setMetrics({objects="-", edges="-", score="-"}){
      document.getElementById("metaObjects").textContent = objects;
      document.getElementById("metaEdges").textContent = edges;
      document.getElementById("metaScore").textContent = score;
    }

    function resetSimProgress() {
      document.getElementById("simProgressTitle").textContent = "Generation Progress";
      document.getElementById("simProgressPct").textContent = "0%";
      document.getElementById("simProgressFill").style.width = "0%";
      document.getElementById("simProgressMeta").textContent = "Waiting...";
    }

    function resetReal2SimLog(startOffset = 0, logPath = "real2sim.log") {
      real2simLogState.offset = Number.isFinite(Number(startOffset)) ? Number(startOffset) : 0;
      real2simLogState.path = logPath || "real2sim.log";
      document.getElementById("real2simLogStatus").textContent = "Waiting...";
      document.getElementById("real2simLog").textContent = `[log] watching ${real2simLogState.path}\n`;
    }

    function appendReal2SimLog(text) {
      if (!text) return;
      const el = document.getElementById("real2simLog");
      const combined = el.textContent + text;
      el.textContent = combined.length > 120000 ? combined.slice(-120000) : combined;
      el.scrollTop = el.scrollHeight;
    }

    async function refreshReal2SimLog() {
      const qs = new URLSearchParams({
        offset: String(real2simLogState.offset || 0),
        limit: "65536"
      });
      const res = await fetch(`/real2sim/log?${qs.toString()}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error((data && (data.msg || data.error)) || "Failed to fetch Real2Sim log");
      }
      if (typeof data.next_offset === "number") {
        real2simLogState.offset = data.next_offset;
      }
      if (data.content) {
        appendReal2SimLog(data.content);
      }
      document.getElementById("real2simLogStatus").textContent =
        data.truncated ? "Streaming..." : "Live";
    }

    function setSimProgress(progress = {}) {
      const phase = progress.phase || "running";
      const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
      const expected = progress.expected_objects;
      const generated = Number(progress.generated_objects || 0);
      const merged = !!progress.has_merged_scene;

      const phaseTextMap = {
        queued: "Queued",
        segmenting: "Segmenting",
        generating_glbs: "Generating GLBs",
        merging_scene: "Merging Scene",
        finalizing: "Finalizing",
        completed: "Completed",
        failed: "Failed"
      };
      const title = phaseTextMap[phase] || "Running";
      document.getElementById("simProgressTitle").textContent = title;
      document.getElementById("simProgressPct").textContent = `${percent}%`;
      document.getElementById("simProgressFill").style.width = `${percent}%`;

      const expectedText = (expected === null || expected === undefined) ? "?" : String(expected);
      document.getElementById("simProgressMeta").textContent = `Objects: ${generated}/${expectedText}${merged ? " • merged ready" : ""}`;
    }

    function normalizeSceneGraph(json) {
      let objects = [];
      if (Array.isArray(json?.objects)) {
        objects = json.objects.map((o) => ({
          path: o?.path,
          class_name: o?.class_name || o?.class,
          id: o?.id
        }));
      } else if (json?.obj && typeof json.obj === "object") {
        objects = Object.entries(json.obj).map(([path, meta]) => ({
          path,
          class_name: meta?.class || meta?.class_name || meta?.caption,
          id: meta?.id
        }));
      }

      const idToPath = new Map();
      for (const obj of objects) {
        if (obj?.id !== undefined && obj?.id !== null && obj?.path) {
          idToPath.set(String(obj.id), obj.path);
        }
      }

      const resolveEndpoint = (value) => {
        if (value === undefined || value === null) return value;
        const key = String(value);
        return idToPath.get(key) || value;
      };

      let edges = [];
      if (Array.isArray(json?.edges)) {
        edges = json.edges.map((e) => ({
          ...e,
          source: resolveEndpoint(e?.source),
          target: resolveEndpoint(e?.target),
        }));
      } else if (json?.edges && typeof json.edges === "object") {
        edges = Array.isArray(json.edges["obj-obj"])
          ? json.edges["obj-obj"].map((e) => ({
              ...e,
              source: resolveEndpoint(e?.source),
              target: resolveEndpoint(e?.target),
            }))
          : [];
      }

      return { objects, edges };
    }

    function analyzeSceneJson(json) {
      const normalized = normalizeSceneGraph(json);
      const objects = normalized.objects;
      const edges = normalized.edges;
      const warnings = [];
      if (!objects.length) warnings.push("`objects` is empty — graph nodes may not render.");
      if (!edges.length) warnings.push("`edges` is empty — relations are weak.");

      const unnamed = objects.filter(o => !o?.class_name).length;
      const noPath = objects.filter(o => !o?.path).length;
      const badEdge = edges.filter(e => !e?.source || !e?.target || !e?.relation).length;

      if (unnamed) warnings.push(`${unnamed} object(s) missing class_name.`);
      if (noPath) warnings.push(`${noPath} object(s) missing path.`);
      if (badEdge) warnings.push(`${badEdge} edge(s) missing source/target/relation.`);

      const score = Math.max(0, 100 - warnings.length * 12);
      return { objects: objects.length, edges: edges.length, warnings, score };
    }

    async function getSceneGraphAnalysisFallback() {
      try {
        const res = await fetch("/scene_graph");
        const graph = await res.json();
        if (!res.ok) throw new Error(graph?.error || "Failed to load scene graph");
        return analyzeSceneJson(graph);
      } catch (err) {
        console.warn("Fallback analysis failed:", err);
        return null;
      }
    }

