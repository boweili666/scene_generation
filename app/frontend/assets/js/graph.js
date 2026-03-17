    /* ===== Cytoscape render ===== */
    function renderSceneGraph(graph){
      const normalized = normalizeSceneGraph(graph);
      const elements = [];
      for (const obj of (normalized.objects || [])) {
        const depth = String(obj.path || "").split("/").filter(Boolean).length;
        elements.push({
          data: { id: obj.path, label: obj.class_name, depth }
        });
      }
      for (const e of (normalized.edges || [])) {
        elements.push({
          data: { id: e.source + "->" + e.target, source: e.source, target: e.target, label: e.relation }
        });
      }

      if (cy) cy.destroy();

      cy = cytoscape({
        container: document.getElementById("sceneGraph"),
        elements,
        style: [
          {
            selector: "node",
            style: {
              "background-color": "mapData(depth, 1, 6, #93c5fd, #1d4ed8)",
              "border-color": "#1d4ed8",
              "border-width": "1.5px",
              "label": "data(label)",
              "color": "#ffffff",
              "text-valign": "center",
              "shape": "round-rectangle",
              "padding": "10px",
              "font-size": "11px",
              "text-wrap": "wrap",
              "text-max-width": "130px",
              "text-outline-width": 2,
              "text-outline-color": "rgba(15, 23, 42, 0.18)",
              "shadow-color": "rgba(29, 78, 216, 0.25)",
              "shadow-blur": 14,
              "shadow-offset-y": 5,
              "transition-property": "background-color, border-color, shadow-blur, shadow-color, opacity",
              "transition-duration": "180ms"
            }
          },
          {
            selector: "edge",
            style: {
              "curve-style": "unbundled-bezier",
              "target-arrow-shape": "triangle",
              "label": "data(label)",
              "font-size": "10px",
              "line-color": "rgba(37, 99, 235, 0.55)",
              "target-arrow-color": "rgba(37, 99, 235, 0.55)",
              "width": 2,
              "arrow-scale": 0.9,
              "text-rotation": "autorotate",
              "text-background-color": "#ffffff",
              "text-background-opacity": 0.92,
              "text-background-padding": "4px",
              "color": "#0f172a",
              "text-border-width": 1,
              "text-border-color": "rgba(148, 163, 184, 0.55)",
              "text-margin-y": -3,
              "transition-property": "line-color, target-arrow-color, width, opacity",
              "transition-duration": "180ms"
            }
          },
          { selector: ".dim", style: { "opacity": 0.16 } },
          {
            selector: ".focus-node",
            style: {
              "border-color": "#f59e0b",
              "border-width": 3,
              "shadow-color": "rgba(245, 158, 11, 0.45)",
              "shadow-blur": 22
            }
          },
          {
            selector: ".focus-edge",
            style: {
              "line-color": "#f59e0b",
              "target-arrow-color": "#f59e0b",
              "width": 3,
              "opacity": 1
            }
          }
        ],
        layout: {
          name: "cose",
          animate: true,
          animationDuration: 420,
          fit: true,
          padding: 28,
          nodeRepulsion: 5200,
          edgeElasticity: 130,
          idealEdgeLength: 110,
          gravity: 0.2,
          numIter: 900
        }
      });

      const clearFocus = () => cy.elements().removeClass("dim focus-node focus-edge");
      cy.on("tap", "node", (evt) => {
        const node = evt.target;
        clearFocus();
        cy.elements().addClass("dim");
        node.removeClass("dim").addClass("focus-node");
        node.connectedEdges().removeClass("dim").addClass("focus-edge");
        node.neighborhood().removeClass("dim");
      });
      cy.on("tap", (evt) => { if (evt.target === cy) clearFocus(); });
    }

    /* ===== init load ===== */
    async function loadSceneGraph(){
      try{
        await ensureCytoscape();
        setPill("graph", "", "Loading...");
        const res = await fetch("/scene_graph");
        const graph = await res.json();
        if (!res.ok) throw new Error(graph?.error || "Failed to load /scene_graph");
        renderSceneGraph(graph);
        const analysis = analyzeSceneJson(graph);
        setMetrics(analysis);
        document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
        setFeedback(analysis.warnings.length ? analysis.warnings : ["Looks good. Add more relations for finer control."]);
        setPill("graph", "ok", "Ready");
      }catch(err){
        console.error(err);
        setPill("graph", "err", "Error");
        toast("err","Graph load failed", String(err));
      }
    }

    /* ===== Generate graph from prompt/image ===== */
    async function generateSceneGraph(opts = {}) {
      const text = (opts.text ?? document.getElementById("sceneInput").value).trim();
      const fileInput = document.getElementById("imageInput");
      const classDirPicker = document.getElementById("classDirPicker");

      const formData = new FormData();
      if (text) formData.append("text", text);

      if (fileInput.files.length > 0) formData.append("image", fileInput.files[0]);

      if (classDirPicker.files.length > 0) {
        const names = Array.from(classDirPicker.files)
          .map((f) => {
            const parts = f.name.split(".");
            parts.pop();
            return parts.join(".");
          })
          .filter(Boolean);
        formData.append("class_names", JSON.stringify(names));
      }

      try {
        await ensureCytoscape();
        const res = await fetch("/scene_from_input", { method:"POST", body: formData });
        const graph = await res.json();
        if (!res.ok) {
          const msg = graph.error || "Generation failed";
          if (!opts.silentError) toast("err","Generation failed", msg);
          return { ok:false, error: msg };
        }
        renderSceneGraph(graph);
        const analysis = analyzeSceneJson(graph);
        setMetrics(analysis);
        document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
        setFeedback(analysis.warnings.length ? analysis.warnings : ["Graph looks complete. You can now run Real2Sim or Scene Service."]);
        return { ok:true, graph, analysis };
      } catch (err) {
        console.error(err);
        const msg = "Failed to generate scene graph";
        if (!opts.silentError) toast("err","Network error", msg);
        return { ok:false, error: String(err || msg) };
      }
    }

    async function generateFromPrompt(){
      const text = document.getElementById("sceneInput").value.trim();
      if (!text && document.getElementById("imageInput").files.length === 0){
        toast("warn","Missing input","Please enter a prompt or upload a reference image.");
        return;
      }

      setPill("graph","warn","Generating...");
      document.getElementById("jsonStatus").textContent = "Generating graph...";
      document.getElementById("btnGenerate").disabled = true;

      const result = await generateSceneGraph({ text, silentError: true });

      document.getElementById("btnGenerate").disabled = false;
      if (!result || !result.ok){
        setPill("graph","err","Failed");
        document.getElementById("jsonStatus").textContent = "Failed";
        toast("err","Graph generation failed", result?.error || "Unknown error");
        return;
      }

      setPill("graph","ok","Ready");
      document.getElementById("jsonStatus").textContent = "Graph updated";
      toast("ok","Graph updated", `${result.analysis.objects} objects • ${result.analysis.edges} edges • score ${result.analysis.score}`);
      if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
    }

