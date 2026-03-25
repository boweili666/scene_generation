    const GRAPH_NODE_SOURCES = ["real2sim", "retrieval"];
    const GRAPH_OBJ_OBJ_RELATIONS = [
      "left",
      "right",
      "in front of",
      "behind",
      "face to",
      "face same as",
      "supported by",
      "supports",
      "center aligned",
      "adjacent",
    ];
    const GRAPH_OBJ_WALL_RELATIONS = ["against wall", "in corner"];
    const GRAPH_RELATION_REVERSE = {
      "left": "right",
      "right": "left",
      "in front of": "behind",
      "behind": "in front of",
      "supported by": "supports",
      "supports": "supported by",
    };
    function cloneSceneGraph(graph) {
      return JSON.parse(JSON.stringify(graph));
    }

    function defaultSceneMetadata() {
      return {
        room_type: "room",
        dimensions: { length: 5, width: 5, height: 3, unit: "m" },
        materials: { floor: "wood", walls: "paint" },
      };
    }

    function sanitizeClassName(value) {
      return String(value || "")
        .trim()
        .toLowerCase()
        .replace(/\s+/g, "_")
        .replace(/[^a-z0-9_]/g, "_")
        .replace(/_+/g, "_")
        .replace(/^_+|_+$/g, "");
    }

    function objectLabel(meta, path) {
      const raw = meta?.class || meta?.class_name || meta?.caption || String(path || "").split("/").pop() || "object";
      return String(raw).replace(/_/g, " ");
    }

    function edgeElementId(kind, source, target, relation) {
      return [kind, source || "", target || "", relation || ""].join("::");
    }

    function pairedRelationFamily(relation) {
      if (relation === "left" || relation === "right") return new Set(["left", "right"]);
      if (relation === "in front of" || relation === "behind") return new Set(["in front of", "behind"]);
      if (relation === "supported by" || relation === "supports") return new Set(["supported by", "supports"]);
      return null;
    }

    function extractObjectId(path, meta) {
      if (Number.isFinite(Number(meta?.id))) return Number(meta.id);
      const match = String(path || "").match(/_(\d+)$/);
      return match ? Number(match[1]) : 0;
    }

    function buildObjectPath(className, numericId) {
      return `/World/${className}_${numericId}`;
    }

    function nextObjectId(graph) {
      const ids = Object.entries(graph?.obj || {}).map(([path, meta]) => extractObjectId(path, meta));
      return ids.length ? Math.max(...ids) + 1 : 0;
    }

    function findWallRelation(graph, source) {
      const edges = graph?.edges?.["obj-wall"] || [];
      const match = edges.find((edge) => edge?.source === source);
      return match ? String(match.relation || "") : "";
    }

    function normalizeObjObjEdges(edges = []) {
      const deduped = [];
      const seen = new Set();
      for (const edge of edges) {
        if (!edge?.source || !edge?.target || !edge?.relation) continue;
        const source = String(edge.source);
        const target = String(edge.target);
        const relation = String(edge.relation);
        const key = edgeElementId("obj-obj", source, target, relation);
        if (seen.has(key)) continue;
        seen.add(key);
        deduped.push({ source, target, relation });
      }
      return deduped;
    }

    function normalizeObjWallEdges(edges = []) {
      const deduped = [];
      const seen = new Set();
      for (const edge of edges) {
        if (!edge?.source || !edge?.relation) continue;
        const source = String(edge.source);
        const relation = String(edge.relation);
        const target = edge?.target ? String(edge.target) : "";
        const key = edgeElementId("obj-wall", source, target, relation);
        if (seen.has(key)) continue;
        seen.add(key);
        const payload = { source, relation };
        if (target) payload.target = target;
        deduped.push(payload);
      }
      return deduped;
    }

    function toEditableSceneGraph(json) {
      const scene = (json?.scene && typeof json.scene === "object")
        ? cloneSceneGraph(json.scene)
        : defaultSceneMetadata();

      const obj = {};
      if (Array.isArray(json?.objects)) {
        for (const item of json.objects) {
          if (!item?.path) continue;
          obj[item.path] = {
            id: item.id,
            class: item.class || item.class_name || String(item.path).split("/").pop() || "object",
            caption: item.caption || item.class || item.class_name || "object",
            source: item.source || "retrieval",
          };
        }
      } else if (json?.obj && typeof json.obj === "object") {
        for (const [path, meta] of Object.entries(json.obj)) {
          obj[path] = {
            id: meta?.id,
            class: meta?.class || meta?.class_name || meta?.caption || String(path).split("/").pop() || "object",
            caption: meta?.caption || meta?.class || meta?.class_name || "object",
            source: meta?.source || "retrieval",
          };
        }
      }

      let objObjEdges = [];
      let objWallEdges = [];
      if (Array.isArray(json?.edges)) {
        objObjEdges = normalizeObjObjEdges(json.edges);
      } else if (json?.edges && typeof json.edges === "object") {
        objObjEdges = normalizeObjObjEdges(json.edges["obj-obj"] || []);
        objWallEdges = normalizeObjWallEdges(json.edges["obj-wall"] || []);
      }

      return {
        scene,
        obj,
        edges: {
          "obj-obj": objObjEdges,
          "obj-wall": objWallEdges,
        },
      };
    }

    function serializeSceneGraphForSave(graph) {
      const scene = cloneSceneGraph(graph?.scene || defaultSceneMetadata());
      const obj = {};
      for (const path of Object.keys(graph?.obj || {}).sort()) {
        const meta = graph.obj[path] || {};
        obj[path] = {
          id: extractObjectId(path, meta),
          class: sanitizeClassName(meta.class || meta.class_name || "object") || "object",
          caption: String(meta.caption || meta.class || "object").trim() || "object",
          source: GRAPH_NODE_SOURCES.includes(String(meta.source || "").toLowerCase())
            ? String(meta.source).toLowerCase()
            : "retrieval",
        };
      }
      const objObjEdges = normalizeObjObjEdges(graph?.edges?.["obj-obj"] || []).sort((a, b) =>
        `${a.source}|${a.target}|${a.relation}`.localeCompare(`${b.source}|${b.target}|${b.relation}`)
      );
      const objWallEdges = normalizeObjWallEdges(graph?.edges?.["obj-wall"] || []).sort((a, b) =>
        `${a.source}|${a.relation}`.localeCompare(`${b.source}|${b.relation}`)
      );
      return {
        scene,
        obj,
        edges: {
          "obj-obj": objObjEdges,
          "obj-wall": objWallEdges,
        },
      };
    }

    function getCurrentSceneGraph() {
      return interactionState.currentSceneGraph
        ? cloneSceneGraph(interactionState.currentSceneGraph)
        : toEditableSceneGraph({ scene: defaultSceneMetadata(), obj: {}, edges: { "obj-obj": [], "obj-wall": [] } });
    }

    function setCurrentSceneGraph(graph) {
      interactionState.currentSceneGraph = toEditableSceneGraph(graph);
      return interactionState.currentSceneGraph;
    }

    function captureGraphPositions() {
      const positions = {};
      if (!cy) return positions;
      cy.nodes().forEach((node) => {
        positions[node.id()] = { ...node.position() };
      });
      return positions;
    }

    function buildGraphElements(graph, positions = {}) {
      const elements = [];
      const objectEntries = Object.entries(graph?.obj || {});

      objectEntries.forEach(([path, meta], index) => {
        const defaultPosition = {
          x: ((index % 4) - 1.5) * 170,
          y: Math.floor(index / 4) * 130 - 80,
        };
        elements.push({
          data: {
            id: path,
            kind: "object",
            label: objectLabel(meta, path),
            className: meta?.class || "",
            caption: meta?.caption || "",
            source: meta?.source || "retrieval",
            nodeId: extractObjectId(path, meta),
            wallRelation: findWallRelation(graph, path),
          },
          position: positions[path] || defaultPosition,
        });
      });

      for (const edge of (graph?.edges?.["obj-obj"] || [])) {
        elements.push({
          data: {
            id: edgeElementId("obj-obj", edge.source, edge.target, edge.relation),
            kind: "obj-obj",
            source: edge.source,
            target: edge.target,
            label: edge.relation,
            relation: edge.relation,
          }
        });
      }

      return elements;
    }

    function refreshGraphEditorClasses() {
      if (!cy) return;
      cy.elements().removeClass("selected-edit link-source");
      if (graphEditorState.selectedElementId) {
        const selected = cy.getElementById(graphEditorState.selectedElementId);
        if (selected && selected.length) selected.addClass("selected-edit");
      }
      if (graphEditorState.linkSourceId) {
        const source = cy.getElementById(graphEditorState.linkSourceId);
        if (source && source.length) source.addClass("link-source");
      }
    }

    function resetGraphEditorSelection() {
      graphEditorState.selectedElementType = "";
      graphEditorState.selectedElementId = "";
      graphEditorState.linkSourceId = "";
      refreshGraphEditorClasses();
    }

    function setGraphSelection(type, id) {
      graphEditorState.selectedElementType = type || "";
      graphEditorState.selectedElementId = id || "";
      refreshGraphEditorClasses();
    }

    function registerGraphTap(targetKey) {
      const now = Date.now();
      const isDouble = graphEditorState.lastTapKey === targetKey && (now - graphEditorState.lastTapAt) < 320;
      graphEditorState.lastTapKey = isDouble ? "" : targetKey;
      graphEditorState.lastTapAt = isDouble ? 0 : now;
      return isDouble;
    }

    function updateSceneGraphUi(graph, options = {}) {
      const analysis = analyzeSceneJson(graph);
      setMetrics(analysis);
      document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
      if (options.statusText) {
        document.getElementById("jsonStatus").textContent = options.statusText;
      }
      if (options.previewPayload) {
        document.getElementById("jsonPreview").textContent = JSON.stringify(options.previewPayload, null, 2);
      }
      if (options.feedbackLines) {
        setFeedback(options.feedbackLines);
      } else if (analysis.warnings.length) {
        setFeedback(analysis.warnings);
      }
      return analysis;
    }

    function promptForNodeMeta(existing = {}) {
      const classInput = window.prompt(
        "Object class (lowercase, letters/numbers/underscore):",
        existing.className || existing.class || "object"
      );
      if (classInput === null) return null;
      const className = sanitizeClassName(classInput);
      if (!className) {
        toast("err", "Invalid class", "Class name cannot be empty.");
        return null;
      }

      const captionInput = window.prompt(
        "Caption:",
        existing.caption || className.replace(/_/g, " ")
      );
      if (captionInput === null) return null;
      const caption = String(captionInput).trim() || className.replace(/_/g, " ");

      const sourceInput = window.prompt(
        "Source: real2sim or retrieval",
        existing.source || "retrieval"
      );
      if (sourceInput === null) return null;
      const source = String(sourceInput).trim().toLowerCase();
      if (!GRAPH_NODE_SOURCES.includes(source)) {
        toast("err", "Invalid source", "Source must be real2sim or retrieval.");
        return null;
      }

      const wallInput = window.prompt(
        "Wall relation: none, against wall, or in corner",
        existing.wallRelation || "none"
      );
      if (wallInput === null) return null;
      const wallRelation = String(wallInput).trim().toLowerCase();
      if (wallRelation && wallRelation !== "none" && !GRAPH_OBJ_WALL_RELATIONS.includes(wallRelation)) {
        toast("err", "Invalid wall relation", "Use none, against wall, or in corner.");
        return null;
      }

      return {
        className,
        caption,
        source,
        wallRelation: wallRelation === "none" ? "" : wallRelation,
      };
    }

    function promptForRelation(kind, currentRelation = "") {
      const allowed = kind === "obj-wall" ? GRAPH_OBJ_WALL_RELATIONS : GRAPH_OBJ_OBJ_RELATIONS;
      const response = window.prompt(
        `Relation (${allowed.join(", ")}):`,
        currentRelation || allowed[0]
      );
      if (response === null) return null;
      const relation = String(response).trim().toLowerCase();
      if (!allowed.includes(relation)) {
        toast("err", "Invalid relation", `Allowed: ${allowed.join(", ")}`);
        return null;
      }
      return relation;
    }

    function updateEdgeEndpoints(graph, previousPath, nextPath) {
      graph.edges["obj-obj"] = (graph.edges["obj-obj"] || []).map((edge) => ({
        ...edge,
        source: edge.source === previousPath ? nextPath : edge.source,
        target: edge.target === previousPath ? nextPath : edge.target,
      }));
      graph.edges["obj-wall"] = (graph.edges["obj-wall"] || []).map((edge) => (
        edge.source === previousPath ? { ...edge, source: nextPath } : edge
      ));
    }

    function setWallRelation(graph, source, relation) {
      graph.edges["obj-wall"] = (graph.edges["obj-wall"] || []).filter((edge) => edge.source !== source);
      if (relation) {
        graph.edges["obj-wall"].push({ source, relation });
      }
    }

    function removeObjObjRelation(graph, source, target, relation) {
      const family = pairedRelationFamily(relation);
      graph.edges["obj-obj"] = (graph.edges["obj-obj"] || []).filter((edge) => {
        if (family) {
          const matchesPair = (
            (edge.source === source && edge.target === target) ||
            (edge.source === target && edge.target === source)
          );
          return !(matchesPair && family.has(edge.relation));
        }
        return !(edge.source === source && edge.target === target && edge.relation === relation);
      });
    }

    function setObjObjRelation(graph, source, target, relation) {
      const family = pairedRelationFamily(relation);
      if (family) {
        graph.edges["obj-obj"] = (graph.edges["obj-obj"] || []).filter((edge) => {
          const matchesPair = (
            (edge.source === source && edge.target === target) ||
            (edge.source === target && edge.target === source)
          );
          return !(matchesPair && family.has(edge.relation));
        });
        graph.edges["obj-obj"].push({ source, target, relation });
        graph.edges["obj-obj"].push({
          source: target,
          target: source,
          relation: GRAPH_RELATION_REVERSE[relation],
        });
        return;
      }

      const exists = (graph.edges["obj-obj"] || []).some((edge) =>
        edge.source === source && edge.target === target && edge.relation === relation
      );
      if (!exists) {
        graph.edges["obj-obj"].push({ source, target, relation });
      }
    }

    async function saveSceneGraphDraft(graph, options = {}) {
      graphEditorState.saveInFlight = true;
      setPill("graph", "warn", "Saving...");
      document.getElementById("jsonStatus").textContent = "Saving graph...";
      try {
        const payload = serializeSceneGraphForSave(graph);
        const res = await fetch("/scene_graph", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scene_graph: payload }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.error || "Failed to save scene graph");
        }

        const feedbackLines = [];
        if (options.feedbackLine) feedbackLines.push(options.feedbackLine);
        if (Array.isArray(data.warnings) && data.warnings.length) {
          feedbackLines.push(...data.warnings);
        } else {
          feedbackLines.push("Graph saved to runtime/scene_graph/current_scene_graph.json");
        }

        renderSceneGraph(data.scene_graph, { preservePositions: true });
        const analysis = updateSceneGraphUi(data.scene_graph, {
          statusText: "Graph saved",
          previewPayload: {
            scene_graph: data.scene_graph,
            placements: data.placements || {},
          },
          feedbackLines,
        });
        setPill("graph", "ok", "Ready");
        toast(
          "ok",
          options.toastTitle || "Graph saved",
          `${analysis.objects} objects • ${analysis.edges} edges`
        );
        return data;
      } catch (err) {
        console.error(err);
        setPill("graph", "err", "Failed");
        document.getElementById("jsonStatus").textContent = "Save failed";
        toast("err", "Graph save failed", String(err));
        throw err;
      } finally {
        graphEditorState.saveInFlight = false;
      }
    }

    async function applyGraphMutation(mutator, options = {}) {
      if (graphEditorState.saveInFlight) {
        toast("warn", "Save in progress", "Wait for the current graph save to finish.");
        return null;
      }
      const draft = getCurrentSceneGraph();
      const result = mutator(draft);
      if (result === false || result?.cancelled) return null;
      return saveSceneGraphDraft(draft, options);
    }

    function handleGraphDeleteRequest() {
      if (!graphEditorState.selectedElementId) return;
      if (graphEditorState.selectedElementType === "node") {
        const nodeId = graphEditorState.selectedElementId;
        const current = getCurrentSceneGraph();
        const meta = current.obj[nodeId];
        if (!meta) return;
        if (!window.confirm(`Delete node ${objectLabel(meta, nodeId)} and its relations?`)) return;
        applyGraphMutation((draft) => {
          delete draft.obj[nodeId];
          draft.edges["obj-obj"] = (draft.edges["obj-obj"] || []).filter((edge) => (
            edge.source !== nodeId && edge.target !== nodeId
          ));
          draft.edges["obj-wall"] = (draft.edges["obj-wall"] || []).filter((edge) => edge.source !== nodeId);
          return true;
        }, {
          toastTitle: "Node deleted",
          feedbackLine: `Deleted ${nodeId}`,
        }).catch(() => {});
        return;
      }

      if (graphEditorState.selectedElementType === "edge") {
        const edge = cy ? cy.getElementById(graphEditorState.selectedElementId) : null;
        if (!edge || !edge.length) return;
        const edgeKind = edge.data("kind");
        const source = edge.data("source");
        const target = edge.data("target");
        const relation = edge.data("relation");
        if (!window.confirm(`Delete relation "${relation}"?`)) return;
        applyGraphMutation((draft) => {
          if (edgeKind === "obj-wall") {
            draft.edges["obj-wall"] = (draft.edges["obj-wall"] || []).filter((item) => !(
              item.source === source && item.relation === relation
            ));
          } else {
            removeObjObjRelation(draft, source, target, relation);
          }
          return true;
        }, {
          toastTitle: "Relation deleted",
          feedbackLine: `Deleted relation ${relation}`,
        }).catch(() => {});
      }
    }

    function bindGraphKeyboardShortcuts() {
      if (window.__sceneGraphEditorKeybound) return;
      window.__sceneGraphEditorKeybound = true;
      document.addEventListener("keydown", (evt) => {
        const tag = String(evt.target?.tagName || "").toLowerCase();
        if (tag === "input" || tag === "textarea" || tag === "select") return;
        if (evt.key === "Delete" || evt.key === "Backspace") {
          if (!graphEditorState.selectedElementId) return;
          evt.preventDefault();
          handleGraphDeleteRequest();
        }
        if (evt.key === "Escape" && graphEditorState.linkSourceId) {
          graphEditorState.linkSourceId = "";
          refreshGraphEditorClasses();
          toast("info", "Link cancelled", "Node-to-node edge creation was cancelled.");
        }
      });
    }

    async function handleNodeEdit(nodeId) {
      const current = getCurrentSceneGraph();
      const meta = current.obj[nodeId];
      if (!meta) return;
      const edited = promptForNodeMeta({
        className: meta.class,
        caption: meta.caption,
        source: meta.source,
        wallRelation: findWallRelation(current, nodeId),
      });
      if (!edited) return;

      await applyGraphMutation((draft) => {
        const existing = draft.obj[nodeId];
        if (!existing) return false;
        const numericId = extractObjectId(nodeId, existing);
        const nextPath = buildObjectPath(edited.className, numericId);
        if (nextPath !== nodeId && draft.obj[nextPath]) {
          toast("err", "Path collision", `${nextPath} already exists.`);
          return false;
        }
        const nextMeta = {
          ...existing,
          id: numericId,
          class: edited.className,
          caption: edited.caption,
          source: edited.source,
        };
        if (nextPath !== nodeId) {
          delete draft.obj[nodeId];
          draft.obj[nextPath] = nextMeta;
          updateEdgeEndpoints(draft, nodeId, nextPath);
          setWallRelation(draft, nextPath, edited.wallRelation);
        } else {
          draft.obj[nodeId] = nextMeta;
          setWallRelation(draft, nodeId, edited.wallRelation);
        }
        return true;
      }, {
        toastTitle: "Node updated",
        feedbackLine: `Updated node ${nodeId}`,
      });
    }

    async function handleAddNode() {
      const meta = promptForNodeMeta({
        className: "object",
        caption: "object",
        source: "retrieval",
        wallRelation: "",
      });
      if (!meta) return;

      await applyGraphMutation((draft) => {
        const numericId = nextObjectId(draft);
        const path = buildObjectPath(meta.className, numericId);
        draft.obj[path] = {
          id: numericId,
          class: meta.className,
          caption: meta.caption,
          source: meta.source,
        };
        setWallRelation(draft, path, meta.wallRelation);
        return true;
      }, {
        toastTitle: "Node added",
        feedbackLine: `Added ${meta.className}`,
      });
    }

    async function handleEdgeEdit(edgeId) {
      if (!cy) return;
      const edge = cy.getElementById(edgeId);
      if (!edge || !edge.length) return;
      const edgeKind = edge.data("kind");
      const relation = edge.data("relation");
      const nextRelation = promptForRelation(edgeKind, relation);
      if (!nextRelation || nextRelation === relation) return;

      await applyGraphMutation((draft) => {
        if (edgeKind === "obj-wall") {
          setWallRelation(draft, edge.data("source"), nextRelation);
        } else {
          removeObjObjRelation(draft, edge.data("source"), edge.data("target"), relation);
          setObjObjRelation(draft, edge.data("source"), edge.data("target"), nextRelation);
        }
        return true;
      }, {
        toastTitle: "Relation updated",
        feedbackLine: `Updated relation to ${nextRelation}`,
      });
    }

    async function handleNodeLinkTap(nodeId) {
      if (!graphEditorState.linkSourceId) {
        graphEditorState.linkSourceId = nodeId;
        refreshGraphEditorClasses();
        toast("info", "Link start selected", "Shift-click another node to create or update a relation.");
        return;
      }

      if (graphEditorState.linkSourceId === nodeId) {
        graphEditorState.linkSourceId = "";
        refreshGraphEditorClasses();
        toast("info", "Link cancelled", "Selected the same node twice.");
        return;
      }

      const sourceId = graphEditorState.linkSourceId;
      graphEditorState.linkSourceId = "";
      refreshGraphEditorClasses();

      const relation = promptForRelation("obj-obj");
      if (!relation) return;

      await applyGraphMutation((draft) => {
        setObjObjRelation(draft, sourceId, nodeId, relation);
        return true;
      }, {
        toastTitle: "Relation added",
        feedbackLine: `Linked ${sourceId} -> ${nodeId} as ${relation}`,
      });
    }

    function bindSceneGraphInteractions() {
      if (!cy) return;
      bindGraphKeyboardShortcuts();

      const clearFocus = () => cy.elements().removeClass("dim focus-node focus-edge");

      cy.on("tap", "node", (evt) => {
        const node = evt.target;
        if (node.data("kind") !== "object") return;
        if (evt.originalEvent && evt.originalEvent.shiftKey) {
          handleNodeLinkTap(node.id());
          return;
        }
        const isDoubleTap = registerGraphTap(`node:${node.id()}`);
        clearFocus();
        cy.elements().addClass("dim");
        node.removeClass("dim").addClass("focus-node");
        node.connectedEdges().removeClass("dim").addClass("focus-edge");
        node.neighborhood().removeClass("dim");
        setGraphSelection("node", node.id());
        if (isDoubleTap) {
          handleNodeEdit(node.id());
        }
      });

      cy.on("tap", "edge", (evt) => {
        const edge = evt.target;
        const isDoubleTap = registerGraphTap(`edge:${edge.id()}`);
        clearFocus();
        cy.elements().addClass("dim");
        edge.removeClass("dim").addClass("focus-edge");
        edge.connectedNodes().removeClass("dim").addClass("focus-node");
        setGraphSelection("edge", edge.id());
        if (isDoubleTap) {
          handleEdgeEdit(edge.id());
        }
      });

      cy.on("tap", (evt) => {
        if (evt.target === cy) {
          const isDoubleTap = registerGraphTap("canvas");
          clearFocus();
          setGraphSelection("", "");
          if (isDoubleTap) {
            handleAddNode();
          }
        }
      });

      cy.on("cxttap", "node", (evt) => {
        const node = evt.target;
        if (node.data("kind") !== "object") return;
        setGraphSelection("node", node.id());
        handleGraphDeleteRequest();
      });

      cy.on("cxttap", "edge", (evt) => {
        setGraphSelection("edge", evt.target.id());
        handleGraphDeleteRequest();
      });

      refreshGraphEditorClasses();
    }

    /* ===== Cytoscape render ===== */
    function renderSceneGraph(graph, opts = {}){
      const editableGraph = setCurrentSceneGraph(graph);
      const positions = opts.preservePositions ? captureGraphPositions() : {};
      const elements = buildGraphElements(editableGraph, positions);

      if (cy) cy.destroy();

      cy = cytoscape({
        container: document.getElementById("sceneGraph"),
        elements,
        style: [
          {
            selector: 'node[kind = "object"]',
            style: {
              "background-color": "#5b6f95",
              "border-color": "#32435f",
              "border-width": "1.8px",
              "label": "data(label)",
              "color": "#ffffff",
              "text-valign": "center",
              "shape": "round-rectangle",
              "padding": "11px",
              "font-size": "11px",
              "font-weight": 700,
              "text-wrap": "wrap",
              "text-max-width": "130px",
              "text-outline-width": 2.5,
              "text-outline-color": "rgba(15, 23, 42, 0.26)",
              "shadow-color": "rgba(34, 44, 64, 0.18)",
              "shadow-blur": 18,
              "shadow-offset-y": 6,
              "transition-property": "background-color, border-color, shadow-blur, shadow-color, opacity",
              "transition-duration": "180ms"
            }
          },
          {
            selector: 'node[source = "real2sim"]',
            style: {
              "background-color": "#1f8a70",
              "border-color": "#145a4d",
              "shadow-color": "rgba(21, 92, 78, 0.22)"
            }
          },
          {
            selector: 'node[source = "retrieval"]',
            style: {
              "background-color": "#5967b3",
              "border-color": "#384584",
              "shadow-color": "rgba(56, 69, 132, 0.22)"
            }
          },
          {
            selector: 'edge[kind = "obj-obj"]',
            style: {
              "curve-style": "unbundled-bezier",
              "target-arrow-shape": "triangle",
              "label": "data(label)",
              "font-size": "10px",
              "line-color": "rgba(78, 108, 173, 0.72)",
              "target-arrow-color": "rgba(78, 108, 173, 0.72)",
              "width": 2.1,
              "arrow-scale": 0.95,
              "text-rotation": "autorotate",
              "text-background-color": "rgba(255,255,255,0.94)",
              "text-background-opacity": 1,
              "text-background-padding": "4px",
              "color": "#334155",
              "text-border-width": 1,
              "text-border-color": "rgba(148, 163, 184, 0.42)",
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
          },
          {
            selector: 'node.selected-edit',
            style: {
              "border-color": "#2563eb",
              "border-width": 4,
              "shadow-color": "rgba(37, 99, 235, 0.28)",
              "shadow-blur": 24
            }
          },
          {
            selector: 'edge.selected-edit',
            style: {
              "line-color": "#2563eb",
              "target-arrow-color": "#2563eb",
              "width": 3.2,
              "opacity": 1
            }
          },
          {
            selector: ".link-source",
            style: {
              "border-color": "#10b981",
              "border-width": 4,
              "shadow-color": "rgba(16, 185, 129, 0.35)",
              "shadow-blur": 24,
            }
          }
        ],
        layout: opts.preservePositions ? {
          name: "preset",
          fit: true,
          padding: 28,
          animate: true,
          animationDuration: 220,
        } : {
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

      resetGraphEditorSelection();
      bindSceneGraphInteractions();
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
        updateSceneGraphUi(graph, {
          statusText: "Graph loaded",
          previewPayload: { scene_graph: toEditableSceneGraph(graph) },
          feedbackLines: [
            "Double-click background to add node.",
            "Double-click node or edge to edit it.",
            "Shift-click two nodes to create a relation. Delete/Backspace removes the selected item."
          ],
        });
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
        const analysis = updateSceneGraphUi(graph, {
          previewPayload: { scene_graph: toEditableSceneGraph(graph) },
          feedbackLines: ["Graph looks complete. You can now run Real2Sim or Scene Service."],
          statusText: "Graph updated",
        });
        return { ok:true, graph, analysis };
      } catch (err) {
        console.error(err);
        const msg = "Failed to generate scene graph";
        if (!opts.silentError) toast("err","Network error", msg);
        return { ok:false, error: String(err || msg) };
      }
    }

    async function generateFromPrompt(){
      return applyInstruction();
    }
