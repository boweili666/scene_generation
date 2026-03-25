    /* ===== Unified Scene Instruction ===== */
    async function applyInstruction(opts = {}) {
      const status = document.getElementById("jsonStatus");
      const preview = document.getElementById("jsonPreview");
      const button = document.getElementById("btnApplyInstruction");
      const instruction = (opts.instruction && opts.instruction.trim()) || document.getElementById("sceneInput").value.trim();
      const fileInput = document.getElementById("imageInput");
      const classDirPicker = document.getElementById("classDirPicker");
      const hadReferenceImage = fileInput.files.length > 0;

      if (!instruction && fileInput.files.length === 0) {
        toast("warn","Missing input","Please enter an instruction or upload a reference image.");
        return;
      }

      interactionState.startedAt = performance.now();
      interactionState.lastInstruction = instruction;

      const formData = new FormData();
      if (instruction) formData.append("text", instruction);
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

      button.disabled = true;
      setPill("model","warn","Routing...");
      setPill("graph","warn","Applying...");
      status.textContent = "Applying instruction...";
      preview.textContent = "Waiting for unified instruction result...";
      setFeedback([
        "Routing instruction…",
        "Applying graph / placement edit…",
        "Validating scene state…"
      ]);

      try {
        const res = await fetch("/apply_instruction", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        if (!res.ok) {
          const msg = data.error || data.detail || "Instruction failed";
          setPill("model","err","Failed");
          setPill("graph","err","Failed");
          status.textContent = "Failed";
          preview.textContent = msg;
          toast("err","Instruction failed", msg);
          return { ok:false, error: msg };
        }

        await ensureCytoscape();
        const graph = data.scene_graph;
        renderSceneGraph(graph);
        const analysis = analyzeSceneJson(graph);
        const elapsed = Math.round(performance.now() - interactionState.startedAt);
        const previewPayload = {
          route: data.route,
          placements: data.placements || {},
          scene_graph: graph,
        };

        interactionState.lastJson = previewPayload;
        preview.textContent = JSON.stringify(previewPayload, null, 2);
        setMetrics(analysis);
        document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
        status.textContent = `Done (${elapsed} ms)`;
        setPill("model","ok","Ready");
        setPill("graph","ok","Ready");

        const feedbackLines = [];
        if (data.route?.mode) {
          feedbackLines.push(`Route: ${data.route.mode} (${Math.round((data.route.confidence || 0) * 100)}%)`);
        }
        if (data.route?.reason) {
          feedbackLines.push(data.route.reason);
        }
        if (Array.isArray(data.warnings) && data.warnings.length) {
          feedbackLines.push(...data.warnings);
        } else if (analysis.warnings.length) {
          feedbackLines.push(...analysis.warnings.slice(0, 4));
        } else {
          feedbackLines.push("Instruction applied. Use Edit Scene or Resample when geometry needs refresh.");
        }
        setFeedback(feedbackLines);

        if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
        if (hadReferenceImage) {
          clearReferenceImageInput();
        }
        toast(
          "ok",
          "Instruction applied",
          `${data.route?.mode || "graph"} • ${analysis.objects} objects • ${analysis.edges} edges • ${elapsed} ms`
        );
        return { ok:true, data, analysis, elapsed };
      } catch (err) {
        console.error(err);
        setPill("model","err","Failed");
        setPill("graph","err","Failed");
        status.textContent = "Failed";
        preview.textContent = "Request failed";
        setFeedback([String(err)]);
        toast("err","Network error","Please confirm backend is reachable.");
        return { ok:false, error: String(err) };
      } finally {
        button.disabled = false;
      }
    }

    async function editJson(opts = {}) {
      return applyInstruction(opts);
    }
