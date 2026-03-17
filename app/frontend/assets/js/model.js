    /* ===== LLM JSON Editing ===== */
    async function editJson(opts = {}) {
      const status = document.getElementById("jsonStatus");
      const preview = document.getElementById("jsonPreview");
      const instruction = (opts.instruction && opts.instruction.trim()) || document.getElementById("sceneInput").value.trim();

      if (!instruction) {
        toast("warn","Missing prompt","Please enter a description first.");
        return;
      }

      interactionState.startedAt = performance.now();
      interactionState.lastInstruction = instruction;

      document.getElementById("btnEditJson").disabled = true;
      setPill("model","warn","Calling...");
      status.textContent = "Calling model...";
      preview.textContent = "Waiting for LLM response...";
      setFeedback([
        "Parsing prompt intent…",
        "Building objects / edges…",
        "Validating JSON…"
      ]);

      try {
        const res = await fetch("/edit_json", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ instruction }),
        });
        const data = await res.json();
        if (!res.ok) {
          const msg = data.error || data.detail || "Edit failed";
          setPill("model","err","Failed");
          status.textContent = "Failed";
          preview.textContent = msg;
          toast("err","Model call failed", msg);
          return { ok:false, error: msg };
        }

        const elapsed = Math.round(performance.now() - interactionState.startedAt);
        const fallbackAnalysis = await getSceneGraphAnalysisFallback();
        const analysis = fallbackAnalysis || analyzeSceneJson(data.json);
        interactionState.lastJson = data.json;

        const pretty = JSON.stringify(data.json, null, 2);
        preview.textContent = pretty;

        setMetrics(analysis);
        document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
        status.textContent = `Done (${elapsed} ms)`;
        setPill("model","ok","Ready");

        if (analysis.warnings.length){
          setFeedback(analysis.warnings.slice(0, 6));
          toast("warn","JSON generated with warnings", `${analysis.score}/100 • see diagnostics drawer`);
        }else{
          setFeedback([`JSON looks complete (${analysis.score}/100).`, "Tip: add more spatial constraints for finer control."]);
          toast("ok","JSON generated", `${analysis.objects} objects • ${analysis.edges} edges • ${elapsed} ms`);
        }

        if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
        return { ok:true, data, analysis, pretty, elapsed };
      } catch (err) {
        console.error(err);
        setPill("model","err","Failed");
        status.textContent = "Failed";
        preview.textContent = "Request failed";
        setFeedback([String(err)]);
        toast("err","Network error","Please confirm backend is reachable.");
        return { ok:false, error: String(err) };
      } finally {
        document.getElementById("btnEditJson").disabled = false;
      }
    }

