    function collectClassNamesPayload(classDirPicker) {
      if (!classDirPicker || classDirPicker.files.length === 0) return "";
      const names = Array.from(classDirPicker.files)
        .map((f) => {
          const parts = f.name.split(".");
          parts.pop();
          return parts.join(".");
        })
        .filter(Boolean);
      return JSON.stringify(names);
    }

    function refreshRuntimeRenderImage() {
      const img = document.getElementById("renderImage");
      if (!img) return;
      img.src = withRuntimeQuery("/render_image", { ts: Date.now() });
    }

    async function handleAgentResponse(data, options = {}) {
      if (data?.session_id || data?.run_id) {
        applyRuntimeContext({
          session_id: data.session_id,
          run_id: data.run_id,
        });
      }

      const agent = data?.agent || {};
      updateAgentPanel(agent);
      if (Array.isArray(data?.session_state?.history)) {
        renderAgentTranscript(data.session_state.history);
      }
      const feedbackLines = [];
      if (agent.message) feedbackLines.push(`Agent: ${agent.message}`);
      if (agent.question) feedbackLines.push(`Question: ${agent.question}`);
      if (agent.reason) feedbackLines.push(`Reason: ${agent.reason}`);
      if (Array.isArray(data?.warnings) && data.warnings.length) {
        feedbackLines.push(...data.warnings);
      }
      updateArtifactPanel(data);
      setAgentErrorInfo(data?.error_info || null);

      if (data?.scene_graph) {
        await ensureCytoscape();
        renderSceneGraph(data.scene_graph);
        const analysis = analyzeSceneJson(data.scene_graph);
        interactionState.lastJson = {
          agent,
          scene_graph: data.scene_graph,
          placements: data.placements || {},
        };
        document.getElementById("jsonPreview").textContent = JSON.stringify(interactionState.lastJson, null, 2);
        setMetrics(analysis);
        document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
        setPill("graph", "ok", "Ready");
        if (!feedbackLines.length && analysis.warnings.length) {
          feedbackLines.push(...analysis.warnings);
        }
      }

      if (data?.scene_result) {
        document.getElementById("sceneSvcResult").textContent = JSON.stringify(data.scene_result, null, 2);
        setSceneDebug(data.scene_result.debug || {});
        document.getElementById("sceneSvcStatus").textContent = "Done";
        document.getElementById("statusSimText").textContent = "Scene generated";
        setPill("sim", "ok", "Ready");
        showImagePreview();
        refreshRuntimeRenderImage();
      }

      if (data?.real2sim_job?.job_id) {
        await monitorReal2SimJob(data.real2sim_job);
      }

      if (!feedbackLines.length) {
        feedbackLines.push("Agent request completed.");
      }
      setFeedback(feedbackLines);
      if (!Array.isArray(data?.session_state?.history)) {
        const transcriptMeta = [];
        if (agent.reason) transcriptMeta.push(`Reason: ${agent.reason}`);
        if (agent.question) transcriptMeta.push(`Question: ${agent.question}`);
        appendAgentTranscript("assistant", agent.message || "Agent request completed.", transcriptMeta.join("\n"));
      }

      if (options.hadReferenceImage) {
        clearReferenceImageInput();
      }

      if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
      return data;
    }

    async function restoreAgentState() {
      await ensureRuntimeContext();
      const response = await fetch(withRuntimeQuery("/agent/state"));
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || "Failed to restore agent state");
      }
      return handleAgentResponse(data, { hadReferenceImage: false });
    }

    async function sendAgentMessage(options = {}) {
      await ensureRuntimeContext();

      const status = document.getElementById("jsonStatus");
      const preview = document.getElementById("jsonPreview");
      const instruction = (options.instruction && options.instruction.trim()) || "";
      const action = options.action || "";
      const resampleMode = options.resampleMode || "";
      const sceneEndpoint = options.sceneEndpoint || "";
      const imageFile = options.imageFile || null;
      const classNamesRaw = options.classNamesRaw || "";
      const hadReferenceImage = !!imageFile;

      const formData = new FormData();
      if (instruction) formData.append("text", instruction);
      if (action) formData.append("action", action);
      if (resampleMode) formData.append("resample_mode", resampleMode);
      if (sceneEndpoint) formData.append("scene_endpoint", sceneEndpoint);
      if (classNamesRaw) formData.append("class_names", classNamesRaw);
      if (imageFile) formData.append("image", imageFile);
      appendRuntimeToFormData(formData);

      const userSummary = instruction || (action ? `[${action}]` : imageFile ? "[image]" : "");
      if (userSummary) {
        appendAgentTranscript("user", userSummary);
      }

      status.textContent = "Agent running...";
      preview.textContent = "Waiting for agent response...";
      setPill("model", "warn", "Agent...");
      setAgentErrorInfo(null);
      updateAgentPanel({
        state: "understand_request",
        intent: action || "understand_request",
        message: "Inspecting the current request and runtime context.",
        reason: "The agent is classifying intent before choosing the next pipeline step.",
        question: "",
        options: [],
      });

      const response = await fetch(withRuntimeQuery("/agent/message"), {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || "Agent request failed");
      }

      if (data?.session_id || data?.run_id) {
        applyRuntimeContext({
          session_id: data.session_id,
          run_id: data.run_id,
        });
      }
      preview.textContent = JSON.stringify(data, null, 2);
      return handleAgentResponse(data, { hadReferenceImage });
    }

    /* ===== Unified Scene Instruction ===== */
    async function applyInstruction(opts = {}) {
      const status = document.getElementById("jsonStatus");
      const preview = document.getElementById("jsonPreview");
      const button = document.getElementById("btnApplyInstruction");
      const fileInput = document.getElementById("imageInput");
      const classDirPicker = document.getElementById("classDirPicker");
      const instruction = (opts.instruction && opts.instruction.trim()) || document.getElementById("sceneInput").value.trim();
      const imageFile = opts.imageFile || (fileInput.files.length > 0 ? fileInput.files[0] : null);

      if (!instruction && !imageFile) {
        toast("warn","Missing input","Please enter an instruction or upload a reference image.");
        return;
      }

      interactionState.startedAt = performance.now();
      interactionState.lastInstruction = instruction;

      button.disabled = true;
      setPill("graph","warn","Applying...");
      status.textContent = "Agent applying instruction...";
      preview.textContent = "Waiting for unified agent result...";
      setFeedback([
        "Routing request through Scene Agent…",
        "Selecting graph / Real2Sim / scene generation flow…",
      ]);

      try {
        const data = await sendAgentMessage({
          instruction,
          action: opts.action || "",
          resampleMode: opts.resampleMode || "",
          sceneEndpoint: opts.sceneEndpoint || "",
          imageFile,
          classNamesRaw: collectClassNamesPayload(classDirPicker),
        });
        const elapsed = Math.round(performance.now() - interactionState.startedAt);
        const state = data?.agent?.state || "completed";
        if (state === "needs_clarification" || state === "await_layout_strategy") {
          setPill("model","warn","Question");
          setPill("graph","warn","Waiting");
          status.textContent = `Agent question (${elapsed} ms)`;
          toast("info", "Agent needs input", data.agent.question || data.agent.message || "Answer the follow-up question.");
        } else if (state === "run_real2sim") {
          setPill("model","warn","Real2Sim");
          status.textContent = `Real2Sim started (${elapsed} ms)`;
          toast("ok", "Agent started Real2Sim", data?.agent?.message || "Real2Sim job started.");
        } else {
          setPill("model","ok","Ready");
          status.textContent = `Done (${elapsed} ms)`;
          toast("ok", "Agent done", data?.agent?.message || "Request completed.");
        }
        return { ok:true, data, elapsed };
      } catch (err) {
        console.error(err);
        setPill("model","err","Failed");
        setPill("graph","err","Failed");
        status.textContent = "Failed";
        preview.textContent = "Agent request failed";
        setFeedback([String(err)]);
        setAgentErrorInfo(null, String(err));
        appendAgentTranscript("assistant", "Request failed.", String(err));
        toast("err","Agent request failed", String(err));
        return { ok:false, error: String(err) };
      } finally {
        button.disabled = false;
      }
    }

    async function editJson(opts = {}) {
      return applyInstruction(opts);
    }
