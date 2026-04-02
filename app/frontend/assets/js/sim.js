    async function monitorReal2SimJob(jobInfo = {}) {
      const jobId = jobInfo.job_id;
      if (!jobId) {
        throw new Error("Missing Real2Sim job id.");
      }

      const resultEl = document.getElementById("sceneSvcResult");
      const statusEl = document.getElementById("sceneSvcStatus");
      const logStartOffset = Number(jobInfo.log_start_offset || 0);
      const logPath = jobInfo.log_path || "real2sim.log";

      resetReal2SimLog(logStartOffset, logPath);
      statusEl.textContent = `Job ${jobId.slice(0, 8)} running`;
      resultEl.textContent = JSON.stringify({ job_id: jobId, status: "running" }, null, 2);

      let done = false;
      let finalJobStatus = null;
      let loadedObjectCount = 0;
      let loadedMerged = false;
      while (!done) {
        await new Promise((resolve) => setTimeout(resolve, 1200));
        try {
          await refreshReal2SimLog();
        } catch (logErr) {
          console.warn("Real2Sim log refresh failed:", logErr);
          document.getElementById("real2simLogStatus").textContent = "Log retrying...";
        }

        const pollRes = await fetch(withRuntimeQuery(`/real2sim/status/${jobId}`));
        const pollData = await pollRes.json();
        if (!pollRes.ok || !pollData?.job) {
          throw new Error((pollData && (pollData.msg || pollData.error)) || "Failed to poll job status");
        }

        const job = pollData.job;
        const progress = job.progress || {};
        const artifacts = job.artifacts || {};
        const objectUrls = Array.isArray(artifacts.object_glb_urls) ? artifacts.object_glb_urls : [];
        if (Array.isArray(job?.session_state?.history)) {
          renderAgentTranscript(job.session_state.history);
        }
        updateArtifactPanel({ job });
        setAgentErrorInfo(job.error_info || null);

        const statusText = `${job.status} • objects ${objectUrls.length} • queue ${viewerState.loadQueue.length}${artifacts.scene_glb_url ? " • merged ready" : ""}`;
        document.getElementById("statusSimText").textContent = statusText;
        statusEl.textContent = statusText;
        setSimProgress(progress);
        resultEl.textContent = JSON.stringify(job, null, 2);

        for (const url of objectUrls) {
          const queued = enqueueGlbLoad(url);
          if (queued) loadedObjectCount += 1;
        }

        if (job.status === "succeeded") {
          if (artifacts.scene_glb_url && !loadedMerged) {
            const mergedManifest = await loadArtifactJson(artifacts.manifest_json_url);
            const mergedQueued = enqueueGlbLoad(artifacts.scene_glb_url, {
              isMerged: true,
              manifest: mergedManifest,
            });
            if (mergedQueued) loadedMerged = true;
          }
          done = true;
          finalJobStatus = job.status;
          setSimProgress({ ...(job.progress || {}), phase: "completed", percent: 100 });
          setPill("sim","ok","Ready");
          document.getElementById("real2simLogStatus").textContent = "Completed";
          toast("ok","Real2Sim done", `Loaded ${loadedObjectCount} object GLB(s)${loadedMerged ? " + merged scene" : ""}.`);
        } else if (job.status === "failed") {
          done = true;
          finalJobStatus = job.status;
          setSimProgress({ ...(job.progress || {}), phase: "failed" });
          setPill("sim","err","Failed");
          document.getElementById("real2simLogStatus").textContent = "Failed";
          const failureMessage = job?.error_info?.user_message || job.error || "Unknown error";
          setAgentErrorInfo(job.error_info || null, job.error || failureMessage);
          toast("err","Real2Sim failed", failureMessage);
        }
      }

      try {
        await refreshReal2SimLog();
      } catch (logErr) {
        console.warn("Final Real2Sim log refresh failed:", logErr);
      }
      if (finalJobStatus === "succeeded") {
        document.getElementById("real2simLogStatus").textContent = "Completed";
      } else if (finalJobStatus === "failed") {
        document.getElementById("real2simLogStatus").textContent = "Failed";
      }
    }

    /* ===== Real2Sim ===== */
    async function runReal2Sim() {
      const btn = document.getElementById("real2simBtn");
      const resultEl = document.getElementById("sceneSvcResult");
      const statusEl = document.getElementById("sceneSvcStatus");

      btn.disabled = true;
      setPill("sim","warn","Real2Sim...");
      document.getElementById("statusSimText").textContent = "Running Real2Sim...";
      statusEl.textContent = "Real2Sim running";
      resultEl.textContent = "Submitting Real2Sim job...";
      setAgentErrorInfo(null);
      showThreeViewer().then(() => clearThreeViewer()).catch((viewerErr) => {
        console.warn("Three viewer unavailable:", viewerErr);
        setPreviewMessage("Three.js viewer failed to load. Check browser console / CDN access.");
      });
      if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
      resetSimProgress();
      resetReal2SimLog();
      setSimProgress({ phase: "queued", percent: 1, generated_objects: 0, expected_objects: null, has_merged_scene: false });

      try {
        await sendAgentMessage({
          instruction: "Run Real2Sim on the current scene graph.",
          action: "run_real2sim",
        });
      } catch (err) {
        console.error(err);
        setPill("sim","err","Failed");
        document.getElementById("statusSimText").textContent = "Failed";
        document.getElementById("real2simLogStatus").textContent = "Unavailable";
        setAgentErrorInfo(null, String(err));
        toast("err","Real2Sim error", "Check backend logs / service status.");
      } finally {
        btn.disabled = false;
      }
    }

    /* ===== Isaac Scene Service ===== */
    function setResampleMode(mode) {
      const normalized = mode === "lock_real2sim" ? "lock_real2sim" : "joint";
      const input = document.getElementById("resampleModeSelect");
      if (input) input.value = normalized;

      document.querySelectorAll(".mode-card").forEach((card) => {
        const active = card.dataset.mode === normalized;
        card.classList.toggle("is-active", active);
        card.setAttribute("aria-checked", active ? "true" : "false");
      });
    }

    function getSelectedResampleMode() {
      const input = document.getElementById("resampleModeSelect");
      return input && input.value ? input.value : "joint";
    }

    async function runResample() {
      return callSceneService("scene_new", { resampleMode: getSelectedResampleMode() });
    }

    async function loadArtifactJson(url) {
      if (!url) return null;
      try {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`Failed to fetch artifact JSON: ${res.status}`);
        }
        return await res.json();
      } catch (err) {
        console.warn("Artifact JSON fetch failed:", url, err);
        return null;
      }
    }

    async function callSceneService(endpoint, options = {}) {
      const statusEl = document.getElementById("sceneSvcStatus");
      const resultEl = document.getElementById("sceneSvcResult");
      const resampleMode = options.resampleMode || "joint";

      statusEl.textContent = `Calling /${endpoint}...`;
      setPill("sim","warn","Calling...");
      document.getElementById("statusSimText").textContent =
        endpoint === "scene_new" ? `Calling /${endpoint} (${resampleMode})...` : `Calling /${endpoint}...`;
      if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
      resetSimProgress();
      resetSceneDebug();
      setAgentErrorInfo(null);
      resultEl.textContent = "Requesting...";

      try {
        await sendAgentMessage({
          instruction: endpoint === "scene_new" ? "Generate the scene." : "Edit the scene and keep the current layout.",
          action: "generate_scene",
          sceneEndpoint: endpoint,
          resampleMode,
        });
      } catch (err) {
        console.error(err);
        statusEl.textContent = "Failed";
        resultEl.textContent = String(err);
        setPill("sim","err","Failed");
        setAgentErrorInfo(null, String(err));
        appendAgentTranscript("assistant", "Scene generation failed.", String(err));
        toast("err","Request failed", "Possibly scene service is not running.");
      }
    }
