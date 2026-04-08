    async function hydrateReal2SimLivePreview(payload = {}, options = {}) {
      const artifacts = getReal2SimArtifactsFromPayload(payload) || {};
      const objectUrls = Array.isArray(artifacts.object_glb_urls) ? artifacts.object_glb_urls : [];
      let loadedObjectCount = 0;
      let loadedMerged = false;

      updateArtifactPanel(payload);

      for (const url of objectUrls) {
        const queued = enqueueGlbLoad(url);
        if (queued) loadedObjectCount += 1;
      }

      if (artifacts.scene_glb_url && viewerState.mergedUrl !== artifacts.scene_glb_url && !viewerState.loadingMerged) {
        const mergedManifest = await loadArtifactJson(artifacts.manifest_json_url);
        const mergedQueued = enqueueGlbLoad(artifacts.scene_glb_url, {
          isMerged: true,
          manifest: mergedManifest,
        });
        if (mergedQueued) loadedMerged = true;
      }

      let review = null;
      if (options.refreshReview) {
        review = await refreshMaskAssignmentReview({
          silent: options.silentReview !== false,
          auto: options.autoReview !== false,
        });
      }

      return {
        artifacts,
        objectUrls,
        loadedObjectCount,
        loadedMerged,
        review,
      };
    }

    async function monitorReal2SimJob(jobInfo = {}) {
      const jobId = jobInfo.job_id;
      if (!jobId) {
        throw new Error("Missing Real2Sim job id.");
      }
      if (real2simMonitorState.activeJobId === jobId && real2simMonitorState.activePromise) {
        return real2simMonitorState.activePromise;
      }
      const monitorToken = real2simMonitorState.token;

      const monitorPromise = (async () => {
        const resultEl = document.getElementById("sceneSvcResult");
        const statusEl = document.getElementById("sceneSvcStatus");
        const logStartOffset = Number(jobInfo.log_start_offset || 0);
        const logPath = jobInfo.log_path || "real2sim.log";

        resetReal2SimLog(logStartOffset, logPath);
        setPill("sim","warn","Real2Sim");
        statusEl.textContent = `Job ${jobId.slice(0, 8)} running`;
        resultEl.textContent = JSON.stringify({ job_id: jobId, status: "running" }, null, 2);

        let done = false;
        let finalJobStatus = null;
        let finalJob = null;
        let loadedObjectCount = 0;
        let loadedMerged = false;
        while (!done) {
          if (monitorToken !== real2simMonitorState.token) {
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, 1200));
          if (monitorToken !== real2simMonitorState.token) {
            return;
          }
          try {
            await refreshReal2SimLog();
          } catch (logErr) {
            console.warn("Real2Sim log refresh failed:", logErr);
            document.getElementById("real2simLogStatus").textContent = "Log retrying...";
          }

          const pollRes = await fetch(withRuntimeQuery(`/real2sim/status/${jobId}`));
          const pollData = await pollRes.json();
          if (monitorToken !== real2simMonitorState.token) {
            return;
          }
          if (!pollRes.ok || !pollData?.job) {
            throw new Error((pollData && (pollData.msg || pollData.error)) || "Failed to poll job status");
          }

          const job = pollData.job;
          finalJob = job;
          const progress = job.progress || {};
          if (Array.isArray(job?.session_state?.history)) {
            renderAgentTranscript(job.session_state.history);
          }
          setAgentErrorInfo(job.error_info || null);

          const livePreview = await hydrateReal2SimLivePreview(
            { job },
            {
              refreshReview: job.status === "running" || job.status === "queued",
              silentReview: true,
              autoReview: true,
            }
          );
          const artifacts = livePreview.artifacts || {};
          const objectUrls = Array.isArray(livePreview.objectUrls) ? livePreview.objectUrls : [];
          loadedObjectCount += livePreview.loadedObjectCount || 0;
          loadedMerged = loadedMerged || !!livePreview.loadedMerged;

          const statusText = `${job.status} • objects ${objectUrls.length} • queue ${viewerState.loadQueue.length}${artifacts.scene_glb_url ? " • merged ready" : ""}`;
          document.getElementById("statusSimText").textContent = statusText;
          statusEl.textContent = statusText;
          setSimProgress(progress);
          resultEl.textContent = JSON.stringify(job, null, 2);

          if (job.status === "succeeded") {
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
        try {
          const review = await refreshMaskAssignmentReview({ silent: true, auto: false });
          if (review?.needs_attention) {
            if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
            toast(
              "warn",
              "Review mask mapping",
              "Real2Sim finished, but the mask-to-node assignment still needs confirmation."
            );
          }
        } catch (reviewErr) {
          console.warn("Mask assignment review refresh failed:", reviewErr);
        }
        if (finalJobStatus === "succeeded") {
          document.getElementById("real2simLogStatus").textContent = "Completed";
        } else if (finalJobStatus === "failed") {
          document.getElementById("real2simLogStatus").textContent = "Failed";
          if (finalJob?.error_info?.code === "mask_assignment_failed") {
            if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
            toast("warn", "Mask assignment needs review", "Open the drawer and correct the assignment before continuing.");
          }
        }
      })();
      real2simMonitorState.activeJobId = jobId;
      real2simMonitorState.activePromise = monitorPromise;
      try {
        return await monitorPromise;
      } finally {
        if (real2simMonitorState.activeJobId === jobId && monitorToken === real2simMonitorState.token) {
          real2simMonitorState.activeJobId = "";
          real2simMonitorState.activePromise = null;
        }
      }
    }

    function prefillAgentShortcut(prompt, options = {}) {
      const input = document.getElementById("sceneInput");
      if (!input) return;
      input.value = prompt;
      updateInputMeta();
      if (options.resampleMode) {
        setResampleMode(options.resampleMode);
      }
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      setFeedback([
        "Shortcut drafted a prompt for the single agent entry.",
        "Review the text if needed, then press Apply Instruction.",
      ]);
      toast("info", "Prompt drafted", "Review or edit the prompt, then use Apply Instruction.");
    }

    /* ===== Real2Sim ===== */
    function runReal2Sim() {
      prefillAgentShortcut(
        "Please run Real2Sim on the current scene graph and reference image, then show me the numbered mask assignment for review."
      );
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

    function runResample() {
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

    function callSceneService(endpoint, options = {}) {
      const resampleMode = options.resampleMode || "joint";
      if (endpoint === "scene") {
        prefillAgentShortcut(
          "Please keep layout and generate a scene preview, and tell me if any real2sim nodes are still missing from the manifest."
        );
        return;
      }

      if (resampleMode === "lock_real2sim") {
        prefillAgentShortcut(
          "Please generate a fresh scene preview in lock real2sim mode, keep the observed Real2Sim support chains rigid, and retry automatically if layout conflicts happen.",
          { resampleMode }
        );
        return;
      }

      prefillAgentShortcut(
        "Please generate a fresh scene preview with joint resampling, and retry automatically if the layout collides.",
        { resampleMode }
      );
    }
