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

    // Right-column tab switcher (Step 4). Tabs: sim / logs / diag.
    function switchRightTab(name) {
      const want = String(name || "sim").toLowerCase();
      const buttons = document.querySelectorAll(".right-tab-btn");
      buttons.forEach((btn) => {
        const target = String(btn.getAttribute("data-tab-target") || "");
        btn.dataset.active = target === want ? "true" : "false";
      });
      const panes = document.querySelectorAll(".right-tab-pane");
      panes.forEach((p) => {
        const tab = String(p.getAttribute("data-tab") || "");
        p.dataset.active = tab === want ? "true" : "false";
      });
      const title = document.getElementById("rightPanelTitle");
      if (title) {
        title.textContent =
          want === "logs" ? "Logs"
          : want === "diag" ? "Diagnostics"
          : "Simulation Preview";
      }
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
      document.getElementById("uploadStatus").textContent = imageCount ? `${imageCount} image selected` : "No image selected";
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

    function clearReferenceImageInput() {
      const fileInput = document.getElementById("imageInput");
      if (fileInput) {
        fileInput.value = "";
      }
      clearReferenceImagePreview();
      updateInputMeta();
    }

    function setFeedback(lines){
      const box = document.getElementById("feedbackBox");
      box.textContent = (lines && lines.length) ? lines.map(s => "• " + s).join("\n") : "-";
    }

    function clearAgentTranscript() {
      const root = document.getElementById("agentTranscript");
      if (!root) return;
      root.innerHTML = "";
      const empty = document.createElement("div");
      empty.className = "agent-transcript-empty";
      empty.textContent = "No conversation yet.";
      root.appendChild(empty);
    }

    function appendAgentTranscript(role, body, meta = "") {
      const root = document.getElementById("agentTranscript");
      if (!root) return;
      const content = String(body || "").trim();
      const metaText = String(meta || "").trim();
      if (!content && !metaText) return;

      const empty = root.querySelector(".agent-transcript-empty");
      if (empty) empty.remove();

      const item = document.createElement("div");
      item.className = `agent-chat-msg ${role === "user" ? "user" : "assistant"}`;

      const roleEl = document.createElement("div");
      roleEl.className = "agent-chat-role";
      roleEl.textContent = role === "user" ? "User" : "Agent";

      const bodyEl = document.createElement("div");
      bodyEl.className = "agent-chat-body";
      bodyEl.textContent = content || "-";

      item.appendChild(roleEl);
      item.appendChild(bodyEl);

      if (metaText) {
        const metaEl = document.createElement("div");
        metaEl.className = "agent-chat-meta";
        metaEl.textContent = metaText;
        item.appendChild(metaEl);
      }

      root.appendChild(item);
      root.scrollTop = root.scrollHeight;
    }

    // ---- Failure bubble (Step 2 of UI redesign) ----------------------------
    // Renders a richer chat card for job failures: stage tag chips, collapsed
    // technical detail, and action buttons (Retry / Copy log path / View log).
    // Dedupes by (kind, job_id, code, user_message) so multiple poll-tick
    // detections of the same failure don't stack.

    let _lastFailureCardKey = "";

    function _failureCardKey(failure) {
      return [
        failure?.kind || "",
        failure?.job_id || "",
        failure?.error_info?.code || "",
        failure?.error_info?.user_message || failure?.user_message || "",
      ].join("|");
    }

    function resetFailureCardDedupe() {
      _lastFailureCardKey = "";
    }

    function appendAgentFailureBubble(failure, opts = {}) {
      const root = document.getElementById("agentTranscript");
      if (!root || !failure || !failure.kind) return;
      const key = _failureCardKey(failure);
      if (!opts.force && key && key === _lastFailureCardKey) return;
      _lastFailureCardKey = key;

      const empty = root.querySelector(".agent-transcript-empty");
      if (empty) empty.remove();

      const card = document.createElement("div");
      card.className = "agent-chat-msg assistant agent-failure";
      card.dataset.kind = failure.kind;

      const role = document.createElement("div");
      role.className = "agent-chat-role";
      role.textContent = "Agent";
      card.appendChild(role);

      const head = document.createElement("div");
      head.className = "agent-failure-head";
      const stageLabel = failure.kind === "scene_robot" ? "scene_robot" : "Real2Sim";
      head.innerHTML = `<span class="agent-failure-icon">⚠</span> ${escapeHtml(stageLabel)} failed`;
      card.appendChild(head);

      const ei = failure.error_info || {};
      const tags = [];
      if (ei.code) tags.push(escapeHtml(String(ei.code)));
      if (ei.step) tags.push("step=" + escapeHtml(String(ei.step)));
      if (ei.retryable === true) tags.push("retryable");
      if (ei.retryable === false) tags.push("not retryable");
      if (tags.length) {
        const subhead = document.createElement("div");
        subhead.className = "agent-failure-subhead";
        subhead.innerHTML = tags.map((t) => `<span class="agent-failure-tag">${t}</span>`).join("");
        card.appendChild(subhead);
      }

      const body = document.createElement("div");
      body.className = "agent-chat-body";
      body.textContent =
        ei.user_message || failure.user_message || `${stageLabel} job failed.`;
      card.appendChild(body);

      const tech = _buildFailureTechBlock(failure);
      if (tech) card.appendChild(tech);

      card.appendChild(_buildFailureActions(failure));

      root.appendChild(card);
      root.scrollTop = root.scrollHeight;

      // Step 4: a fresh failure switches the right panel to Logs so the user
      // sees the relevant log without a manual click. Avoid yanking the view
      // when the bubble is being re-rendered (dedupe handled above already).
      if (typeof switchRightTab === "function") switchRightTab("logs");
    }

    function _buildFailureTechBlock(failure) {
      const ei = failure.error_info || {};
      const dig = failure.log_digest || ei.log_digest || {};
      const lines = [];
      if (dig.last_exception) lines.push(`last_exception: ${dig.last_exception}`);
      if (typeof dig.exit_code === "number") lines.push(`exit_code: ${dig.exit_code}`);
      if (Array.isArray(dig.hostports) && dig.hostports.length) {
        lines.push(`hostports: ${dig.hostports.join(", ")}`);
      }
      if (Array.isArray(dig.urls) && dig.urls.length) {
        lines.push(`urls: ${dig.urls.join(", ")}`);
      }
      const logPath = failure.log_path || dig.log_path;
      if (logPath) lines.push(`log: ${logPath}`);
      if (!lines.length && ei.technical_detail) {
        lines.push(`technical_detail: ${String(ei.technical_detail).slice(0, 600)}`);
      }
      if (Array.isArray(dig.error_lines) && dig.error_lines.length) {
        lines.push("error_lines (last):");
        for (const ln of dig.error_lines.slice(-6)) {
          lines.push(`  | ${String(ln).slice(0, 300)}`);
        }
      }
      if (!lines.length) return null;

      const wrap = document.createElement("details");
      wrap.className = "agent-failure-tech";
      const summary = document.createElement("summary");
      summary.textContent = "Technical detail";
      wrap.appendChild(summary);
      const pre = document.createElement("pre");
      pre.className = "agent-failure-tech-pre";
      pre.textContent = lines.join("\n");
      wrap.appendChild(pre);
      return wrap;
    }

    function _buildFailureActions(failure) {
      const actions = document.createElement("div");
      actions.className = "agent-failure-actions";
      const ei = failure.error_info || {};

      if (ei.retryable !== false && typeof retryFailedStage === "function") {
        const retryBtn = document.createElement("button");
        retryBtn.type = "button";
        retryBtn.className = "agent-failure-btn primary";
        retryBtn.textContent = "↻ Retry";
        retryBtn.addEventListener("click", async () => {
          retryBtn.disabled = true;
          try {
            await retryFailedStage(failure.kind, failure);
          } catch (err) {
            toast("err", "Retry failed", String(err?.message || err));
          } finally {
            retryBtn.disabled = false;
          }
        });
        actions.appendChild(retryBtn);
      }

      const dig = failure.log_digest || ei.log_digest || {};
      const logPath = failure.log_path || dig.log_path;
      if (logPath) {
        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "agent-failure-btn";
        copyBtn.textContent = "📋 Copy log path";
        copyBtn.addEventListener("click", () => _copyFailureLogPath(logPath));
        actions.appendChild(copyBtn);
      }

      const viewBtn = document.createElement("button");
      viewBtn.type = "button";
      viewBtn.className = "agent-failure-btn";
      viewBtn.textContent = "📂 View log";
      viewBtn.addEventListener("click", () => _viewFailureLog(failure));
      actions.appendChild(viewBtn);

      return actions;
    }

    async function _copyFailureLogPath(path) {
      try {
        await navigator.clipboard.writeText(String(path));
        toast("ok", "Log path copied", String(path));
      } catch (err) {
        // Fallback: select-and-copy via a hidden textarea
        try {
          const ta = document.createElement("textarea");
          ta.value = String(path);
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          document.body.removeChild(ta);
          toast("ok", "Log path copied", String(path));
        } catch (err2) {
          toast("err", "Copy failed", String(err2?.message || err2));
        }
      }
    }

    function _viewFailureLog(failure) {
      // Step 4: logs live in the right column's Logs tab now.
      if (typeof switchRightTab === "function") switchRightTab("logs");
      const targetId = failure.kind === "scene_robot" ? "sceneRobotLog" : "real2simLog";
      const el = document.getElementById(targetId);
      if (el && typeof el.scrollIntoView === "function") {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }

    // ---- Tool steps (Step 3): collapsible "what tools the agent ran" ------
    // The backend records `tool_steps: [{tool, args, summary}]` in every
    // /agent/message response. We render them as a <details> block attached
    // to the latest assistant chat bubble, so the user can see *what*
    // actually ran without leaving the chat. Args are truncated to keep the
    // UI compact; full args fit in the scroll-y area when expanded.

    function _toolStepStatus(summary) {
      const s = String(summary || "");
      if (s.startsWith("error")) return "err";
      if (s.startsWith("started job")) return "running";
      return "ok";
    }

    function _toolStepIcon(status) {
      switch (status) {
        case "running": return "⏳";
        case "err":     return "✗";
        default:        return "✓";
      }
    }

    function _formatToolArgs(args) {
      if (args == null) return "";
      let text;
      if (typeof args === "string") {
        const trimmed = args.trim();
        if (!trimmed || trimmed === "{}") return "";
        // Pretty-print JSON when we can; otherwise show the raw string.
        try {
          const parsed = JSON.parse(trimmed);
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed) && Object.keys(parsed).length === 0) {
            return "";
          }
          text = JSON.stringify(parsed, null, 2);
        } catch (e) {
          text = trimmed;
        }
      } else if (typeof args === "object") {
        if (Array.isArray(args) ? args.length === 0 : Object.keys(args).length === 0) return "";
        try { text = JSON.stringify(args, null, 2); } catch (e) { text = String(args); }
      } else {
        text = String(args);
      }
      // Cap at ~600 chars; users rarely need more than this in a glance.
      if (text.length > 600) text = text.slice(0, 600) + "\n…";
      return text;
    }

    function _buildToolStepsBlock(steps) {
      if (!Array.isArray(steps) || !steps.length) return null;
      const wrap = document.createElement("details");
      wrap.className = "agent-tool-steps";
      const summary = document.createElement("summary");
      summary.textContent = `Tools used (${steps.length})`;
      wrap.appendChild(summary);

      const list = document.createElement("div");
      list.className = "agent-tool-steps-list";

      for (const step of steps) {
        if (!step || typeof step !== "object") continue;
        const status = _toolStepStatus(step.summary);
        const row = document.createElement("div");
        row.className = "tool-step";
        row.dataset.status = status;

        const icon = document.createElement("span");
        icon.className = "tool-step-icon";
        icon.textContent = _toolStepIcon(status);
        row.appendChild(icon);

        const name = document.createElement("span");
        name.className = "tool-step-name";
        name.textContent = String(step.tool || "(unknown)");
        row.appendChild(name);

        const summ = document.createElement("span");
        summ.className = "tool-step-summary";
        summ.dataset.status = status;
        summ.textContent = String(step.summary || "");
        summ.title = String(step.summary || "");
        row.appendChild(summ);

        // Reserve the right-hand grid column even when empty so rows align.
        const spacer = document.createElement("span");
        row.appendChild(spacer);

        const argsText = _formatToolArgs(step.args);
        if (argsText) {
          const argsEl = document.createElement("pre");
          argsEl.className = "tool-step-args";
          argsEl.textContent = argsText;
          row.appendChild(argsEl);
        }

        list.appendChild(row);
      }
      wrap.appendChild(list);
      return wrap;
    }

    function appendToolStepsToLatestAssistantBubble(steps) {
      const root = document.getElementById("agentTranscript");
      if (!root) return;
      if (!Array.isArray(steps) || !steps.length) return;
      // Pick the latest assistant message; that's what these tools belong to.
      const candidates = root.querySelectorAll(".agent-chat-msg.assistant");
      const target = candidates[candidates.length - 1];
      if (!target) return;
      // Replace any previously-attached tool-steps block (re-render on update).
      const existing = target.querySelector(":scope > .agent-tool-steps");
      if (existing) existing.remove();
      const block = _buildToolStepsBlock(steps);
      if (!block) return;
      target.appendChild(block);
      root.scrollTop = root.scrollHeight;
    }

    function renderAgentTranscript(history) {
      const root = document.getElementById("agentTranscript");
      if (!root) return;
      root.innerHTML = "";
      const turns = Array.isArray(history) ? history.filter((entry) => entry && (entry.content || entry.role)) : [];
      if (!turns.length) {
        clearAgentTranscript();
        return;
      }
      for (const turn of turns) {
        appendAgentTranscript(turn.role === "user" ? "user" : "assistant", turn.content || "");
      }
    }

    function resetArtifactPanel() {
      const meta = document.getElementById("artifactMeta");
      const root = document.getElementById("artifactList");
      if (meta) meta.textContent = "No outputs yet";
      if (!root) return;
      root.innerHTML = "";
      const empty = document.createElement("div");
      empty.className = "artifact-empty";
      empty.textContent = "Run Real2Sim or Scene Service to collect artifact links.";
      root.appendChild(empty);
    }

    function setAgentErrorInfo(errorInfo, fallbackMessage = "") {
      const meta = document.getElementById("agentErrorMeta");
      const box = document.getElementById("agentErrorBox");
      if (!meta || !box) return;

      if (!errorInfo && !fallbackMessage) {
        meta.textContent = "No active errors";
        box.textContent = "No errors.";
        return;
      }

      const info = errorInfo && typeof errorInfo === "object" ? errorInfo : {};
      const lines = [];
      if (info.user_message) lines.push(`Message: ${info.user_message}`);
      if (info.code) lines.push(`Code: ${info.code}`);
      if (info.step) lines.push(`Step: ${info.step}`);
      if (typeof info.retryable === "boolean") lines.push(`Retryable: ${info.retryable ? "yes" : "no"}`);
      if (info.technical_detail) lines.push(`Detail: ${info.technical_detail}`);
      if (!lines.length && fallbackMessage) lines.push(String(fallbackMessage));

      meta.textContent = info.code || (fallbackMessage ? "Request failed" : "No active errors");
      box.textContent = lines.join("\n");
    }

    function buildArtifactGroups(payload = {}) {
      const groups = [];
      const sessionState = payload?.session_state || payload?.job?.session_state || null;
      const currentRun = sessionState?.current_run || {};
      const real2sim = currentRun?.real2sim?.artifacts || payload?.job?.artifacts || payload?.real2sim_artifacts || null;
      const scene = currentRun?.scene_generation?.outputs || payload?.scene_result || null;

      if (scene && (scene.saved_usd_url || scene.render_image_url || scene.placements_url)) {
        const links = [];
        if (scene.saved_usd_url) links.push({ label: "Scene USD", url: scene.saved_usd_url });
        if (scene.render_image_url) links.push({ label: "Render", url: scene.render_image_url });
        if (scene.placements_url) links.push({ label: "Placements JSON", url: scene.placements_url });
        groups.push({ title: "Scene Generation", links });
      }

      if (real2sim) {
        const links = [];
        if (real2sim.assignment_json_url) links.push({ label: "Assignment JSON", url: real2sim.assignment_json_url });
        if (real2sim.poses_json_url) links.push({ label: "Poses JSON", url: real2sim.poses_json_url });
        if (real2sim.manifest_json_url) links.push({ label: "Manifest JSON", url: real2sim.manifest_json_url });
        if (real2sim.scene_glb_url) links.push({ label: "Merged Scene GLB", url: real2sim.scene_glb_url });
        if (real2sim.scene_usd_url) links.push({ label: "Merged Scene USD", url: real2sim.scene_usd_url });
        const objectGlbUrls = Array.isArray(real2sim.object_glb_urls) ? real2sim.object_glb_urls : [];
        const objectUsdUrls = Array.isArray(real2sim.object_usd_urls) ? real2sim.object_usd_urls : [];
        objectGlbUrls.slice(0, 4).forEach((url, index) => links.push({ label: `Object GLB ${index + 1}`, url }));
        objectUsdUrls.slice(0, 4).forEach((url, index) => links.push({ label: `Object USD ${index + 1}`, url }));
        if (links.length) groups.push({ title: "Real2Sim", links });
      }

      return groups;
    }

    function updateArtifactPanel(payload = {}) {
      const meta = document.getElementById("artifactMeta");
      const root = document.getElementById("artifactList");
      if (!meta || !root) return;

      const groups = buildArtifactGroups(payload);
      root.innerHTML = "";
      if (!groups.length) {
        resetArtifactPanel();
        return;
      }

      meta.textContent = `${groups.length} output group${groups.length === 1 ? "" : "s"}`;
      for (const group of groups) {
        const wrap = document.createElement("div");
        wrap.className = "artifact-group";

        const title = document.createElement("div");
        title.className = "artifact-group-title";
        title.textContent = group.title;
        wrap.appendChild(title);

        const links = document.createElement("div");
        links.className = "artifact-links";
        for (const item of group.links) {
          if (!item?.url) continue;
          const anchor = document.createElement("a");
          anchor.className = "artifact-link";
          anchor.href = item.url;
          anchor.target = "_blank";
          anchor.rel = "noreferrer";
          anchor.textContent = item.label || item.url;
          links.appendChild(anchor);
        }
        wrap.appendChild(links);
        root.appendChild(wrap);
      }
    }

    function formatSceneObjectOptionLabel(obj = {}) {
      const parts = [obj.path || ""];
      if (obj.class) parts.push(obj.class);
      if (obj.caption && obj.caption !== obj.class) parts.push(obj.caption);
      return parts.filter(Boolean).join(" • ");
    }

    function resetMaskAssignmentReview(message = "Run Real2Sim to inspect or correct mask-to-node assignment.") {
      assignmentReviewState.data = null;
      assignmentReviewState.dirty = false;
      assignmentReviewState.signature = "";
      const meta = document.getElementById("maskReviewMeta");
      const panel = document.getElementById("maskReviewPanel");
      const overlayMeta = document.getElementById("maskOverlayMeta");
      const overlayImage = document.getElementById("maskOverlayImage");
      const overlayEmpty = document.getElementById("maskOverlayEmpty");
      if (meta) meta.textContent = "No review data";
      if (panel) {
        panel.innerHTML = "";
        const empty = document.createElement("div");
        empty.className = "artifact-empty";
        empty.textContent = message;
        panel.appendChild(empty);
      }
      if (overlayMeta) overlayMeta.textContent = "Unavailable";
      if (overlayImage) {
        overlayImage.removeAttribute("src");
        overlayImage.style.display = "none";
      }
      if (overlayEmpty) {
        overlayEmpty.style.display = "flex";
        overlayEmpty.textContent = "No numbered mask overlay available yet.";
      }
    }

    function getMaskAssignmentReviewSignature(review) {
      if (!review || typeof review !== "object") return "";
      const signaturePayload = {
        overlay_image_url: review.overlay_image_url || "",
        assignments: Array.isArray(review.assignments)
          ? review.assignments.map((row) => ({
              scene_path: row?.scene_path || "",
              mask_label: Number(row?.mask_label || 0),
              output_name: row?.output_name || "",
              confidence: Number(row?.confidence || 0),
            }))
          : [],
        unmatched_scene_paths: Array.isArray(review.unmatched_scene_paths) ? review.unmatched_scene_paths : [],
        unmatched_mask_labels: Array.isArray(review.unmatched_mask_labels) ? review.unmatched_mask_labels : [],
        manifest: {
          unmatched_scene_paths: Array.isArray(review?.manifest?.unmatched_scene_paths) ? review.manifest.unmatched_scene_paths : [],
          unmatched_outputs: Array.isArray(review?.manifest?.unmatched_outputs) ? review.manifest.unmatched_outputs : [],
        },
      };
      return JSON.stringify(signaturePayload);
    }

    function renderMaskAssignmentReview(review) {
      const meta = document.getElementById("maskReviewMeta");
      const panel = document.getElementById("maskReviewPanel");
      const overlayMeta = document.getElementById("maskOverlayMeta");
      const overlayImage = document.getElementById("maskOverlayImage");
      const overlayEmpty = document.getElementById("maskOverlayEmpty");
      if (!meta || !panel || !overlayMeta || !overlayImage || !overlayEmpty) return;

      if (!review || typeof review !== "object") {
        resetMaskAssignmentReview();
        return;
      }

      assignmentReviewState.data = review;
      assignmentReviewState.signature = getMaskAssignmentReviewSignature(review);
      assignmentReviewState.dirty = false;
      const summary = review.summary || {};
      const manifest = review.manifest || {};
      const matched = Number(summary.matched_assignments || 0);
      const sceneObjects = Number(summary.scene_objects || 0);
      const maskLabels = Number(summary.mask_labels || 0);
      const unmatchedNodes = Number(summary.unmatched_scene_paths || 0);
      const unmatchedMasks = Number(summary.unmatched_mask_labels || 0);
      const lowConfidence = Number(summary.low_confidence_assignments || 0);
      meta.textContent = `${matched}/${sceneObjects} nodes matched • ${maskLabels} masks`;

      panel.innerHTML = "";

      const toolbar = document.createElement("div");
      toolbar.className = "assignment-review-toolbar";

      const summaryCard = document.createElement("div");
      summaryCard.className = `assignment-review-summary${review.needs_attention ? " needs-attention" : ""}`;
      summaryCard.textContent =
        `Nodes ${sceneObjects} • Masks ${maskLabels} • Unmatched nodes ${unmatchedNodes} • ` +
        `Unmatched masks ${unmatchedMasks} • Low confidence ${lowConfidence}`;
      toolbar.appendChild(summaryCard);

      const actions = document.createElement("div");
      actions.className = "assignment-review-actions";

      const reloadBtn = document.createElement("button");
      reloadBtn.type = "button";
      reloadBtn.className = "toolbtn toolbtn-quiet";
      reloadBtn.textContent = "Reload Review";
      reloadBtn.addEventListener("click", async () => {
        try {
          await refreshMaskAssignmentReview();
        } catch (err) {
          console.error("Reload mask review failed:", err);
        }
      });
      actions.appendChild(reloadBtn);

      const saveBtn = document.createElement("button");
      saveBtn.type = "button";
      saveBtn.className = "toolbtn primary";
      saveBtn.textContent = assignmentReviewState.saving ? "Saving..." : "Save Mapping";
      saveBtn.disabled = !!assignmentReviewState.saving;
      saveBtn.addEventListener("click", async () => {
        try {
          await saveMaskAssignmentReview();
        } catch (err) {
          console.error("Save mask review failed:", err);
        }
      });
      actions.appendChild(saveBtn);
      toolbar.appendChild(actions);
      panel.appendChild(toolbar);

      if (Array.isArray(review.unmatched_scene_paths) && review.unmatched_scene_paths.length) {
        const note = document.createElement("div");
        note.className = "assignment-review-note";
        note.textContent = `Missing scene nodes in assignment: ${review.unmatched_scene_paths.join(", ")}`;
        panel.appendChild(note);
      }

      if (Array.isArray(manifest.unmatched_scene_paths) && manifest.unmatched_scene_paths.length) {
        const note = document.createElement("div");
        note.className = "assignment-review-note warning";
        note.textContent = `Manifest still misses: ${manifest.unmatched_scene_paths.join(", ")}`;
        panel.appendChild(note);
      }

      if (Array.isArray(manifest.unmatched_outputs) && manifest.unmatched_outputs.length) {
        const note = document.createElement("div");
        note.className = "assignment-review-note";
        note.textContent = `Unbound outputs: ${manifest.unmatched_outputs.join(", ")}`;
        panel.appendChild(note);
      }

      const assignmentByMask = new Map();
      for (const row of Array.isArray(review.assignments) ? review.assignments : []) {
        if (!row || typeof row !== "object") continue;
        assignmentByMask.set(Number(row.mask_label), row);
      }

      const list = document.createElement("div");
      list.className = "assignment-review-list";
      const sceneObjectsList = Array.isArray(review.scene_objects) ? review.scene_objects : [];

      for (const mask of Array.isArray(review.mask_labels) ? review.mask_labels : []) {
        const current = assignmentByMask.get(Number(mask.mask_label)) || null;
        const row = document.createElement("div");
        row.className = "assignment-review-row";

        const header = document.createElement("div");
        header.className = "assignment-review-row-head";

        const title = document.createElement("div");
        title.className = "assignment-review-row-title";
        title.textContent = `Mask ${mask.mask_label} • output ${mask.output_name}`;
        header.appendChild(title);

        const prompt = document.createElement("div");
        prompt.className = "assignment-review-row-meta";
        const promptParts = [];
        if (mask.prompt) promptParts.push(`prompt ${mask.prompt}`);
        if (Array.isArray(mask.bbox_xyxy)) promptParts.push(`bbox ${mask.bbox_xyxy.join(", ")}`);
        if (current && Number.isFinite(Number(current.confidence))) {
          promptParts.push(`confidence ${(Number(current.confidence) * 100).toFixed(0)}%`);
        }
        prompt.textContent = promptParts.join(" • ") || "No prompt metadata";
        header.appendChild(prompt);
        row.appendChild(header);

        const select = document.createElement("select");
        select.className = "toolselect assignment-review-select";
        select.dataset.maskAssignmentSelect = String(mask.mask_label);

        const emptyOption = document.createElement("option");
        emptyOption.value = "";
        emptyOption.textContent = "Leave Unassigned";
        select.appendChild(emptyOption);

        for (const obj of sceneObjectsList) {
          const option = document.createElement("option");
          option.value = obj.path || "";
          option.textContent = formatSceneObjectOptionLabel(obj);
          if (current && current.scene_path === obj.path) {
            option.selected = true;
          }
          select.appendChild(option);
        }
        select.addEventListener("change", () => {
          assignmentReviewState.dirty = true;
        });
        row.appendChild(select);

        const reason = document.createElement("div");
        reason.className = "assignment-review-row-meta";
        reason.textContent = current?.reason || "No reviewer note.";
        row.appendChild(reason);
        list.appendChild(row);
      }
      panel.appendChild(list);

      if (review.overlay_image_url) {
        overlayImage.src = review.overlay_image_url;
        overlayImage.style.display = "block";
        overlayEmpty.style.display = "none";
        overlayMeta.textContent = review.needs_attention ? "Needs confirmation" : "Ready";
      } else {
        overlayImage.removeAttribute("src");
        overlayImage.style.display = "none";
        overlayEmpty.style.display = "flex";
        overlayMeta.textContent = "Unavailable";
      }
    }

    function collectMaskAssignmentSelections() {
      const selections = [];
      document.querySelectorAll("[data-mask-assignment-select]").forEach((selectEl) => {
        const maskLabel = Number(selectEl.dataset.maskAssignmentSelect || 0);
        const scenePath = String(selectEl.value || "").trim();
        if (!maskLabel || !scenePath) return;
        selections.push({
          mask_label: maskLabel,
          scene_path: scenePath,
          confidence: 1.0,
          reason: "Confirmed via manual review.",
        });
      });
      selections.sort((a, b) => a.mask_label - b.mask_label);
      return selections;
    }

    async function refreshMaskAssignmentReview(options = {}) {
      const silent = !!options.silent;
      const auto = !!options.auto;
      if (auto && assignmentReviewState.dirty) {
        return assignmentReviewState.data;
      }
      await ensureRuntimeContext();
      const response = await fetch(withRuntimeQuery("/real2sim/assignment"));
      const data = await response.json();
      if (!response.ok) {
        if (!silent) {
          resetMaskAssignmentReview(data?.error || "Mask assignment review is not available yet.");
        }
        return null;
      }
      const review = data?.review || null;
      const nextSignature = getMaskAssignmentReviewSignature(review);
      if (auto && nextSignature && nextSignature === assignmentReviewState.signature) {
        assignmentReviewState.data = review;
        return review;
      }
      renderMaskAssignmentReview(review);
      return review;
    }

    async function saveMaskAssignmentReview() {
      if (assignmentReviewState.saving) return null;
      assignmentReviewState.saving = true;
      try {
        const response = await fetch(withRuntimeQuery("/real2sim/assignment"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ assignments: collectMaskAssignmentSelections() }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data?.error || "Failed to save the reviewed mask assignment.");
        }
        assignmentReviewState.dirty = false;
        renderMaskAssignmentReview(data?.review || null);
        toast("ok", "Mask mapping saved", "Updated assignment.json and refreshed the Real2Sim manifest.");
        return data?.review || null;
      } finally {
        assignmentReviewState.saving = false;
        if (assignmentReviewState.data) {
          renderMaskAssignmentReview(assignmentReviewState.data);
        }
      }
    }

    function formatAgentStateLabel(state, completedState = "") {
      const value = String(state || completedState || "idle").trim();
      const labels = {
        understand_request: "Understand Request",
        needs_clarification: "Needs Clarification",
        create_scene_graph: "Create Scene Graph",
        edit_scene_graph: "Edit Scene Graph",
        run_real2sim: "Run Real2Sim",
        await_layout_strategy: "Await Layout Strategy",
        generate_scene: "Generate Scene",
        completed: completedState ? `Completed: ${completedState.replaceAll("_", " ")}` : "Completed",
        failed: "Failed",
        idle: "Idle",
      };
      return labels[value] || value.replaceAll("_", " ");
    }

    function updateAgentPanel(agent = {}) {
      const badge = document.getElementById("agentStateBadge");
      const intent = document.getElementById("agentIntentText");
      const message = document.getElementById("agentMessageText");
      const reason = document.getElementById("agentReasonText");
      const questionWrap = document.getElementById("agentQuestionWrap");
      const questionText = document.getElementById("agentQuestionText");
      const optionList = document.getElementById("agentOptionList");
      if (!badge || !intent || !message || !reason || !questionWrap || !questionText || !optionList) return;

      const state = String(agent.state || "idle");
      const completedState = String(agent.completed_state || "");
      const intentText = String(agent.intent || "idle");
      badge.textContent = formatAgentStateLabel(state, completedState);
      badge.classList.remove("is-ok", "is-warn", "is-err");
      if (state === "completed") {
        badge.classList.add("is-ok");
      } else if (state === "failed") {
        badge.classList.add("is-err");
      } else if (state !== "idle") {
        badge.classList.add("is-warn");
      }

      intent.textContent = intentText ? intentText.replaceAll("_", " ") : "Waiting for a request";
      message.textContent = agent.message || "The agent is idle.";
      reason.textContent = agent.reason || "No reasoning yet.";

      const question = agent.question || "";
      questionText.textContent = question || "-";
      optionList.innerHTML = "";
      const options = Array.isArray(agent.options) ? agent.options : [];
      questionWrap.hidden = !question;
      for (const option of options) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "agent-option";
        const title = document.createElement("b");
        title.textContent = option.label || option.id || "Choose";
        const desc = document.createElement("span");
        desc.textContent = option.description || "";
        btn.appendChild(title);
        if (desc.textContent) btn.appendChild(desc);
        btn.addEventListener("click", async () => {
          try {
            await applyInstruction({
              instruction: option.reply || option.label || "",
              action: option.action || "",
              resampleMode: option.resample_mode || "",
              sceneEndpoint: option.scene_endpoint || "",
            });
          } catch (err) {
            console.error("Agent option failed:", err);
          }
        });
        optionList.appendChild(btn);
      }
    }

    function resetAgentPanel() {
      updateAgentPanel({
        state: "idle",
        intent: "",
        message: "The agent will decide whether to create or edit the scene graph, run Real2Sim, or generate the scene.",
        reason: "No reasoning yet.",
        question: "",
        options: [],
      });
      clearAgentTranscript();
      resetArtifactPanel();
      setAgentErrorInfo(null);
      resetMaskAssignmentReview();
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

    function resetSceneDebug() {
      const box = document.getElementById("sceneDebugBox");
      const meta = document.getElementById("sceneDebugMeta");
      if (box) box.textContent = "Waiting for scene generation debug...";
      if (meta) meta.textContent = "No scene run yet";
    }

    function setSceneDebug(payload) {
      const box = document.getElementById("sceneDebugBox");
      const meta = document.getElementById("sceneDebugMeta");
      if (!box || !meta) return;

      if (!payload || typeof payload !== "object") {
        box.textContent = "No debug payload returned.";
        meta.textContent = "Unavailable";
        return;
      }

      const mode = payload.resample_mode || "joint";
      const counts = payload.asset_resolution_counts || {};
      meta.textContent =
        `mode ${mode} • real2sim ${counts.real2sim || 0} • retrieval ${counts.retrieval || 0} • fallback ${counts.fallback || 0} • missing ${counts.missing || 0}`;
      box.textContent = JSON.stringify(payload, null, 2);
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
      await ensureRuntimeContext();
      const qs = new URLSearchParams({
        offset: String(real2simLogState.offset || 0),
        limit: "65536"
      });
      const res = await fetch(withRuntimeQuery("/real2sim/log", {
        offset: qs.get("offset"),
        limit: qs.get("limit"),
      }));
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

    function resetSceneRobotLog(startOffset = 0, logPath = "scene_robot.log") {
      sceneRobotLogState.offset = Number.isFinite(Number(startOffset)) ? Number(startOffset) : 0;
      sceneRobotLogState.path = logPath || "scene_robot.log";
      document.getElementById("sceneRobotLogStatus").textContent = "Waiting...";
      document.getElementById("sceneRobotLog").textContent = `[log] watching ${sceneRobotLogState.path}\n`;
    }

    function appendSceneRobotLog(text) {
      if (!text) return;
      const el = document.getElementById("sceneRobotLog");
      const combined = el.textContent + text;
      el.textContent = combined.length > 120000 ? combined.slice(-120000) : combined;
      el.scrollTop = el.scrollHeight;
    }

    async function refreshSceneRobotLog() {
      await ensureRuntimeContext();
      const qs = new URLSearchParams({
        offset: String(sceneRobotLogState.offset || 0),
        limit: "65536"
      });
      const res = await fetch(withRuntimeQuery("/scene_robot/log", {
        offset: qs.get("offset"),
        limit: qs.get("limit"),
      }));
      const data = await res.json();
      if (!res.ok) {
        throw new Error((data && (data.msg || data.error)) || "Failed to fetch scene_robot log");
      }
      if (typeof data.next_offset === "number") {
        sceneRobotLogState.offset = data.next_offset;
      }
      if (data.content) {
        appendSceneRobotLog(data.content);
      }
      document.getElementById("sceneRobotLogStatus").textContent =
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
          id: o?.id,
          source: o?.source
        }));
      } else if (json?.obj && typeof json.obj === "object") {
        objects = Object.entries(json.obj).map(([path, meta]) => ({
          path,
          class_name: meta?.class || meta?.class_name || meta?.caption,
          id: meta?.id,
          source: meta?.source
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
        const objObjEdges = Array.isArray(json.edges["obj-obj"])
          ? json.edges["obj-obj"].map((e) => ({
              ...e,
              kind: "obj-obj",
              source: resolveEndpoint(e?.source),
              target: resolveEndpoint(e?.target),
            }))
          : [];
        const objWallEdges = Array.isArray(json.edges["obj-wall"])
          ? json.edges["obj-wall"].map((e) => ({
              ...e,
              kind: "obj-wall",
              source: resolveEndpoint(e?.source),
              target: e?.target || `__wall_anchor__:${String(e?.relation || "").replace(/\s+/g, "_")}`,
            }))
          : [];
        edges = [...objObjEdges, ...objWallEdges];
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
        await ensureRuntimeContext();
        const res = await fetch(withRuntimeQuery("/scene_graph"));
        const graph = await res.json();
        if (!res.ok) throw new Error(graph?.error || "Failed to load scene graph");
        return analyzeSceneJson(graph);
      } catch (err) {
        console.warn("Fallback analysis failed:", err);
        return null;
      }
    }
