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

    const PLAN_TOOL_NAMES = [
      "inspect_state",
      "create_scene_graph",
      "run_real2sim",
      "generate_scene",
      "run_scene_robot_collect",
    ];
    const PLAN_STATUS_VISUAL = {
      pending: { glyph: "·", color: "#7d8696", weight: "normal" },
      ok: { glyph: "✓", color: "#1f8b4c", weight: "bold" },
      running: { glyph: "↻", color: "#2266cc", weight: "bold" },
      failed: { glyph: "✗", color: "#c0392b", weight: "bold" },
    };

    function planStepVisualStatus(step) {
      const stepStatus = step?.status || "pending";
      if (stepStatus === "failed") return "failed";
      if (stepStatus !== "ok") return "pending";

      if (step?.result?.job_id) {
        const live = planStepLiveStatus(step);
        if (live === "succeeded") return "ok";
        if (live === "failed") return "failed";
        if (live === "running" || live === "queued") return "running";
        // No live data — fall back to the snapshot taken at execute time.
        return step.result?.status === "running" ? "running" : "ok";
      }
      return "ok";
    }

    function buildPlanStepRowDom(step, options = {}) {
      const li = document.createElement("li");
      const visual = planStepVisualStatus(step);
      const meta = PLAN_STATUS_VISUAL[visual];
      li.style.color = meta.color;
      li.style.lineHeight = "1.5";
      li.style.transition = "color 0.4s ease";
      if (step?.id) li.dataset.stepId = step.id;
      li.dataset.visualStatus = visual;

      const badge = document.createElement("span");
      badge.className = "plan-step-badge";
      badge.textContent = meta.glyph;
      badge.style.cssText = `display:inline-block; min-width:14px; font-weight:${meta.weight}; margin-right:6px; transition: color 0.4s ease;`;
      li.appendChild(badge);

      const argsPretty = step.args ? JSON.stringify(step.args) : "{}";
      const resultPart = step?.result?.error
        ? ` | error: ${step.result.error}`
        : step?.result?.job_id
          ? ` | job ${String(step.result.job_id).slice(0, 8)} (${step.result.status || "?"})`
          : "";
      const showWhy = options.includeWhy !== false;
      const text = document.createElement("span");
      text.className = "plan-step-text";
      text.textContent = `${step.tool}${argsPretty === "{}" ? "" : " " + argsPretty}${showWhy ? " — " + (step.why || "") : ""}${resultPart}`;
      li.appendChild(text);
      return li;
    }

    function updatePlanStepVisualsInPlace() {
      const plan = planEditState.planSnapshot;
      if (!plan) return false;
      const stepsList = document.getElementById("planStepsList");
      if (!stepsList) return false;
      const steps = Array.isArray(plan.steps) ? plan.steps : [];
      const rows = stepsList.querySelectorAll("li[data-step-id]");
      if (rows.length !== steps.length) return false;
      const pendingStart = Number(plan.current_step_index || 0);
      for (let i = 0; i < steps.length; i++) {
        const li = rows[i];
        const step = steps[i];
        if (li.dataset.stepId !== step.id) return false;
        const annotated = step.status
          ? step
          : { ...step, status: i < pendingStart ? "ok" : "pending" };
        const visual = planStepVisualStatus(annotated);
        if (li.dataset.visualStatus === visual) continue;
        const meta = PLAN_STATUS_VISUAL[visual];
        li.style.color = meta.color;
        const badge = li.querySelector(".plan-step-badge");
        if (badge) {
          badge.textContent = meta.glyph;
          badge.style.fontWeight = meta.weight;
        }
        const textSpan = li.querySelector(".plan-step-text");
        if (textSpan) {
          const argsPretty = annotated.args ? JSON.stringify(annotated.args) : "{}";
          const resultPart = annotated?.result?.error
            ? ` | error: ${annotated.result.error}`
            : annotated?.result?.job_id
              ? ` | job ${String(annotated.result.job_id).slice(0, 8)} (${annotated.result.status || "?"})`
              : "";
          textSpan.textContent = `${annotated.tool}${argsPretty === "{}" ? "" : " " + argsPretty} — ${annotated.why || ""}${resultPart}`;
        }
        li.dataset.visualStatus = visual;
      }
      return true;
    }
    const planEditState = {
      editing: false,
      target: "plan",
      draftSteps: [],
      pendingStartIndex: 0,
      planSnapshot: null,
      lastReflection: null,
    };

    function renderPlanStepRow(step, idx, isEditable) {
      const li = document.createElement("li");
      li.dataset.idx = String(idx);

      if (planEditState.editing && isEditable) {
        li.style.cssText = "margin-bottom:8px; list-style:none;";
        const draft = planEditState.draftSteps[idx - planEditState.pendingStartIndex] || step;

        const row = document.createElement("div");
        row.style.cssText = "display:flex; flex-direction:column; gap:4px; padding:6px; border:1px dashed #ccc; border-radius:4px;";

        const topRow = document.createElement("div");
        topRow.style.cssText = "display:flex; gap:6px; align-items:center;";

        const toolSelect = document.createElement("select");
        PLAN_TOOL_NAMES.forEach((name) => {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          if (draft.tool === name) opt.selected = true;
          toolSelect.appendChild(opt);
        });
        toolSelect.style.cssText = "min-width:180px;";
        toolSelect.dataset.field = "tool";
        toolSelect.addEventListener("change", () => updateDraftStep(idx, "tool", toolSelect.value));
        topRow.appendChild(toolSelect);

        const upBtn = document.createElement("button");
        upBtn.textContent = "↑";
        upBtn.className = "toolbtn";
        upBtn.disabled = idx <= planEditState.pendingStartIndex;
        upBtn.addEventListener("click", () => movePlanStep(idx, -1));
        topRow.appendChild(upBtn);

        const downBtn = document.createElement("button");
        downBtn.textContent = "↓";
        downBtn.className = "toolbtn";
        downBtn.disabled = idx >= planEditState.pendingStartIndex + planEditState.draftSteps.length - 1;
        downBtn.addEventListener("click", () => movePlanStep(idx, 1));
        topRow.appendChild(downBtn);

        const delBtn = document.createElement("button");
        delBtn.textContent = "×";
        delBtn.className = "toolbtn";
        delBtn.title = "Delete step";
        delBtn.addEventListener("click", () => removePlanStep(idx));
        topRow.appendChild(delBtn);

        row.appendChild(topRow);

        const whyInput = document.createElement("input");
        whyInput.type = "text";
        whyInput.value = draft.why || "";
        whyInput.placeholder = "why (one sentence)";
        whyInput.style.cssText = "width:100%; padding:4px 6px;";
        whyInput.addEventListener("input", () => updateDraftStep(idx, "why", whyInput.value));
        row.appendChild(whyInput);

        const argsArea = document.createElement("textarea");
        argsArea.rows = 2;
        argsArea.value = JSON.stringify(draft.args || {}, null, 2);
        argsArea.placeholder = '{"key": "value"}';
        argsArea.style.cssText = "width:100%; font-family:monospace; font-size:12px; padding:4px 6px;";
        argsArea.addEventListener("input", () => updateDraftStep(idx, "args_text", argsArea.value));
        row.appendChild(argsArea);

        li.appendChild(row);
      } else {
        const annotated = step.status
          ? step
          : { ...step, status: idx < (planEditState.planSnapshot?.current_step_index || 0) ? "ok" : "pending" };
        return buildPlanStepRowDom(annotated);
      }
      return li;
    }

    function renderPlanPanel(plan, reflection) {
      const panel = document.getElementById("planPanel");
      if (!panel) return;
      if (!plan || (plan.status === "cancelled" && (!plan.steps || plan.steps.length === 0))) {
        panel.style.display = "none";
        planEditState.editing = false;
        planEditState.planSnapshot = null;
        return;
      }
      panel.style.display = "block";
      planEditState.planSnapshot = plan;
      planEditState.lastReflection = reflection || null;
      const goalEl = document.getElementById("planGoalText");
      const statusBadge = document.getElementById("planStatusBadge");
      const stepsList = document.getElementById("planStepsList");
      const reflectionBox = document.getElementById("planReflectionBox");
      const editBtn = document.getElementById("planEditBtn");
      const editControls = document.getElementById("planEditControls");
      const runBtn = document.getElementById("planRunBtn");
      const cancelBtn = document.getElementById("planCancelBtn");

      goalEl.textContent = plan.goal || "(no goal)";
      const status = String(plan.status || "");
      statusBadge.textContent = status || "idle";

      const steps = Array.isArray(plan.steps) ? plan.steps : [];
      const pendingStart = Number(plan.current_step_index || 0);
      const editingPlan = planEditState.editing && planEditState.target === "plan";
      const editingFollowUp = planEditState.editing && planEditState.target === "follow_up";
      if (editingPlan) {
        planEditState.pendingStartIndex = pendingStart;
      }

      stepsList.innerHTML = "";
      steps.forEach((step, idx) => {
        const isEditable = editingPlan && idx >= pendingStart;
        stepsList.appendChild(renderPlanStepRow(step, idx, isEditable));
      });

      const planEditable = ["proposed", "needs_user", "paused"].includes(status);
      editBtn.disabled = !planEditable || editingFollowUp;
      editBtn.style.display = editingPlan ? "none" : "inline-block";
      editControls.style.display = editingPlan ? "inline-flex" : "none";

      const isReviewable = status === "proposed" || status === "needs_user";
      const isResumable = status === "paused";
      runBtn.disabled = planEditState.editing || !(isReviewable || isResumable);
      runBtn.textContent = isResumable ? "Continue Plan" : "Run Plan";
      cancelBtn.disabled = status === "cancelled" || status === "completed" || status === "failed";

      if (reflection && (reflection.summary || reflection.next_action)) {
        reflectionBox.style.display = "block";
        const lines = [];
        if (reflection.summary) lines.push("Reflection: " + reflection.summary);
        if (reflection.next_action) lines.push("Next: " + reflection.next_action);
        if (reflection.ask_user) lines.push("Question: " + reflection.ask_user);
        reflectionBox.textContent = lines.join("\n");
      } else {
        reflectionBox.style.display = "none";
        reflectionBox.textContent = "";
      }

      const followUpBox = document.getElementById("planFollowUpBox");
      const followUpList = document.getElementById("planFollowUpList");
      const followUpCount = document.getElementById("planFollowUpCount");
      const followUpControls = document.getElementById("planFollowUpControls");
      const followUpEditControls = document.getElementById("planFollowUpEditControls");
      const followUpEditBtn = document.getElementById("planFollowUpEditBtn");
      const followUpClearBtn = document.getElementById("planFollowUpClearBtn");
      const followUpAcceptBtn = document.getElementById("planAcceptFollowUpBtn");
      const followUpSteps = Array.isArray(plan.follow_up_plan) ? plan.follow_up_plan : [];

      const showFollowUp = editingFollowUp || (!editingPlan && followUpSteps.length > 0);
      if (showFollowUp) {
        followUpBox.style.display = "block";
        const renderedSteps = editingFollowUp
          ? planEditState.draftSteps.map((d) => ({ tool: d.tool, args: d.args, why: d.why }))
          : followUpSteps;
        followUpCount.textContent = `(${renderedSteps.length} step(s))`;
        followUpList.innerHTML = "";
        if (editingFollowUp) {
          planEditState.pendingStartIndex = 0;
          renderedSteps.forEach((step, idx) => {
            followUpList.appendChild(renderPlanStepRow(step, idx, true));
          });
        } else {
          renderedSteps.forEach((step) => {
            followUpList.appendChild(buildPlanStepRowDom({ ...step, status: step.status || "pending" }));
          });
        }
        followUpControls.style.display = editingFollowUp ? "none" : "flex";
        followUpEditControls.style.display = editingFollowUp ? "flex" : "none";
        followUpEditBtn.disabled = editingPlan || followUpSteps.length === 0;
        followUpClearBtn.disabled = editingPlan || followUpSteps.length === 0;
        followUpAcceptBtn.disabled = editingPlan || followUpSteps.length === 0;
      } else {
        followUpBox.style.display = "none";
        followUpList.innerHTML = "";
      }
    }

    function renderPlanHistoryEntry(plan) {
      const wrapper = document.createElement("div");
      wrapper.style.cssText = "border-bottom:1px solid #e6e6e6; padding:6px 0;";

      const header = document.createElement("div");
      header.style.cssText = "display:flex; gap:8px; align-items:center; cursor:pointer;";
      const chevron = document.createElement("span");
      chevron.textContent = "▸";
      chevron.style.cssText = "width:14px; display:inline-block;";
      header.appendChild(chevron);

      const summary = document.createElement("div");
      summary.style.cssText = "flex:1; font-size:13px;";
      const stepCount = Array.isArray(plan.steps) ? plan.steps.length : 0;
      const ts = (plan.archived_at || plan.updated_at || plan.created_at || "").replace("T", " ").replace("+00:00", "Z");
      const goal = (plan.goal || "(no goal)").slice(0, 80);
      summary.textContent = `[${plan.status || "?"}] ${goal} — ${stepCount} step(s) — ${ts}`;
      header.appendChild(summary);

      if (plan.derived_from) {
        const derivedBadge = document.createElement("span");
        derivedBadge.className = "hint";
        derivedBadge.textContent = `↳ from ${String(plan.derived_from).slice(0, 12)}`;
        header.appendChild(derivedBadge);
      }

      const detail = document.createElement("div");
      detail.style.cssText = "display:none; margin-top:4px; padding:0 0 0 14px; font-family:monospace; font-size:12px;";
      const stepRows = plan.steps || [];
      if (stepRows.length === 0) {
        detail.textContent = "(no steps)";
      } else {
        const ul = document.createElement("ul");
        ul.style.cssText = "margin:0; padding-left:14px;";
        stepRows.forEach((step) => {
          ul.appendChild(buildPlanStepRowDom({ ...step, status: step.status || "pending" }));
        });
        detail.appendChild(ul);
      }

      header.addEventListener("click", () => {
        const open = detail.style.display !== "none";
        detail.style.display = open ? "none" : "block";
        chevron.textContent = open ? "▸" : "▾";
      });

      wrapper.appendChild(header);
      wrapper.appendChild(detail);
      return wrapper;
    }

    function renderJobAuditList() {
      const list = document.getElementById("planJobAuditList");
      const count = document.getElementById("planJobAuditCount");
      if (!list || !count) return;
      const audit = (latestSessionState && latestSessionState.job_audit) || {};
      const entries = Object.keys(audit).map((jobId) => ({
        job_id: jobId,
        ...audit[jobId],
      }));
      // Newest first by updated_at (or first_seen_at fallback)
      entries.sort((a, b) => {
        const ta = a.updated_at || a.first_seen_at || "";
        const tb = b.updated_at || b.first_seen_at || "";
        return tb.localeCompare(ta);
      });
      count.textContent = `(${entries.length})`;
      list.innerHTML = "";
      if (entries.length === 0) {
        const empty = document.createElement("div");
        empty.className = "hint";
        empty.textContent = "No jobs recorded yet.";
        list.appendChild(empty);
        return;
      }
      entries.forEach((entry) => {
        const row = document.createElement("div");
        row.style.cssText =
          "display:flex; align-items:center; gap:8px; padding:4px 0; border-bottom:1px solid #eee; font-size:13px;";

        const visualKey = entry.status === "succeeded" ? "ok" : (entry.status === "failed" ? "failed" : (entry.status === "running" || entry.status === "queued") ? "running" : "pending");
        const meta = PLAN_STATUS_VISUAL[visualKey];
        const badge = document.createElement("span");
        badge.textContent = meta.glyph;
        badge.style.cssText = `min-width:14px; font-weight:${meta.weight}; color:${meta.color};`;
        row.appendChild(badge);

        const kindBadge = document.createElement("span");
        kindBadge.textContent = entry.kind || "?";
        kindBadge.style.cssText =
          "padding:1px 6px; border-radius:8px; background:#eef2f7; font-size:11px; min-width:80px; text-align:center;";
        row.appendChild(kindBadge);

        const idLabel = document.createElement("span");
        idLabel.textContent = String(entry.job_id || "").slice(0, 12);
        idLabel.style.cssText = "font-family:monospace; font-size:12px; min-width:100px;";
        idLabel.title = entry.job_id;
        row.appendChild(idLabel);

        const statusText = document.createElement("span");
        statusText.textContent = entry.status || "?";
        statusText.style.cssText = `color:${meta.color}; font-weight:${meta.weight}; min-width:80px;`;
        row.appendChild(statusText);

        const ts = document.createElement("span");
        const tsValue = (entry.updated_at || entry.first_seen_at || "").replace("T", " ").replace("+00:00", "Z");
        ts.textContent = tsValue;
        ts.className = "hint";
        ts.style.cssText = "flex:1; text-align:right; font-size:11px;";
        row.appendChild(ts);

        if (entry.error) {
          const errLine = document.createElement("div");
          errLine.textContent = `error: ${entry.error}`;
          errLine.style.cssText = "padding:2px 0 4px 22px; color:#c0392b; font-size:11px; width:100%;";
          const wrapper = document.createElement("div");
          wrapper.style.cssText = "display:flex; flex-direction:column; border-bottom:1px solid #eee;";
          wrapper.appendChild(row);
          row.style.borderBottom = "none";
          wrapper.appendChild(errLine);
          list.appendChild(wrapper);
        } else {
          list.appendChild(row);
        }
      });
    }

    async function togglePlanHistory() {
      const box = document.getElementById("planHistoryBox");
      const isOpen = box.style.display !== "none";
      if (isOpen) {
        box.style.display = "none";
        return;
      }
      try {
        await ensureRuntimeContext();
        const res = await fetch(withRuntimeQuery("/agent/plan/history"));
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.error || "Failed to load plan history");
        }
        refreshSessionStateCache(data?.session_state);
        const plans = Array.isArray(data?.plans) ? data.plans : [];
        const list = document.getElementById("planHistoryList");
        const count = document.getElementById("planHistoryCount");
        list.innerHTML = "";
        count.textContent = `(${plans.length})`;
        if (plans.length === 0) {
          const empty = document.createElement("div");
          empty.className = "hint";
          empty.textContent = "No archived plans yet.";
          list.appendChild(empty);
        } else {
          plans.forEach((plan) => list.appendChild(renderPlanHistoryEntry(plan)));
        }
        renderJobAuditList();
        box.style.display = "block";
      } catch (err) {
        toast("err", "History fetch failed", String(err.message || err));
      }
    }

    async function clearJobAudit() {
      if (!window.confirm("Clear all recorded job statuses? This does not affect plans or active jobs.")) return;
      try {
        await ensureRuntimeContext();
        const res = await fetch(withRuntimeQuery("/agent/job/audit/clear"), { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.error || "Clear job audit failed");
        }
        refreshSessionStateCache(data?.session_state);
        renderJobAuditList();
        toast("ok", "Job audit cleared", "Recorded job statuses have been wiped.");
      } catch (err) {
        toast("err", "Clear failed", String(err.message || err));
      }
    }

    async function clearPlanFollowUp() {
      if (!window.confirm("Clear the suggested follow-up plan?")) return;
      const btn = document.getElementById("planFollowUpClearBtn");
      btn.disabled = true;
      try {
        await ensureRuntimeContext();
        const res = await fetch(withRuntimeQuery("/agent/plan/follow_up/update"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ steps: [] }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.error || "Clear follow-up failed");
        }
        refreshSessionStateCache(data?.session_state);
        renderPlanPanel(data?.active_plan, null);
        toast("ok", "Follow-up cleared", "Reflector suggestion dismissed.");
      } catch (err) {
        toast("err", "Clear follow-up failed", String(err.message || err));
        btn.disabled = false;
      }
    }

    async function acceptPlanFollowUp() {
      const btn = document.getElementById("planAcceptFollowUpBtn");
      btn.disabled = true;
      try {
        await ensureRuntimeContext();
        const res = await fetch(withRuntimeQuery("/agent/plan/accept_follow_up"), {
          method: "POST",
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.error || "Accept follow-up failed");
        }
        refreshSessionStateCache(data?.session_state);
        renderPlanPanel(data?.active_plan, null);
        toast("ok", "Follow-up accepted", "New plan ready for review.");
      } catch (err) {
        toast("err", "Accept follow-up failed", String(err.message || err));
        btn.disabled = false;
      }
    }

    function rerenderActivePlanIfShown() {
      if (planEditState.planSnapshot && !planEditState.editing) {
        // Try in-place visual update first so CSS color transitions can
        // animate. Fall back to wholesale rebuild if step structure has
        // changed (e.g. user accepted a follow-up between polls).
        if (!updatePlanStepVisualsInPlace()) {
          renderPlanPanel(planEditState.planSnapshot, planEditState.lastReflection);
        }
      }
      const historyBox = document.getElementById("planHistoryBox");
      if (historyBox && historyBox.style.display !== "none") {
        renderJobAuditList();
      }
    }

    function enterPlanEditMode(target = "plan") {
      const plan = planEditState.planSnapshot;
      if (!plan) return;
      let baseSteps;
      let pendingStart;
      if (target === "follow_up") {
        baseSteps = Array.isArray(plan.follow_up_plan) ? plan.follow_up_plan : [];
        pendingStart = 0;
      } else {
        pendingStart = Number(plan.current_step_index || 0);
        baseSteps = (plan.steps || []).slice(pendingStart);
      }
      planEditState.target = target;
      planEditState.draftSteps = baseSteps.map((s) => ({
        tool: s.tool,
        why: s.why || "",
        args: s.args || {},
        args_text: JSON.stringify(s.args || {}, null, 2),
      }));
      planEditState.pendingStartIndex = pendingStart;
      planEditState.editing = true;
      renderPlanPanel(plan, null);
    }

    function discardPlanEdits() {
      planEditState.editing = false;
      planEditState.target = "plan";
      planEditState.draftSteps = [];
      renderPlanPanel(planEditState.planSnapshot, null);
    }

    function updateDraftStep(globalIdx, field, value) {
      const localIdx = globalIdx - planEditState.pendingStartIndex;
      const draft = planEditState.draftSteps[localIdx];
      if (!draft) return;
      if (field === "args_text") {
        draft.args_text = value;
        try {
          const parsed = JSON.parse(value || "{}");
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            draft.args = parsed;
          }
        } catch (err) {
          // Wait until valid JSON; keep last good draft.args
        }
      } else {
        draft[field] = value;
      }
    }

    function movePlanStep(globalIdx, delta) {
      const localIdx = globalIdx - planEditState.pendingStartIndex;
      const newLocal = localIdx + delta;
      if (newLocal < 0 || newLocal >= planEditState.draftSteps.length) return;
      const arr = planEditState.draftSteps;
      [arr[localIdx], arr[newLocal]] = [arr[newLocal], arr[localIdx]];
      renderPlanPanel(planEditState.planSnapshot, null);
    }

    function removePlanStep(globalIdx) {
      const localIdx = globalIdx - planEditState.pendingStartIndex;
      planEditState.draftSteps.splice(localIdx, 1);
      renderPlanPanel(planEditState.planSnapshot, null);
    }

    function addPlanStep() {
      planEditState.draftSteps.push({
        tool: "inspect_state",
        why: "",
        args: {},
        args_text: "{}",
      });
      renderPlanPanel(planEditState.planSnapshot, null);
    }

    async function savePlanEdits() {
      const editingFollowUp = planEditState.target === "follow_up";
      if (!planEditState.draftSteps.length && !editingFollowUp) {
        toast("warn", "Empty plan", "Add at least one step before saving.");
        return;
      }
      const stepsPayload = [];
      for (let i = 0; i < planEditState.draftSteps.length; i += 1) {
        const draft = planEditState.draftSteps[i];
        let args = draft.args;
        if (typeof draft.args_text === "string") {
          try {
            const parsed = JSON.parse(draft.args_text || "{}");
            if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
              args = parsed;
            } else {
              throw new Error("args must be a JSON object");
            }
          } catch (err) {
            toast("err", "Invalid args JSON", `Step ${i + 1}: ${err.message || err}`);
            return;
          }
        }
        stepsPayload.push({
          tool: draft.tool,
          args,
          why: draft.why || "",
        });
      }
      try {
        await ensureRuntimeContext();
        const url = editingFollowUp ? "/agent/plan/follow_up/update" : "/agent/plan/update";
        const res = await fetch(withRuntimeQuery(url), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ steps: stepsPayload }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.error || "Plan update failed");
        }
        refreshSessionStateCache(data?.session_state);
        planEditState.editing = false;
        planEditState.target = "plan";
        planEditState.draftSteps = [];
        renderPlanPanel(data?.active_plan, data?.reflection || null);
        toast(
          "ok",
          editingFollowUp ? "Follow-up updated" : "Plan updated",
          editingFollowUp
            ? `Saved ${stepsPayload.length} follow-up step(s).`
            : `Saved ${stepsPayload.length} step(s).`,
        );
      } catch (err) {
        toast("err", "Plan update failed", String(err.message || err));
      }
    }

    async function executeAgentPlan() {
      const runBtn = document.getElementById("planRunBtn");
      runBtn.disabled = true;
      try {
        await ensureRuntimeContext();
        const response = await fetch(withRuntimeQuery("/agent/plan/execute"), {
          method: "POST",
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data?.error || "Plan execute failed");
        }
        await handleAgentResponse(data, { hadReferenceImage: false });
      } catch (err) {
        toast("err", "Plan execute failed", String(err.message || err));
      } finally {
        runBtn.disabled = false;
      }
    }

    async function cancelAgentPlan() {
      try {
        await ensureRuntimeContext();
        const response = await fetch(withRuntimeQuery("/agent/plan/cancel"), {
          method: "POST",
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data?.error || "Plan cancel failed");
        }
        refreshSessionStateCache(data?.session_state);
        renderPlanPanel(data?.active_plan, null);
      } catch (err) {
        toast("err", "Plan cancel failed", String(err.message || err));
      }
    }

    async function handleAgentResponse(data, options = {}) {
      if (data?.session_id || data?.run_id) {
        applyRuntimeContext({
          session_id: data.session_id,
          run_id: data.run_id,
        });
      }
      refreshSessionStateCache(data?.session_state);

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

      // Surface fresh failures as a structured chat card with action buttons.
      // Dedupes inside appendAgentFailureBubble; safe to call on every turn.
      _maybeAppendFailureBubbleFromPayload(data);

      if (data?.active_plan || data?.reflection) {
        renderPlanPanel(data.active_plan, data.reflection);
      }

      const real2simJobInfo = getReal2SimJobInfoFromPayload(data);
      const shouldRefreshMaskReview =
        !!real2simJobInfo?.job_id ||
        !!getReal2SimArtifactsFromPayload(data) ||
        !!data?.job?.artifacts?.assignment_json_url ||
        !!data?.session_state?.current_run?.real2sim?.artifacts?.assignment_json_url ||
        data?.error_info?.code === "mask_assignment_failed";
      if (shouldRefreshMaskReview || real2simJobInfo?.job_id) {
        try {
          const livePreview = await hydrateReal2SimLivePreview(data, {
            refreshReview: shouldRefreshMaskReview,
            silentReview: true,
            autoReview: true,
          });
          const review = livePreview?.review || null;
          const real2simState = getReal2SimStateFromPayload(data);
          const real2simStatus = String(real2simState?.status || "");
          if (real2simStatus === "queued" || real2simStatus === "running") {
            setPill("sim", "warn", "Real2Sim");
            document.getElementById("sceneSvcStatus").textContent = `Job ${String(real2simState?.job_id || "").slice(0, 8)} running`;
          } else if (real2simStatus === "succeeded") {
            setPill("sim", "ok", "Ready");
          } else if (real2simStatus === "failed") {
            setPill("sim", "err", "Failed");
          }
          if (review?.needs_attention) {
            feedbackLines.push("Mask assignment still needs confirmation before you trust downstream layout steps.");
          }
        } catch (reviewErr) {
          console.warn("Real2Sim live preview restore unavailable:", reviewErr);
        }
      }

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

      if (real2simJobInfo?.job_id) {
        await monitorReal2SimJob(real2simJobInfo);
      }

      const sceneRobotJobInfo = getSceneRobotJobInfoFromPayload(data);
      if (sceneRobotJobInfo?.job_id) {
        await monitorSceneRobotJob(sceneRobotJobInfo);
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

      // Step 3: surface tool calls used in this turn under the latest agent
      // bubble (or the failure card if that's the most recent). Backend
      // already records `tool_steps: [{tool, args, summary}]` in every
      // /agent/message response.
      if (typeof appendToolStepsToLatestAssistantBubble === "function" && Array.isArray(data?.tool_steps) && data.tool_steps.length) {
        appendToolStepsToLatestAssistantBubble(data.tool_steps);
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
      const selectedMode = document.querySelector('input[name="agentMode"]:checked');
      const modeValue = selectedMode ? selectedMode.value : "loop";
      formData.append("mode", modeValue);
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
      const rawText = await response.text();
      let data;
      try {
        data = rawText ? JSON.parse(rawText) : {};
      } catch (parseErr) {
        const snippet = (rawText || "(empty body)").slice(0, 800);
        throw new Error(
          `HTTP ${response.status} ${response.statusText} — non-JSON response from /agent/message:\n${snippet}`
        );
      }
      if (!response.ok) {
        throw new Error(
          data?.error || `HTTP ${response.status} ${response.statusText}`
        );
      }

      if (data?.session_id || data?.run_id) {
        applyRuntimeContext({
          session_id: data.session_id,
          run_id: data.run_id,
        });
      }
      preview.textContent = JSON.stringify(data, null, 2);

      const proposedPlan = data?.active_plan;
      if (
        modeValue === "plan" &&
        instruction &&
        proposedPlan &&
        Array.isArray(proposedPlan.steps) &&
        proposedPlan.steps.length > 0
      ) {
        saveRecentPlanPrompt(instruction);
        refreshRepeatPlanButton();
      }

      return handleAgentResponse(data, { hadReferenceImage });
    }

    function refreshRepeatPlanButton() {
      const btn = document.getElementById("repeatPlanPromptBtn");
      if (!btn) return;
      const last = getLastPlanPrompt();
      btn.disabled = !last;
      if (last) {
        const preview = last.prompt.length > 60 ? last.prompt.slice(0, 60) + "…" : last.prompt;
        btn.title = `Repeat: ${preview}`;
      } else {
        btn.title = "No saved plan prompt yet — propose a plan first.";
      }
    }

    function renderTemplateChips() {
      const wrap = document.getElementById("planTemplatesChips");
      if (!wrap) return;
      const templates = getNamedTemplates();
      wrap.innerHTML = "";
      if (templates.length === 0) {
        const empty = document.createElement("span");
        empty.className = "hint";
        empty.textContent = "No saved templates yet.";
        wrap.appendChild(empty);
        return;
      }
      templates.forEach((t) => {
        const chip = document.createElement("span");
        chip.style.cssText =
          "display:inline-flex; align-items:center; gap:4px; padding:2px 6px; margin:2px 4px 2px 0;" +
          "border:1px solid #b6c4d6; border-radius:12px; background:#f5f8fc; font-size:12px;";
        chip.title = t.prompt;

        const name = document.createElement("span");
        name.textContent = t.name;
        name.style.cssText = "cursor:pointer;";
        name.addEventListener("click", () => runNamedTemplate(t.name));
        chip.appendChild(name);

        const edit = document.createElement("span");
        edit.textContent = "✎";
        edit.style.cssText = "cursor:pointer; color:#1f6feb; padding:0 2px;";
        edit.title = "Edit template";
        edit.addEventListener("click", (evt) => {
          evt.stopPropagation();
          editNamedTemplateInline(t.name);
        });
        chip.appendChild(edit);

        const del = document.createElement("span");
        del.textContent = "×";
        del.style.cssText = "cursor:pointer; color:#c0392b; font-weight:bold; padding:0 2px;";
        del.title = "Delete template";
        del.addEventListener("click", (evt) => {
          evt.stopPropagation();
          deleteNamedTemplateAndRerender(t.name);
        });
        chip.appendChild(del);

        wrap.appendChild(chip);
      });
    }

    const templateFormState = {
      mode: "create",     // "create" | "edit"
      originalName: null,
    };

    function _openTemplateForm({ mode, name, prompt }) {
      const form = document.getElementById("savePlanTemplateForm");
      const nameInput = document.getElementById("savePlanTemplateNameInput");
      const promptInput = document.getElementById("savePlanTemplatePromptInput");
      const modeLabel = document.getElementById("savePlanTemplateModeLabel");
      if (!form || !nameInput || !promptInput || !modeLabel) return;
      templateFormState.mode = mode;
      templateFormState.originalName = mode === "edit" ? name : null;
      modeLabel.textContent = mode === "edit" ? `Edit template "${name}"` : "Save as new template";
      nameInput.value = name || "";
      if (mode === "edit") {
        promptInput.style.display = "block";
        promptInput.value = prompt || "";
      } else {
        promptInput.style.display = "none";
        promptInput.value = "";
      }
      form.style.display = "block";
      nameInput.focus();
      nameInput.select?.();
    }

    function saveCurrentAsTemplate() {
      const input = document.getElementById("sceneInput");
      const prompt = (input?.value || "").trim();
      if (!prompt) {
        toast("warn", "Empty prompt", "Type a prompt in the box first, then save it as a template.");
        return;
      }
      _openTemplateForm({ mode: "create", name: "", prompt });
    }

    function editNamedTemplateInline(name) {
      const tpl = findNamedTemplate(name);
      if (!tpl) {
        toast("err", "Template missing", `Template "${name}" not found.`);
        renderTemplateChips();
        return;
      }
      _openTemplateForm({ mode: "edit", name: tpl.name, prompt: tpl.prompt });
    }

    function cancelSaveTemplate() {
      const form = document.getElementById("savePlanTemplateForm");
      if (form) form.style.display = "none";
      templateFormState.mode = "create";
      templateFormState.originalName = null;
    }

    function confirmSaveTemplate() {
      const nameInput = document.getElementById("savePlanTemplateNameInput");
      const promptInput = document.getElementById("savePlanTemplatePromptInput");
      const name = (nameInput?.value || "").trim();
      if (!name) {
        nameInput?.focus();
        toast("warn", "Name required", "Template name cannot be empty.");
        return;
      }

      let prompt;
      if (templateFormState.mode === "edit") {
        prompt = (promptInput?.value || "").trim();
        if (!prompt) {
          promptInput?.focus();
          toast("warn", "Empty prompt", "Template prompt cannot be empty.");
          return;
        }
      } else {
        const inputBox = document.getElementById("sceneInput");
        prompt = (inputBox?.value || "").trim();
        if (!prompt) {
          toast("warn", "Empty prompt", "Prompt was cleared while saving — type one and try again.");
          return;
        }
      }

      const original = templateFormState.originalName;
      const isRename = templateFormState.mode === "edit" && original && original !== name;
      const collision = findNamedTemplate(name);
      if (collision && (templateFormState.mode === "create" || isRename)) {
        if (!window.confirm(`Overwrite existing template "${name}"?`)) return;
      }

      if (isRename) {
        deleteNamedTemplate(original);
      }

      if (saveNamedTemplate(name, prompt)) {
        const verb = templateFormState.mode === "edit" ? "updated" : "saved";
        cancelSaveTemplate();
        renderTemplateChips();
        toast("ok", `Template ${verb}`, `"${name}" — click the chip to run it.`);
      } else {
        toast("err", "Save failed", "Could not write to localStorage.");
      }
    }

    function deleteNamedTemplateAndRerender(name) {
      if (!window.confirm(`Delete template "${name}"?`)) return;
      deleteNamedTemplate(name);
      renderTemplateChips();
    }

    async function runNamedTemplate(name) {
      const tpl = findNamedTemplate(name);
      if (!tpl) {
        toast("err", "Template missing", `Template "${name}" not found.`);
        renderTemplateChips();
        return;
      }
      const planRadio = document.getElementById("agentModePlan");
      if (planRadio) planRadio.checked = true;
      const input = document.getElementById("sceneInput");
      if (input) input.value = tpl.prompt;
      try {
        await applyInstruction({ instruction: tpl.prompt });
      } catch (err) {
        toast("err", "Template run failed", String(err.message || err));
      }
    }

    async function repeatLastPlanPrompt() {
      const last = getLastPlanPrompt();
      if (!last) {
        toast("warn", "No saved prompt", "Propose a plan first; the prompt will be remembered.");
        return;
      }
      const planRadio = document.getElementById("agentModePlan");
      if (planRadio) planRadio.checked = true;
      const input = document.getElementById("sceneInput");
      if (input) input.value = last.prompt;
      try {
        await applyInstruction({ instruction: last.prompt });
      } catch (err) {
        toast("err", "Repeat failed", String(err.message || err));
      }
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

    // ---- Step 2: failure bubble + retry plumbing ---------------------------

    function _composeFailureFromSessionState(sessionState, payloadFallback = {}) {
      const r2s = sessionState?.current_run?.real2sim || {};
      const sr  = sessionState?.current_run?.scene_robot || {};
      const r2sFailed = String(r2s.status || "").toLowerCase() === "failed";
      const srFailed  = String(sr.status  || "").toLowerCase() === "failed";
      if (!r2sFailed && !srFailed) return null;
      const kind = r2sFailed ? "real2sim" : "scene_robot";
      const block = r2sFailed ? r2s : sr;
      return {
        kind,
        job_id: block.job_id || "",
        error_info: block.error_info || payloadFallback?.error_info || null,
        log_digest:
          block.log_digest ||
          (block.error_info && block.error_info.log_digest) ||
          (payloadFallback?.error_info && payloadFallback.error_info.log_digest) ||
          null,
        log_path: block.log_path || "",
        user_message:
          (block.error_info && block.error_info.user_message) ||
          payloadFallback?.error_info?.user_message ||
          block.error ||
          "",
      };
    }

    function _maybeAppendFailureBubbleFromPayload(payload) {
      if (typeof appendAgentFailureBubble !== "function") return;
      const sessionState = payload?.session_state || payload?.job?.session_state || null;
      if (!sessionState) return;
      const failure = _composeFailureFromSessionState(sessionState, payload);
      if (failure) appendAgentFailureBubble(failure);
    }

    async function retryFailedStage(kind, failureContext = {}) {
      const sceneInput = document.getElementById("sceneInput");
      const friendly = kind === "scene_robot" ? "scene_robot" : "Real2Sim";
      const code = failureContext?.error_info?.code;
      // Compose a clear instruction the agent can pick up from inspect_state.
      const reasonHint = code ? ` (last failure code: ${code})` : "";
      const promptText =
        `Please retry the previous failed ${friendly} job for the current run` +
        `${reasonHint}. Inspect the current state first; if the underlying issue ` +
        `is still present (e.g. remote service unreachable), tell me so before retrying.`;
      if (sceneInput) {
        sceneInput.value = promptText;
        if (typeof updateInputMeta === "function") updateInputMeta();
      }
      try {
        await applyInstruction({ instruction: promptText, action: "retry_" + (kind || "real2sim") });
      } catch (err) {
        toast("err", "Retry failed", String(err?.message || err));
      }
    }
