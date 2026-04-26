    (async function bootSceneUi() {
      document.getElementById("sceneInput").addEventListener("input", updateInputMeta);
      document.getElementById("imageInput").addEventListener("change", () => {
        updateInputMeta();
        updateReferenceImagePreview();
      });
      document.getElementById("classDirPicker").addEventListener("change", updateInputMeta);

      document.getElementById("sceneInput").addEventListener("keydown", (evt) => {
        if (evt.key === "Enter" && !evt.shiftKey) {
          evt.preventDefault();
          applyInstruction();
        }
      });

      updateInputMeta();
      setPill("model","", "Idle");
      setPill("graph","", "Idle");
      setPill("sim","", "Idle");
      setResampleMode("joint");
      resetSimProgress();
      resetSceneDebug();
      clearReferenceImagePreview();
      resetAgentPanel();
      refreshRepeatPlanButton();
      renderTemplateChips();

      const tplNameInput = document.getElementById("savePlanTemplateNameInput");
      if (tplNameInput) {
        tplNameInput.addEventListener("keydown", (evt) => {
          if (evt.key === "Enter") {
            evt.preventDefault();
            confirmSaveTemplate();
          } else if (evt.key === "Escape") {
            evt.preventDefault();
            cancelSaveTemplate();
          }
        });
      }
      const tplPromptInput = document.getElementById("savePlanTemplatePromptInput");
      if (tplPromptInput) {
        tplPromptInput.addEventListener("keydown", (evt) => {
          if (evt.key === "Escape") {
            evt.preventDefault();
            cancelSaveTemplate();
          }
        });
      }

      try {
        await ensureRuntimeContext();
        refreshRuntimeRenderImage();
        const restored = await restoreAgentState();
        if (!restored?.scene_graph) {
          await loadSceneGraph();
        }
      } catch (err) {
        console.error(err);
        toast("err", "Session init failed", String(err));
      }
    })();
