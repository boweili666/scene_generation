    /* ===== Wire events ===== */
    document.getElementById("sceneInput").addEventListener("input", updateInputMeta);
    document.getElementById("imageInput").addEventListener("change", () => {
      updateInputMeta();
      updateReferenceImagePreview();
    });
    document.getElementById("classDirPicker").addEventListener("change", updateInputMeta);

    document.getElementById("sceneInput").addEventListener("keydown", (evt) => {
      if (evt.key === "Enter" && !evt.shiftKey) {
        evt.preventDefault();
        generateFromPrompt();
      }
    });

    updateInputMeta();
    setPill("model","", "Idle");
    setPill("graph","", "Idle");
    setPill("sim","", "Idle");
    resetSimProgress();
    clearReferenceImagePreview();
    loadSceneGraph();
