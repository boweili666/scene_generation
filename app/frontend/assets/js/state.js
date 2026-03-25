    let cy = null;

    const interactionState = {
      startedAt: null,
      lastInstruction: "",
      lastJson: null,
      currentSceneGraph: null
    };

    const graphEditorState = {
      selectedElementType: "",
      selectedElementId: "",
      linkSourceId: "",
      saveInFlight: false,
      lastTapKey: "",
      lastTapAt: 0
    };

    const externalScriptState = new Map();

    const viewerState = {
      initialized: false,
      scene: null,
      camera: null,
      controls: null,
      renderer: null,
      resize: null,
      loader: null,
      container: null,
      pmremGenerator: null,
      studioEnvTarget: null,
      stageGrid: null,
      stageFloor: null,
      stageShadow: null,
      spinRoots: [],
      loadedUrls: new Set(),
      loadingUrls: new Set(),
      queuedUrls: new Set(),
      loadQueue: [],
      queueBusy: false,
      loadingMerged: false,
      mergedUrl: null
    };

    const real2simLogState = {
      offset: 0,
      path: "real2sim.log"
    };
    let imagePreviewObjectUrl = null;
    const PREVIEW_FLOOR_Y = -1.2;
    const PREVIEW_FLOOR_CLEARANCE = 0.015;
    const PREVIEW_STAGE_RADIUS = 14;
    const PREVIEW_STAGE_GRID_RADIUS = 13.2;
    const PREVIEW_STAGE_THICKNESS = 0.62;

    function setPreviewMessage(message) {
      const placeholder = document.getElementById("imagePlaceholder");
      placeholder.textContent = message;
      placeholder.style.display = "flex";
      document.getElementById("threeViewport").style.display = "none";
      document.getElementById("renderImage").style.display = "none";
    }

    function ensureExternalScript(src, isReady, timeoutMs = 10000) {
      try {
        if (isReady()) return Promise.resolve();
      } catch (err) {
        console.warn("External script readiness check failed:", src, err);
      }

      if (externalScriptState.has(src)) {
        return externalScriptState.get(src);
      }

      const promise = new Promise((resolve, reject) => {
        const existing = Array.from(document.querySelectorAll("script[src]")).find((el) => el.src === src);
        const script = existing || document.createElement("script");
        let timeoutId = null;

        const cleanup = () => {
          if (timeoutId !== null) {
            clearTimeout(timeoutId);
          }
          script.removeEventListener("load", onLoad);
          script.removeEventListener("error", onError);
        };

        const onLoad = () => {
          cleanup();
          if (isReady()) {
            resolve();
          } else {
            externalScriptState.delete(src);
            reject(new Error(`Script loaded but dependency missing: ${src}`));
          }
        };

        const onError = () => {
          cleanup();
          externalScriptState.delete(src);
          reject(new Error(`Failed to load script: ${src}`));
        };

        timeoutId = window.setTimeout(() => {
          cleanup();
          externalScriptState.delete(src);
          reject(new Error(`Timed out loading script: ${src}`));
        }, timeoutMs);

        script.addEventListener("load", onLoad);
        script.addEventListener("error", onError);

        if (!existing) {
          script.src = src;
          script.async = true;
          document.head.appendChild(script);
        }
      });

      externalScriptState.set(src, promise);
      return promise;
    }

    async function ensureExternalScriptAny(sources, isReady, timeoutMs = 10000) {
      let lastError = null;
      for (const src of sources) {
        try {
          await ensureExternalScript(src, isReady, timeoutMs);
          return;
        } catch (err) {
          lastError = err;
          console.warn("External script source failed:", src, err);
        }
      }
      throw lastError || new Error(`Failed to load dependency from all sources: ${sources.join(", ")}`);
    }

    function ensureCytoscape() {
      return ensureExternalScriptAny([
        "/assets/vendor/cytoscape.min.js",
        "https://cdn.jsdelivr.net/npm/cytoscape/dist/cytoscape.min.js",
        "https://unpkg.com/cytoscape/dist/cytoscape.min.js"
      ],
        () => typeof window.cytoscape === "function"
      );
    }

    async function ensureThreeViewerDeps() {
      await ensureExternalScriptAny([
        "/assets/vendor/three.min.js",
        "https://cdn.jsdelivr.net/npm/three@0.124.0/build/three.min.js",
        "https://unpkg.com/three@0.124.0/build/three.min.js"
      ],
        () => typeof window.THREE !== "undefined"
      );
      await ensureExternalScriptAny([
        "/assets/vendor/GLTFLoader.js",
        "https://cdn.jsdelivr.net/npm/three@0.124.0/examples/js/loaders/GLTFLoader.js",
        "https://unpkg.com/three@0.124.0/examples/js/loaders/GLTFLoader.js"
      ],
        () => typeof window.THREE !== "undefined" && typeof window.THREE.GLTFLoader === "function"
      );
      await ensureExternalScriptAny([
        "/assets/vendor/OrbitControls.js",
        "https://cdn.jsdelivr.net/npm/three@0.124.0/examples/js/controls/OrbitControls.js",
        "https://unpkg.com/three@0.124.0/examples/js/controls/OrbitControls.js"
      ],
        () => typeof window.THREE !== "undefined" && typeof window.THREE.OrbitControls === "function"
      );
    }

    async function initThreeViewer() {
      await ensureThreeViewerDeps();
      if (viewerState.initialized) return;
      const container = document.getElementById("threeViewport");
      const scene = new THREE.Scene();
      scene.background = null;
      scene.fog = new THREE.Fog(0xeaf1fb, 18, 46);

      const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
      camera.position.set(0, 4, 12);
      camera.lookAt(0, 0, 0);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.08;
      renderer.physicallyCorrectLights = true;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      container.appendChild(renderer.domElement);
      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.target.set(0, 0, 0);
      controls.update();

      const hemi = new THREE.HemisphereLight(0xfdfefe, 0xb8c8dc, 1.05);
      hemi.userData.keepPreviewStage = true;
      scene.add(hemi);
      const key = new THREE.DirectionalLight(0xfffaf2, 1.7);
      key.position.set(5.5, 9.5, 6.5);
      key.castShadow = true;
      key.shadow.mapSize.width = 1024;
      key.shadow.mapSize.height = 1024;
      key.shadow.camera.near = 0.5;
      key.shadow.camera.far = 40;
      key.shadow.camera.left = -8;
      key.shadow.camera.right = 8;
      key.shadow.camera.top = 8;
      key.shadow.camera.bottom = -8;
      key.userData.keepPreviewStage = true;
      scene.add(key);
      const fill = new THREE.DirectionalLight(0xe6f0ff, 0.72);
      fill.position.set(-6, 4, 5);
      fill.userData.keepPreviewStage = true;
      scene.add(fill);
      const rim = new THREE.DirectionalLight(0xffffff, 0.58);
      rim.position.set(-4, 6, -7);
      rim.userData.keepPreviewStage = true;
      scene.add(rim);
      const top = new THREE.SpotLight(0xffffff, 0.82, 0, Math.PI / 4, 0.32, 1);
      top.position.set(0, 10.5, 3.5);
      top.target.position.set(0, -0.6, 0);
      top.userData.keepPreviewStage = true;
      top.target.userData.keepPreviewStage = true;
      top.castShadow = true;
      top.shadow.mapSize.width = 1024;
      top.shadow.mapSize.height = 1024;
      scene.add(top);
      scene.add(top.target);

      const pmremGenerator = new THREE.PMREMGenerator(renderer);
      const studioEnvTarget = createStudioEnvironment(pmremGenerator);
      scene.environment = studioEnvTarget.texture;

      const floor = new THREE.Mesh(
        createStagePlatformGeometry(PREVIEW_STAGE_RADIUS, PREVIEW_STAGE_THICKNESS),
        new THREE.MeshPhysicalMaterial({
          color: 0xfbfcfe,
          roughness: 0.72,
          metalness: 0.015,
          clearcoat: 0.08,
          clearcoatRoughness: 0.58,
          envMapIntensity: 0.34,
        })
      );
      floor.position.y = PREVIEW_FLOOR_Y - PREVIEW_STAGE_THICKNESS / 2;
      floor.castShadow = true;
      floor.receiveShadow = true;
      floor.userData.keepPreviewStage = true;
      scene.add(floor);

      const shadowCatcher = new THREE.Mesh(
        new THREE.PlaneGeometry(19, 19),
        new THREE.MeshBasicMaterial({
          color: 0xffffff,
          map: createStageContactShadowTexture(),
          transparent: true,
          opacity: 0.24,
          depthWrite: false,
        })
      );
      shadowCatcher.rotation.x = -Math.PI / 2;
      shadowCatcher.position.y = PREVIEW_FLOOR_Y - PREVIEW_STAGE_THICKNESS - 0.02;
      shadowCatcher.renderOrder = -2;
      shadowCatcher.userData.keepPreviewStage = true;
      scene.add(shadowCatcher);

      const grid = new THREE.Mesh(
        new THREE.CircleGeometry(PREVIEW_STAGE_GRID_RADIUS, 96),
        createFadingGridMaterial()
      );
      grid.rotation.x = -Math.PI / 2;
      grid.position.y = PREVIEW_FLOOR_Y + 0.004;
      grid.renderOrder = 2;
      grid.userData.keepPreviewStage = true;
      scene.add(grid);

      viewerState.initialized = true;
      viewerState.scene = scene;
      viewerState.camera = camera;
      viewerState.controls = controls;
      viewerState.renderer = renderer;
      viewerState.resize = resize;
      viewerState.loader = new THREE.GLTFLoader();
      viewerState.container = container;
      viewerState.pmremGenerator = pmremGenerator;
      viewerState.studioEnvTarget = studioEnvTarget;
      viewerState.stageGrid = grid;
      viewerState.stageFloor = floor;
      viewerState.stageShadow = shadowCatcher;

      function resize() {
        const w = Math.max(container.clientWidth, 10);
        const h = Math.max(container.clientHeight, 10);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
      }
      window.addEventListener("resize", resize);
      resize();

      function animate() {
        requestAnimationFrame(animate);
        controls.update();
        for (const root of viewerState.spinRoots) {
          root.rotation.y += 0.01;
        }
        renderer.render(scene, camera);
      }
      animate();
    }

    async function showThreeViewer() {
      document.getElementById("threeViewport").style.display = "block";
      document.getElementById("renderImage").style.display = "none";
      document.getElementById("imagePlaceholder").style.display = "none";
      await initThreeViewer();
      if (viewerState.resize) {
        viewerState.resize();
        requestAnimationFrame(() => {
          if (viewerState.resize) viewerState.resize();
        });
      }
    }

    function showImagePreview() {
      document.getElementById("threeViewport").style.display = "none";
      document.getElementById("renderImage").style.display = "block";
      document.getElementById("imagePlaceholder").style.display = "none";
    }

    function clearThreeViewer() {
      if (!viewerState.initialized) return;
      const keep = [];
      for (const obj of viewerState.scene.children) {
        if (obj.userData?.keepPreviewStage || obj.type === "HemisphereLight" || obj.type === "DirectionalLight" || obj.type === "SpotLight") {
          keep.push(obj);
        }
      }
      viewerState.scene.clear();
      keep.forEach((x) => viewerState.scene.add(x));
      viewerState.spinRoots = [];
      viewerState.loadedUrls = new Set();
      viewerState.loadingUrls = new Set();
      viewerState.queuedUrls = new Set();
      viewerState.loadQueue = [];
      viewerState.queueBusy = false;
      viewerState.loadingMerged = false;
      viewerState.mergedUrl = null;
      if (viewerState.controls) {
        viewerState.controls.target.set(0, 0, 0);
        viewerState.controls.update();
      }
    }

    function layoutObjectRoot(root, index, totalCount = 1) {
      const cols = Math.min(4, Math.max(1, totalCount));
      const gap = cols === 1 ? 0 : 3.0;
      const row = Math.floor(index / cols);
      const col = index % cols;
      const x = (col - (cols - 1) / 2) * gap;
      const z = -row * gap;
      const groundLift = Number(root.userData?.groundLift || 0);
      root.position.set(x, PREVIEW_FLOOR_Y + PREVIEW_FLOOR_CLEARANCE + groundLift, z);
      root.scale.setScalar(1.0);
    }

    function relayoutObjectRoots() {
      const objectRoots = viewerState.spinRoots.filter((root) => !root.userData?.isMerged);
      const totalCount = objectRoots.length;
      objectRoots.forEach((root, index) => layoutObjectRoot(root, index, totalCount));
    }

    function framePreviewCamera() {
      if (!viewerState.camera || viewerState.spinRoots.length === 0) return;

      const box = new THREE.Box3();
      let hasBounds = false;
      for (const root of viewerState.spinRoots) {
        box.expandByObject(root);
        hasBounds = true;
      }
      if (!hasBounds || box.isEmpty()) return;

      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);

      const maxDim = Math.max(size.x, size.y, size.z, 1);
      const fov = (viewerState.camera.fov * Math.PI) / 180;
      const distance = Math.max(4.5, ((maxDim * 0.5) / Math.tan(fov / 2)) * 1.35);

      viewerState.camera.position.set(
        center.x,
        center.y + Math.max(1.2, size.y * 0.35),
        center.z + distance
      );
      viewerState.camera.lookAt(center);
      viewerState.camera.far = Math.max(200, distance + maxDim * 8);
      viewerState.camera.updateProjectionMatrix();
      if (viewerState.controls) {
        viewerState.controls.target.copy(center);
        viewerState.controls.update();
      }
    }

    function fitRootToUnit(root) {
      const box = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z, 1e-3);
      const scale = 1.6 / maxDim;
      root.scale.multiplyScalar(scale);
      box.setFromObject(root);
      const center = new THREE.Vector3();
      box.getCenter(center);
      root.position.sub(center);
      box.setFromObject(root);
      root.userData.groundLift = -box.min.y;
    }

    function createStagePlatformGeometry(radius, height) {
      const bevel = Math.min(radius * 0.04, height * 0.34);
      const baseInset = radius * 0.05;
      const points = [
        new THREE.Vector2(0, -height / 2),
        new THREE.Vector2(radius - baseInset, -height / 2),
        new THREE.Vector2(radius, -height / 2 + bevel * 0.75),
        new THREE.Vector2(radius, height / 2 - bevel * 0.85),
        new THREE.Vector2(radius - bevel, height / 2),
        new THREE.Vector2(0, height / 2),
      ];
      return new THREE.LatheGeometry(points, 96);
    }

    function createStageContactShadowTexture() {
      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new Error("Failed to create canvas context for stage contact shadow.");
      }

      const gradient = ctx.createRadialGradient(256, 256, 28, 256, 256, 240);
      gradient.addColorStop(0, "rgba(0,0,0,0.7)");
      gradient.addColorStop(0.45, "rgba(0,0,0,0.24)");
      gradient.addColorStop(0.78, "rgba(0,0,0,0.08)");
      gradient.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const texture = new THREE.CanvasTexture(canvas);
      texture.encoding = THREE.sRGBEncoding;
      texture.magFilter = THREE.LinearFilter;
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.needsUpdate = true;
      return texture;
    }

    function createFadingGridMaterial() {
      const material = new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        uniforms: {
          uColor: { value: new THREE.Color(0xd3deef) },
          uDivisions: { value: 26.0 },
          uLineWidth: { value: 0.028 },
          uFadeStart: { value: 0.52 },
          uFadeEnd: { value: 0.98 },
          uOpacity: { value: 0.92 },
        },
        vertexShader: `
          varying vec2 vUv;

          void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: `
          uniform vec3 uColor;
          uniform float uDivisions;
          uniform float uLineWidth;
          uniform float uFadeStart;
          uniform float uFadeEnd;
          uniform float uOpacity;

          varying vec2 vUv;

          float gridLine(float coordinate, float width) {
            float local = abs(fract(coordinate - 0.5) - 0.5);
            float aa = max(fwidth(coordinate), 0.0001) * 0.85;
            return 1.0 - smoothstep(width, width + aa, local);
          }

          void main() {
            vec2 centered = vUv - 0.5;
            float radius = length(centered) * 2.0;
            float outerMask = 1.0 - smoothstep(0.975, 1.0, radius);
            float fade = 1.0 - smoothstep(uFadeStart, uFadeEnd, radius);
            vec2 gridUv = centered * uDivisions * 2.0;
            float lines = max(gridLine(gridUv.x, uLineWidth), gridLine(gridUv.y, uLineWidth));
            float alpha = lines * fade * outerMask * uOpacity;

            if (alpha <= 0.001) discard;
            gl_FragColor = vec4(uColor, alpha);
          }
        `
      });
      material.extensions.derivatives = true;
      return material;
    }

    function createStudioEnvironment(pmremGenerator) {
      const canvas = document.createElement("canvas");
      canvas.width = 1024;
      canvas.height = 512;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new Error("Failed to create canvas context for studio environment.");
      }

      const base = ctx.createLinearGradient(0, 0, 0, canvas.height);
      base.addColorStop(0, "#fefefe");
      base.addColorStop(0.3, "#eef4fc");
      base.addColorStop(0.68, "#dbe7f7");
      base.addColorStop(1, "#c8d9ee");
      ctx.fillStyle = base;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const glowLeft = ctx.createRadialGradient(220, 120, 40, 220, 120, 220);
      glowLeft.addColorStop(0, "rgba(255,255,255,0.96)");
      glowLeft.addColorStop(0.4, "rgba(248,250,255,0.65)");
      glowLeft.addColorStop(1, "rgba(248,250,255,0)");
      ctx.fillStyle = glowLeft;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const glowRight = ctx.createRadialGradient(800, 140, 30, 800, 140, 240);
      glowRight.addColorStop(0, "rgba(255,255,255,0.94)");
      glowRight.addColorStop(0.5, "rgba(245,249,255,0.52)");
      glowRight.addColorStop(1, "rgba(245,249,255,0)");
      ctx.fillStyle = glowRight;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const glowCenter = ctx.createRadialGradient(512, 70, 20, 512, 70, 150);
      glowCenter.addColorStop(0, "rgba(255,255,255,0.92)");
      glowCenter.addColorStop(0.55, "rgba(255,255,255,0.34)");
      glowCenter.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = glowCenter;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.fillRect(84, 106, 126, 52);
      ctx.fillRect(812, 122, 134, 56);
      ctx.fillRect(432, 52, 160, 40);

      const texture = new THREE.CanvasTexture(canvas);
      texture.mapping = THREE.EquirectangularReflectionMapping;
      texture.encoding = THREE.sRGBEncoding;
      texture.magFilter = THREE.LinearFilter;
      texture.minFilter = THREE.LinearMipmapLinearFilter;

      const envTarget = pmremGenerator.fromEquirectangular(texture);
      texture.dispose();
      return envTarget;
    }

    function tuneTextureForPreview(texture) {
      if (!texture || !viewerState.renderer) return;
      texture.anisotropy = Math.max(texture.anisotropy || 1, viewerState.renderer.capabilities.getMaxAnisotropy());
      texture.needsUpdate = true;
    }

    function upgradePreviewMaterial(material) {
      if (!material) return material;

      const textureProps = [
        "map",
        "normalMap",
        "roughnessMap",
        "metalnessMap",
        "emissiveMap",
        "aoMap",
        "alphaMap",
      ];
      const hasTexture = textureProps.some((key) => !!material[key]);

      if (hasTexture) {
        const next = typeof material.clone === "function" ? material.clone() : material;
        for (const key of textureProps) {
          tuneTextureForPreview(next[key]);
        }
        if ("envMapIntensity" in next && next.envMapIntensity !== undefined) {
          next.envMapIntensity = Math.max(next.envMapIntensity || 0, 0.85);
        }
        if ("roughness" in next && next.roughness !== undefined) {
          next.roughness = Math.min(Math.max(next.roughness, 0.3), 0.92);
        }
        if ("metalness" in next && next.metalness !== undefined) {
          next.metalness = Math.min(next.metalness, 0.12);
        }
        next.needsUpdate = true;
        return next;
      }

      const alreadyPhysical = material.isMeshPhysicalMaterial === true;
      const next = alreadyPhysical ? material.clone() : new THREE.MeshPhysicalMaterial();

      if (material.name && !next.name) next.name = material.name;
      if (material.color && next.color) next.color.copy(material.color);
      if (material.emissive && next.emissive) next.emissive.copy(material.emissive);
      if (material.map) next.map = material.map;
      if (material.normalMap) next.normalMap = material.normalMap;
      if (material.roughnessMap) next.roughnessMap = material.roughnessMap;
      if (material.metalnessMap) next.metalnessMap = material.metalnessMap;
      if (material.emissiveMap) next.emissiveMap = material.emissiveMap;
      if (material.aoMap) next.aoMap = material.aoMap;
      if (material.alphaMap) next.alphaMap = material.alphaMap;
      if (material.transparent !== undefined) next.transparent = material.transparent;
      if (material.opacity !== undefined) next.opacity = material.opacity;
      if (material.side !== undefined) next.side = material.side;
      if (material.alphaTest !== undefined) next.alphaTest = material.alphaTest;
      if (material.normalScale && next.normalScale) next.normalScale.copy(material.normalScale);
      if (material.envMapIntensity !== undefined) next.envMapIntensity = Math.max(material.envMapIntensity, 1.35);
      else next.envMapIntensity = 1.35;

      next.roughness = material.roughness !== undefined ? Math.min(Math.max(material.roughness, 0.24), 0.82) : 0.46;
      next.metalness = material.metalness !== undefined ? Math.min(material.metalness, 0.18) : 0.06;
      next.clearcoat = Math.max(next.clearcoat || 0, 0.22);
      next.clearcoatRoughness = Math.min(next.clearcoatRoughness || 0.18, 0.26);
      next.reflectivity = Math.max(next.reflectivity || 0.5, 0.68);
      next.sheen = 0;
      next.transmission = next.transmission || 0;
      next.needsUpdate = true;

      return next;
    }

    async function addGlbToViewer(url, opts = {}) {
      await initThreeViewer();
      if (!url) return false;
      if (opts.isMerged) {
        if (viewerState.mergedUrl === url || viewerState.loadingMerged) return false;
        viewerState.loadingMerged = true;
      } else {
        if (viewerState.loadedUrls.has(url) || viewerState.loadingUrls.has(url)) return false;
        viewerState.loadingUrls.add(url);
      }
      try {
        const gltf = await new Promise((resolve, reject) => {
          viewerState.loader.load(url, resolve, undefined, reject);
        });
        const root = gltf.scene;
        root.traverse((node) => {
          if (!node.isMesh) return;
          node.castShadow = true;
          node.receiveShadow = true;
          if (node.material) {
            const materials = Array.isArray(node.material) ? node.material : [node.material];
            const upgraded = materials.map((material) => upgradePreviewMaterial(material));
            node.material = Array.isArray(node.material) ? upgraded : upgraded[0];
          }
        });
        fitRootToUnit(root);
        if (opts.isMerged) {
          root.userData.isMerged = true;
          root.scale.multiplyScalar(2.3);
          const mergedBox = new THREE.Box3().setFromObject(root);
          root.userData.groundLift = -mergedBox.min.y;
          root.position.set(0, PREVIEW_FLOOR_Y + PREVIEW_FLOOR_CLEARANCE + root.userData.groundLift, 5.2);
          viewerState.mergedUrl = url;
        } else {
          root.userData.isMerged = false;
          viewerState.loadedUrls.add(url);
        }
        viewerState.spinRoots.push(root);
        viewerState.scene.add(root);
        if (!opts.isMerged) {
          relayoutObjectRoots();
        }
        framePreviewCamera();
        return true;
      } finally {
        if (opts.isMerged) {
          viewerState.loadingMerged = false;
        } else {
          viewerState.loadingUrls.delete(url);
        }
      }
    }

    function enqueueGlbLoad(url, opts = {}) {
      if (!url) return false;
      const isMerged = !!opts.isMerged;
      if (isMerged) {
        if (viewerState.mergedUrl === url || viewerState.loadingMerged) return false;
      } else {
        if (
          viewerState.loadedUrls.has(url) ||
          viewerState.loadingUrls.has(url) ||
          viewerState.queuedUrls.has(url)
        ) {
          return false;
        }
      }
      viewerState.loadQueue.push({ url, opts: { isMerged } });
      if (!isMerged) {
        viewerState.queuedUrls.add(url);
      }
      processGlbQueue();
      return true;
    }

    async function processGlbQueue() {
      if (viewerState.queueBusy) return;
      viewerState.queueBusy = true;
      try {
        while (viewerState.loadQueue.length > 0) {
          const item = viewerState.loadQueue.shift();
          if (!item) continue;
          const { url, opts } = item;
          try {
            const added = await addGlbToViewer(url, opts);
            if (added && !opts?.isMerged) {
              viewerState.queuedUrls.delete(url);
            }
          } catch (e) {
            viewerState.queuedUrls.delete(url);
            console.error("GLB load failed:", url, e);
            setPreviewMessage("GLB load failed in browser. Open DevTools Console for details.");
          }
        }
      } finally {
        viewerState.queueBusy = false;
      }
    }
