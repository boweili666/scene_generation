    let cy = null;

    const interactionState = {
      startedAt: null,
      lastInstruction: "",
      lastJson: null
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
      document.getElementById("uploadStatus").textContent = imageCount ? `${imageCount} image selected` : "Using text";
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

    function setFeedback(lines){
      const box = document.getElementById("feedbackBox");
      box.textContent = (lines && lines.length) ? lines.map(s => "• " + s).join("\n") : "-";
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
      const qs = new URLSearchParams({
        offset: String(real2simLogState.offset || 0),
        limit: "65536"
      });
      const res = await fetch(`/real2sim/log?${qs.toString()}`);
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
          id: o?.id
        }));
      } else if (json?.obj && typeof json.obj === "object") {
        objects = Object.entries(json.obj).map(([path, meta]) => ({
          path,
          class_name: meta?.class || meta?.class_name || meta?.caption,
          id: meta?.id
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
        edges = Array.isArray(json.edges["obj-obj"])
          ? json.edges["obj-obj"].map((e) => ({
              ...e,
              source: resolveEndpoint(e?.source),
              target: resolveEndpoint(e?.target),
            }))
          : [];
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
        const res = await fetch("/scene_graph");
        const graph = await res.json();
        if (!res.ok) throw new Error(graph?.error || "Failed to load scene graph");
        return analyzeSceneJson(graph);
      } catch (err) {
        console.warn("Fallback analysis failed:", err);
        return null;
      }
    }

    /* ===== Cytoscape render ===== */
    function renderSceneGraph(graph){
      const normalized = normalizeSceneGraph(graph);
      const elements = [];
      for (const obj of (normalized.objects || [])) {
        const depth = String(obj.path || "").split("/").filter(Boolean).length;
        elements.push({
          data: { id: obj.path, label: obj.class_name, depth }
        });
      }
      for (const e of (normalized.edges || [])) {
        elements.push({
          data: { id: e.source + "->" + e.target, source: e.source, target: e.target, label: e.relation }
        });
      }

      if (cy) cy.destroy();

      cy = cytoscape({
        container: document.getElementById("sceneGraph"),
        elements,
        style: [
          {
            selector: "node",
            style: {
              "background-color": "mapData(depth, 1, 6, #93c5fd, #1d4ed8)",
              "border-color": "#1d4ed8",
              "border-width": "1.5px",
              "label": "data(label)",
              "color": "#ffffff",
              "text-valign": "center",
              "shape": "round-rectangle",
              "padding": "10px",
              "font-size": "11px",
              "text-wrap": "wrap",
              "text-max-width": "130px",
              "text-outline-width": 2,
              "text-outline-color": "rgba(15, 23, 42, 0.18)",
              "shadow-color": "rgba(29, 78, 216, 0.25)",
              "shadow-blur": 14,
              "shadow-offset-y": 5,
              "transition-property": "background-color, border-color, shadow-blur, shadow-color, opacity",
              "transition-duration": "180ms"
            }
          },
          {
            selector: "edge",
            style: {
              "curve-style": "unbundled-bezier",
              "target-arrow-shape": "triangle",
              "label": "data(label)",
              "font-size": "10px",
              "line-color": "rgba(37, 99, 235, 0.55)",
              "target-arrow-color": "rgba(37, 99, 235, 0.55)",
              "width": 2,
              "arrow-scale": 0.9,
              "text-rotation": "autorotate",
              "text-background-color": "#ffffff",
              "text-background-opacity": 0.92,
              "text-background-padding": "4px",
              "color": "#0f172a",
              "text-border-width": 1,
              "text-border-color": "rgba(148, 163, 184, 0.55)",
              "text-margin-y": -3,
              "transition-property": "line-color, target-arrow-color, width, opacity",
              "transition-duration": "180ms"
            }
          },
          { selector: ".dim", style: { "opacity": 0.16 } },
          {
            selector: ".focus-node",
            style: {
              "border-color": "#f59e0b",
              "border-width": 3,
              "shadow-color": "rgba(245, 158, 11, 0.45)",
              "shadow-blur": 22
            }
          },
          {
            selector: ".focus-edge",
            style: {
              "line-color": "#f59e0b",
              "target-arrow-color": "#f59e0b",
              "width": 3,
              "opacity": 1
            }
          }
        ],
        layout: {
          name: "cose",
          animate: true,
          animationDuration: 420,
          fit: true,
          padding: 28,
          nodeRepulsion: 5200,
          edgeElasticity: 130,
          idealEdgeLength: 110,
          gravity: 0.2,
          numIter: 900
        }
      });

      const clearFocus = () => cy.elements().removeClass("dim focus-node focus-edge");
      cy.on("tap", "node", (evt) => {
        const node = evt.target;
        clearFocus();
        cy.elements().addClass("dim");
        node.removeClass("dim").addClass("focus-node");
        node.connectedEdges().removeClass("dim").addClass("focus-edge");
        node.neighborhood().removeClass("dim");
      });
      cy.on("tap", (evt) => { if (evt.target === cy) clearFocus(); });
    }

    /* ===== init load ===== */
    async function loadSceneGraph(){
      try{
        await ensureCytoscape();
        setPill("graph", "", "Loading...");
        const res = await fetch("/scene_graph");
        const graph = await res.json();
        if (!res.ok) throw new Error(graph?.error || "Failed to load /scene_graph");
        renderSceneGraph(graph);
        const analysis = analyzeSceneJson(graph);
        setMetrics(analysis);
        document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
        setFeedback(analysis.warnings.length ? analysis.warnings : ["Looks good. Add more relations for finer control."]);
        setPill("graph", "ok", "Ready");
      }catch(err){
        console.error(err);
        setPill("graph", "err", "Error");
        toast("err","Graph load failed", String(err));
      }
    }

    /* ===== Generate graph from prompt/image ===== */
    async function generateSceneGraph(opts = {}) {
      const text = (opts.text ?? document.getElementById("sceneInput").value).trim();
      const fileInput = document.getElementById("imageInput");
      const classDirPicker = document.getElementById("classDirPicker");

      const formData = new FormData();
      if (text) formData.append("text", text);

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

      try {
        await ensureCytoscape();
        const res = await fetch("/scene_from_input", { method:"POST", body: formData });
        const graph = await res.json();
        if (!res.ok) {
          const msg = graph.error || "Generation failed";
          if (!opts.silentError) toast("err","Generation failed", msg);
          return { ok:false, error: msg };
        }
        renderSceneGraph(graph);
        const analysis = analyzeSceneJson(graph);
        setMetrics(analysis);
        document.getElementById("diagText").textContent = `Objects ${analysis.objects} • Edges ${analysis.edges} • Score ${analysis.score}`;
        setFeedback(analysis.warnings.length ? analysis.warnings : ["Graph looks complete. You can now run Real2Sim or Scene Service."]);
        return { ok:true, graph, analysis };
      } catch (err) {
        console.error(err);
        const msg = "Failed to generate scene graph";
        if (!opts.silentError) toast("err","Network error", msg);
        return { ok:false, error: String(err || msg) };
      }
    }

    async function generateFromPrompt(){
      const text = document.getElementById("sceneInput").value.trim();
      if (!text && document.getElementById("imageInput").files.length === 0){
        toast("warn","Missing input","Please enter a prompt or upload a reference image.");
        return;
      }

      setPill("graph","warn","Generating...");
      document.getElementById("jsonStatus").textContent = "Generating graph...";
      document.getElementById("btnGenerate").disabled = true;

      const result = await generateSceneGraph({ text, silentError: true });

      document.getElementById("btnGenerate").disabled = false;
      if (!result || !result.ok){
        setPill("graph","err","Failed");
        document.getElementById("jsonStatus").textContent = "Failed";
        toast("err","Graph generation failed", result?.error || "Unknown error");
        return;
      }

      setPill("graph","ok","Ready");
      document.getElementById("jsonStatus").textContent = "Graph updated";
      toast("ok","Graph updated", `${result.analysis.objects} objects • ${result.analysis.edges} edges • score ${result.analysis.score}`);
      if (!document.getElementById("drawer").classList.contains("open")) toggleDrawer();
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
      showThreeViewer().then(() => clearThreeViewer()).catch((viewerErr) => {
        console.warn("Three viewer unavailable:", viewerErr);
        setPreviewMessage("Three.js viewer failed to load. Check browser console / CDN access.");
      });
      resetSimProgress();
      resetReal2SimLog();
      setSimProgress({ phase: "queued", percent: 1, generated_objects: 0, expected_objects: null, has_merged_scene: false });

      try {
        const startRes = await fetch("/real2sim/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
        const startData = await startRes.json();
        if (!startRes.ok || !startData?.job_id) {
          setPill("sim","err","Failed");
          toast("err","Real2Sim failed", (startData && (startData.detail || startData.msg)) || "Failed to start job");
          return;
        }
        const jobId = startData.job_id;
        resetReal2SimLog(startData.log_start_offset || 0, startData.log_path || "real2sim.log");
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
          const pollRes = await fetch(`/real2sim/status/${jobId}`);
          const pollData = await pollRes.json();
          if (!pollRes.ok || !pollData?.job) {
            throw new Error((pollData && (pollData.msg || pollData.error)) || "Failed to poll job status");
          }

          const job = pollData.job;
          const progress = job.progress || {};
          const artifacts = job.artifacts || {};
          const objectUrls = Array.isArray(artifacts.object_glb_urls) ? artifacts.object_glb_urls : [];

          const statusText = `${job.status} • objects ${objectUrls.length} • queue ${viewerState.loadQueue.length}${artifacts.scene_glb_url ? " • merged ready" : ""}`;
          document.getElementById("statusSimText").textContent = statusText;
          statusEl.textContent = statusText;
          setSimProgress(progress);
          resultEl.textContent = JSON.stringify(job, null, 2);

          for (const url of objectUrls) {
            const queued = enqueueGlbLoad(url);
            if (queued) loadedObjectCount += 1;
          }

          if (artifacts.scene_glb_url && !loadedMerged) {
            const mergedQueued = enqueueGlbLoad(artifacts.scene_glb_url, { isMerged: true });
            if (mergedQueued) loadedMerged = true;
          }

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
            toast("err","Real2Sim failed", job.error || "Unknown error");
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
      } catch (err) {
        console.error(err);
        setPill("sim","err","Failed");
        document.getElementById("statusSimText").textContent = "Failed";
        document.getElementById("real2simLogStatus").textContent = "Unavailable";
        toast("err","Real2Sim error", "Check backend logs / service status.");
      } finally {
        btn.disabled = false;
      }
    }

    /* ===== Isaac Scene Service ===== */
    async function callSceneService(endpoint) {
      const statusEl = document.getElementById("sceneSvcStatus");
      const resultEl = document.getElementById("sceneSvcResult");
      const img = document.getElementById("renderImage");

      const base = "http://127.0.0.1:8001";
      const payload = {
        camera_eye: [0.0, -18.0, 18.0],
        camera_target: [0.0, 0.0, 1.0],
        frames: 20,
        capture_frame: 10,
        resolution: [1280, 720],
        use_default_ground: true,
        default_ground_z_offset: -0.01,
        generate_room: true,
        room_include_back_wall: true,
        room_include_left_wall: false,
        room_include_right_wall: true,
        room_include_front_wall: true
      };

      statusEl.textContent = `Calling /${endpoint}...`;
      setPill("sim","warn","Calling...");
      document.getElementById("statusSimText").textContent = `Calling /${endpoint}...`;
      resetSimProgress();
      resultEl.textContent = "Requesting...";

      try {
        const res = await fetch(`${base}/${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();

        statusEl.textContent = res.ok ? "Done" : "Failed";
        resultEl.textContent = JSON.stringify(data, null, 2);

        if (!res.ok) {
          setPill("sim","err","Failed");
          toast("err","Scene service failed", data.detail || data.error || "Request failed");
        } else {
          showImagePreview();
          img.src = "/render_image?ts=" + Date.now();
          setPill("sim","ok","Ready");
          toast("ok","Scene service done", `Endpoint /${endpoint} finished.`);
        }
      } catch (err) {
        console.error(err);
        statusEl.textContent = "Failed";
        resultEl.textContent = String(err);
        setPill("sim","err","Failed");
        toast("err","Request failed", "Possibly CORS / port / service not running.");
      }
    }

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
