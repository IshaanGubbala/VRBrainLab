
// viz3d.js - Three.js implementation for 3D Brain Visualization

class BrainVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x16161e); // Match sidebar bg
        // Add some fog for depth
        this.scene.fog = new THREE.Fog(0x16161e, 200, 1000);

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 2000);
        this.camera.position.set(0, 150, 200);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 0.5;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(100, 100, 50);
        this.scene.add(dirLight);

        // State w/ geometry
        this.nodes = []; // Meshes
        this.edges = []; // Line segments
        this.nodeData = []; // Metadata
        
        // Resize handler
        window.addEventListener('resize', () => this.onResize());
        
        // Kickoff loop
        this.animate();
    }
    
    onResize() {
        this.width = this.container.clientWidth;
        this.height = this.container.clientHeight;
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.width, this.height);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    async loadBrain() {
        try {
            // Get Info (Centers)
            const infoRes = await fetch('/api/brain/info');
            const info = await infoRes.json();
            
            // Get Connectivity (Weights)
            const connRes = await fetch('/api/brain/connectivity');
            const conn = await connRes.json();
            
            this.clearScene();
            this.buildNetwork(info.centers, conn.weights, info.region_labels);
            
            console.log(`Viz3D: Loaded ${info.num_regions} nodes, ${conn.num_connections} edges.`);
            
        } catch(e) {
            console.error("Viz3D Load Error:", e);
        }
    }
    
    clearScene() {
        // Remove old meshes
        this.nodes.forEach(m => this.scene.remove(m));
        this.edges.forEach(m => this.scene.remove(m));
        this.nodes = [];
        this.edges = [];
    }
    
    buildNetwork(centers, weights, labels) {
        const numRegions = centers.length;
        this.nodeData = [];
        
        // Geometry for nodes
        const geometry = new THREE.SphereGeometry(3.5, 32, 32); 
        
        // Create Nodes
        for(let i=0; i<numRegions; i++) {
            const material = new THREE.MeshPhongMaterial({ 
                color: 0x5d87e5, 
                emissive: 0x000000,
                specular: 0x111111,
                shininess: 30
            });
            const mesh = new THREE.Mesh(geometry, material);
            
            // Set pos
            mesh.position.set(centers[i][0], centers[i][1], centers[i][2]);
            
            this.scene.add(mesh);
            this.nodes.push(mesh);
            this.nodeData.push({ id: i, label: labels[i] });
        }
        
        // Create Edges (optimized with LineSegments?)
        // For aesthetic thickness, we might need cylinder meshes or thick lines.
        // For performance, simple lines are best for now.
        // Let's filter weak connections to avoid clutter.
        
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x565f89,
            transparent: true,
            opacity: 0.15,
            blending: THREE.AdditiveBlending
        });
        
        // Collect points for line segments
        const points = [];
        // Or individual lines if we want varying opacity? 
        // Let's do individual lines for now to control visual weight of strong connections.
        
        let edgeCount = 0;
        const maxWeight = Math.max(...weights.flat());
        
        for(let i=0; i<numRegions; i++) {
            for(let j=i+1; j<numRegions; j++) {
                const w = weights[i][j];
                if(w > 0.05) { // Threshold
                    const start = new THREE.Vector3(centers[i][0], centers[i][1], centers[i][2]);
                    const end = new THREE.Vector3(centers[j][0], centers[j][1], centers[j][2]);
                    
                    const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
                    const opacity = 0.1 + (w/maxWeight) * 0.4;
                    const mat = new THREE.LineBasicMaterial({ color: 0x565f89, transparent: true, opacity });
                    const line = new THREE.Line(geo, mat);
                    
                    this.scene.add(line);
                    this.edges.push(line);
                    edgeCount++;
                }
            }
        }
    }
    
    updateActivity(activityArray) {
        // activityArray: array of floats 0.0-1.0 per region
        if(!activityArray || activityArray.length !== this.nodes.length) return;
        
        for(let i=0; i<this.nodes.length; i++) {
            const act = activityArray[i];
            const mesh = this.nodes[i];
            
            // Lerp Color: Blue (inactive) -> Red (active)
            // 0 -> 0x5d87e5 (Blueish)
            // 1 -> 0xf7768e (Reddish/Pink)
            
            const r = THREE.MathUtils.lerp(0x5d/255, 0xf7/255, act);
            const g = THREE.MathUtils.lerp(0x87/255, 0x76/255, act);
            const b = THREE.MathUtils.lerp(0xe5/255, 0x8e/255, act);
            
            mesh.material.color.setRGB(r, g, b);
            
            // Pulse size slightly?
            const scale = 1.0 + act * 0.3;
            mesh.scale.setScalar(scale);
            
            // Emissive glow for high activity
            if(act > 0.5) {
                mesh.material.emissive.setHex(0xf7768e);
                mesh.material.emissiveIntensity = (act - 0.5);
            } else {
                mesh.material.emissive.setHex(0x000000);
            }
        }
    }
}

// Global hook
window.BrainViz = BrainVisualizer;
