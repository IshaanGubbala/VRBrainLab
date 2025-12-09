
// app.js - VR Brain Lab Dashboard Logic

const API_BASE = '/api';

// State
let simulationRunning = false;
let pollingInterval = null;
let charts = {};
let lastStreamIdx = 0;
let brainViz = null;
let tuningRunning = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    
    // Init 3D Viz (after scripts load)
    if(window.BrainViz) {
        brainViz = new BrainViz('brain-viz');
        brainViz.loadBrain(); // Load geometry
    }

    checkStatus();
    loadBrainInfo();
    
    // Start polling loop
    setInterval(updateDashboard, 1000);
    setInterval(checkTuningStatus, 2000); // Check tuner every 2s
});

// --- Tab Navigation ---
function switchTab(tabId) {
    document.querySelectorAll('.nav-links li').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    
    const navItems = document.querySelectorAll('.nav-links li');
    const tabs = ['dashboard', 'experiments', 'tuning', 'ai', 'intervention'];
    const idx = tabs.indexOf(tabId);
    if(idx >= 0) navItems[idx].classList.add('active');
    
    document.getElementById(`tab-${tabId}`).classList.add('active');
}

// --- Charts ---
function initCharts() {
    const ctx = document.getElementById('activityChart').getContext('2d');
    charts.activity = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Global Activity (E)',
                data: [],
                borderColor: '#7aa2f7',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { display: false },
                y: { min: 0, max: 1.0, grid: { color: '#24283b' } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // 2. Latent Space (Scatter)
    const ctx2 = document.getElementById('latentChart').getContext('2d');
    charts.latent = new Chart(ctx2, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Activity Trajectory',
                data: [],
                borderColor: '#e0af68',
                backgroundColor: 'rgba(224, 175, 104, 0.2)',
                showLine: true,
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#24283b' }, title: {display: true, text: 'PC 1'} },
                y: { grid: { color: '#24283b' }, title: {display: true, text: 'PC 2'} }
            },
            plugins: { legend: { display: false } }
        }
    });

    // 3. Therapy Optimization
    const ctx3 = document.getElementById('therapyChart').getContext('2d');
    charts.therapy = new Chart(ctx3, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Score (Alpha Power)',
                data: [],
                borderColor: '#9ece6a',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: '#24283b' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// --- Simulation Control ---
async function startSimulation() {
    try {
        const res = await fetch(`${API_BASE}/simulation/run`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ duration: 5000 })
        });
        const data = await res.json();
        if(data.status === 'started') {
            simulationRunning = true;
            lastStreamIdx = 0;
            updateStatus('Running', true);
            charts.activity.data.labels = [];
            charts.activity.data.datasets[0].data = [];
            charts.activity.update();
        }
    } catch(e) {
        console.error("Start failed", e);
    }
}

async function stopSimulation() {
    simulationRunning = false;
    updateStatus('Stopped', false);
}

async function checkStatus() {
    try {
        const res = await fetch(`${API_BASE}/simulation/status`);
        const data = await res.json();
        simulationRunning = data.running;
        updateStatus(data.running ? 'Running' : 'Ready', data.running);
    } catch(e) {
        updateStatus('Disconnected', false);
    }
}

function updateStatus(text, isRunning) {
    document.getElementById('status-text').innerText = text;
    const ind = document.getElementById('sim-indicator');
    if(isRunning) ind.classList.add('running');
    else ind.classList.remove('running');
}

// --- Data Polling ---
async function updateDashboard() {
    if(!simulationRunning) return;

    // 1. Get Stream
    try {
        const res = await fetch(`${API_BASE}/simulation/activity_stream?start=${lastStreamIdx}&limit=100`);
        const data = await res.json();
        
        if(data.error) return; // No results yet
        
        if(data.activity && data.activity.length > 0) {
            const newTime = data.time;
            const newAct = data.activity.map(row => 
                row.reduce((a,b) => a+b, 0) / row.length // Mean field
            );
            
            // Append to chart
            const maxPoints = 200;
            let labels = charts.activity.data.labels;
            let values = charts.activity.data.datasets[0].data;
            
            labels.push(...newTime);
            values.push(...newAct);
            
            if(labels.length > maxPoints) {
                 labels = labels.slice(-maxPoints);
                 values = values.slice(-maxPoints);
            }
            
            charts.activity.data.labels = labels;
            charts.activity.data.datasets[0].data = values;
            charts.activity.update();
            
            lastStreamIdx = data.next_start || lastStreamIdx;
            
            // Update metrics (simple current value)
            const currentMean = newAct[newAct.length-1];
            document.getElementById('m-mean').innerText = currentMean.toFixed(3);

            // Update 3D Viz (use latest frame)
            if(brainViz && data.activity.length > 0) {
               const latestActivity = data.activity[data.activity.length-1];
               brainViz.updateActivity(latestActivity);
            }
        }
        
        if(data.done && simulationRunning) {
             simulationRunning = false;
             updateStatus('Complete', false);
             // Final full metrics update
             updateDetailedMetrics();
        }
        
    } catch(e) {
        console.error("Poll error", e);
    }
}

async function updateDetailedMetrics() {
     const res = await fetch(`${API_BASE}/analysis/metrics`);
     const data = await res.json();
     if(data.temporal) {
         document.getElementById('m-meta').innerText = data.temporal.metastability?.toFixed(3) || '--';
         document.getElementById('m-sync').innerText = data.temporal.synchrony?.toFixed(3) || '--';
     }
}

async function loadBrainInfo() {
    const res = await fetch(`${API_BASE}/brain/info`);
    const data = await res.json();
    document.getElementById('b-regions').innerText = data.num_regions;
    document.getElementById('b-conns').innerText = data.num_connections;
}

// --- Experiments ---
async function runValidation() {
    const consoleDiv = document.getElementById('validation-results');
    consoleDiv.innerText = "Running validation suite... please wait...";
    
    try {
        const res = await fetch(`${API_BASE}/experiments/validate`, { method: 'POST' });
        const data = await res.json();
        
        // Format output
        let output = `Validation Result: ${data.overall_pass ? 'PASS ✅' : 'FAIL ❌'}\n\n`;
        data.tests.forEach(test => {
            output += `[${test.status}] ${test.name}\n   -> ${test.message}\n`;
        });
        consoleDiv.innerText = output;
        
    } catch(e) {
        consoleDiv.innerText = "Error: " + e.message;
    }
}

async function runRegimeSweep() {
    const canvas = document.getElementById('regimeChart');
    const ctx = canvas.getContext('2d');
    
    // Clear
    ctx.fillStyle = '#000';
    ctx.fillRect(0,0, canvas.width, canvas.height);
    ctx.fillStyle = '#fff';
    ctx.fillText("Generating map... might take ~30s...", 20, 50);
    
    try {
        const res = await fetch(`${API_BASE}/experiments/regime_map`, { method: 'POST' });
        const data = await res.json();
        const heatmap = data.heatmap; // 2D array of codes 0-4
        
        // colors: 0:Quiet(Blue), 1:Stable(Green), 2:Osc(Yellow), 3:Meta(Red), 4:Sat(Black)
        const colors = ['#1f77b4', '#2ca02c', '#e6ce00', '#d62728', '#ffffff'];
        
        // Render
        const rows = heatmap.length;
        const cols = heatmap[0].length;
        const w = canvas.width / cols;
        const h = canvas.height / rows;
        
        // Fix coordinates for canvas
        for(let r=0; r<rows; r++) { // Input (Y)
             for(let c=0; c<cols; c++) { // Coupling (X)
                 const val = heatmap[r][c];
                 ctx.fillStyle = colors[val] || '#fff';
                 // Invert Y for drawing
                 const y = canvas.height - (r+1)*h;
                 const x = c*w;
                 ctx.fillRect(x, y, w, h);
             }
        }
        
    } catch(e) {
        console.error("Map failed", e);
    }
}

async function runAIDiagnostics() {
    const label = document.getElementById('ai-state-label');
    const conf = document.getElementById('ai-confidence');
    
    label.innerText = "Analyzing...";
    label.style.color = "#7aa2f7";
    
    try {
        const res = await fetch(`${API_BASE}/experiments/ai_diagnostics`, { method: 'POST' });
        const data = await res.json();
        
        if(data.error) {
             label.innerText = "Error: " + data.error;
             return;
        }
        
        // 1. Update text
        const cls = data.classification;
        label.innerText = cls.label;
        conf.innerText = `Confidence: ${(cls.confidence * 100).toFixed(1)}%`;
        
        // Color code
        if(cls.label.includes("Healthy")) label.style.color = "#9ece6a";
        else if(cls.label.includes("Seizure")) label.style.color = "#f7768e";
        else label.style.color = "#e0af68";
        
        // Metrics
        document.getElementById('ai-m-alpha').innerText = data.metrics.alpha_power?.toFixed(3);
        document.getElementById('ai-m-meta').innerText = data.metrics.metastability?.toFixed(3);
        
        // 2. Update Plot
        const traj = data.latent_trajectory;
        const pts = traj.x.map((x, i) => ({ x: x, y: traj.y[i] }));
        
        charts.latent.data.datasets[0].data = pts;
        charts.latent.update();
        
    } catch(e) {
        console.error("AI Diag failed", e);
        label.innerText = "Failed";
    }
}


// --- Interventions ---
async function applyStimulation() {
    const region = document.getElementById('stim-region').value;
    const amp = document.getElementById('stim-amp').value;
    await fetch(`${API_BASE}/intervention/stimulate`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ region_idx: parseInt(region), amplitude: parseFloat(amp) })
    });
    alert("Stimulation applied!");
}

async function applyLesion() {
     const region = document.getElementById('lesion-region').value;
     await fetch(`${API_BASE}/intervention/lesion`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ region_idx: parseInt(region), type: 'region' })
    });
    alert("Lesion applied!");
}

async function resetInterventions() {
    await fetch(`${API_BASE}/intervention/reset`, { method: 'POST' });
    alert("Reset done");
}

async function runTherapy() {
    const btn = document.querySelector('button[onclick="runTherapy()"]');
    const oldText = btn.innerHTML;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Optimizing...';
    btn.disabled = true;
    
    try {
        const res = await fetch(`${API_BASE}/experiments/therapy`, { method: 'POST' });
        const data = await res.json();
        
        // Update Metrics
        document.getElementById('opt-best-amp').innerText = data.best_params.amplitude.toFixed(2);
        document.getElementById('opt-best-score').innerText = data.best_score.toFixed(4);
        
        // Update Chart
        const hist = data.history;
        charts.therapy.data.labels = hist.map(h => h.step);
        charts.therapy.data.datasets[0].data = hist.map(h => h.score);
        charts.therapy.update();
        
        alert(`Optimization Complete! Best Amplitude: ${data.best_params.amplitude.toFixed(2)}`);
        
    } catch(e) {
        console.error("Therapy failed", e);
        alert("Optimization failed.");
    } finally {
        btn.innerHTML = oldText;
        btn.disabled = false;
    }
}



async function startTuning() {
    try {
        const res = await fetch(`${API_BASE}/experiments/tune/start`, { method: 'POST' });
        const data = await res.json();
        if(data.status === 'started') {
            tuningRunning = true;
            document.getElementById('tuning-status').innerText = "Status: Running...";
            document.getElementById('tuning-status').style.color = "#e0af68";
            alert("Tuner started in background. This may take 1-2 minutes.");
        } else {
            alert("Could not start tuner: " + data.message);
        }
    } catch(e) {
        console.error("Tune start failed", e);
    }
}

async function checkTuningStatus() {
    if(!tuningRunning) return;
    
    try {
        const res = await fetch(`${API_BASE}/experiments/tune/status`);
        const data = await res.json();
        
        if(data.status === 'complete') {
            tuningRunning = false;
            document.getElementById('tuning-status').innerText = "Status: Complete ✅";
            document.getElementById('tuning-status').style.color = "#9ece6a";
            renderTunerResults(data.result);
        } else if(data.status === 'failed') {
            tuningRunning = false;
            document.getElementById('tuning-status').innerText = "Status: Failed ❌";
            document.getElementById('tuning-status').style.color = "#f7768e";
        }
        // If running, do nothing (wait)
    } catch(e) {
        console.error("Tune check failed", e);
    }
}

function renderTunerResults(result) {
    // Params
    const pList = document.getElementById('tune-params-list');
    pList.innerHTML = '';
    for(const [k, v] of Object.entries(result.params)) {
        pList.innerHTML += `
            <div class="metric">
                <span class="label">${k}</span>
                <span class="val">${v.toFixed(3)}</span>
            </div>`;
    }
    
    // Metrics (Loss)
    document.getElementById('tune-loss').innerText = result.loss.toFixed(4);
    
    const mList = document.getElementById('tune-metrics-list');
    mList.innerHTML = '';
    for(const [k, v] of Object.entries(result.metrics)) {
         mList.innerHTML += `
            <div class="metric">
                <span class="label">${k}</span>
                <span class="val">${v.toFixed(3)}</span>
            </div>`;
    }
}

// ... existing resize code ...
/* Helper: Resize canvas to parent */
function resizeCanvas() {
    const cvs = document.querySelectorAll('canvas');
    cvs.forEach(c => {
        const p = c.parentElement;
        c.width = p.clientWidth;
        c.height = p.clientHeight;
    });
}
window.addEventListener('resize', resizeCanvas);
setTimeout(resizeCanvas, 100);
