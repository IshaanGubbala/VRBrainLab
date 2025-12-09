
// app.js - VR Brain Lab Dashboard Logic

const API_BASE = '/api';

// State
let simulationRunning = false;
let pollingInterval = null;
let charts = {};
let lastStreamIdx = 0;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    checkStatus();
    loadBrainInfo();
    
    // Start polling loop
    setInterval(updateDashboard, 1000);
});

// --- Tab Navigation ---
function switchTab(tabId) {
    document.querySelectorAll('.nav-links li').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    
    // Find nav item by text (hacky but works for now) or index. 
    // Actually, I attached onClick inline.
    // Let's just find the one with matching onclick
    // Simplified: Just restart active class logic
    const navItems = document.querySelectorAll('.nav-links li');
    const tabs = ['dashboard', 'experiments', 'tuning', 'intervention'];
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
    // Current backend doesn't support "stop", it just runs fixed duration.
    // We can just stop polling.
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
        
        // Fix coordinates for canvas (origin top-left vs plot bottom-left)
        // Usually index [0][0] corresponds to min x, min y.
        // Heatmap array is likely [input_row][coupling_col] based on sweep loop.
        
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
