const socket = io();
let trainingChart = null;

socket.on('connect', () => {
    console.log('Connected to server');
    updateSystemStatus();
});

socket.on('training_progress', (data) => {
    console.log('Training progress:', data);
    updateTrainingProgress(data);
});

socket.on('training_complete', (data) => {
    console.log('Training complete:', data);
    completeTraining(data);
});

function startTraining() {
    const profile = document.getElementById('cost-profile').value;
    const epochs = document.getElementById('epochs').value;
    
    // Fix: Find button by onclick attribute instead of ID
    const trainBtn = document.getElementById('trainBtn') || document.querySelector('button[onclick="startTraining()"]');
    
    if (trainBtn) {
        // Change button appearance
        trainBtn.disabled = true;
        trainBtn.style.background = '#27ae60';
        trainBtn.textContent = 'Training...';
    }
    
    // Initialize progress display if function exists
    if (typeof initializeProgress === 'function') {
        initializeProgress();
    }
    
    fetch('/api/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cost_profile: profile, epochs: parseInt(epochs)})
    })
    .then(r => r.json())
    .then(data => {
        console.log('Training started:', data);
    })
    .catch(err => {
        console.error('Error starting training:', err);
        // Reset button on error
        if (trainBtn) {
            trainBtn.disabled = false;
            trainBtn.style.background = '#667eea';
            trainBtn.textContent = 'Start Training';
        }
    });
}

function updateSystemStatus() {
    fetch('/api/system/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('system-status').innerHTML = `
                GPU: ${data.gpu.available ? data.gpu.name : 'Not available'}<br>
                Memory: ${data.gpu.memory_used.toFixed(1)}/${data.gpu.memory_total.toFixed(1)} GB
            `;
        })
        .catch(err => {
            console.error('Error fetching status:', err);
        });
}

// Add the missing functions
function initializeProgress() {
    console.log('Initializing progress display');
}

function updateTrainingProgress(data) {
    console.log('Updating progress:', data);
}

function completeTraining(data) {
    console.log('Training complete:', data);
    // Reset button
    const trainBtn = document.getElementById('trainBtn') || document.querySelector('button[onclick="startTraining()"]');
    if (trainBtn) {
        trainBtn.disabled = false;
        trainBtn.style.background = '#667eea';
        trainBtn.textContent = 'Start Training';
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    updateSystemStatus();
    setInterval(updateSystemStatus, 5000);
});
