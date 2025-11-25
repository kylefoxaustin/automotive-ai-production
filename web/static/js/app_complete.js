const socket = io();
let trainingChart = null;

socket.on('connect', () => {
    console.log('Connected to server');
    updateSystemStatus();
});

socket.on('training_progress', (data) => {
    console.log('Progress received:', data);
    updateProgressDisplay(data);
});

socket.on('training_complete', (data) => {
    console.log('Training complete:', data);
    completeTraining(data);
});

function updateProgressDisplay(data) {
    // Create progress UI if it doesn't exist
    let progressContainer = document.getElementById('progress-container');
    if (!progressContainer) {
        const progressPanel = document.querySelector('.panel.full-width');
        if (progressPanel) {
            progressPanel.innerHTML = `
                <h2>Training Progress</h2>
                <div id="progress-container">
                    <div style="background: #e0e0e0; border-radius: 10px; height: 30px; overflow: hidden; margin: 20px;">
                        <div id="progress-bar" style="background: linear-gradient(90deg, #667eea, #764ba2); 
                             height: 100%; width: 0%; transition: width 0.3s; display: flex; 
                             align-items: center; justify-content: center; color: white; font-weight: bold;">
                            0%
                        </div>
                    </div>
                    <p id="epoch-info" style="text-align: center; font-size: 18px;">Epoch 0/0</p>
                    <p id="loss-info" style="text-align: center; font-size: 16px;">Loss: --</p>
                    <p id="accuracy-info" style="text-align: center; font-size: 16px;">Accuracy: --</p>
                    <canvas id="loss-chart" width="700" height="300"></canvas>
                </div>
            `;
            
            // Initialize chart
            const ctx = document.getElementById('loss-chart').getContext('2d');
            trainingChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false
                        }
                    }
                }
            });
        }
    }
    
    // Update progress bar
    const progress = (data.epoch / data.total_epochs) * 100;
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        progressBar.style.width = progress + '%';
        progressBar.textContent = Math.round(progress) + '%';
    }
    
    // Update text info
    const epochInfo = document.getElementById('epoch-info');
    if (epochInfo) {
        epochInfo.textContent = `Epoch ${data.epoch}/${data.total_epochs}`;
    }
    
    const lossInfo = document.getElementById('loss-info');
    if (lossInfo) {
        lossInfo.textContent = `Loss: ${data.loss.toFixed(4)}`;
    }
    
    const accuracyInfo = document.getElementById('accuracy-info');
    if (accuracyInfo && data.metrics && data.metrics.accuracy) {
        accuracyInfo.textContent = `Accuracy: ${(data.metrics.accuracy * 100).toFixed(1)}%`;
    }
    
    // Update chart
    if (trainingChart && data.history) {
        trainingChart.data.labels = data.history.map((_, i) => i + 1);
        trainingChart.data.datasets[0].data = data.history;
        trainingChart.update('none');
    }
}

function startTraining() {
    const trainBtn = document.getElementById('trainBtn') || document.querySelector('button[onclick="startTraining()"]');
    
    if (trainBtn) {
        trainBtn.disabled = true;
        trainBtn.style.background = '#27ae60';
        trainBtn.textContent = 'Training...';
    }
    
    const profile = document.getElementById('cost-profile').value;
    const epochs = document.getElementById('epochs').value;
    
    fetch('/api/train', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({cost_profile: profile, epochs: parseInt(epochs)})
    })
    .then(r => r.json())
    .then(data => {
        console.log('Training started:', data);
    });
}

function completeTraining(data) {
    const trainBtn = document.getElementById('trainBtn') || document.querySelector('button[onclick="startTraining()"]');
    if (trainBtn) {
        trainBtn.disabled = false;
        trainBtn.style.background = '#667eea';
        trainBtn.textContent = 'Start Training';
    }
    
    const epochInfo = document.getElementById('epoch-info');
    if (epochInfo) {
        epochInfo.textContent = `✅ Training Complete! Final Loss: ${data.final_loss.toFixed(4)}`;
        epochInfo.style.color = '#27ae60';
    }
}

function updateSystemStatus() {
    fetch('/api/system/status')
        .then(r => r.json())
        .then(data => {
            const statusElement = document.getElementById('system-status');
            if (statusElement) {
                statusElement.innerHTML = `
                    GPU: ${data.gpu.available ? data.gpu.name : 'Not available'}<br>
                    Memory: ${data.gpu.memory_used.toFixed(1)}/${data.gpu.memory_total.toFixed(1)} GB
                `;
            }
        });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    updateSystemStatus();
    setInterval(updateSystemStatus, 5000);
});
