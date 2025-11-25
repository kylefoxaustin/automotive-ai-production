#!/usr/bin/env python3
"""
Automotive Body Control AI - Production System
Complete training, serving, and monitoring platform
"""

import torch
import torch.nn as nn
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import redis
import json
import os
import logging
from datetime import datetime
import numpy as np
import threading

# Import our models
from models.advanced_models import UnifiedBodyControlAI
from training.trainer import AutomotiveTrainer
from data.can_bus_loader import CANBusDataLoader
from exporters.onnx_exporter import ONNXExporter
from utils.metrics import MetricsCollector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='../web/templates', static_folder='../web/static')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///automotive_ai.db')
app.config['SECRET_KEY'] = 'automotive-ai-secret-key'

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

# Redis connection - use 'redis' hostname in container
redis_client = redis.StrictRedis(
    host=os.getenv('REDIS_HOST', 'redis'),  # Changed from 'localhost' to 'redis'
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Global model and trainer
model = None
trainer = None
training_state = {
    'is_training': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'current_loss': 0,
    'history': []
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system/status')
def system_status():
    """Get complete system status"""
    try:
        redis_status = redis_client.ping()
    except:
        redis_status = False
    
    status = {
        'gpu': {
            'available': torch.cuda.is_available(),
            'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'memory_used': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        },
        'services': {
            'database': db.session.is_active,
            'redis': redis_status,
            'can_bus': False
        },
        'model': {
            'loaded': model is not None,
            'parameters': model.get_total_parameters() if model else 0
        }
    }
    return jsonify(status)

@app.route('/api/train', methods=['POST'])
def start_training():
    """Start model training with specified configuration"""
    global model, trainer, training_state
    
    if training_state['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    data = request.json
    config = {
        'epochs': data.get('epochs', 100),
        'batch_size': data.get('batch_size', 32),
        'learning_rate': data.get('learning_rate', 0.001),
        'cost_profile': data.get('cost_profile', 'balanced'),
        'use_can_data': data.get('use_can_data', False)
    }
    
    # Initialize model and trainer
    model = UnifiedBodyControlAI().to(device)
    trainer = AutomotiveTrainer(model, config, device)
    
    # Update training state
    training_state['is_training'] = True
    training_state['current_epoch'] = 0
    training_state['total_epochs'] = config['epochs']
    training_state['history'] = []
    
    # Define progress callback
    def training_progress(epoch, loss, metrics):
        """Emit training progress via WebSocket"""
        training_state['current_epoch'] = epoch
        training_state['current_loss'] = loss
        training_state['history'].append(loss)
        
        # Emit to all connected clients
        socketio.emit('training_progress', {
            'epoch': epoch,
            'total_epochs': training_state['total_epochs'],
            'loss': loss,
            'metrics': metrics,
            'history': training_state['history'][-50:],  # Last 50 points
            'timestamp': datetime.now().isoformat()
        })
    
    # Start training in background thread
    def train_thread():
        try:
            trainer.train(progress_callback=training_progress)
        finally:
            training_state['is_training'] = False
            socketio.emit('training_complete', {
                'final_loss': training_state['current_loss'],
                'total_epochs': training_state['total_epochs']
            })
    
    thread = threading.Thread(target=train_thread)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Training started', 'config': config})

@app.route('/api/status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_state)

@app.route('/api/export/<format>')
def export_model(format):
    """Export trained model"""
    if model is None:
        return jsonify({'error': 'No model trained yet'}), 400
    
    exporter = ONNXExporter(model)
    
    if format == 'onnx':
        path = exporter.export_onnx('models/exported/automotive_ai.onnx')
    elif format == 'tensorrt':
        path = exporter.export_tensorrt('models/exported/automotive_ai.trt')
    else:
        return jsonify({'error': f'Unsupported format: {format}'}), 400
    
    return jsonify({'path': path, 'format': format})

@app.route('/api/can/connect', methods=['POST'])
def connect_can_bus():
    """Connect to CAN bus interface"""
    data = request.json
    interface = data.get('interface', 'can0')
    bitrate = data.get('bitrate', 500000)
    
    try:
        can_loader = CANBusDataLoader(interface, bitrate)
        can_loader.connect()
        return jsonify({'status': 'Connected', 'interface': interface})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('connected', {'data': 'Connected to Automotive AI server'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🚗 AUTOMOTIVE BODY CONTROL AI - PRODUCTION SYSTEM")
    logger.info("="*60)
    logger.info("Starting all services...")
    
    # Run with WebSocket support
    socketio.run(app, host='0.0.0.0', port=8080, debug=False, allow_unsafe_werkzeug=True)
