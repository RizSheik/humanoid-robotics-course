---
title: Practical Lab - Digital Twin Implementation
description: Hands-on lab exercises implementing core concepts of digital twin systems
sidebar_position: 102
---

# Practical Lab - Digital Twin Implementation

## Lab Overview

This lab provides hands-on experience implementing and experimenting with digital twin systems. Students will work with simulation environments and real sensors to create virtual replicas of physical systems, implement real-time synchronization, and validate twin accuracy. The lab emphasizes practical implementation skills while reinforcing theoretical concepts of digital twin architecture and operation.

## Lab Objectives

By completing this lab, students will be able to:
- Implement a basic digital twin for a physical system
- Create real-time synchronization between physical and virtual systems
- Validate digital twin accuracy using appropriate metrics
- Design and implement twin-to-twin communication protocols
- Evaluate performance of digital twin systems

## Prerequisites and Setup

### Software Requirements
- Python 3.8+ with libraries: numpy, pandas, matplotlib, flask, kafka-python
- Node.js and npm for visualization components
- Docker for containerized simulation environments
- Git for version control
- MQTT broker (Mosquitto or HiveMQ)
- Digital twin platform (Azure Digital Twins SDK or AWS IoT TwinMaker SDK)

### Hardware Requirements (Physical Components)
- Raspberry Pi 4 or similar single-board computer
- Temperature sensor (DHT22 or DS18B20)
- Humidity sensor (optional)
- LED and button for actuator simulation
- Breadboard and jumper wires
- Laptop/computer for running simulation and visualization

### Simulation Environment Setup
```bash
# Create project directory
mkdir digital-twin-lab
cd digital-twin-lab

# Create virtual environment
python -m venv dt_env
source dt_env/bin/activate  # On Windows: dt_env\Scripts\activate

# Install required packages
pip install numpy pandas matplotlib flask kafka-python paho-mqtt requests

# Set up MQTT broker (using docker)
docker run -it -p 1883:1883 --name mqtt-broker eclipse-mosquitto
```

## Lab Exercise 1: Basic Digital Twin Creation

### Objective
Create a basic digital twin of a simple physical system with temperature sensing capabilities.

### Steps
1. Set up the physical system:
```python
# sensor_emulator.py - Simulates physical sensor readings
import time
import json
import random
from paho.mqtt import client as mqtt_client

class PhysicalSystemEmulator:
    def __init__(self, broker='localhost', port=1883):
        self.broker = broker
        self.port = port
        self.client = self.connect_mqtt()
        
    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT broker")
            else:
                print(f"Failed to connect, return code {rc}")
        
        client = mqtt_client.Client()
        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        client.loop_start()
        return client
    
    def simulate_sensor_reading(self):
        # Simulate realistic temperature readings with noise
        base_temp = 22.0  # Base temperature in Celsius
        # Add some variation and noise
        temperature = base_temp + random.gauss(0, 1) + 2 * random.sin(time.time() / 300)  # Slow oscillation
        humidity = 45 + random.gauss(0, 2)  # Humidity with noise
        
        reading = {
            'timestamp': time.time(),
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'device_id': 'lab_sensor_001'
        }
        
        return reading
    
    def publish_sensor_data(self):
        while True:
            reading = self.simulate_sensor_reading()
            self.client.publish('sensor_data/physical', json.dumps(reading))
            print(f"Published: {reading}")
            time.sleep(2)  # Publish every 2 seconds

if __name__ == '__main__':
    emulator = PhysicalSystemEmulator()
    emulator.publish_sensor_data()
```

2. Create the digital twin system:
```python
# digital_twin.py - Implementing the digital twin
import json
import time
from threading import Thread
from flask import Flask, jsonify, request
import paho.mqtt.client as mqtt_client

class DigitalTwin:
    def __init__(self):
        self.state = {
            'temperature': 20.0,
            'humidity': 50.0,
            'timestamp': time.time()
        }
        self.state_history = []
        self.mqtt_client = None
        self.setup_mqtt()
        
    def setup_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Twin connected to MQTT")
                client.subscribe('sensor_data/physical')
            else:
                print(f"Failed to connect to MQTT, return code {rc}")
        
        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload.decode())
                self.update_state(data)
                print(f"Twin updated: {data}")
            except json.JSONDecodeError:
                print("Failed to decode message")
        
        self.mqtt_client = mqtt_client.Client()
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        self.mqtt_client.connect('localhost', 1883, 60)
        self.mqtt_client.loop_start()
    
    def update_state(self, sensor_data):
        # Update twin state with new sensor data
        self.state.update({
            'temperature': sensor_data['temperature'],
            'humidity': sensor_data['humidity'],
            'timestamp': sensor_data['timestamp']
        })
        
        # Store in history with timestamp
        self.state_history.append({
            'temperature': sensor_data['temperature'],
            'humidity': sensor_data['humidity'],
            'timestamp': sensor_data['timestamp'],
            'received_at': time.time()
        })
        
        # Keep only last 1000 readings
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
    
    def get_current_state(self):
        return self.state
    
    def get_state_history(self, limit=50):
        return self.state_history[-limit:]

# Create Flask web API for the twin
app = Flask(__name__)
twin = DigitalTwin()

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify(twin.get_current_state())

@app.route('/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', default=50, type=int)
    return jsonify(twin.get_state_history(limit))

@app.route('/predict', methods=['POST'])
def predict():
    # Simple prediction based on current trend
    current_state = twin.get_current_state()
    # Naive prediction: assume temperature will continue current trend
    # In a real system, this would use ML models
    
    # For demo purposes, just return current state + small change
    prediction = {
        'predicted_temperature': current_state['temperature'] + 0.1,
        'predicted_humidity': current_state['humidity'] - 0.05,
        'prediction_time': time.time() + 60  # Predict 1 minute ahead
    }
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

3. Run the physical system and digital twin:
```bash
# Terminal 1: Run sensor emulator
python sensor_emulator.py

# Terminal 2: Run digital twin
python digital_twin.py

# Terminal 3: Test the API
curl http://localhost:5000/state
curl http://localhost:5000/history
```

### Deliverables
- Working physical sensor emulator
- Digital twin implementation with MQTT communication
- API endpoints for accessing twin state
- Basic visualization of sensor readings vs. twin predictions

## Lab Exercise 2: Advanced Synchronization and Validation

### Objective
Implement advanced synchronization mechanisms and validate twin accuracy using appropriate metrics.

### Steps
1. Create enhanced synchronization system:
```python
# enhanced_twin.py - Advanced twin with synchronization
import threading
import time
import json
import numpy as np
from scipy import stats
from collections import deque

class EnhancedDigitalTwin:
    def __init__(self, temporal_tolerance=0.05):
        self.state = {
            'temperature': 20.0,
            'humidity': 50.0,
            'timestamp': time.time()
        }
        self.temporal_tolerance = temporal_tolerance
        self.physical_states = deque(maxlen=100)  # Store last 100 readings
        self.twin_predictions = deque(maxlen=100)  # Store twin predictions
        self.validation_metrics = {
            'rmse': float('inf'),
            'mae': float('inf'),
            'correlation': 0.0
        }
        
        # Synchronization thread
        self.sync_lock = threading.Lock()
        self.running = True
        
    def add_physical_reading(self, reading):
        with self.sync_lock:
            self.physical_states.append(reading)
    
    def predict_next_state(self, time_ahead=5.0):
        """Predict state for time_ahead seconds in the future"""
        if len(self.physical_states) < 2:
            return self.state.copy()
        
        with self.sync_lock:
            # Extract recent data for trend analysis
            recent = list(self.physical_states)[-10:]  # Last 10 readings
            
            if len(recent) < 2:
                return self.state.copy()
            
            # Calculate time and temperature arrays
            times = np.array([r['timestamp'] for r in recent])
            temps = np.array([r['temperature'] for r in recent])
            hums = np.array([r['humidity'] for r in recent])
            
            # Perform linear regression to predict trend
            if len(times) > 1:
                # Temperature trend
                temp_slope, temp_intercept, _, _, _ = stats.linregress(times, temps)
                # Humidity trend
                hum_slope, hum_intercept, _, _, _ = stats.linregress(times, hums)
                
                # Predict at future time
                future_time = times[-1] + time_ahead
                future_temp = temp_slope * future_time + temp_intercept
                future_hum = hum_slope * future_time + hum_intercept
                
                prediction = {
                    'predicted_temperature': float(future_temp),
                    'predicted_humidity': float(future_hum),
                    'prediction_time': future_time,
                    'confidence': 0.8  # Placeholder confidence value
                }
                
                self.twin_predictions.append({
                    'predicted': prediction,
                    'actual_time': time.time()
                })
                
                return prediction
        
        # Fallback: return current state if prediction fails
        return {
            'predicted_temperature': self.state['temperature'],
            'predicted_humidity': self.state['humidity'],
            'prediction_time': time.time(),
            'confidence': 0.5
        }
    
    def validate_prediction(self, actual_reading):
        """Validate previous predictions against actual readings"""
        if not self.twin_predictions:
            return
        
        # Find the most recent prediction that should have occurred
        actual_time = actual_reading['timestamp']
        
        with self.sync_lock:
            for i, pred_data in enumerate(list(self.twin_predictions)):
                predicted = pred_data['predicted']
                
                # Check if this prediction time is close to actual reading time
                if abs(predicted['prediction_time'] - actual_time) < 2.0:  # 2 second tolerance
                    # Calculate validation metrics
                    temp_error = abs(predicted['predicted_temperature'] - actual_reading['temperature'])
                    hum_error = abs(predicted['predicted_humidity'] - actual_reading['humidity'])
                    
                    # Update metrics (simplified approach)
                    if self.validation_metrics['rmse'] == float('inf'):
                        self.validation_metrics['rmse'] = np.sqrt(temp_error**2 + hum_error**2)
                        self.validation_metrics['mae'] = (temp_error + hum_error) / 2
                    else:
                        # Running averages (exponential moving average)
                        alpha = 0.1  # Smoothing factor
                        current_rmse = np.sqrt(temp_error**2 + hum_error**2)
                        self.validation_metrics['rmse'] = (
                            alpha * current_rmse + 
                            (1 - alpha) * self.validation_metrics['rmse']
                        )
                        self.validation_metrics['mae'] = (
                            alpha * (temp_error + hum_error) / 2 +
                            (1 - alpha) * self.validation_metrics['mae']
                        )
                    
                    # Remove processed prediction
                    self.twin_predictions.remove(pred_data)
                    break
    
    def get_validation_report(self):
        return self.validation_metrics

# Enhanced MQTT client for synchronization
class SynchronizedTwinClient:
    def __init__(self):
        self.twin = EnhancedDigitalTwin()
        self.setup_mqtt()
        
    def setup_mqtt(self):
        import paho.mqtt.client as mqtt_client
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Synchronized twin connected to MQTT")
                client.subscribe('sensor_data/physical')
                client.subscribe('control_commands')  # For sending control back to physical system
            else:
                print(f"Failed to connect, return code {rc}")
        
        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload.decode())
                
                if msg.topic == 'sensor_data/physical':
                    self.twin.add_physical_reading(data)
                    self.twin.validate_prediction(data)
                    print(f"Physical data: {data}")
                    print(f"Validation metrics: {self.twin.get_validation_report()}")
                    
                elif msg.topic == 'control_commands':
                    # Process control commands from twin (if any)
                    print(f"Control command: {data}")
                    
            except json.JSONDecodeError:
                print("Failed to decode message")
        
        self.client = mqtt_client.Client()
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.connect('localhost', 1883, 60)
        self.client.loop_start()
    
    def run_prediction_loop(self):
        while True:
            prediction = self.twin.predict_next_state(time_ahead=5.0)
            print(f"Prediction: {prediction}")
            
            # Send prediction via MQTT for visualization or other systems
            self.client.publish('twin_predictions', json.dumps(prediction))
            
            time.sleep(10)  # Update prediction every 10 seconds

# Run the synchronized twin
if __name__ == '__main__':
    twin_client = SynchronizedTwinClient()
    
    # Start prediction loop in a separate thread
    prediction_thread = threading.Thread(target=twin_client.run_prediction_loop)
    prediction_thread.daemon = True
    prediction_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
```

2. Implement validation API:
```python
# validation_api.py - API for monitoring twin validation metrics
from flask import Flask, jsonify
import threading

app = Flask(__name__)

# Global instance of the enhanced twin (in a real system, you'd use a proper service architecture)
global_twin = None

@app.route('/validation', methods=['GET'])
def get_validation():
    if global_twin is None:
        return jsonify({'error': 'Twin not initialized'}), 500
    
    return jsonify(global_twin.get_validation_report())

@app.route('/validation/history', methods=['GET'])
def get_validation_history():
    # In a real implementation, this would return historical validation data
    return jsonify({
        'history': [
            {'timestamp': time.time(), 'rmse': 0.5, 'mae': 0.3},
            {'timestamp': time.time() - 10, 'rmse': 0.6, 'mae': 0.4},
            {'timestamp': time.time() - 20, 'rmse': 0.4, 'mae': 0.25}
        ]
    })

if __name__ == '__main__':
    # In a real system, this would be integrated with the main twin system
    app.run(host='0.0.0.0', port=5001, debug=False)
```

### Deliverables
- Enhanced synchronization system with temporal alignment
- Validation metrics calculation and reporting
- Prediction capabilities with accuracy assessment
- API endpoints for monitoring validation performance

## Lab Exercise 3: Multi-Twin Communication

### Objective
Implement communication between multiple digital twins to simulate a network of interconnected systems.

### Steps
1. Create a multi-twin communication system:
```python
# multi_twin_network.py - Network of interconnected twins
import json
import time
import threading
from typing import Dict, List
import random

class NetworkedDigitalTwin:
    def __init__(self, twin_id: str, broker='localhost', port=1883):
        self.twin_id = twin_id
        self.local_state = {
            'temperature': 20.0 + random.uniform(-5, 5),
            'humidity': 50.0 + random.uniform(-10, 10),
            'timestamp': time.time()
        }
        self.connected_twins: Dict[str, dict] = {}
        self.broker = broker
        self.port = port
        self.setup_mqtt()
        
    def setup_mqtt(self):
        import paho.mqtt.client as mqtt_client
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"Twin {self.twin_id} connected to MQTT")
                # Subscribe to messages for this twin
                client.subscribe(f'twin_state/{self.twin_id}')
                # Subscribe to general twin updates
                client.subscribe('twin_network/updates')
            else:
                print(f"Twin {self.twin_id} failed to connect, return code {rc}")
        
        def on_message(client, userdata, msg):
            try:
                data = json.loads(msg.payload.decode())
                
                if msg.topic.startswith('twin_state/'):
                    # Update connected twin states
                    source_twin_id = msg.topic.split('/')[-1]
                    if source_twin_id != self.twin_id:
                        self.connected_twins[source_twin_id] = data
                        
                elif msg.topic == 'twin_network/updates':
                    # General network updates
                    if data.get('twin_id') != self.twin_id:
                        self.handle_network_update(data)
                        
            except json.JSONDecodeError:
                print(f"Twin {self.twin_id}: Failed to decode message")
        
        self.client = mqtt_client.Client()
        self.client.on_connect = on_connect
        self.client.on_message = on_message
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()
    
    def handle_network_update(self, update_data):
        """Handle updates from other twins in the network"""
        print(f"Twin {self.twin_id} received update: {update_data}")
        # In a real system, this would update interconnected models
        
    def broadcast_state(self):
        """Broadcast current state to other twins"""
        state_to_broadcast = {
            'twin_id': self.twin_id,
            'state': self.local_state,
            'timestamp': time.time()
        }
        
        self.client.publish(f'twin_state/{self.twin_id}', json.dumps(state_to_broadcast))
        self.client.publish('twin_network/updates', json.dumps(state_to_broadcast))
        
        print(f"Twin {self.twin_id} broadcasted state")
    
    def update_local_state(self):
        """Simulate local state changes"""
        # Add some realistic variation
        temp_change = random.uniform(-0.1, 0.1)
        hum_change = random.uniform(-0.2, 0.2)
        
        self.local_state['temperature'] += temp_change
        self.local_state['humidity'] += hum_change
        self.local_state['timestamp'] = time.time()
        
    def get_connected_state_summary(self):
        """Get summary of all connected twins"""
        summary = {
            'local_twin': self.twin_id,
            'local_state': self.local_state,
            'connected_twins': len(self.connected_twins),
            'twin_states': self.connected_twins.copy()
        }
        return summary

class TwinNetworkCoordinator:
    def __init__(self, broker='localhost', port=1883):
        self.broker = broker
        self.port = port
        self.twins: Dict[str, NetworkedDigitalTwin] = {}
        self.network_running = False
        
    def create_twin(self, twin_id: str) -> NetworkedDigitalTwin:
        twin = NetworkedDigitalTwin(twin_id, self.broker, self.port)
        self.twins[twin_id] = twin
        return twin
    
    def start_network(self):
        """Start coordinated network updates"""
        self.network_running = True
        
        def network_update_loop():
            while self.network_running:
                # Update all local states
                for twin in self.twins.values():
                    twin.update_local_state()
                    twin.broadcast_state()
                
                time.sleep(5)  # Update every 5 seconds
        
        network_thread = threading.Thread(target=network_update_loop)
        network_thread.daemon = True
        network_thread.start()
        
        print(f"Created {len(self.twins)} twins in network")
    
    def get_network_summary(self):
        """Get summary of entire network"""
        summary = {
            'network_size': len(self.twins),
            'twin_summaries': {}
        }
        
        for twin_id, twin in self.twins.items():
            summary['twin_summaries'][twin_id] = twin.get_connected_state_summary()
        
        return summary

# Example usage
if __name__ == '__main__':
    coordinator = TwinNetworkCoordinator()
    
    # Create multiple interconnected twins
    coordinator.create_twin('sensor_twin_001')
    coordinator.create_twin('actuator_twin_002')
    coordinator.create_twin('environment_twin_003')
    
    # Start network coordination
    coordinator.start_network()
    
    # Run for demonstration
    try:
        while True:
            # Print network summary every 15 seconds
            summary = coordinator.get_network_summary()
            print(f"\nNetwork Summary - {len(summary['twin_summaries'])} twins:")
            for twin_id, twin_summary in summary['twin_summaries'].items():
                print(f"  {twin_id}: Temp={twin_summary['local_state']['temperature']:.2f}°C, "
                      f"Hum={twin_summary['local_state']['humidity']:.2f}%, "
                      f"Connected: {twin_summary['connected_twins']}")
            
            time.sleep(15)
    except KeyboardInterrupt:
        print("Shutting down twin network...")
        coordinator.network_running = False
```

2. Create visualization for the twin network:
```python
# twin_visualization.py - Simple visualization of twin network
from flask import Flask, render_template_string, jsonify
import threading
import time

app = Flask(__name__)

# Global coordinator reference
global_coordinator = None

@app.route('/')
def index():
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Digital Twin Network Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .twin-card { 
                border: 1px solid #ddd; 
                margin: 10px; 
                padding: 15px; 
                border-radius: 5px;
                display: inline-block;
                min-width: 200px;
                vertical-align: top;
            }
            .chart-container { width: 400px; height: 200px; margin: 10px; }
        </style>
    </head>
    <body>
        <h1>Digital Twin Network Dashboard</h1>
        <div id="twin-status"></div>
        <div id="charts"></div>
        
        <script>
            async function updateDashboard() {
                try {
                    const response = await fetch('/network/summary');
                    const data = await response.json();
                    
                    // Update twin status display
                    let statusHTML = '';
                    for (const [twinId, twinData] of Object.entries(data.twin_summaries)) {
                        statusHTML += `
                            <div class="twin-card">
                                <h3>${twinId}</h3>
                                <p>Temperature: ${twinData.local_state.temperature.toFixed(2)}°C</p>
                                <p>Humidity: ${twinData.local_state.humidity.toFixed(2)}%</p>
                                <p>Connected Twins: ${twinData.connected_twins}</p>
                                <p>Timestamp: ${new Date(twinData.local_state.timestamp * 1000).toLocaleTimeString()}</p>
                            </div>
                        `;
                    }
                    document.getElementById('twin-status').innerHTML = statusHTML;
                    
                    // Update charts (simplified)
                    document.getElementById('charts').innerHTML = '<p>Charts would be displayed here in a full implementation</p>';
                    
                } catch (error) {
                    console.error('Error updating dashboard:', error);
                }
            }
            
            // Update every 3 seconds
            setInterval(updateDashboard, 3000);
            updateDashboard(); // Initial update
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/network/summary')
def get_network_summary():
    if global_coordinator is None:
        return jsonify({'error': 'Coordinator not initialized'}), 500
    
    return jsonify(global_coordinator.get_network_summary())

if __name__ == '__main__':
    # In a real implementation, this would be integrated with the main system
    app.run(host='0.0.0.0', port=5002, debug=False)
```

### Deliverables
- Working multi-twin network with communication protocol
- State broadcasting and synchronization between twins
- Network monitoring and visualization capabilities
- Performance metrics for networked twin operations

## Lab Exercise 4: Integration with Simulation

### Objective
Integrate the digital twin system with a physics-based simulation to demonstrate model-in-the-loop capabilities.

### Steps
1. Create a physics-based simulation model:
```python
# physics_simulation.py - Physics simulation for digital twin
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import threading
import time

class PhysicsBasedSimulator:
    def __init__(self):
        # Physical parameters
        self.thermal_mass = 1000.0  # J/K (thermal mass of the system)
        self.heat_transfer_coeff = 5.0  # W/K (heat transfer to environment)
        self.ambient_temp = 20.0  # °C (ambient temperature)
        
        # State variables
        self.temperature = 22.0  # Current temperature in °C
        self.power_input = 0.0  # Current power input in W
        
    def physics_model(self, temp, t, power_input):
        """
        Physics model: dT/dt = (P - h(T - T_ambient)) / C
        Where:
        - T: Temperature
        - P: Power input
        - h: Heat transfer coefficient
        - C: Thermal mass
        - T_ambient: Ambient temperature
        """
        rate_of_change = (power_input - self.heat_transfer_coeff * (temp - self.ambient_temp)) / self.thermal_mass
        return rate_of_change
    
    def simulate_step(self, dt=1.0):
        """Simulate one time step"""
        # Current power input affects the system
        t = [0, dt]
        solution = odeint(self.physics_model, [self.temperature], t, args=(self.power_input,))
        
        # Update temperature based on simulation
        self.temperature = float(solution[-1][0])
        return self.temperature

class SimulationTwinAdapter:
    def __init__(self):
        self.simulator = PhysicsBasedSimulator()
        self.simulation_thread = None
        self.running = False
        
    def start_simulation(self):
        """Start the simulation in a background thread"""
        self.running = True
        
        def sim_loop():
            while self.running:
                # Update simulation based on current power input
                current_temp = self.simulator.simulate_step(dt=1.0)
                
                # In a real system, this would update the digital twin state
                print(f"Simulation: Temp={current_temp:.2f}°C, Power={self.simulator.power_input:.2f}W")
                
                time.sleep(1.0)  # Real-time simulation
        
        self.simulation_thread = threading.Thread(target=sim_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def set_power_input(self, power_watts):
        """Set power input to simulation (e.g., from heater, cooling system)"""
        self.simulator.power_input = max(0.0, power_watts)  # Ensure non-negative
    
    def get_temperature(self):
        """Get current temperature from simulation"""
        return self.simulator.temperature
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)

# Example integration with digital twin
class PhysicsEnhancedTwin:
    def __init__(self):
        self.sim_adapter = SimulationTwinAdapter()
        self.target_temperature = 25.0  # Desired temperature
        self.control_power = 0.0  # Current control output
        
    def start_system(self):
        """Start the physics-based twin system"""
        self.sim_adapter.start_simulation()
        
        # Start control loop
        self.start_control_loop()
    
    def start_control_loop(self):
        """Start closed-loop control"""
        def control_loop():
            while self.sim_adapter.running:
                # Get current temperature from simulation
                current_temp = self.sim_adapter.get_temperature()
                
                # Simple PID-like control
                error = self.target_temperature - current_temp
                # Proportional control
                control_output = 100 * error  # Adjust based on error
                control_output = max(0, min(500, control_output))  # Limit to 0-500W
                
                # Apply to simulation
                self.sim_adapter.set_power_input(control_output)
                self.control_power = control_output
                
                print(f"Control: Setpoint={self.target_temperature}°C, "
                      f"Current={current_temp:.2f}°C, Output={control_output:.2f}W, "
                      f"Error={error:.2f}°C")
                
                time.sleep(2.0)  # Control loop every 2 seconds
        
        control_thread = threading.Thread(target=control_loop)
        control_thread.daemon = True
        control_thread.start()
    
    def set_target_temperature(self, temp):
        """Set the target temperature for the system"""
        self.target_temperature = temp

if __name__ == '__main__':
    twin = PhysicsEnhancedTwin()
    print("Starting physics-enhanced digital twin...")
    
    # Start the system
    twin.start_system()
    
    try:
        # Allow system to run and observe behavior
        while True:
            time.sleep(10)
            print(f"System running - Target: {twin.target_temperature}°C, "
                  f"Current: {twin.sim_adapter.get_temperature():.2f}°C, "
                  f"Power: {twin.control_power:.2f}W")
    except KeyboardInterrupt:
        print("Stopping physics-enhanced twin...")
        twin.sim_adapter.stop_simulation()
```

2. Integrate with the existing digital twin system:
```python
# integration_demo.py - Complete integration demo
import threading
import time
import json
from enhanced_twin import EnhancedDigitalTwin
from physics_simulation import PhysicsEnhancedTwin

class IntegratedTwinSystem:
    def __init__(self):
        self.enhanced_twin = EnhancedDigitalTwin()
        self.physics_twin = PhysicsEnhancedTwin()
        self.system_running = False
        
    def start_system(self):
        """Start the integrated system"""
        self.system_running = True
        print("Starting integrated digital twin system...")
        
        # Start physics-based twin
        self.physics_twin.start_system()
        
        # Periodic synchronization between enhanced and physics twins
        def sync_loop():
            while self.system_running:
                # Get current state from physics simulation
                phys_temp = self.physics_twin.sim_adapter.get_temperature()
                
                # Create sensor-like reading
                reading = {
                    'temperature': phys_temp,
                    'humidity': 45.0 + 5 * (phys_temp - 20) / 10,  # Correlated humidity
                    'timestamp': time.time(),
                    'device_id': 'physics_simulated'
                }
                
                # Update enhanced twin
                self.enhanced_twin.add_physical_reading(reading)
                self.enhanced_twin.validate_prediction(reading)
                
                print(f"Synchronized: Phys Temp={phys_temp:.2f}°C, "
                      f"Enhanced Temp={reading['temperature']:.2f}°C")
                
                time.sleep(5)  # Sync every 5 seconds
        
        sync_thread = threading.Thread(target=sync_loop)
        sync_thread.daemon = True
        sync_thread.start()
    
    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            'physics_twin': {
                'temperature': self.physics_twin.sim_adapter.get_temperature(),
                'power_input': self.physics_twin.control_power,
                'target': self.physics_twin.target_temperature
            },
            'enhanced_twin': {
                'validation_metrics': self.enhanced_twin.get_validation_report(),
                'state_count': len(self.enhanced_twin.physical_states)
            }
        }

if __name__ == '__main__':
    system = IntegratedTwinSystem()
    system.start_system()
    
    try:
        while True:
            status = system.get_system_status()
            print(f"\nSystem Status:")
            print(f"  Physics Twin: {status['physics_twin']['temperature']:.2f}°C "
                  f"(Target: {status['physics_twin']['target']:.2f}°C)")
            print(f"  Validation RMSE: {status['enhanced_twin']['validation_metrics']['rmse']:.4f}")
            print(f"  State History: {status['enhanced_twin']['state_count']} readings")
            
            time.sleep(15)
    except KeyboardInterrupt:
        print("Stopping integrated system...")
        system.system_running = False
```

### Deliverables
- Physics-based simulation model integrated with digital twin
- Closed-loop control system with simulation feedback
- Synchronization between simulation and enhanced twin
- Performance evaluation of the integrated system

## Assessment Rubric

### Exercise 1: Basic Digital Twin (25 points)
- **Implementation Quality**: Proper use of MQTT for communication (10 points)
- **API Development**: Working endpoints for state access (10 points)
- **Documentation**: Clear code comments and setup instructions (5 points)

### Exercise 2: Advanced Synchronization (25 points)
- **Synchronization Algorithm**: Proper temporal alignment (10 points)
- **Validation Metrics**: Correct implementation of accuracy measures (10 points)
- **Prediction System**: Working prediction with accuracy assessment (5 points)

### Exercise 3: Multi-Twin Network (25 points)
- **Network Protocol**: Proper twin-to-twin communication (10 points)
- **State Management**: Handling of multiple twin states (10 points)
- **Visualization**: Working dashboard or monitoring interface (5 points)

### Exercise 4: Physics Integration (25 points)
- **Physics Model**: Accurate simulation of physical system (10 points)
- **Integration Quality**: Proper coupling between twin and simulation (10 points)
- **Control System**: Working closed-loop control (5 points)

## Additional Resources

### Recommended Reading
- "Digital Twin: Manufacturing Excellence through Virtual Factory Replication"
- "Internet of Things and Digital Twins: Concepts, Technologies, and Applications"
- Research papers on real-time synchronization in digital twin systems

### Troubleshooting Guide
- Check MQTT broker connectivity
- Verify timestamp synchronization
- Monitor resource usage for performance issues
- Validate sensor data quality and ranges

This lab provides comprehensive hands-on experience with digital twin implementation, from basic concept to advanced integration with physics-based simulation models.