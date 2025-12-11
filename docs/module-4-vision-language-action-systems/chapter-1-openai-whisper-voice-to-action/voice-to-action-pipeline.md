---
id: module-4-chapter-1-voice-to-action-pipeline
title: 'Module 4 — Vision-Language-Action Systems | Chapter 1 — Voice to Action Pipeline'
sidebar_label: 'Chapter 1 — Voice to Action Pipeline'
---

# Chapter 1 — Voice to Action Pipeline

## Complete Pipeline for Voice-Controlled Robot Actions

The voice-to-action pipeline transforms spoken human commands into robot actions through multiple processing stages. This pipeline includes audio preprocessing, speech recognition, natural language understanding, action mapping, and robot control execution.

### Overview of the Voice-to-Action Pipeline

The complete voice-to-action pipeline consists of these stages:

```
Voice Input → Audio Preprocessing → Speech Recognition → Natural Language Processing → Action Mapping → Robot Control → Action Execution
```

Each stage is crucial for converting natural language to robot actions.

### Stage 1: Audio Input and Preprocessing

```python
import pyaudio
import numpy as np
import threading
import queue
import webrtcvad
from scipy import signal
import resampy

class AudioInputProcessor:
    def __init__(self, sample_rate=16000, chunk_duration=0.03):
        """
        Initialize audio input processor
        sample_rate: Target sample rate for Whisper (16kHz recommended)
        chunk_duration: Duration of audio chunks in seconds
        """
        self.rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.rate * chunk_duration)
        
        # Initialize PyAudio for microphone access
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Initialize VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # Aggressiveness mode (0-3)
        
        # Audio buffers
        self.audio_buffer = queue.Queue()
        self.voice_chunks = []
        self.listening = False
        self.recording = False
    
    def start_listening(self):
        """Start listening for voice commands"""
        self.listening = True
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,  # Mono
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Start audio processing thread
        self.listen_thread = threading.Thread(target=self._listen_worker)
        self.listen_thread.daemon = True
        self.listen_thread.start()
        
        print("Audio input processor started")
    
    def _listen_worker(self):
        """Thread worker for continuous audio processing"""
        silence_frames = 0
        max_silence_frames = 50  # Max frames of silence before command is complete
        
        while self.listening:
            # Read audio chunk
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Check for voice activity
            if self._is_voice_present(audio_chunk):
                # Voice detected, add to voice chunks
                self.voice_chunks.extend(audio_chunk)
                silence_frames = 0  # Reset silence counter
            else:
                # Silence detected
                silence_frames += 1
                
                # If we have accumulated voice data and sufficient silence
                if len(self.voice_chunks) > 0 and silence_frames > max_silence_frames:
                    # Process the collected voice command
                    voice_command = np.array(self.voice_chunks)
                    
                    # Add to buffer for processing
                    self.audio_buffer.put(voice_command)
                    
                    # Clear voice chunks for next command
                    self.voice_chunks = []
                    silence_frames = 0
        
        # Clean up
        self.stream.stop_stream()
        self.stream.close()
    
    def _is_voice_present(self, audio_chunk):
        """Check if voice is present in audio chunk using VAD"""
        # Convert to int16 for VAD (it requires int16)
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # VAD requires specific frame sizes (10ms, 20ms, or 30ms)
        frame_size = int(self.rate * 0.01)  # 10ms frames
        if len(audio_int16) < frame_size:
            return False
        
        # Process in smaller frames
        num_frames = len(audio_int16) // frame_size
        voice_frames = 0
        
        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = (i + 1) * frame_size
            frame = audio_int16[start_idx:end_idx]
            
            if self.vad.is_speech(frame.tobytes(), self.rate):
                voice_frames += 1
        
        # Consider voice present if more than 30% of frames have voice
        return voice_frames / max(num_frames, 1) > 0.3
    
    def get_voice_command(self, timeout=5.0):
        """Get the next voice command from the buffer"""
        try:
            return self.audio_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_listening(self):
        """Stop listening for audio"""
        self.listening = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
```

### Stage 2: Speech Recognition with Whisper

```python
import whisper
import torch
import os

class SpeechRecognizer:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper speech recognizer
        model_size: tiny, base, small, medium, large
        """
        print(f"Loading Whisper {model_size} model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size).to(self.device)
        print(f"Speech recognizer initialized on {self.device}")
        
        # Set model to evaluation mode
        self.model.eval()
    
    def transcribe_audio(self, audio_array, language="en", temperature=0.0):
        """
        Transcribe audio array to text using Whisper
        """
        with torch.no_grad():
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_array,
                language=language,
                temperature=temperature,
                best_of=1,
                patience=1.0
            )
        
        return {
            'text': result["text"].strip(),
            'segments': result["segments"] if "segments" in result else [],
            'language': result["language"] if "language" in result else language,
            'confidence': 1.0  # Whisper doesn't provide confidence scores
        }
    
    def transcribe_audio_file(self, file_path, language="en"):
        """
        Transcribe audio file to text
        """
        result = self.model.transcribe(file_path, language=language)
        return {
            'text': result["text"].strip(),
            'segments': result["segments"],
            'language': result["language"],
            'confidence': 1.0
        }
```

### Stage 3: Natural Language Processing

```python
import re
from typing import Dict, List, Optional

class NaturalLanguageProcessor:
    def __init__(self):
        """Initialize NLP processor for command understanding"""
        self.action_patterns = self._define_action_patterns()
        self.object_patterns = self._define_object_patterns()
        self.location_patterns = self._define_location_patterns()
        self.quantity_patterns = self._define_quantity_patterns()
    
    def _define_action_patterns(self):
        """Define patterns for recognizing robot actions"""
        return {
            # Navigation actions
            'move_forward': [
                r'move forward',
                r'go forward', 
                r'go straight',
                r'go ahead',
                r'proceed',
                r'advance'
            ],
            'move_backward': [
                r'move backward',
                r'go backward',
                r'go back',
                r'reverse',
                r'move back'
            ],
            'turn_left': [
                r'turn left',
                r'rotate left',
                r'pivot left',
                r'go left'
            ],
            'turn_right': [
                r'turn right',
                r'rotate right',
                r'pivot right',
                r'go right'
            ],
            'navigate_to': [
                r'go to the (\w+)',  # Extract location
                r'go to (\w+)',      # Generic location
                r'go to (.+)',       # Any location
                r'move to the (\w+)',
                r'go towards (\w+)'
            ],
            'grasp_object': [
                r'pick up the (\w+)',
                r'take the (\w+)',
                r'grasp the (\w+)',
                r'grab the (\w+)',
                r'lift the (\w+)',
                r'pick up (.+)',
                r'take (.+)',
                r'grasp (.+)'
            ],
            'place_object': [
                r'place.*on',
                r'put.*on',
                r'place.*at',
                r'put.*at',
                r'set down',
                r'put down',
                r'release'
            ],
            'follow_me': [
                r'follow me',
                r'come with me',
                r'follow',
                r'accompany me'
            ],
            'stop': [
                r'stop',
                r'freeze',
                r'halt',
                r'cease'
            ],
            'wave': [
                r'wave hello',
                r'wave goodbye',
                r'wave',
                r'hello'
            ],
            'introduce': [
                r'introduce yourself',
                r'tell me about yourself',
                r'who are you',
                r'what are you'
            ]
        }
    
    def _define_object_patterns(self):
        """Define patterns for recognizing objects"""
        return {
            'common_objects': [
                r'red cup', r'blue cup', r'green cup',
                r'book', r'bottle', r'pen', r'paper', r'box',
                r'ball', r'toy', r'phone', r'laptop', r'keys',
                r'cup', r'plate', r'bowl', r'knife', r'fork',
                r'spoon', r'table', r'chair', r'door'
            ],
            'object_colors': [
                r'red', r'blue', r'green', r'yellow', r'black',
                r'white', r'orange', r'purple', r'pink', r'gray'
            ],
            'object_shapes': [
                r'cube', r'sphere', r'cylinder', r'cone', r'box'
            ]
        }
    
    def _define_location_patterns(self):
        """Define patterns for recognizing locations"""
        return [
            r'kitchen', r'living room', r'bedroom', r'office', r'bathroom',
            r'dining room', r'garage', r'hallway', r'corridor', r'garden',
            r'outside', r'inside', r'here', r'there', r'over there',
            r'table', r'counter', r'shelf', r'desk', r'couch', r'bed'
        ]
    
    def _define_quantity_patterns(self):
        """Define patterns for recognizing quantities"""
        return {
            r'one': 1, r'two': 2, r'three': 3, r'four': 4, r'five': 5,
            r'small': 'small', r'medium': 'medium', r'large': 'large',
            r'big': 'large', r'tiny': 'small'
        }
    
    def parse_command(self, text: str) -> Dict:
        """Parse natural language command into structured action"""
        text_lower = text.lower().strip()
        result = {
            'raw_text': text,
            'intent': 'unknown',
            'entities': {},
            'confidence': 0.0,
            'action_sequence': []
        }
        
        # Find the best matching action
        best_intent = self._identify_intent(text_lower)
        if best_intent:
            result['intent'] = best_intent['intent']
            result['entities'] = best_intent['entities']
            result['confidence'] = best_intent['confidence']
        
        # Create action sequence based on intent
        result['action_sequence'] = self._create_action_sequence(result)
        
        return result
    
    def _identify_intent(self, text: str) -> Optional[Dict]:
        """Identify the intent from the text"""
        best_match = None
        best_score = 0.0
        
        for intent, patterns in self.action_patterns.items():
            for pattern in patterns:
                # Use regex with word boundaries for better matching
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Calculate confidence based on match strength
                    score = len(matches) * 0.8  # Base score for matches
                    if score > best_score:
                        best_score = score
                        best_match = {
                            'intent': intent,
                            'entities': self._extract_entities(text),
                            'confidence': min(score, 1.0)
                        }
        
        return best_match if best_match else None
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract entities (objects, locations, quantities) from text"""
        entities = {
            'objects': [],
            'locations': [],
            'quantities': [],
            'other': []
        }
        
        # Extract objects
        for obj_pattern in self.object_patterns['common_objects']:
            matches = re.findall(obj_pattern, text, re.IGNORECASE)
            entities['objects'].extend(matches)
        
        # Extract colors
        for color_pattern in self.object_patterns['object_colors']:
            matches = re.findall(r'\b' + color_pattern + r'\b', text, re.IGNORECASE)
            entities['objects'].extend(matches)
        
        # Extract locations
        for loc_pattern in self.location_patterns:
            matches = re.findall(r'\b' + loc_pattern + r'\b', text, re.IGNORECASE)
            entities['locations'].extend(matches)
        
        # Extract quantities
        for quantity_word, quantity_value in self.quantity_patterns.items():
            if re.search(r'\b' + quantity_word + r'\b', text, re.IGNORECASE):
                entities['quantities'].append(quantity_value)
        
        return entities
    
    def _create_action_sequence(self, parsed_result: Dict) -> List[Dict]:
        """Create action sequence from parsed result"""
        sequence = []
        intent = parsed_result['intent']
        
        if intent == 'move_forward':
            sequence.append({
                'action': 'NAVIGATE',
                'direction': 'forward',
                'distance': 'medium'
            })
        elif intent == 'move_backward':
            sequence.append({
                'action': 'NAVIGATE',
                'direction': 'backward',
                'distance': 'medium'
            })
        elif intent == 'turn_left':
            sequence.append({
                'action': 'ROTATE',
                'direction': 'left',
                'angle': 90
            })
        elif intent == 'turn_right':
            sequence.append({
                'action': 'ROTATE',
                'direction': 'right',
                'angle': 90
            })
        elif intent == 'navigate_to':
            # Extract location from entities
            locations = parsed_result['entities'].get('locations', [])
            if locations:
                sequence.append({
                    'action': 'NAVIGATE_TO_LOCATION',
                    'destination': locations[0],
                    'waypoints': []
                })
        elif intent == 'grasp_object':
            # Extract object from entities
            objects = parsed_result['entities'].get('objects', [])
            if objects:
                sequence.append({
                    'action': 'GRASP_OBJECT',
                    'target_object': objects[0],
                    'approach_method': 'reach_and_grasp'
                })
        elif intent == 'place_object':
            locations = parsed_result['entities'].get('locations', [])
            sequence.append({
                'action': 'PLACE_OBJECT',
                'destination': locations[0] if locations else 'table',
                'placement_method': 'safe_placement'
            })
        elif intent == 'follow_me':
            sequence.append({
                'action': 'FOLLOW_LEADER',
                'tracking_method': 'visual_tracking',
                'max_distance': 2.0
            })
        elif intent == 'stop':
            sequence.append({
                'action': 'STOP_CURRENT_ACTION',
                'reason': 'user_request'
            })
        elif intent == 'wave':
            sequence.append({
                'action': 'PERFORM_GESTURE',
                'gesture_type': 'wave',
                'duration': 3.0
            })
        elif intent == 'introduce':
            sequence.append({
                'action': 'PROVIDE_INFORMATION',
                'info_type': 'self_introduction',
                'content': 'I am a humanoid robot designed to assist with daily tasks.'
            })
        
        # Add default action if no specific action identified
        if not sequence:
            sequence.append({
                'action': 'UNKNOWN_COMMAND',
                'raw_input': parsed_result['raw_text']
            })
        
        return sequence
```

### Stage 4: Action Mapping and Validation

```python
import json
from datetime import datetime
from typing import Any

class ActionMapper:
    def __init__(self):
        """Initialize action mapper to translate to robot-specific commands"""
        self.robot_capabilities = self._load_robot_capabilities()
        self.action_validators = self._initialize_validators()
    
    def _load_robot_capabilities(self):
        """Load robot's actual capabilities and limitations"""
        return {
            'navigation': {
                'max_speed': 0.5,  # m/s
                'min_turn_radius': 0.3,  # meters
                'max_distance': 10.0  # meters
            },
            'manipulation': {
                'reachable_space': {'min': [-1.0, -1.0, 0.1], 'max': [1.0, 1.0, 2.0]},  # meters
                'max_load': 5.0,  # kg
                'gripper_types': ['parallel', 'suction']
            },
            'sensors': ['camera', 'lidar', 'imu', 'force_torque'],
            'navigation_modes': ['go_to_pose', 'move_base', 'path_follow'],
            'interaction_modes': ['gesture', 'speech', 'physical'] 
        }
    
    def _initialize_validators(self):
        """Initialize action validators"""
        return {
            'NAVIGATE': self._validate_navigation,
            'ROTATE': self._validate_rotation,
            'NAVIGATE_TO_LOCATION': self._validate_location_nav,
            'GRASP_OBJECT': self._validate_grasp,
            'PLACE_OBJECT': self._validate_place,
            'FOLLOW_LEADER': self._validate_follow,
            'PERFORM_GESTURE': self._validate_gesture,
            'STOP_CURRENT_ACTION': self._validate_stop,
            'PROVIDE_INFORMATION': self._validate_info_provide
        }
    
    def map_to_robot_actions(self, action_sequence: List[Dict]) -> List[Dict]:
        """Map abstract actions to robot-specific commands"""
        robot_actions = []
        
        for action in action_sequence:
            action_type = action.get('action', 'UNKNOWN')
            validator = self.action_validators.get(action_type, self._default_validator)
            
            # Validate action
            is_valid, validation_error = validator(action)
            
            if is_valid:
                # Map to robot-specific command
                robot_action = self._convert_to_robot_command(action)
                robot_actions.append(robot_action)
            else:
                # Add error handling action
                robot_actions.append({
                    'robot_action': 'ERROR_HANDLING',
                    'error': validation_error,
                    'original_action': action
                })
        
        return robot_actions
    
    def _validate_navigation(self, action: Dict) -> tuple:
        """Validate navigation action"""
        direction = action.get('direction', 'forward')
        distance = action.get('distance', 1.0)
        
        if distance > self.robot_capabilities['navigation']['max_distance']:
            return False, f"Distance {distance}m exceeds maximum {self.robot_capabilities['navigation']['max_distance']}m"
        
        if direction not in ['forward', 'backward', 'left', 'right']:
            return False, f"Invalid direction: {direction}"
        
        return True, None
    
    def _validate_rotation(self, action: Dict) -> tuple:
        """Validate rotation action"""
        angle = action.get('angle', 0)
        
        if abs(angle) > 360:
            return False, f"Rotation angle {angle}° exceeds valid range"
        
        return True, None
    
    def _validate_location_nav(self, action: Dict) -> tuple:
        """Validate location navigation"""
        destination = action.get('destination', '').lower()
        
        # Check if location is known/predefined
        known_locations = [
            'kitchen', 'living_room', 'bedroom', 'office', 'bathroom',
            'dining_room', 'garage', 'hallway', 'garden'
        ]
        
        if destination.replace(' ', '_') not in known_locations:
            return False, f"Unknown destination: {destination}"
        
        return True, None
    
    def _validate_grasp(self, action: Dict) -> tuple:
        """Validate grasp action"""
        target_obj = action.get('target_object', '')
        
        if not target_obj:
            return False, "No target object specified for grasp"
        
        # In a real system, check if object is reachable and graspable
        return True, None
    
    def _validate_place(self, action: Dict) -> tuple:
        """Validate place action"""
        destination = action.get('destination', '')
        
        if not destination:
            return False, "No destination specified for placement"
        
        return True, None
    
    def _validate_follow(self, action: Dict) -> tuple:
        """Validate follow action"""
        max_dist = action.get('max_distance', 2.0)
        
        if max_dist > 5.0:  # Arbitrary safety limit
            return False, f"Max following distance {max_dist}m is too large"
        
        return True, None
    
    def _validate_gesture(self, action: Dict) -> tuple:
        """Validate gesture action"""
        gesture_type = action.get('gesture_type', '')
        
        valid_gestures = ['wave', 'point', 'shake_hand', 'nod', 'raise_arm']
        
        if gesture_type not in valid_gestures:
            return False, f"Invalid gesture type: {gesture_type}"
        
        return True, None
    
    def _validate_stop(self, action: Dict) -> tuple:
        """Validate stop action"""
        return True, None  # Stop is always valid
    
    def _validate_info_provide(self, action: Dict) -> tuple:
        """Validate information provision"""
        return True, None  # Info provision is always valid
    
    def _default_validator(self, action: Dict) -> tuple:
        """Default validator for unknown actions"""
        return False, f"Unknown action type: {action.get('action', 'UNKNOWN')}"
    
    def _convert_to_robot_command(self, action: Dict) -> Dict:
        """Convert abstract action to robot command"""
        action_type = action['action']
        
        # This is where we'd use ROS2 topics/services for actual robot control
        robot_command = {
            'timestamp': datetime.now().isoformat(),
            'action_type': action_type,
            'command': self._generate_ros_command(action),
            'parameters': action,
            'status': 'pending'
        }
        
        return robot_command
    
    def _generate_ros_command(self, action: Dict) -> Dict:
        """Generate ROS2 command for the action"""
        action_type = action['action']
        
        if action_type == 'NAVIGATE':
            return {
                'topic': '/move_base_simple/goal',
                'message_type': 'geometry_msgs/PoseStamped',
                'data': self._generate_navigation_goal(action)
            }
        elif action_type == 'ROTATE':
            return {
                'topic': '/cmd_vel',
                'message_type': 'geometry_msgs/Twist',
                'data': self._generate_rotation_command(action)
            }
        elif action_type == 'GRASP_OBJECT':
            return {
                'topic': '/robot/arm_controller/grasp',
                'message_type': 'control_msgs/GripperCommand',
                'data': self._generate_grasp_command(action)
            }
        elif action_type == 'PROVIDE_INFORMATION':
            return {
                'topic': '/speech/output',
                'message_type': 'std_msgs/String',
                'data': {
                    'text': action.get('content', 'No information provided')
                }
            }
        else:
            # For other actions, we'd generate appropriate ROS messages
            return {
                'topic': '/default_robot_action',
                'message_type': 'std_msgs/String',
                'data': action
            }
    
    def _generate_navigation_goal(self, action: Dict) -> Dict:
        """Generate navigation goal based on action"""
        direction = action.get('direction', 'forward')
        distance = action.get('distance', 1.0)
        
        # In a real system, this would calculate actual pose
        # For simplicity, we'll create a general movement command
        return {
            'relative_movement': {
                'direction': direction,
                'distance': distance,
                'speed': 0.3
            }
        }
    
    def _generate_rotation_command(self, action: Dict) -> Dict:
        """Generate rotation command based on action"""
        angle_deg = action.get('angle', 0)
        angle_rad = angle_deg * 3.14159 / 180.0
        
        return {
            'angular_velocity': {
                'z': angle_rad,  # Just an example - in reality would need proper velocity
                'duration': 2.0  # Estimated time to complete rotation
            }
        }
    
    def _generate_grasp_command(self, action: Dict) -> Dict:
        """Generate grasp command based on action"""
        target = action.get('target_object', 'object')
        
        return {
            'command': 'grasp',
            'target': target,
            'approach_method': action.get('approach_method', 'default')
        }
```

### Stage 5: Robot Control Interface

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String
from builtin_interfaces.msg import Time
import threading
import time

class RobotController(Node):
    def __init__(self):
        """Initialize robot controller for executing commands"""
        super().__init__('voice_to_action_controller')
        
        # Publishers for robot control
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav_goal_publisher = self.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
        self.speech_publisher = self.create_publisher(String, '/speech/output', 10)
        
        # Status tracking
        self.current_action = None
        self.action_status = 'idle'
        self.is_executing = False
        
        # Action execution thread
        self.execution_thread = None
        self.execution_lock = threading.Lock()
        
        self.get_logger().info("Robot controller initialized")
    
    def execute_action_sequence(self, robot_actions: List[Dict]):
        """Execute sequence of robot actions"""
        with self.execution_lock:
            if self.is_executing:
                self.get_logger().warn("Action sequence already executing, skipping")
                return False
            
            self.is_executing = True
            self.current_action = None
        
        # Execute in separate thread to not block main process
        self.execution_thread = threading.Thread(
            target=self._execute_action_sequence_worker,
            args=(robot_actions,)
        )
        self.execution_thread.start()
        
        return True
    
    def _execute_action_sequence_worker(self, robot_actions: List[Dict]):
        """Worker function to execute action sequence"""
        self.get_logger().info(f"Executing sequence with {len(robot_actions)} actions")
        
        for action in robot_actions:
            if not self.is_executing:
                self.get_logger().info("Execution cancelled")
                break
            
            self.current_action = action
            self.action_status = 'executing'
            
            success = self._execute_single_action(action)
            
            if not success:
                self.get_logger().error(f"Action failed: {action}")
                # In a real system, we might have error recovery here
                break
            
            # Small delay between actions
            time.sleep(0.5)
        
        self.action_status = 'completed'
        self.is_executing = False
        self.current_action = None
    
    def _execute_single_action(self, robot_action: Dict) -> bool:
        """Execute a single robot action"""
        cmd_type = robot_action['action_type']
        
        try:
            if cmd_type == 'NAVIGATE':
                return self._execute_navigation(robot_action)
            elif cmd_type == 'ROTATE':
                return self._execute_rotation(robot_action)
            elif cmd_type == 'NAVIGATE_TO_LOCATION':
                return self._execute_location_navigation(robot_action)
            elif cmd_type == 'PROVIDE_INFORMATION':
                return self._execute_speech_output(robot_action)
            elif cmd_type == 'STOP_CURRENT_ACTION':
                return self._execute_stop(robot_action)
            else:
                # For other command types, we could use services or other interfaces
                self.get_logger().warning(f"Unsupported action type: {cmd_type}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error executing action {robot_action}: {e}")
            return False
    
    def _execute_navigation(self, robot_action: Dict) -> bool:
        """Execute navigation command"""
        params = robot_action['parameters']
        direction = params.get('direction', 'forward')
        distance = params.get('distance', 1.0)
        
        self.get_logger().info(f"Navigating {direction} for {distance}m")
        
        # Create Twist message for movement
        twist_msg = Twist()
        
        if direction == 'forward':
            twist_msg.linear.x = 0.3  # Forward speed (m/s)
        elif direction == 'backward':
            twist_msg.linear.x = -0.3
        elif direction == 'left':
            twist_msg.linear.y = 0.3
        elif direction == 'right':
            twist_msg.linear.y = -0.3
        
        # Calculate time to travel distance
        travel_time = distance / 0.3  # time = distance / speed
        
        start_time = time.time()
        while time.time() - start_time < travel_time and self.is_executing:
            self.cmd_vel_publisher.publish(twist_msg)
            time.sleep(0.1)
        
        # Stop the robot
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)
        
        return True
    
    def _execute_rotation(self, robot_action: Dict) -> bool:
        """Execute rotation command"""
        params = robot_action['parameters']
        angle = params.get('angle', 0)
        
        self.get_logger().info(f"Rotating by {angle} degrees")
        
        # Convert angle to radians
        angle_rad = angle * 3.14159 / 180.0
        
        # Calculate approximate rotation time (assuming 0.5 rad/s)
        rotation_time = abs(angle_rad) / 0.5
        
        # Create Twist message for rotation
        twist_msg = Twist()
        twist_msg.angular.z = 0.5 if angle > 0 else -0.5  # Direction based on angle sign
        
        start_time = time.time()
        while time.time() - start_time < rotation_time and self.is_executing:
            self.cmd_vel_publisher.publish(twist_msg)
            time.sleep(0.1)
        
        # Stop rotation
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)
        
        return True
    
    def _execute_location_navigation(self, robot_action: Dict) -> bool:
        """Execute navigation to specific location"""
        params = robot_action['parameters']
        destination = params.get('destination', '').lower()
        
        self.get_logger().info(f"Navigating to {destination}")
        
        # In a real system, this would look up the coordinates for the location
        # For this example, we'll just log the intent
        self.get_logger().info(f"Looking up coordinates for '{destination}'")
        
        # For now, we'll publish a placeholder goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = "map"
        # Placeholder coordinates - in real system these would come from map
        goal_msg.pose.position.x = 1.0  # Example location
        goal_msg.pose.position.y = 1.0
        goal_msg.pose.orientation.w = 1.0
        
        self.nav_goal_publisher.publish(goal_msg)
        
        # Wait for navigation to complete (in reality, would check nav status)
        time.sleep(5.0)
        
        return True
    
    def _execute_speech_output(self, robot_action: Dict) -> bool:
        """Execute speech output command"""
        params = robot_action['parameters']
        content = params.get('content', 'Hello')
        
        self.get_logger().info(f"Speaking: {content}")
        
        # Publish speech command
        speech_msg = String()
        speech_msg.data = content
        self.speech_publisher.publish(speech_msg)
        
        # Wait for speech to complete
        time.sleep(len(content.split()) * 0.5)  # Rough estimate of speech duration
        
        return True
    
    def _execute_stop(self, robot_action: Dict) -> bool:
        """Execute stop command"""
        self.get_logger().info("Stopping current action")
        
        # Stop all movement
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)
        
        # Cancel any navigation goals
        # In a real system, we'd cancel the navigation goal here
        
        return True
    
    def cancel_current_action(self):
        """Cancel the currently executing action"""
        with self.execution_lock:
            self.is_executing = False
            self.action_status = 'cancelled'