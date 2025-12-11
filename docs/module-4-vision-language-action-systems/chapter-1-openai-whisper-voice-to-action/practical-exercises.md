---
id: module-4-chapter-1-practical-exercises
title: 'Module 4 — Vision-Language-Action Systems | Chapter 1 — Practical Exercises'
sidebar_label: 'Chapter 1 — Practical Exercises'
sidebar_position: 5
---

# Chapter 1 — Practical Exercises

## OpenAI Whisper Voice-to-Action: Advanced Implementation

This practical lab focuses on implementing a complete voice-to-action pipeline using OpenAI Whisper for humanoid robotics applications, emphasizing real-time processing, command interpretation, and robot execution.

### Exercise 1: Real-Time Whisper Integration

#### Objective
Implement a real-time voice command system using OpenAI Whisper for robot control.

#### Steps
1. Set up Whisper model for real-time inference
2. Implement audio preprocessing pipeline
3. Create command classification system
4. Integrate with robot control interface

```python
# advanced_whisper_integration.py
import whisper
import torch
import pyaudio
import numpy as np
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import json

@dataclass
class VoiceCommand:
    """Structure to hold voice command information"""
    text: str
    confidence: float
    timestamp: float
    command_type: str
    parameters: Dict[str, Any]

class RealTimeWhisperProcessor:
    def __init__(self, model_size="base", sample_rate=16000, chunk_duration=0.5):
        """
        Initialize real-time Whisper processor
        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
            sample_rate: Audio sampling rate
            chunk_duration: Duration of audio chunks for processing
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Load Whisper model
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("Whisper model loaded successfully")
        
        # Audio processing setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        
        # Voice activity detection thresholds
        self.energy_threshold = 0.01
        self.silence_threshold = 10  # Number of silent chunks before processing
        self.min_voice_chunks = 5    # Minimum voiced chunks for valid command
        
        # Result queue for processed commands
        self.result_queue = queue.Queue()
        
        # Command patterns for parsing
        self.command_patterns = {
            'navigation': [
                r'go to the (\w+)',           # Go to kitchen/living room
                r'move to the (\w+)',         # Move to bedroom
                r'go forward',                # Move forward
                r'move backward',             # Move backward
                r'turn left',                 # Turn left
                r'turn right',                # Turn right
                r'stop',                      # Stop
            ],
            'manipulation': [
                r'pick up the (\w+)',        # Pick up the ball
                r'grasp the (\w+)',          # Grasp the cup  
                r'place it on the (\w+)',    # Place on table
                r'put down',                 # Put down
                r'release',                  # Release object
            ],
            'interaction': [
                r'wave hello',               # Wave hello
                r'wave goodbye',             # Wave goodbye
                r'say hello',                # Speak hello
                r'speak',                    # Speak
                r'tell me',                  # Tell information
            ],
            'system': [
                r'shutdown',                # Shutdown system
                r'power off',               # Power off
                r'restart',                 # Restart
                r'calibrate',               # Calibrate
            ]
        }
    
    def start_listening(self):
        """Start real-time audio listening"""
        if self.stream is not None:
            return  # Already running
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.is_listening = True
        
        # Start audio capture thread
        self.capture_thread = threading.Thread(target=self._capture_worker)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Real-time listening started")
    
    def _capture_worker(self):
        """Worker thread for audio capture"""
        voice_chunks = []
        silence_chunks = 0
        recording = False
        
        while self.is_listening:
            try:
                # Read audio chunk
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                # Compute energy for VAD
                energy = np.mean(audio_chunk ** 2)
                is_voice = energy > self.energy_threshold
                
                if is_voice:
                    # Voice detected - add to buffer
                    voice_chunks.append(audio_chunk)
                    silence_chunks = 0
                    
                    if not recording:
                        # Start of speech detected
                        recording = True
                        silence_chunks = 0
                        print("Voice detected, buffering...")
                else:
                    # Silence detected
                    silence_chunks += 1
                    
                    if recording:
                        if len(voice_chunks) >= self.min_voice_chunks:
                            # Valid speech segment - submit for processing
                            full_audio = np.concatenate(voice_chunks)
                            self.audio_queue.put(full_audio)
                            print(f"Submitted {len(voice_chunks)} chunks for processing")
                        
                        # Reset for next segment
                        voice_chunks = []
                        recording = False
                        silence_chunks = 0
                        
            except Exception as e:
                print(f"Audio capture error: {e}")
                time.sleep(0.01)
    
    def _processing_worker(self):
        """Worker thread for Whisper processing"""
        while self.is_listening:
            try:
                # Get audio from queue
                audio_data = self.audio_queue.get(timeout=1.0)
                
                # Process with Whisper
                result = self.model.transcribe(
                    audio_data,
                    fp16=torch.cuda.is_available(),  # Use FP16 on GPU
                    language='en',
                    temperature=0.0
                )
                
                # Process transcription
                transcription = result['text'].strip().lower()
                if transcription:  # Only create command if non-empty
                    command = self.parse_command(transcription)
                    if command:
                        self.result_queue.put(command)
                        print(f"Command recognized: {command.text}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse transcribed text into structured command"""
        if not text:
            return None
        
        # Classify command type based on patterns
        command_type = self.classify_command_type(text)
        
        # Extract parameters using regex
        parameters = self.extract_parameters(text, command_type)
        
        # Calculate confidence (simplified - in reality would use model confidence)
        confidence = self.estimate_confidence(text)
        
        return VoiceCommand(
            text=text,
            confidence=confidence,
            timestamp=time.time(),
            command_type=command_type,
            parameters=parameters
        )
    
    def classify_command_type(self, text: str) -> str:
        """Classify command type based on text patterns"""
        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                import re
                if re.search(pattern, text, re.IGNORECASE):
                    return cmd_type
        
        return "unknown"
    
    def extract_parameters(self, text: str, command_type: str) -> Dict[str, Any]:
        """Extract parameters from command text"""
        params = {}
        
        # Extract object names for manipulation commands
        if command_type == "manipulation":
            import re
            # Look for object mentions
            object_match = re.search(r'the (\w+)', text)
            if object_match:
                params['object'] = object_match.group(1)
        
        # Extract location names for navigation commands  
        elif command_type == "navigation":
            import re
            # Look for location mentions
            location_match = re.search(r'the (\w+)', text)
            if location_match:
                params['location'] = location_match.group(1)
        
        # Extract content for interaction commands
        elif command_type == "interaction":
            if 'say' in text or 'speak' in text:
                # Extract what to say
                import re
                say_match = re.search(r'(say|speak|tell me)\s+(.*)', text)
                if say_match:
                    params['content'] = say_match.group(2).strip()
        
        return params
    
    def estimate_confidence(self, text: str) -> float:
        """Estimate confidence in transcription (simplified)"""
        # In a real system, this would use Whisper's confidence scores
        # For now, use simple heuristics
        words = text.strip().split()
        
        if len(words) == 0:
            return 0.0
        elif len(words) == 1:
            # Single word commands might be lower confidence
            return 0.6
        else:
            # Longer, more complex commands tend to be more confident
            return min(0.95, 0.5 + len(words) * 0.1)
    
    def get_command(self, timeout: Optional[float] = None) -> Optional[VoiceCommand]:
        """Get next recognized command"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_listening(self):
        """Stop real-time listening"""
        self.is_listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        
        print("Real-time listening stopped")

def main():
    """Demonstrate real-time Whisper processing"""
    print("Initializing real-time Whisper processor...")
    
    # Create processor
    processor = RealTimeWhisperProcessor(model_size="base")
    
    try:
        # Start listening
        processor.start_listening()
        
        print("Voice command system active. Say commands like:")
        print("  - 'Go to the kitchen'")
        print("  - 'Pick up the red ball'") 
        print("  - 'Wave hello'")
        print("  - 'Say hello'")
        print("\nPress Ctrl+C to stop...")
        
        # Process commands
        while True:
            command = processor.get_command(timeout=0.1)
            if command:
                print(f"\nRECEIVED: '{command.text}'")
                print(f"  Type: {command.command_type}")
                print(f"  Confidence: {command.confidence:.2f}")
                print(f"  Params: {command.parameters}")
                
                # In a real implementation, you would send command to robot here
                # execute_robot_command(command)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        processor.stop_listening()
        print("Shutdown complete")

if __name__ == "__main__":
    main()
```

### Exercise 2: Voice Command Classification and Intent Recognition

#### Objective
Implement advanced command classification using NLP techniques to improve intent recognition accuracy.

#### Steps
1. Create intent classification system
2. Implement named entity recognition
3. Add context awareness
4. Validate against robot command vocabulary

```python
# voice_classification.py
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional
import re
import pickle
from dataclasses import dataclass

@dataclass
class ClassifiedCommand:
    """Classified voice command with intent and entities"""
    intent: str
    entities: Dict[str, str]
    confidence: float
    original_text: str

class VoiceCommandClassifier:
    def __init__(self):
        """Initialize voice command classifier"""
        # Load spaCy English model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Define robot command intents
        self.intents = {
            'navigation_move_forward': ['go forward', 'move forward', 'go ahead', 'move ahead', 'proceed'],
            'navigation_move_backward': ['go backward', 'move backward', 'go back', 'move back', 'reverse'],
            'navigation_turn_left': ['turn left', 'rotate left', 'go left', 'pivot left'],
            'navigation_turn_right': ['turn right', 'rotate right', 'go right', 'pivot right'],
            'navigation_goto_room': ['go to the (.+)', 'move to the (.+)', 'go to (.+)', 'navigate to (.+)'],
            'manipulation_pick_object': ['pick up the (.+)', 'take the (.+)', 'grasp the (.+)', 'get the (.+)'],
            'manipulation_place_object': ['place it on the (.+)', 'put it on the (.+)', 'set it on the (.+)'],
            'manipulation_release_object': ['release', 'let go', 'put down', 'place down'],
            'interaction_wave': ['wave', 'wave hello', 'wave goodbye', 'waving'],
            'interaction_speak': ['say (.+)', 'speak (.+)', 'tell (.+)', 'speak'],
            'system_stop': ['stop', 'halt', 'freeze', 'cease'],
            'system_pause': ['pause', 'wait', 'hold on', 'stand by'],
        }
        
        # Initialize training data
        self.training_sentences = []
        self.training_labels = []
        
        # Robot-specific entities
        self.known_rooms = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'dining room']
        self.known_objects = ['ball', 'cup', 'book', 'bottle', 'box', 'chair', 'table', 'phone', 'keys']
        
        # Build classifier
        self.classifier = self._build_classifier()
        
        # Entity patterns
        self.entity_patterns = {
            'room': '|'.join(self.known_rooms),
            'object': '|'.join(self.known_objects),
            'direction': 'north|south|east|west|left|right|forward|backward|up|down',
            'color': 'red|blue|green|yellow|purple|orange|pink|black|white|gray'
        }
    
    def _build_classifier(self):
        """Build classification pipeline"""
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except necessary ones
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract named entities from text"""
        if not self.nlp:
            # Fallback without spaCy
            return self._fallback_entity_extraction(text)
        
        doc = self.nlp(text)
        entities = {}
        
        # Extract named entities using spaCy
        for ent in doc.ents:
            # Look for room or object mentions
            if ent.text in self.known_rooms:
                entities['room'] = ent.text
            elif ent.text in self.known_objects:
                entities['object'] = ent.text
        
        # Extract entities using regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match and entity_type not in entities:
                entities[entity_type] = match.group(0)
        
        return entities
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, str]:
        """Fallback entity extraction without spaCy"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities[entity_type] = match.group(0)
        
        # Check for known objects and rooms
        for room in self.known_rooms:
            if room in text.lower():
                entities['room'] = room
                break
        
        for obj in self.known_objects:
            if obj in text.lower():
                entities['object'] = obj
                break
        
        return entities
    
    def prepare_training_data(self):
        """Prepare training data for classifier"""
        # Generate training examples from intent patterns
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                # Remove regex-specific characters for training examples
                clean_pattern = re.sub(r'\(.+\)', 'something', pattern)
                
                # Generate variations
                variations = self._generate_sentence_variations(clean_pattern)
                
                for sentence in variations:
                    self.training_sentences.append(sentence)
                    self.training_labels.append(intent)
    
    def _generate_sentence_variations(self, base_sentence: str) -> List[str]:
        """Generate sentence variations for training data"""
        variations = [base_sentence]
        
        # Add common prefixes and suffixes
        prefixes = ['', 'please ', 'can you ', 'robot, ', 'hey robot ']
        suffixes = ['', ' now', ' please', ' immediately', ' right now']
        
        for prefix in prefixes:
            for suffix in suffixes:
                variations.append(prefix + base_sentence + suffix)
        
        return variations
    
    def train_classifier(self):
        """Train the intent classifier"""
        if not self.training_sentences:
            self.prepare_training_data()
        
        print(f"Training classifier with {len(self.training_sentences)} examples...")
        
        # Train the model
        self.classifier.fit(self.training_sentences, self.training_labels)
        print("Classifier training complete")
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent for given text"""
        processed_text = self.preprocess_text(text)
        
        # Get prediction and probability
        predicted_intent = self.classifier.predict([processed_text])[0]
        prediction_proba = self.classifier.predict_proba([processed_text])[0]
        confidence = max(prediction_proba)
        
        return predicted_intent, confidence
    
    def classify_command(self, text: str) -> Optional[ClassifiedCommand]:
        """Classify voice command into intent and entities"""
        # Predict intent
        intent, confidence = self.predict_intent(text)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Create classified command
        classified = ClassifiedCommand(
            intent=intent,
            entities=entities,
            confidence=confidence,
            original_text=text
        )
        
        return classified
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        model_data = {
            'classifier': self.classifier,
            'training_sentences': self.training_sentences,
            'training_labels': self.training_labels,
            'nlp_model': self.nlp,
            'known_rooms': self.known_rooms,
            'known_objects': self.known_objects,
            'entity_patterns': self.entity_patterns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.training_sentences = model_data['training_sentences'] 
        self.training_labels = model_data['training_labels']
        self.nlp = model_data['nlp_model']
        self.known_rooms = model_data['known_rooms']
        self.known_objects = model_data['known_objects'] 
        self.entity_patterns = model_data['entity_patterns']
        
        print(f"Model loaded from {filepath}")

def main():
    """Test the voice command classifier"""
    print("Initializing voice command classifier...")
    
    classifier = VoiceCommandClassifier()
    
    # Train the classifier
    classifier.train_classifier()
    
    # Test sentences
    test_sentences = [
        "Go forward slowly",
        "Move to the kitchen now",
        "Pick up the red ball",
        "Please turn left at the corner",
        "Wave hello to everyone",
        "Stop immediately", 
        "Tell me about the weather",
        "Place it on the table",
        "Can you go to the living room?",
        "Take the blue cup from the shelf"
    ]
    
    print("\nTesting command classification:")
    print("=" * 50)
    
    for sentence in test_sentences:
        classified = classifier.classify_command(sentence)
        if classified:
            print(f"Input: '{sentence}'")
            print(f"  Intent: {classified.intent}")
            print(f"  Entities: {classified.entities}")
            print(f"  Confidence: {classified.confidence:.3f}")
            print("-" * 40)

if __name__ == "__main__":
    main()
```

### Exercise 3: Action Mapping and Robot Control Integration

#### Objective
Create a mapping system that translates classified commands to robot actions and integrates with robot control systems.

#### Steps
1. Define robot action space and capabilities
2. Create action mapping system
3. Implement safety checks and validation
4. Integrate with robot middleware (ROS 2)

```python
# action_mapping_system.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
import time
from typing import Dict, Any, Optional
import numpy as np

class ActionMappingSystem(Node):
    def __init__(self):
        super().__init__('action_mapping_system')
        
        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_cmd_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self.speech_pub = self.create_publisher(String, '/speech_output', 10)
        
        # Subscribers for robot status
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)
        self.odometry_sub = self.create_subscription(
            Odometry, '/odom', self.odometry_callback, 10)
        
        # Robot state tracking
        self.current_joints = JointState()
        self.current_pose = Pose()
        
        # Command mapping dictionary
        self.command_mappings = {
            'navigation_move_forward': self._execute_move_forward,
            'navigation_move_backward': self._execute_move_backward,
            'navigation_turn_left': self._execute_turn_left,
            'navigation_turn_right': self._execute_turn_right,
            'navigation_goto_room': self._execute_goto_room,
            'manipulation_pick_object': self._execute_pick_object,
            'manipulation_place_object': self._execute_place_object,
            'manipulation_release_object': self._execute_release_object,
            'interaction_wave': self._execute_wave,
            'interaction_speak': self._execute_speak,
            'system_stop': self._execute_stop,
            'system_pause': self._execute_pause,
        }
        
        # Robot capabilities and limits
        self.robot_caps = {
            'max_linear_speed': 0.5,      # m/s
            'max_angular_speed': 1.0,     # rad/s
            'min_translation_dist': 0.1,  # m
            'min_rotation_angle': 0.1,    # rad
            'reachable_workspace': {      # Robot reach limits (m)
                'x': (-1.0, 1.0),
                'y': (-0.5, 0.5), 
                'z': (0.1, 2.0)
            }
        }
        
        # Safety parameters
        self.safety_enabled = True
        self.emergency_stop_active = False
        self.safe_zones = []  # Define safe zones for robot operation
        
        self.get_logger().info('Action mapping system initialized')
    
    def joint_state_callback(self, msg: JointState):
        """Update joint state"""
        self.current_joints = msg
    
    def odometry_callback(self, msg: Odometry):
        """Update robot pose"""
        self.current_pose = msg.pose.pose
    
    def map_command_to_action(self, classified_command, confidence_threshold=0.7):
        """Map classified command to robot action with validation"""
        
        if classified_command.confidence < confidence_threshold:
            self.get_logger().warn(
                f'Command confidence too low: {classified_command.confidence:.3f} '
                f'< {confidence_threshold}. Rejecting command.'
            )
            return False
        
        # Check if robot is in safe state to execute command
        if self.emergency_stop_active:
            self.get_logger().warn('Emergency stop active, rejecting command')
            self.speak_response("Cannot execute command, emergency stop is active")
            return False
        
        # Check if intent is supported
        intent = classified_command.intent
        if intent not in self.command_mappings:
            self.get_logger().warn(f'Unsupported command intent: {intent}')
            self.speak_response(f"Sorry, I don't know how to {intent.replace('_', ' ')}")
            return False
        
        # Validate command parameters
        if not self.validate_command_parameters(classified_command):
            self.get_logger().warn(f'Command parameters validation failed: {classified_command}')
            return False
        
        try:
            # Execute the command
            success = self.command_mappings[intent](classified_command)
            
            if success:
                self.get_logger().info(f'Successfully executed command: {intent}')
                self.speak_response("Command executed successfully")
                return True
            else:
                self.get_logger().error(f'Failed to execute command: {intent}')
                self.speak_response("Sorry, I couldn't complete that command")
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error executing command {intent}: {str(e)}')
            self.speak_response("Something went wrong executing the command")
            return False
    
    def validate_command_parameters(self, command) -> bool:
        """Validate command parameters for safety and feasibility"""
        
        intent = command.intent
        entities = command.entities
        
        if intent.startswith('navigation'):
            # Validate navigation commands
            if 'room' in entities:
                room = entities['room']
                # Check if room is in known safe locations
                if room not in ['kitchen', 'living room', 'bedroom', 'office', 'bathroom']:
                    self.get_logger().warn(f'Unknown or unsafe room: {room}')
                    return False
        
        elif intent.startswith('manipulation'):
            # Validate manipulation commands
            if 'object' in entities:
                obj = entities['object']
                # Check if object is known and graspable
                if obj not in ['ball', 'cup', 'book', 'bottle', 'box']:
                    self.get_logger().warn(f'Unknown or ungraspable object: {obj}')
                    return False
        
        elif intent.startswith('interaction'):
            if 'content' in entities and len(entities['content']) > 100:
                # Limit speech content length for safety
                self.get_logger().warn('Speech content too long')
                return False
        
        return True
    
    def _execute_move_forward(self, command):
        """Execute forward movement command"""
        params = command.parameters
        distance = params.get('distance', 1.0)  # Default 1m if not specified
        speed = params.get('speed', 0.3)       # Default 0.3 m/s if not specified
        
        # Normalize speed to limits
        speed = min(speed, self.robot_caps['max_linear_speed'])
        
        # Calculate duration for movement
        duration = distance / speed
        
        twist_msg = Twist()
        twist_msg.linear.x = speed
        twist_msg.angular.z = 0.0
        
        # Execute movement
        self.get_logger().info(f'Moving forward {distance}m at {speed}m/s')
        
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            if self.emergency_stop_active:
                break
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(0.1)
        
        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        return True
    
    def _execute_move_backward(self, command):
        """Execute backward movement command"""
        params = command.parameters
        distance = params.get('distance', 1.0)
        speed = params.get('speed', 0.3)
        
        speed = min(speed, self.robot_caps['max_linear_speed'])
        duration = distance / speed
        
        twist_msg = Twist()
        twist_msg.linear.x = -speed  # Negative for backward
        twist_msg.angular.z = 0.0
        
        self.get_logger().info(f'Moving backward {distance}m at {speed}m/s')
        
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            if self.emergency_stop_active:
                break
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(0.1)
        
        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        return True
    
    def _execute_turn_left(self, command):
        """Execute left turn command"""
        params = command.parameters
        angle = params.get('angle', np.pi/2)  # Default 90 degrees
        speed = params.get('speed', 0.5)      # Angular speed in rad/s
        
        speed = min(speed, self.robot_caps['max_angular_speed'])
        duration = angle / speed
        
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = speed  # Positive for left turn
        
        self.get_logger().info(f'Turning left {np.degrees(angle):.1f}° at {speed}rad/s')
        
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            if self.emergency_stop_active:
                break
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(0.1)
        
        # Stop rotation
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        return True
    
    def _execute_turn_right(self, command):
        """Execute right turn command"""
        params = command.parameters
        angle = params.get('angle', np.pi/2)  # Default 90 degrees
        speed = params.get('speed', 0.5)      # Angular speed in rad/s
        
        speed = min(speed, self.robot_caps['max_angular_speed'])
        duration = angle / speed
        
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = -speed  # Negative for right turn
        
        self.get_logger().info(f'Turning right {np.degrees(angle):.1f}° at {speed}rad/s')
        
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            if self.emergency_stop_active:
                break
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(0.1)
        
        # Stop rotation
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        return True
    
    def _execute_goto_room(self, command):
        """Execute navigation to specific room"""
        if 'room' not in command.entities:
            self.get_logger().error('No room specified in command')
            return False
        
        room = command.entities['room']
        self.get_logger().info(f'Navigating to {room}')
        
        # In a real implementation, this would interface with navigation stack
        # For this example, we'll use mock navigation
        
        # Mock room positions (in a real system these would come from map)
        room_positions = {
            'kitchen': (-2.0, 1.0),
            'living room': (0.0, 0.0),
            'bedroom': (2.0, -1.0),
            'office': (-1.0, -2.0),
            'bathroom': (1.0, 2.0)
        }
        
        if room not in room_positions:
            self.get_logger().error(f'Unknown room: {room}')
            self.speak_response(f"Sorry, I don't know where the {room} is")
            return False
        
        target_x, target_y = room_positions[room]
        
        # Calculate movement to target (simplified - no obstacle avoidance)
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        dx = target_x - current_x
        dy = target_y - current_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < 0.5:  # Already close enough
            self.get_logger().info(f'Already at {room}')
            self.speak_response(f"I'm already in the {room}")
            return True
        
        # Move toward target
        duration = distance / 0.3  # Assume 0.3 m/s speed
        linear_speed = 0.3
        angular_speed = np.arctan2(dy, dx)  # Heading toward target
        
        self.get_logger().info(f'Moving to {room} at ({target_x}, {target_y})')
        
        # In a real system, we'd use move_base or similar navigation framework
        # For this example, we'll just publish movement commands
        twist_msg = Twist()
        twist_msg.linear.x = linear_speed
        twist_msg.angular.z = angular_speed
        
        start_time = time.time()
        while time.time() - start_time < duration and rclpy.ok():
            if self.emergency_stop_active:
                break
            self.cmd_vel_pub.publish(twist_msg)
            time.sleep(0.1)
        
        # Stop robot
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        self.speak_response(f"I've reached the {room}")
        return True
    
    def _execute_pick_object(self, command):
        """Execute object pick-up command"""
        if 'object' not in command.entities:
            self.get_logger().error('No object specified in command')
            return False
        
        obj = command.entities['object']
        self.get_logger().info(f'Attempting to pick up {obj}')
        
        # In a real implementation, this would:
        # 1. Localize the object using vision
        # 2. Plan a grasp trajectory
        # 3. Execute the grasp with manipulator
        
        # For this example, we'll simulate the action
        self.speak_response(f"Attempting to pick up the {obj}")
        
        # Simulate grasp time
        time.sleep(2.0)
        
        self.speak_response(f"Successfully picked up the {obj}")
        return True
    
    def _execute_place_object(self, command):
        """Execute object placement command"""
        if 'object' not in command.entities:
            self.get_logger().error('No object specified in command')
            return False
        
        obj = command.entities['object']
        location = command.entities.get('location', 'default location')
        
        self.get_logger().info(f'Attempting to place {obj} at {location}')
        
        # In a real implementation, this would:
        # 1. Plan a placement trajectory
        # 2. Execute the placement with manipulator
        # 3. Release the object
        
        # For this example, we'll simulate the action
        self.speak_response(f"Placing the {obj} at {location}")
        
        # Simulate placement time
        time.sleep(1.5)
        
        self.speak_response(f"Successfully placed the {obj}")
        return True
    
    def _execute_release_object(self, command):
        """Execute object release command"""
        self.get_logger().info('Releasing held object')
        
        # In a real implementation, this would release the gripper
        self.speak_response("Releasing object")
        
        # Simulate release
        time.sleep(0.5)
        
        self.speak_response("Object released")
        return True
    
    def _execute_wave(self, command):
        """Execute waving gesture"""
        self.get_logger().info('Executing waving gesture')
        
        # In a real implementation, this would:
        # 1. Plan a waving motion trajectory
        # 2. Execute with arm joints
        
        # For this example, we'll simulate the action
        self.speak_response("Waving hello!")
        
        # Simulate waving (in a real system would control arm joints)
        time.sleep(2.0)
        
        return True
    
    def _execute_speak(self, command):
        """Execute speech command"""
        if 'content' not in command.entities:
            self.get_logger().error('No speech content specified in command')
            return False
        
        content = command.entities['content']
        self.get_logger().info(f'Speaking: {content}')
        
        # Publish speech content
        speech_msg = String()
        speech_msg.data = content
        self.speech_pub.publish(speech_msg)
        
        return True
    
    def _execute_stop(self, command):
        """Execute stop command"""
        self.get_logger().info('Stopping all robot motion')
        
        # Stop all motion
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        # In a real system, would also stop joint motions
        joint_msg = JointState()
        joint_msg.name = self.current_joints.name  # Keep same joint names
        joint_msg.position = [0.0] * len(self.current_joints.name)  # Zero positions
        self.joint_cmd_pub.publish(joint_msg)
        
        self.speak_response("All motion stopped")
        return True
    
    def _execute_pause(self, command):
        """Execute pause command"""
        self.get_logger().info('Pausing robot execution')
        
        # Stop current motion but maintain position
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        
        self.speak_response("Robot paused, awaiting further commands")
        return True
    
    def speak_response(self, text: str):
        """Publish speech response"""
        speech_msg = String()
        speech_msg.data = text
        self.speech_pub.publish(speech_msg)
        self.get_logger().info(f'Speaking: {text}')

def main(args=None):
    rclpy.init(args=args)
    
    action_mapper = ActionMappingSystem()
    
    try:
        rclpy.spin(action_mapper)
    except KeyboardInterrupt:
        print("Action mapping system interrupted")
    finally:
        action_mapper.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 4: Complete Voice-to-Action Pipeline Integration

#### Objective
Integrate all components into a complete voice-to-action pipeline that can process natural language commands and execute corresponding robot behaviors.

#### Steps
1. Integrate Whisper, classifier, and action mapper
2. Implement error handling and fallback strategies
3. Create validation and testing framework
4. Demonstrate with real robot scenarios

```python
# complete_voice_to_action_system.py
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional, Dict, Any
import queue

class CompleteVoiceToActionSystem:
    def __init__(self):
        """Initialize complete voice-to-action system"""
        # Initialize components
        self.whisper_processor = RealTimeWhisperProcessor(model_size="base")
        self.command_classifier = VoiceCommandClassifier()
        self.action_mapper = ActionMappingSystem()  # Would need to run in separate thread/process
        
        # Training data
        self.command_classifier.train_classifier()
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.running = False
        
        # Command queues
        self.transcription_queue = queue.Queue()
        self.classification_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        
        # Statistics
        self.stats = {
            'commands_processed': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'errors': 0
        }
        
        print("Complete voice-to-action system initialized")
    
    def start_system(self):
        """Start the complete voice-to-action system"""
        self.running = True
        
        # Start Whisper processor
        self.whisper_processor.start_listening()
        
        # Start processing threads
        self.transcription_thread = threading.Thread(target=self._transcription_worker)
        self.classification_thread = threading.Thread(target=self._classification_worker)  
        self.execution_thread = threading.Thread(target=self._execution_worker)
        
        self.transcription_thread.daemon = True
        self.classification_thread.daemon = True
        self.execution_thread.daemon = True
        
        self.transcription_thread.start()
        self.classification_thread.start()
        self.execution_thread.start()
        
        print("Voice-to-action system started")
    
    def _transcription_worker(self):
        """Worker thread for Whisper transcription"""
        while self.running:
            try:
                # Get transcription from Whisper
                command = self.whisper_processor.get_command(timeout=0.1)
                if command:
                    self.transcription_queue.put(command)
            except Exception as e:
                print(f"Transcription worker error: {e}")
                time.sleep(0.01)
    
    def _classification_worker(self):
        """Worker thread for command classification"""
        while self.running:
            try:
                # Get transcription
                if not self.transcription_queue.empty():
                    transcribed_cmd = self.transcription_queue.get()
                    
                    # Classify command
                    classified_cmd = self.command_classifier.classify_command(
                        transcribed_cmd.text
                    )
                    
                    if classified_cmd:
                        classified_cmd.original_text = transcribed_cmd.text
                        classified_cmd.confidence = max(classified_cmd.confidence, transcribed_cmd.confidence)
                        self.classification_queue.put(classified_cmd)
                
                time.sleep(0.01)  # Small sleep to prevent busy waiting
            except Exception as e:
                print(f"Classification worker error: {e}")
                time.sleep(0.01)
    
    def _execution_worker(self):
        """Worker thread for action execution"""
        while self.running:
            try:
                if not self.classification_queue.empty():
                    classified_cmd = self.classification_queue.get()
                    
                    # Log command
                    print(f"\nProcessing command: {classified_cmd.intent}")
                    print(f"  Entities: {classified_cmd.entities}")
                    print(f"  Confidence: {classified_cmd.confidence:.3f}")
                    print(f"  Original: '{classified_cmd.original_text}'")
                    
                    # In a real system, this would interface with the robot
                    # For this example, we'll just print the intended action
                    success = self._mock_robot_execution(classified_cmd)
                    
                    # Update statistics
                    self.stats['commands_processed'] += 1
                    if success:
                        print("✓ Command executed successfully")
                    else:
                        self.stats['errors'] += 1
                        print("✗ Command execution failed")
                    
                    # Update success rate
                    total = self.stats['commands_processed']
                    errors = self.stats['errors']
                    if total > 0:
                        self.stats['success_rate'] = (total - errors) / total
                    else:
                        self.stats['success_rate'] = 0.0
                else:
                    time.sleep(0.01)  # Prevent busy waiting
            except Exception as e:
                print(f"Execution worker error: {e}")
                time.sleep(0.01)
    
    def _mock_robot_execution(self, classified_cmd) -> bool:
        """Mock robot execution for demonstration"""
        intent = classified_cmd.intent
        
        if intent.startswith('navigation'):
            print(f"  → Robot would navigate: {intent}")
            time.sleep(1)  # Simulate movement time
        elif intent.startswith('manipulation'):
            print(f"  → Robot would manipulate: {intent}")
            time.sleep(2)  # Simulate manipulation time
        elif intent.startswith('interaction'):
            print(f"  → Robot would interact: {intent}")
            time.sleep(0.5)  # Simulate interaction time
        elif intent.startswith('system'):
            print(f"  → Robot would execute system command: {intent}")
            time.sleep(0.2)  # Simulate system command time
        else:
            print(f"  → Unknown command type: {intent}")
            return False
        
        return True  # Simulate successful execution
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        return {
            'running': self.running,
            'queue_sizes': {
                'transcription': self.transcription_queue.qsize(),
                'classification': self.classification_queue.qsize(),
                'execution': self.execution_queue.qsize()
            },
            'statistics': self.stats.copy()
        }
    
    def stop_system(self):
        """Stop the complete system"""
        print("Stopping voice-to-action system...")
        
        self.running = False
        
        # Stop Whisper processor
        self.whisper_processor.stop_listening()
        
        # Wait for threads to finish
        self.transcription_thread.join(timeout=2.0)
        self.classification_thread.join(timeout=2.0)
        self.execution_thread.join(timeout=2.0)
        
        print("Voice-to-action system stopped")
    
    def run_demo(self, duration: int = 60):
        """Run a demonstration of the system"""
        print(f"Starting voice-to-action demo for {duration} seconds...")
        print("Speak commands like:")
        print("  - 'Go to the kitchen'")
        print("  - 'Pick up the red ball'") 
        print("  - 'Wave hello'")
        print("  - 'Say hello everyone'")
        print("\nPress Ctrl+C to stop early...\n")
        
        self.start_system()
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration and self.running:
                time.sleep(1)
                
                # Periodically print stats
                if int(time.time() - start_time) % 10 == 0:
                    status = self.get_system_status()
                    print(f"\n[Stats] Success rate: {status['statistics']['success_rate']*100:.1f}%, "
                          f"Processed: {status['statistics']['commands_processed']}")
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            self.stop_system()
            print(f"\nDemo completed. Final stats:")
            final_status = self.get_system_status()
            print(f"  Total commands processed: {final_status['statistics']['commands_processed']}")
            print(f"  Success rate: {final_status['statistics']['success_rate']*100:.1f}%")
            print(f"  Errors: {final_status['statistics']['errors']}")

def main():
    """Main function to run the complete system"""
    print("Initializing Complete Voice-to-Action System")
    print("=" * 50)
    
    system = CompleteVoiceToActionSystem()
    
    try:
        system.run_demo(duration=120)  # Run for 2 minutes
    except Exception as e:
        print(f"Error running demo: {e}")
    finally:
        print("System shutdown complete")

if __name__ == "__main__":
    main()
```

### Exercise 5: Performance Optimization and Validation

#### Objective
Optimize the voice-to-action pipeline for real-time performance and validate its effectiveness.

#### Steps
1. Profile and optimize system performance
2. Validate accuracy and response times
3. Test robustness under various conditions
4. Document performance characteristics

```python
# performance_optimization.py
import time
import cProfile
import pstats
import io
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

class PerformanceAnalyzer:
    def __init__(self, voice_system):
        """Initialize performance analyzer"""
        self.voice_system = voice_system
        self.measurements = {
            'transcription_times': [],
            'classification_times': [], 
            'mapping_times': [],
            'end_to_end_times': [],
            'throughput_rates': [],
            'accuracy_scores': [],
            'memory_usage': []
        }
        self.baseline_performance = self.get_baseline_performance()
    
    def get_baseline_performance(self) -> Dict:
        """Establish baseline performance metrics"""
        return {
            'transcription_time_target': 0.5,  # seconds
            'classification_time_target': 0.1,  # seconds
            'mapping_time_target': 0.05,  # seconds
            'end_to_end_time_target': 1.0,  # seconds
            'throughput_target': 3,  # commands per second
            'accuracy_target': 0.85,  # 85% accuracy
        }
    
    def profile_transcription_performance(self, num_samples=100) -> Dict:
        """Profile Whisper transcription performance"""
        transcription_times = []
        
        print(f"Profiling Whisper transcription over {num_samples} samples...")
        
        # For this example, we'll simulate transcription performance
        # In a real system, this would use actual Whisper processing
        for i in range(num_samples):
            start_time = time.perf_counter()
            
            # Simulate Whisper processing time (in reality would be actual Whisper call)
            time.sleep(np.random.uniform(0.05, 0.3))  # Simulated processing time
            
            end_time = time.perf_counter()
            transcription_times.append(end_time - start_time)
        
        results = {
            'avg_time': np.mean(transcription_times),
            'std_time': np.std(transcription_times),
            'min_time': np.min(transcription_times),
            'max_time': np.max(transcription_times),
            'percentile_95': np.percentile(transcription_times, 95),
            'percentile_99': np.percentile(transcription_times, 99),
        }
        
        return results
    
    def profile_classification_performance(self, num_samples=100) -> Dict:
        """Profile command classification performance"""
        classification_times = []
        
        print(f"Profiling command classification over {num_samples} samples...")
        
        for i in range(num_samples):
            start_time = time.perf_counter()
            
            # Simulate classification time (in reality would be actual classification)
            time.sleep(np.random.uniform(0.01, 0.05))  # Simulated processing time
            
            end_time = time.perf_counter()
            classification_times.append(end_time - start_time)
        
        results = {
            'avg_time': np.mean(classification_times),
            'std_time': np.std(classification_times),
            'min_time': np.min(classification_times),
            'max_time': np.max(classification_times),
            'percentile_95': np.percentile(classification_times, 95),
            'percentile_99': np.percentile(classification_times, 99),
        }
        
        return results
    
    def profile_action_mapping_performance(self, num_samples=100) -> Dict:
        """Profile action mapping performance"""
        mapping_times = []
        
        print(f"Profiling action mapping over {num_samples} samples...")
        
        for i in range(num_samples):
            start_time = time.perf_counter()
            
            # Simulate action mapping time (in reality would be actual mapping)
            time.sleep(np.random.uniform(0.005, 0.025))  # Simulated processing time
            
            end_time = time.perf_counter()
            mapping_times.append(end_time - start_time)
        
        results = {
            'avg_time': np.mean(mapping_times),
            'std_time': np.std(mapping_times),
            'min_time': np.min(mapping_times),
            'max_time': np.max(mapping_times),
            'percentile_95': np.percentile(mapping_times, 95),
            'percentile_99': np.percentile(mapping_times, 99),
        }
        
        return results
    
    def run_full_performance_analysis(self) -> Dict:
        """Run comprehensive performance analysis"""
        print("Starting comprehensive performance analysis...")
        
        # Profile individual components
        transcription_perf = self.profile_transcription_performance(50)
        classification_perf = self.profile_classification_performance(50)
        mapping_perf = self.profile_action_mapping_performance(50)
        
        # Profile end-to-end system
        end_to_end_times = []
        accuracy_scores = []
        
        print("Profiling end-to-end performance...")
        for i in range(30):  # Fewer samples for end-to-end to save time
            # Simulate end-to-end processing
            start_time = time.perf_counter()
            
            # Simulate all components
            time.sleep(transcription_perf['avg_time'] + 
                     classification_perf['avg_time'] + 
                     mapping_perf['avg_time'])
            
            end_time = time.perf_counter()
            end_to_end_times.append(end_time - start_time)
            
            # Simulate accuracy (in real system would be measured against ground truth)
            accuracy = np.random.uniform(0.7, 0.95)
            accuracy_scores.append(accuracy)
        
        end_to_end_results = {
            'avg_time': np.mean(end_to_end_times),
            'std_time': np.std(end_to_end_times),
            'min_time': np.min(end_to_end_times),
            'max_time': np.max(end_to_end_times),
            'percentile_95': np.percentile(end_to_end_times, 95),
            'percentile_99': np.percentile(end_to_end_times, 99),
        }
        
        accuracy_results = {
            'avg_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores),
            'min_accuracy': np.min(accuracy_scores),
            'max_accuracy': np.max(accuracy_scores),
        }
        
        # Calculate throughput
        total_time = sum(end_to_end_times)
        if total_time > 0:
            throughput = len(end_to_end_times) / total_time  # commands per second
        else:
            throughput = 0
        
        # Compile analysis results
        analysis_results = {
            'component_performance': {
                'transcription': transcription_perf,
                'classification': classification_perf,
                'mapping': mapping_perf
            },
            'system_performance': {
                'end_to_end': end_to_end_results,
                'accuracy': accuracy_results,
                'throughput': throughput,
                'total_processed': len(end_to_end_times)
            },
            'comparisons': {
                'transcription_meets_target': transcription_perf['avg_time'] <= self.baseline_performance['transcription_time_target'],
                'classification_meets_target': classification_perf['avg_time'] <= self.baseline_performance['classification_time_target'],
                'mapping_meets_target': mapping_perf['avg_time'] <= self.baseline_performance['mapping_time_target'],
                'end_to_end_meets_target': end_to_end_results['avg_time'] <= self.baseline_performance['end_to_end_time_target'],
                'throughput_meets_target': throughput >= self.baseline_performance['throughput_target'],
                'accuracy_meets_target': accuracy_results['avg_accuracy'] >= self.baseline_performance['accuracy_target'],
            }
        }
        
        return analysis_results
    
    def generate_performance_report(self, results: Dict) -> str:
        """Generate a detailed performance report"""
        report = []
        report.append("VOICE-TO-ACTION SYSTEM PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Component performance
        report.append("1. COMPONENT PERFORMANCE:")
        report.append("")
        
        comp_perf = results['component_performance']
        for comp_name, perf_data in comp_perf.items():
            report.append(f"  {comp_name.upper()}:")
            report.append(f"    Average time: {perf_data['avg_time']:.3f}s ± {perf_data['std_time']:.3f}s")
            report.append(f"    Min/Max: {perf_data['min_time']:.3f}s / {perf_data['max_time']:.3f}s")
            report.append(f"    95th percentile: {perf_data['percentile_95']:.3f}s")
            report.append(f"    99th percentile: {perf_data['percentile_99']:.3f}s")
            report.append("")
        
        # System performance
        report.append("2. SYSTEM PERFORMANCE:")
        report.append("")
        
        sys_perf = results['system_performance']
        report.append(f"  End-to-End Performance:")
        report.append(f"    Average time: {sys_perf['end_to_end']['avg_time']:.3f}s ± {sys_perf['end_to_end']['std_time']:.3f}s")
        report.append(f"    Throughput: {sys_perf['throughput']:.2f} commands/sec")
        report.append(f"    Total processed: {sys_perf['total_processed']} commands")
        report.append("")
        
        report.append(f"  Accuracy Performance:")
        report.append(f"    Average accuracy: {sys_perf['accuracy']['avg_accuracy']:.3f} ± {sys_perf['accuracy']['std_accuracy']:.3f}")
        report.append(f"    Range: {sys_perf['accuracy']['min_accuracy']:.3f} - {sys_perf['accuracy']['max_accuracy']:.3f}")
        report.append("")
        
        # Compliance check
        report.append("3. COMPLIANCE WITH TARGETS:")
        report.append("")
        
        compliance = results['comparisons']
        targets_met = sum(1 for met in compliance.values() if met)
        total_targets = len(compliance)
        
        for target, met in compliance.items():
            status = "✓ PASS" if met else "✗ FAIL"
            target_name = target.replace('_', ' ').title()
            report.append(f"  {target_name}: {status}")
        
        report.append(f"")
        report.append(f"  Overall compliance: {targets_met}/{total_targets} targets met ({targets_met/total_targets*100:.1f}%)")
        
        # Recommendations
        report.append("")
        report.append("4. RECOMMENDATIONS:")
        report.append("")
        
        if not compliance['transcription_meets_target']:
            report.append("  • Transcription component needs optimization - consider smaller model or model quantization")
        
        if not compliance['end_to_end_meets_target']:
            report.append("  • End-to-end latency exceeds target - consider pipeline optimizations")
        
        if not compliance['accuracy_meets_target']:
            report.append("  • Accuracy below target - consider additional training data or model fine-tuning")
        
        if not compliance['throughput_meets_target']:
            report.append("  • Throughput below target - consider parallel processing or computational optimization")
        
        return "\n".join(report)
    
    def plot_performance_metrics(self, results: Dict):
        """Plot performance metrics for visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Component processing times
        comp_perf = results['component_performance']
        components = ['transcription', 'classification', 'mapping']
        avg_times = [comp_perf[c]['avg_time'] for c in components]
        std_times = [comp_perf[c]['std_time'] for c in components]
        
        ax1 = axes[0, 0]
        bars = ax1.bar(components, avg_times, yerr=std_times, capsize=5, alpha=0.7)
        ax1.set_title('Average Processing Time by Component')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, avg_time in zip(bars, avg_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{avg_time:.3f}s', ha='center', va='bottom')
        
        # End-to-end performance
        end_to_end = results['system_performance']['end_to_end']
        ax2 = axes[0, 1]
        stats = ['Min', 'Avg', 'Max', '95%', '99%']
        values = [end_to_end['min_time'], end_to_end['avg_time'], 
                 end_to_end['max_time'], end_to_end['percentile_95'], end_to_end['percentile_99']]
        ax2.bar(stats, values, alpha=0.7, color='orange')
        ax2.set_title('End-to-End Performance Distribution')
        ax2.set_ylabel('Time (seconds)')
        
        # Accuracy distribution
        accuracy = results['system_performance']['accuracy']
        ax3 = axes[1, 0]
        accuracy_data = [accuracy['min_accuracy'], accuracy['avg_accuracy'], accuracy['max_accuracy']]
        ax3.plot(['Min', 'Avg', 'Max'], accuracy_data, marker='o', linewidth=2, markersize=8)
        ax3.set_title('Accuracy Distribution')
        ax3.set_ylabel('Accuracy Score')
        ax3.grid(True, alpha=0.3)
        
        # Compliance radar chart would require more setup for the 6 dimensions
        compliances = list(results['comparisons'].values())
        compliance_labels = [k.replace('_', ' ').title() for k in results['comparisons'].keys()]
        ax4 = axes[1, 1]
        y_pos = np.arange(len(compliance_labels))
        colors = ['green' if c else 'red' for c in compliances]
        ax4.barh(y_pos, [int(c) for c in compliances], color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(compliance_labels)
        ax4.set_xlabel('Pass (1) / Fail (0)')
        ax4.set_title('Target Compliance Status')
        
        # Add value labels
        for i, (v, c) in enumerate(zip([int(c) for c in compliances], compliances)):
            ax4.text(v + 0.05, i, 'PASS' if c else 'FAIL', va='center')
        
        plt.tight_layout()
        plt.savefig('voice_to_action_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Performance charts saved to 'voice_to_action_performance.png'")

def main():
    """Run performance analysis on the voice-to-action system"""
    print("Initializing performance analyzer...")
    
    # Since we don't have an actual voice system running, we'll create mock analyzer
    # In a real scenario, this would connect to a running voice system
    print("Running performance analysis...")
    
    # Create analyzer instance
    analyzer = PerformanceAnalyzer(None)  # Pass None as mock system
    
    # Run analysis
    results = analyzer.run_full_performance_analysis()
    
    # Generate report
    report = analyzer.generate_performance_report(results)
    print(report)
    
    # Plot metrics
    analyzer.plot_performance_metrics(results)
    
    # Save detailed results
    import json
    with open('performance_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed performance analysis saved to 'performance_analysis_results.json'")

if __name__ == "__main__":
    main()
```

## Assessment Rubric

Your implementation will be evaluated based on:

1. **Technical Implementation** (30%)
   - Correct integration of Whisper with robotics systems
   - Proper command classification and intent recognition
   - Valid action mapping to robot capabilities
   - Performance optimization techniques

2. **System Integration** (25%)
   - Seamless integration between components
   - Proper error handling and fallback strategies
   - Robust communication between modules
   - Real-time performance maintenance

3. **Domain Randomization** (20%)
   - Effective implementation of environmental variations
   - Adequate diversity in synthetic data generation
   - Transfer learning capability demonstration
   - Robustness validation

4. **Validation and Testing** (15%)
   - Comprehensive testing procedures
   - Performance validation against baselines
   - Quality assessment of generated data
   - Safety validation procedures

5. **Documentation and Presentation** (10%)
   - Clear technical documentation
   - Understanding of design choices and trade-offs
   - Quality of performance evaluation
   - Professional presentation of results

## Troubleshooting Guide

### Common Issues

1. **Whisper Model Loading Errors**:
   - Verify PyTorch installation compatibility
   - Check GPU availability for CUDA models
   - Ensure sufficient memory for model loading

2. **Audio Processing Problems**:
   - Verify microphone permissions
   - Check audio input format compatibility
   - Validate sample rate settings

3. **ROS Communication Issues**:
   - Ensure proper ROS network configuration
   - Verify message type compatibility
   - Check topic/service connectivity

4. **Performance Bottlenecks**:
   - Profile individual components
   - Optimize for real-time constraints
   - Consider model quantization for edge deployment

### Performance Optimization Tips

1. **Model Optimization**:
   - Use smaller Whisper models (tiny, base) for real-time applications
   - Apply quantization techniques to reduce model size
   - Implement caching for repeated commands

2. **Pipeline Optimization**:
   - Use asynchronous processing where possible
   - Implement efficient buffering strategies
   - Optimize memory allocation patterns

3. **Resource Management**:
   - Monitor CPU/GPU utilization
   - Implement graceful degradation under load
   - Use efficient data structures and algorithms

This practical exercise provides comprehensive experience with implementing vision-language-action systems for humanoid robotics, emphasizing real-world deployment considerations and performance optimization.