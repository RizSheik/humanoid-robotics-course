---
id: module-4-chapter-1-whisper-integration
title: 'Module 4 — Vision-Language-Action Systems | Chapter 1 — Whisper Integration'
sidebar_label: 'Chapter 1 — Whisper Integration'
---

# Chapter 1 — Whisper Integration

## OpenAI Whisper for Voice-to-Action in Humanoid Robotics

OpenAI Whisper is a state-of-the-art speech recognition model that can transcribe audio to text with high accuracy. In humanoid robotics, Whisper can be integrated to enable natural voice-to-action capabilities, allowing users to control robots through spoken commands.

### Understanding Whisper

Whisper is a general-purpose speech recognition model that:
- Supports multiple languages
- Performs well across accents and background noise
- Handles transcription, translation, and language identification
- Works well on both clean and noisy audio

### Whisper Architecture

Whisper uses a transformer-based encoder-decoder architecture:
- **Encoder**: Processes audio spectrograms
- **Decoder**: Generates text tokens conditioned on audio
- **Multilingual capability**: Can transcribe in 99+ languages
- **Robustness**: Trained on diverse datasets for robust performance

### Whisper in Robotics Context

For humanoid robotics:
- **Voice Commands**: Converting spoken instructions to text
- **Human Interaction**: Enabling natural conversation
- **Accessibility**: Providing voice control for users with mobility limitations
- **Hands-Free Operation**: Allowing tasks to be performed without physical input

### Implementation Approaches

#### Offline Approach
Using a locally installed Whisper model for privacy and performance:

```python
import whisper
import torch
import numpy as np

class WhisperRobotInterface:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper with specified model size
        Options: tiny, base, small, medium, large
        """
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        self.options = whisper.DecodingOptions(fp16=torch.cuda.is_available())
        
    def transcribe_audio(self, audio_path):
        """Transcribe audio file to text"""
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # Decode the audio
        _, probs = self.model.encode(mel)
        result = self.model.decode(mel, self.options)
        
        return result.text
    
    def transcribe_audio_tensor(self, audio_tensor):
        """Transcribe audio tensor to text"""
        # Ensure audio is properly formatted
        if audio_tensor.shape[0] > 1:
            # Average across channels if stereo
            audio_tensor = torch.mean(audio_tensor, dim=0)
        
        # Pad or trim to 30-second window
        audio = whisper.pad_or_trim(audio_tensor)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # Perform transcription
        result = self.model.transcribe(audio_tensor.cpu().numpy())
        
        return result["text"]

# Alternative using the higher-level interface
class WhisperRobotInterfaceSimple:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_input):
        """
        Transcribe audio to text
        audio_input: file path or numpy array of audio data
        """
        result = self.model.transcribe(audio_input)
        return result["text"]
    
    def transcribe_with_timestamps(self, audio_input):
        """Transcribe audio with word-level timestamps"""
        result = self.model.transcribe(audio_input, word_timestamps=True)
        return result
```

#### Online/Streaming Approach
For real-time applications:

```python
import pyaudio
import numpy as np
import threading
import queue
import torch
import whisper

class StreamingWhisperInterface:
    def __init__(self, model_size="base", chunk_duration=1.0):
        """
        Initialize streaming Whisper interface
        chunk_duration: Duration of audio chunks in seconds
        """
        self.model = whisper.load_model(model_size)
        self.chunk_duration = chunk_duration
        
        # Audio stream parameters
        self.rate = 16000  # Sample rate
        self.chunk_size = int(self.rate * chunk_duration)
        
        # Buffers
        self.audio_buffer = np.array([])
        self.transcription_queue = queue.Queue()
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def start_streaming(self):
        """Start audio stream for real-time transcription"""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Start streaming thread
        self.streaming_thread = threading.Thread(target=self._streaming_worker)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
    def _streaming_worker(self):
        """Worker thread for processing audio chunks"""
        while True:
            # Read audio chunk
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            
            # Transcribe if we have enough audio (3 seconds minimum recommended)
            if len(self.audio_buffer) >= 3 * self.rate:
                self._transcribe_buffer()
    
    def _transcribe_buffer(self):
        """Transcribe the accumulated audio buffer"""
        # Transcribe the buffer
        result = self.model.transcribe(self.audio_buffer)
        text = result["text"]
        
        # Add to transcription queue
        self.transcription_queue.put(text)
        
        # Keep a small portion of the buffer to maintain context
        keep_duration = 1.0  # Keep 1 second of context
        keep_samples = int(keep_duration * self.rate)
        self.audio_buffer = self.audio_buffer[-keep_samples:] if len(self.audio_buffer) > keep_samples else self.audio_buffer
    
    def get_transcription(self, timeout=None):
        """Get next transcription from queue"""
        try:
            return self.transcription_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop_streaming(self):
        """Stop audio streaming"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
```

### Integration with Robot Control

Connecting Whisper to robot action execution:

```python
class VoiceToActionConverter:
    def __init__(self):
        self.whisper_interface = WhisperRobotInterfaceSimple()
        self.action_mapping = self.create_action_mapping()
    
    def create_action_mapping(self):
        """Define mapping from voice commands to robot actions"""
        return {
            # Navigation actions
            "move forward": "NAVIGATE_FORWARD",
            "go forward": "NAVIGATE_FORWARD",
            "move backward": "NAVIGATE_BACKWARD",
            "go back": "NAVIGATE_BACKWARD",
            "turn left": "TURN_LEFT",
            "rotate left": "TURN_LEFT",
            "turn right": "TURN_RIGHT",
            "rotate right": "TURN_RIGHT",
            "go to the kitchen": "NAVIGATE_TO_KITCHEN",
            "go to the living room": "NAVIGATE_TO_LIVING_ROOM",
            "go to the bedroom": "NAVIGATE_TO_BEDROOM",
            
            # Manipulation actions
            "pick up the red cup": "PICK_UP_RED_CUP",
            "pick up the blue ball": "PICK_UP_BLUE_BALL",
            "put it down": "PLACE_OBJECT",
            "place the object": "PLACE_OBJECT",
            "grasp the item": "GRASP_OBJECT",
            "release the object": "RELEASE_GRASP",
            
            # Interaction actions
            "wave hello": "WAVE_HELLO",
            "say hello": "SPEAK_HELLO",
            "introduce yourself": "INTRODUCE_ROBOT",
            "what's your name": "SPEAK_NAME",
            "how are you": "SPEAK_STATUS",
            
            # System commands
            "stop": "STOP_ROBOT",
            "pause": "PAUSE_EXECUTION",
            "resume": "RESUME_EXECUTION",
            "shutdown": "SHUTDOWN_ROBOT"
        }
    
    def process_voice_command(self, audio_input):
        """
        Process voice command and return robot action
        """
        # Transcribe audio to text
        transcribed_text = self.whisper_interface.transcribe(audio_input)
        
        # Normalize text for matching
        normalized_text = transcribed_text.lower().strip()
        
        # Find matching action (with fuzzy matching for variations)
        action = self.match_command(normalized_text)
        
        return {
            'original_text': transcribed_text,
            'normalized_text': normalized_text,
            'predicted_action': action,
            'confidence': 1.0  # Whisper doesn't provide confidence, assuming successful
        }
    
    def match_command(self, text):
        """Match text to robot action with fuzzy matching"""
        # Exact match first
        if text in self.action_mapping:
            return self.action_mapping[text]
        
        # Fuzzy matching for similar phrases
        best_match = None
        best_score = 0
        
        for command, action in self.action_mapping.items():
            score = self.calculate_similarity(text, command)
            if score > best_score and score > 0.7:  # Threshold for fuzzy matching
                best_score = score
                best_match = action
        
        return best_match if best_match else "UNKNOWN_COMMAND"
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two text strings"""
        # Simple word overlap ratio
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0

# Advanced voice command processor with NLP enhancement
class AdvancedVoiceCommandProcessor:
    def __init__(self):
        self.whisper_interface = WhisperRobotInterfaceSimple()
        self.nlp_model = self.load_nlp_model()  # Could use spaCy, NLTK, etc.
        self.action_parser = ActionParser()
    
    def load_nlp_model(self):
        """Load NLP model for better understanding"""
        # In a real implementation, this could be:
        # import spacy
        # return spacy.load("en_core_web_sm")
        return None
    
    def process_complex_command(self, audio_input):
        """
        Process complex voice commands that may contain multiple actions
        or complex instructions
        """
        # Transcribe audio
        transcribed_text = self.whisper_interface.transcribe(audio_input)
        
        # Parse complex command structure
        parsed_actions = self.action_parser.parse(transcribed_text)
        
        return {
            'raw_text': transcribed_text,
            'parsed_actions': parsed_actions,
            'confidence': 1.0
        }

class ActionParser:
    def parse(self, command_text):
        """Parse natural language command into robot actions"""
        # This would use more sophisticated NLP techniques
        # For this example, we'll do simple pattern matching
        
        actions = []
        
        # Check for spatial references
        if "kitchen" in command_text.lower():
            actions.append({"action": "NAVIGATE_TO_LOCATION", "location": "kitchen"})
        
        if "living room" in command_text.lower():
            actions.append({"action": "NAVIGATE_TO_LOCATION", "location": "living_room"})
        
        # Check for object manipulation
        if "pick up" in command_text.lower() or "grasp" in command_text.lower():
            # Extract object if possible
            object_name = self.extract_object_name(command_text)
            actions.append({
                "action": "PICK_UP_OBJECT",
                "object": object_name
            })
        
        # Check for navigation
        if any(word in command_text.lower() for word in ["move", "go", "walk", "turn", "rotate"]):
            nav_action = self.extract_navigation_action(command_text)
            if nav_action:
                actions.append(nav_action)
        
        return actions
    
    def extract_object_name(self, text):
        """Extract object name from command"""
        # Simple extraction - in reality would use NLP
        common_objects = ["cup", "ball", "box", "bottle", "book", "chair"]
        for obj in common_objects:
            if obj in text.lower():
                # Look for color descriptors
                color = self.extract_color(text)
                return f"{color} {obj}" if color else obj
        return "unknown_object"
    
    def extract_color(self, text):
        """Extract color from text"""
        colors = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
        for color in colors:
            if color in text.lower():
                return color
        return None
    
    def extract_navigation_action(self, text):
        """Extract navigation action from text"""
        text_lower = text.lower()
        
        if "forward" in text_lower or "ahead" in text_lower:
            return {"action": "MOVE_FORWARD", "distance": "medium"}
        
        if "backward" in text_lower or "back" in text_lower:
            return {"action": "MOVE_BACKWARD", "distance": "medium"}
        
        if "left" in text_lower:
            return {"action": "ROTATE_LEFT", "angle": 90}
        
        if "right" in text_lower:
            return {"action": "ROTATE_RIGHT", "angle": 90}
        
        return None
```

### Real-World Integration Considerations

#### Audio Quality Optimization
```python
import sounddevice as sd
from scipy import signal
import resampy

class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000
    
    def preprocess_audio(self, audio_data, original_sr):
        """
        Preprocess audio for Whisper
        """
        # Resample if needed
        if original_sr != self.sample_rate:
            audio_data = resampy.resample(
                audio_data, 
                original_sr, 
                self.sample_rate
            )
        
        # Apply noise reduction (simple high-pass filter)
        # Remove low-frequency noise
        sos = signal.butter(10, 15, 'hp', fs=self.sample_rate, output='sos')
        audio_data = signal.sosfilt(sos, audio_data)
        
        # Normalize audio levels
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    
    def record_audio(self, duration=3):
        """Record audio from microphone"""
        print(f"Recording {duration} seconds of audio...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete
        
        return audio_data.flatten()
```

#### Performance Optimization
```python
import torch
from functools import lru_cache

class OptimizedWhisperInterface:
    def __init__(self, model_size="base"):
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def transcribe_batch(self, audio_list):
        """
        Transcribe multiple audio clips efficiently
        """
        transcriptions = []
        
        with torch.no_grad():  # Disable gradient computation for inference
            for audio in audio_list:
                # Move audio to device if needed
                if isinstance(audio, np.ndarray):
                    audio_tensor = torch.from_numpy(audio).to(self.device)
                else:
                    audio_tensor = audio.to(self.device)
                
                # Transcribe
                result = self.model.transcribe(audio_tensor.cpu().numpy())
                transcriptions.append(result["text"])
        
        return transcriptions
```

### Error Handling and Fallback Strategies

```python
class RobustVoiceInterface:
    def __init__(self):
        self.whisper_interface = WhisperRobotInterfaceSimple()
        self.backup_services = [
            self.manual_input_fallback,
            self.predefined_commands
        ]
    
    def robust_transcribe(self, audio_input, max_retries=3):
        """Transcribe with retry logic and fallbacks"""
        for attempt in range(max_retries):
            try:
                result = self.whisper_interface.transcribe(audio_input)
                if result and len(result.strip()) > 0:
                    return result
            except Exception as e:
                print(f"Whisper transcribe attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:  # Last attempt
                    return self.handle_failure()
        
        return "TRANSCRIPTION_FAILED"
    
    def handle_failure(self):
        """Handle transcribe failure"""
        print("All transcription attempts failed, trying fallback methods...")
        
        for fallback_method in self.backup_services:
            try:
                result = fallback_method()
                if result:
                    return result
            except Exception as e:
                print(f"Fallback method failed: {e}")
        
        return "UNABLE_TO_PROCESS_COMMAND"
    
    def manual_input_fallback(self):
        """Fallback to manual text input"""
        print("Manual input required. Please type the command:")
        # In a real system, this would use GUI or other input method
        return input("Command: ")
    
    def predefined_commands(self):
        """Return common predefined command"""
        # In real system, this could return most common commands
        return None
```

### Privacy and Security Considerations

When implementing Whisper in humanoid robots:

1. **Local Processing**: Use local models to avoid sending audio to cloud services
2. **Data Encryption**: Encrypt stored audio data
3. **Access Controls**: Limit access to audio processing capabilities
4. **Consent Mechanisms**: Implement opt-in for voice recording features

Integrating Whisper into humanoid robotics systems enables natural voice interaction, making robots more accessible and user-friendly. The implementation requires careful consideration of real-time performance, audio quality, and robust error handling.