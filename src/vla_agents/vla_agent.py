# Vision-Language-Action (VLA) Agent for Humanoid Robotics

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import openai
import numpy as np
import json
import time
from threading import Lock

class VLAAgent(Node):
    """
    A Vision-Language-Action agent that integrates perception, language understanding,
    and action planning for humanoid robots using LLMs and computer vision.
    """
    
    def __init__(self):
        super().__init__('vla_agent')
        
        # Initialize components
        self.bridge = CvBridge()
        self.image_data = None
        self.image_lock = Lock()
        
        # Set up publishers for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speech_publisher = self.create_publisher(String, '/speech_output', 10)
        
        # Set up subscribers for sensor data
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )
        
        # Set up subscriber for voice commands
        self.voice_command_subscriber = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )
        
        # Set parameters
        self.declare_parameter('openai_api_key', '')
        self.openai_api_key = self.get_parameter('openai_api_key').value
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        else:
            self.get_logger().warn("No OpenAI API key provided - using mock responses")
        
        # Action execution timer
        self.action_timer = self.create_timer(0.1, self.execute_pending_action)
        self.pending_action = None
        self.last_action_time = 0
        
        self.get_logger().info("VLA Agent initialized")

    def image_callback(self, msg):
        """Store the latest image for processing"""
        with self.image_lock:
            self.image_data = msg

    def voice_command_callback(self, msg):
        """Process voice commands and generate appropriate responses"""
        try:
            command = msg.data
            self.get_logger().info(f"Received voice command: {command}")
            
            # Process the command with vision input if available
            with self.image_lock:
                if self.image_data is not None:
                    cv_image = self.bridge.imgmsg_to_cv2(self.image_data, "bgr8")
                    # For efficiency, we'll compress the image before sending to LLM
                    # In a real implementation, we'd use more sophisticated preprocessing
                    action = self.process_command_with_vision(command, cv_image)
                else:
                    # Process command without vision
                    action = self.process_command(command)
            
            # Execute the determined action
            self.pending_action = action
            self.last_action_time = time.time()
            
        except Exception as e:
            self.get_logger().error(f"Error processing voice command: {str(e)}")

    def process_command(self, command):
        """Process a command without vision input using LLM"""
        prompt = f"""
        You are an AI assistant controlling a humanoid robot. Based on the user's command, determine the appropriate action.

        Command: {command}

        Available actions:
        1. move_forward(distance_meters)
        2. turn_left(degrees)
        3. turn_right(degrees)
        4. speak(text)
        5. stop
        6. wave
        7. nod_head
        
        Respond with a JSON object containing the action type and parameters:
        {{
            "action": "action_type",
            "params": [param1, param2, ...]
        }}
        """
        
        return self.get_llm_response(prompt)

    def process_command_with_vision(self, command, image):
        """Process a command with vision input using multimodal LLM"""
        # In a real implementation, we'd send the image to a multimodal model
        # For this example, we'll simulate a vision analysis
        vision_analysis = self.analyze_scene(image)
        
        prompt = f"""
        You are an AI assistant controlling a humanoid robot. Based on the user's command and visual input, determine the appropriate action.

        Command: {command}
        Visual Analysis: {vision_analysis}

        Available actions:
        1. move_forward(distance_meters)
        2. turn_left(degrees)
        3. turn_right(degrees)
        4. speak(text)
        5. stop
        6. wave
        7. nod_head
        8. pick_up(object_name)
        9. place_at(location)
        
        Respond with a JSON object containing the action type and parameters:
        {{
            "action": "action_type",
            "params": [param1, param2, ...]
        }}
        """
        
        return self.get_llm_response(prompt)

    def analyze_scene(self, image):
        """Analyze the visual scene (simulated for this example)"""
        # In a real implementation, this would run an object detection model
        # For this example, we'll simulate the detection of common objects
        objects = ["table", "chair", "person"]  # Simulated detection results
        return f"Detected objects: {', '.join(objects)}. Room appears to be an office environment."

    def get_llm_response(self, prompt):
        """Get response from LLM (uses mock response if no API key)"""
        if hasattr(openai, 'api_key') and openai.api_key:
            try:
                # In a real implementation, we might use GPT-4 Vision if we have an image
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                return json.loads(response.choices[0].message['content'])
            except Exception as e:
                self.get_logger().error(f"Error calling OpenAI API: {str(e)}")
                # Return a mock response
                return {"action": "speak", "params": ["Sorry, I encountered an error processing your request"]}
        else:
            # For demo purposes without an API key, return a mock response
            # This is just for demonstration, in real usage, an API key is required
            return {"action": "speak", "params": ["I understand your command: " + prompt.split('\n')[2] + ". I'll perform this action."]}
    
    def execute_pending_action(self):
        """Execute the pending action if one exists"""
        if self.pending_action and time.time() - self.last_action_time < 2.0:  # 2 second timeout
            action_type = self.pending_action.get('action', 'stop')
            params = self.pending_action.get('params', [])
            
            if action_type == 'move_forward':
                self.move_forward(params[0] if params else 0.5)
            elif action_type == 'turn_left':
                self.turn(1, params[0] if params else 90)
            elif action_type == 'turn_right':
                self.turn(-1, params[0] if params else 90)
            elif action_type == 'speak':
                self.speak(params[0] if params else "Hello")
            elif action_type == 'stop':
                self.stop()
            elif action_type == 'wave':
                self.wave()
            elif action_type == 'nod_head':
                self.nod_head()
            
            self.pending_action = None
        elif self.pending_action:
            # Timeout action
            self.pending_action = None

    def move_forward(self, distance):
        """Move the robot forward by a specified distance"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.2  # m/s
        cmd_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info(f"Moving forward {distance} meters")

        # Stop after moving for the appropriate time
        time_to_move = distance / 0.2
        self.get_logger().info(f"Will stop in {time_to_move} seconds")
        
        # In a real implementation, we'd use a timer or feedback to stop after distance
        # For this example, we'll just send a stop command after a short time
        self.create_timer(time_to_move, self.stop)

    def turn(self, direction, degrees):
        """Turn the robot left or right"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = direction * 0.5  # rad/s
        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info(f"Turning {direction * degrees} degrees")

        # Calculate time to turn (assuming 0.5 rad/s = 90 degrees in pi/4 seconds ~ 0.785s)
        time_to_turn = np.radians(degrees) / 0.5
        self.create_timer(time_to_turn, self.stop)

    def stop(self):
        """Stop all robot movement"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info("Robot stopped")

    def speak(self, text):
        """Make the robot speak text"""
        msg = String()
        msg.data = text
        self.speech_publisher.publish(msg)
        self.get_logger().info(f"Speaking: {text}")

    def wave(self):
        """Execute waving action (in a real robot, this would control arm joints)"""
        self.get_logger().info("Executing wave action")
        # In a real implementation, this would publish to joint controllers

    def nod_head(self):
        """Execute head nod action (in a real robot, this would control neck joints)"""
        self.get_logger().info("Executing head nod action")
        # In a real implementation, this would publish to joint controllers

def main(args=None):
    rclpy.init(args=args)
    
    vla_agent = VLAAgent()
    
    try:
        rclpy.spin(vla_agent)
    except KeyboardInterrupt:
        pass
    finally:
        vla_agent.get_logger().info('Shutting down VLA Agent')
        vla_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()