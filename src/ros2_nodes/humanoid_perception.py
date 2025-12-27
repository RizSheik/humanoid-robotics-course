# ROS 2 Humanoid Perception Node

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan, Imu
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import cv2
from cv_bridge import CvBridge
import numpy as np
import message_filters

class HumanoidPerception(Node):
    """
    A ROS 2 node that processes sensor data for a humanoid robot,
    including cameras, LiDAR, IMU, and other sensors.
    """
    
    def __init__(self):
        super().__init__('humanoid_perception')
        
        # Initialize CvBridge for image processing
        self.bridge = CvBridge()
        
        # Create transform broadcaster for TF tree
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribers for various sensors
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )
        
        self.lidar_subscriber = self.create_subscription(
            LaserScan,
            '/laser_scan',
            self.lidar_callback,
            10
        )
        
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Publishers for processed data
        self.obstacle_publisher = self.create_publisher(PointStamped, '/obstacles', 10)
        self.floor_publisher = self.create_publisher(PointStamped, '/floor_normal', 10)
        
        # Set parameters for perception algorithms
        self.declare_parameter('image_processing_enabled', True)
        self.declare_parameter('lidar_processing_enabled', True)
        self.declare_parameter('detection_confidence_threshold', 0.7)
        
        self.image_processing_enabled = self.get_parameter('image_processing_enabled').value
        self.lidar_processing_enabled = self.get_parameter('lidar_processing_enabled').value
        self.confidence_threshold = self.get_parameter('detection_confidence_threshold').value
        
        self.get_logger().info('Humanoid Perception node initialized')

    def image_callback(self, msg):
        """Process incoming image data"""
        if not self.image_processing_enabled:
            return
            
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Apply perception algorithms
            # Example: Simple object detection using color thresholding
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # Detect red objects (for example)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            
            mask = mask1 + mask2
            
            # Find contours of detected objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Only consider large enough objects
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw contour and centroid
                        cv2.drawContours(cv_image, [contour], -1, (0, 255, 0), 2)
                        cv2.circle(cv_image, (cx, cy), 7, (255, 255, 255), -1)
                        
                        # Log detection
                        self.get_logger().info(f'Detected red object at ({cx}, {cy}) with area {area}')
            
            # For demonstration, show the processed image
            # In a real application, you might publish the detections instead
            cv2.imshow("Processed Image", cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def lidar_callback(self, msg):
        """Process incoming LiDAR data to detect obstacles and floor"""
        if not self.lidar_processing_enabled:
            return
            
        try:
            # Process LiDAR data
            # Convert to numpy array for easier processing
            angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            distances = np.array(msg.ranges)
            
            # Filter out invalid measurements
            valid_indices = (distances >= msg.range_min) & (distances <= msg.range_max)
            valid_angles = angles[valid_indices]
            valid_distances = distances[valid_indices]
            
            # Detect obstacles (objects within 1 meter)
            obstacle_angles = valid_angles[valid_distances < 1.0]
            obstacle_distances = valid_distances[valid_distances < 1.0]
            
            if len(obstacle_distances) > 0:
                # Publish the closest obstacle
                closest_idx = np.argmin(obstacle_distances)
                closest_angle = obstacle_angles[closest_idx]
                closest_distance = obstacle_distances[closest_idx]
                
                # Convert polar to Cartesian coordinates
                x = closest_distance * np.cos(closest_angle)
                y = closest_distance * np.sin(closest_angle)
                
                obstacle_msg = PointStamped()
                obstacle_msg.header.stamp = self.get_clock().now().to_msg()
                obstacle_msg.header.frame_id = "laser_frame"
                obstacle_msg.point.x = x
                obstacle_msg.point.y = y
                obstacle_msg.point.z = 0.0  # Assuming flat floor
                
                self.obstacle_publisher.publish(obstacle_msg)
                self.get_logger().info(f'Detected obstacle at ({x:.2f}, {y:.2f})')
            
            # Publish floor normal (always pointing up in robot frame)
            floor_msg = PointStamped()
            floor_msg.header.stamp = self.get_clock().now().to_msg()
            floor_msg.header.frame_id = "base_link"
            floor_msg.point.x = 0.0
            floor_msg.point.y = 0.0
            floor_msg.point.z = 1.0  # Normal pointing upward
            
            self.floor_publisher.publish(floor_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {str(e)}')

    def imu_callback(self, msg):
        """Process incoming IMU data for balance and orientation"""
        # Extract orientation from IMU quaternion
        orientation = {
            'x': msg.orientation.x,
            'y': msg.orientation.y,
            'z': msg.orientation.z,
            'w': msg.orientation.w
        }
        
        # Extract angular velocity
        angular_velocity = {
            'x': msg.angular_velocity.x,
            'y': msg.angular_velocity.y,
            'z': msg.angular_velocity.z
        }
        
        # Extract linear acceleration
        linear_acceleration = {
            'x': msg.linear_acceleration.x,
            'y': msg.linear_acceleration.y,
            'z': msg.linear_acceleration.z
        }
        
        # Log important values (like tilt angles)
        # Convert quaternion to Euler angles for easier interpretation
        import math
        sinr_cosp = 2 * (orientation['w'] * orientation['x'] + orientation['y'] * orientation['z'])
        cosr_cosp = 1 - 2 * (orientation['x'] * orientation['x'] + orientation['y'] * orientation['y'])
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (orientation['w'] * orientation['y'] - orientation['z'] * orientation['x'])
        pitch = math.asin(sinp)

        siny_cosp = 2 * (orientation['w'] * orientation['z'] + orientation['x'] * orientation['y'])
        cosy_cosp = 1 - 2 * (orientation['y'] * orientation['y'] + orientation['z'] * orientation['z'])
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.get_logger().info(f'IMU Orientation - Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}')

def main(args=None):
    rclpy.init(args=args)
    
    humanoid_perception = HumanoidPerception()
    
    try:
        rclpy.spin(humanoid_perception)
    except KeyboardInterrupt:
        pass
    finally:
        humanoid_perception.get_logger().info('Shutting down Humanoid Perception')
        cv2.destroyAllWindows()
        humanoid_perception.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()