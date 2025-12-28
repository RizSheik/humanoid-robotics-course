# Chapter 3: Services and Synchronous Communication

<div className="robotDiagram">
  <img src="/static/img/book-image/Flowchart_showing_ROS_2_nodes_communicat_1.jpg" alt="Humanoid Robot" style={{borderRadius:"50px", width: '900px', height: '350px', margin: '10px auto', display: 'block'}} />
</div>


## Learning Objectives

After completing this chapter, students will be able to:
- Implement service servers and clients in ROS 2
- Design appropriate service interfaces for robot operations
- Compare when to use services versus topics or actions
- Handle synchronous communication patterns in robotic systems
- Manage service request/response lifecycles
- Integrate services into humanoid robot control architectures

## 3.1 Understanding Services in ROS 2

Services in ROS 2 implement a request-response communication pattern, which provides synchronous communication between nodes. Unlike topics, which are asynchronous and many-to-many, services are synchronous and typically one-to-one between a client and a server.

### 3.1.1 Characteristics of Service Communication

- **Synchronous**: The client waits for a response from the server
- **Request-Response**: Client sends a request, server processes it and returns a response
- **One-to-One**: Typically one server handles requests from one or more clients
- **Blocking**: The client is blocked until a response is received (unless using async clients)

### 3.1.2 When to Use Services

Services are appropriate when:
- You need a direct answer to a query
- The operation is relatively fast (less than a few seconds)
- Results are deterministic and self-contained
- You need to modify robot state or configuration
- You're implementing configuration or calibration operations

Services are NOT appropriate when:
- The operation takes a long time to complete
- The operation might be preempted
- You need to send continuous feedback during execution
- You're dealing with streaming data

## 3.2 Service Types and Standard Interfaces

### 3.2.1 Standard Service Types

ROS 2 provides standard service types in various packages:
- `std_srvs/srv/Empty`: Simple service with no parameters
- `std_srvs/srv/Trigger`: Returns success state, used for simple commands
- `std_srvs/srv/SetBool`: Set boolean value
- `std_srvs/srv/SetString`: Set string value

### 3.2.2 Common Robot Service Types

- `sensor_msgs/srv/SetCameraInfo`: Set camera calibration
- `geometry_msgs/srv/GetTransform`: Get transformation between frames
- `nav_msgs/srv/GetPlan`: Get path planning result
- `tf2_msgs/srv/FrameGraph`: Get information about TF frames

### 3.2.3 Custom Service Types

Custom services are defined using the `.srv` format with the request and response separated by `---`:

```
# Custom service: ComputeIK.srv
# Request
geometry_msgs/Point target_position
geometry_msgs/Quaternion target_orientation
string chain_name
---
# Response
float64[] joint_angles
bool success
string error_message
```

## 3.3 Implementing Service Servers

### 3.3.1 Basic Service Server (Python)

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(SetBool, 'set_bool', self.set_bool_callback)

    def set_bool_callback(self, request, response):
        response.success = request.data
        if request.data:
            response.message = 'Successfully set to true'
            self.get_logger().info('Set to true')
        else:
            response.message = 'Successfully set to false'
            self.get_logger().info('Set to false')
        
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.3.2 Basic Service Server (C++)

```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/set_bool.hpp"

class MinimalService : public rclcpp::Node
{
public:
    MinimalService() : Node("minimal_service")
    {
        service_ = this->create_service<std_srvs::srv::SetBool>(
            "set_bool",
            [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                   std::shared_ptr<std_srvs::srv::SetBool::Response> response)
            {
                response->success = request->data;
                if (request->data) {
                    response->message = "Successfully set to true";
                    RCLCPP_INFO(this->get_logger(), "Set to true");
                } else {
                    response->message = "Successfully set to false";
                    RCLCPP_INFO(this->get_logger(), "Set to false");
                }
            });
    }

private:
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr service_;
};
```

## 3.4 Implementing Service Clients

### 3.4.1 Basic Service Client (Python)

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(SetBool, 'set_bool')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.req = SetBool.Request()

    def send_request(self, data):
        self.req.data = data
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    
    # Send request
    response = minimal_client.send_request(True)
    minimal_client.get_logger().info(f'Result: {response.success}, {response.message}')
    
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.4.2 Asynchronous Service Client (Python)

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import SetBool

class AsyncMinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(SetBool, 'set_bool')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        
        self.req = SetBool.Request()

    def send_request(self, data):
        self.req.data = data
        self.future = self.cli.call_async(self.req)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f'Result: {response.success}, {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    minimal_client = AsyncMinimalClient()
    
    # Send request asynchronously
    minimal_client.send_request(True)
    
    # Keep running to handle response callback
    rclpy.spin(minimal_client)
    
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 3.5 Advanced Service Concepts

### 3.5.1 Service with Custom Types

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class CustomServiceServer(Node):
    def __init__(self):
        super().__init__('custom_service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request: a={request.a}, b={request.b}')
        self.get_logger().info(f'Sending back response: [{response.sum}]')
        return response

def main(args=None):
    rclpy.init(args=args)
    service = CustomServiceServer()
    rclpy.spin(service)
    service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3.5.2 Batch Processing Service

For services that need to process multiple items, consider design patterns that optimize performance:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger  # Custom service would be better here

class BatchService(Node):
    def __init__(self):
        super().__init__('batch_service')
        self.srv = self.create_service(Trigger, 'batch_process', self.batch_callback)

    def batch_callback(self, request, response):
        import time
        # Simulate batch processing
        time.sleep(0.5)  # Processing time
        
        # Perform batch operations
        # This might involve calibration, configuration, or other bulk operations
        success = True
        message = "Batch processing completed successfully"
        
        response.success = success
        response.message = message
        return response
```

## 3.6 Service Design Patterns for Robotics

### 3.6.1 Configuration Services

Used for setting parameters or configuration values:

```
# SetRobotConfiguration.srv
string robot_name
float64[] joint_positions
float64 velocity_limit
float64 acceleration_limit
---
bool success
string error_message
```

### 3.6.2 Calibration Services

Used for robot calibration tasks:

```
# CalibrateSensor.srv
string sensor_name
string calibration_type
float64[] calibration_parameters
---
bool success
float64[] calculated_parameters
string error_message
```

### 3.6.3 State Management Services

Used for changing robot operational states:

```
# SetRobotState.srv
string requested_state  # e.g., "idle", "active", "error"
bool force_change
---
bool success
string previous_state
string error_message
```

## 3.7 Service Implementation for Humanoid Robots

### 3.7.1 Joint Control Services

For setting joint positions or parameters:

```python
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

class JointControlService(Node):
    def __init__(self):
        super().__init__('joint_control_service')
        self.srv = self.create_service(Trigger, 'initialize_joints', self.init_joints_callback)
        self.joint_pub = self.create_publisher(JointState, 'joint_commands', 10)
        
    def init_joints_callback(self, request, response):
        # Initialize all joints to home position
        try:
            # Send initialization commands to joints
            msg = JointState()
            msg.name = ['joint1', 'joint2', 'joint3']  # Actual joint names
            msg.position = [0.0, 0.0, 0.0]  # Home positions
            self.joint_pub.publish(msg)
            
            response.success = True
            response.message = "Joints initialized successfully"
        except Exception as e:
            response.success = False
            response.message = f"Failed to initialize joints: {str(e)}"
        
        return response
```

### 3.7.2 Sensor Management Services

For managing sensor states:

```python
from std_srvs.srv import SetBool
from sensor_msgs.msg import CameraInfo

class SensorManagementService(Node):
    def __init__(self):
        super().__init__('sensor_management_service')
        self.enable_camera_srv = self.create_service(
            SetBool, 'enable_camera', self.enable_camera_callback)
        
    def enable_camera_callback(self, request, response):
        # Enable or disable camera based on request
        try:
            # Implement camera enable/disable logic
            if request.data:
                # Enable camera
                self.get_logger().info("Camera enabled")
            else:
                # Disable camera
                self.get_logger().info("Camera disabled")
            
            response.success = True
            response.message = f"Camera {'enabled' if request.data else 'disabled'} successfully"
        except Exception as e:
            response.success = False
            response.message = f"Failed to change camera state: {str(e)}"
        
        return response
```

## 3.8 Performance and Reliability Considerations

### 3.8.1 Timeout Handling

Implement proper timeout handling in service clients:

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

class TimeoutClient(Node):
    def __init__(self):
        super().__init__('timeout_client')
        self.cli = self.create_client(Trigger, 'slow_service')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

    def call_with_timeout(self, timeout_sec=5.0):
        request = Trigger.Request()
        future = self.cli.call_async(request)
        
        # Wait with timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        
        if future.done():
            try:
                response = future.result()
                return response
            except Exception as e:
                self.get_logger().error(f'Service call failed: {e}')
                return None
        else:
            self.get_logger().warning('Service call timed out')
            return None
```

### 3.8.2 Error Handling and Recovery

Implement robust error handling in service implementations:

```python
def robust_service_callback(self, request, response):
    try:
        # Validate input parameters
        if not self.validate_request(request):
            response.success = False
            response.message = "Invalid request parameters"
            return response
        
        # Perform the service operation
        result = self.perform_operation(request)
        
        # Return the result
        response.success = True
        response.message = "Operation completed successfully"
        # Include result data if applicable
        response.result_data = result
        
    except ValueError as e:
        # Handle value errors specifically
        response.success = False
        response.message = f"Value error: {str(e)}"
    except RuntimeError as e:
        # Handle runtime errors
        response.success = False
        response.message = f"Runtime error: {str(e)}"
    except Exception as e:
        # Handle unexpected errors
        response.success = False
        response.message = f"Unexpected error: {str(e)}"
        self.get_logger().error(f"Unexpected error in service: {e}")
    
    return response
```

## 3.9 Security Considerations

### 3.9.1 Service Authentication

For security-sensitive services, consider implementing authentication mechanisms:

```python
def secure_service_callback(self, request, response):
    # Verify authentication token if present in request
    if hasattr(request, 'auth_token'):
        if not self.verify_auth_token(request.auth_token):
            response.success = False
            response.message = "Authentication failed"
            return response
    
    # Perform authorized operation
    # ... implementation ...
    return response
```

### 3.9.2 Access Control

Implement access control for sensitive services:

```python
class AccessControlService(Node):
    def __init__(self):
        super().__init__('access_control_service')
        # Define authorized nodes/users
        self.authorized_nodes = ['safe_controller', 'maintenance_client']
        self.srv = self.create_service(Trigger, 'critical_operation', self.op_callback)

    def is_authorized(self, client_info):
        # Check if client is authorized
        # This could involve checking client node name, certificates, etc.
        client_name = client_info.name  # hypothetical client info
        return client_name in self.authorized_nodes
```

## 3.10 Best Practices for Service Development

### 3.10.1 Service Naming Conventions

- Use descriptive names that clearly indicate the service purpose
- Follow the pattern `verb_noun` (e.g., `enable_camera`, `calibrate_sensor`)
- Use lowercase with underscores
- Be consistent across related services

### 3.10.2 Error Handling

- Always return meaningful error messages
- Use appropriate return codes to indicate different types of failures
- Log errors appropriately for debugging
- Implement proper cleanup in error cases

### 3.10.3 Performance Considerations

- Keep service operations short (under a few seconds)
- For longer operations, consider using actions instead
- Use appropriate QoS settings for service communication
- Implement timeouts in clients to prevent hanging

### 3.10.4 Documentation

- Document service interfaces clearly
- Include examples of how to use the service
- Specify expected response times
- Document any dependencies or preconditions

## 3.11 Comparison with Other Communication Patterns

| Feature | Topics | Services | Actions |
|---------|--------|----------|---------|
| Communication Type | Async | Sync | Async (with status) |
| Best For | Streaming data | One-shot queries | Long-running tasks |
| Blocking | No | Yes (sync) | No |
| Feedback | No | No | Yes |
| Preemption | N/A | N/A | Yes |

## Chapter Summary

This chapter covered the fundamentals of service-based communication in ROS 2, including implementation of service servers and clients, design patterns for robotic applications, and considerations for humanoid robot systems. We explored performance, security, and best practices for service development, emphasizing when to use services over other communication patterns.

## Key Terms
- Request-Response Pattern
- Service Server
- Service Client
- Synchronous Communication
- Service Interface Definition
- Service Timeout
- Access Control

## Exercises
1. Implement a service for calibrating robot joints
2. Create a service interface for emergency stop functionality
3. Design a service architecture for humanoid robot state management
4. Implement a client that handles service timeouts gracefully

## References
- ROS 2 Documentation: https://docs.ros.org/
- Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics.
- Quigley, M., Gerkey, B., & Smart, W. D. (2015). Programming robots with ROS.