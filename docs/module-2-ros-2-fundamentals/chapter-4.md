# Chapter 4: Actions and Complex Task Execution


<div className="robotDiagram">
  <img src="../../../img/book-image/Highquality_infographic_of_ROS_2_archite_1.jpg" alt="Humanoid Robot" style={{borderRadius:"50px", width: '900px', height: '350px', margin: '10px auto', display: 'block'}} />
</div>


## Learning Objectives

After completing this chapter, students will be able to:
- Implement action servers and clients in ROS 2 for long-running tasks
- Design appropriate action interfaces for complex robot operations
- Handle goal, feedback, and result patterns in robotic applications
- Manage action lifecycle including preemption and cancellation
- Integrate actions into humanoid robot task planning and execution
- Compare when to use actions versus topics or services

## 4.1 Understanding Actions in ROS 2

Actions in ROS 2 provide a communication pattern for long-running tasks that require feedback and the ability to be preempted. Unlike services, which are synchronous, or topics, which are asynchronous without built-in request-response patterns, actions combine the best of both approaches.

### 4.1.1 Characteristics of Action Communication

- **Long-running**: Designed for tasks that take seconds, minutes, or longer
- **Goal-oriented**: Client sends a goal, server executes it
- **Feedback-enabled**: Server provides continuous feedback during execution
- **Preemptable**: Goals can be canceled or preempted by new goals
- **Result-returning**: Server returns a final result when task completes

### 4.1.2 When to Use Actions

Actions are appropriate when:
- The operation takes a significant amount of time
- You need to monitor progress during execution
- The operation might need to be canceled or changed
- You need to distinguish between different goals for the same task
- The task has intermediate results worth reporting

## 4.2 Action Types and Standard Interfaces

### 4.2.1 Action Message Components

Each action type consists of three message definitions:
- **Goal**: Parameters for the action
- **Feedback**: Status during action execution
- **Result**: Final outcome of the action

Example action definition (`Fibonacci.action`):
```
#goal definition
int32 order
---
#feedback
int32[] sequence
---
#result
int32[] sequence
```

### 4.2.2 Standard Action Types

ROS 2 provides common action types:
- `control_msgs/action/FollowJointTrajectory`: Execute joint trajectories
- `control_msgs/action/PointHead`: Point a sensor in a specific direction
- `move_base_msgs/action/MoveBase`: Navigate to a position
- `nav2_msgs/action/ComputePathToPose`: Plan a navigation path
- `example_interfaces/action/Fibonacci`: Example action for learning

## 4.3 Implementing Action Servers

### 4.3.1 Basic Action Server (Python)

```python
import time
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            callback_group=rclpy.callback_groups.ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def destroy(self):
        self._action_server.destroy()
        super().destroy_node()

    def goal_callback(self, goal_request):
        """Accept or reject a client request to begin an action."""
        self.get_logger().info('Received goal request')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject a client request to cancel an action."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the goal."""
        self.get_logger().info('Executing goal...')
        
        # Feedback and result messages
        feedback_msg = Fibonacci.Feedback()
        result_msg = Fibonacci.Result()
        
        # Initialize Fibonacci sequence
        feedback_msg.sequence = [0, 1]
        sequence = [0, 1]
        
        # Start executing the action
        for i in range(1, goal_handle.request.order):
            # Check if the goal has been canceled
            if goal_handle.is_cancel_requested:
                result_msg.sequence = sequence
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return result_msg

            # Update Fibonacci sequence
            sequence.append(sequence[i] + sequence[i-1])
            feedback_msg.sequence = sequence
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')
            
            # Sleep to simulate work
            time.sleep(1)
        
        # Check if goal was canceled during execution
        if goal_handle.is_cancel_requested:
            result_msg.sequence = sequence
            goal_handle.canceled()
            self.get_logger().info('Goal canceled')
            return result_msg
        
        # Set result and succeed
        result_msg.sequence = sequence
        goal_handle.succeed()
        self.get_logger().info('Returning result: {result_msg.sequence}')
        
        return result_msg

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    
    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        fibonacci_action_server.get_logger().info('Interrupted by user')
    finally:
        fibonacci_action_server.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 4.3.2 Complex Action Server Example

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import QoSProfile
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class JointTrajectoryActionServer(Node):
    def __init__(self):
        super().__init__('joint_trajectory_action_server')
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'joint_trajectory_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

    def goal_callback(self, goal_request):
        """Accept or reject goal based on trajectory validity."""
        # Validate the incoming trajectory
        trajectory = goal_request.trajectory
        
        if len(trajectory.joint_names) == 0:
            self.get_logger().warn('Goal rejected: No joint names specified')
            return GoalResponse.REJECT
        
        if len(trajectory.points) == 0:
            self.get_logger().warn('Goal rejected: No trajectory points specified')
            return GoalResponse.REJECT
        
        self.get_logger().info('Goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept all cancel requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the joint trajectory."""
        self.get_logger().info('Executing joint trajectory goal')
        
        # Get the trajectory from the goal
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        points = trajectory.points
        
        # Initialize feedback
        feedback_msg = FollowJointTrajectory.Feedback()
        feedback_msg.joint_names = joint_names
        feedback_msg.desired = JointTrajectoryPoint()
        feedback_msg.actual = JointTrajectoryPoint()
        feedback_msg.error = JointTrajectoryPoint()
        
        # Result message
        result_msg = FollowJointTrajectory.Result()
        result_msg.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        
        # Execute the trajectory point by point
        for i, point in enumerate(points):
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                result_msg.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                goal_handle.canceled()
                self.get_logger().info('Trajectory execution canceled')
                return result_msg
            
            # Set feedback
            feedback_msg.desired = point
            feedback_msg.actual.positions = [0.0] * len(joint_names)  # Simulated actual positions
            feedback_msg.error.positions = [0.0] * len(joint_names)    # Simulated error
            
            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            
            # Simulate moving to the trajectory point
            # In real implementation, this would control actual joints
            self.move_to_position(point, joint_names)
            
            self.get_logger().info(f'Completed trajectory point {i+1}/{len(points)}')
        
        # Check for final cancellation
        if goal_handle.is_cancel_requested:
            result_msg.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            goal_handle.canceled()
            return result_msg
        
        # Set result and succeed
        goal_handle.succeed()
        self.get_logger().info('Trajectory execution completed successfully')
        
        return result_msg

    def move_to_position(self, point, joint_names):
        """Simulate or execute movement to the specified position."""
        # In a real implementation, this would interface with the robot's
        # joint control system to move to the specified position
        import time
        time.sleep(0.5)  # Simulate movement time

def main(args=None):
    rclpy.init(args=args)
    action_server = JointTrajectoryActionServer()
    
    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        action_server.get_logger().info('Interrupted by user')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 4.4 Implementing Action Clients

### 4.4.1 Basic Action Client (Python)

```python
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        # Wait for the action server to be available
        self._action_client.wait_for_server()
        
        # Create the goal
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order
        
        # Send the goal and get a future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        
        # Get result future
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    
    # Send goal
    action_client.send_goal(10)
    
    # Spin to allow callbacks to be processed
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

### 4.4.2 Asynchronous Action Client with Preemption

```python
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class TrajectoryActionClient(Node):
    def __init__(self):
        super().__init__('trajectory_action_client')
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            'joint_trajectory_controller/follow_joint_trajectory')

    def send_trajectory(self, joint_names, positions_list, time_from_start_list):
        """Send a joint trajectory to the robot."""
        # Wait for the action server to be available
        self._action_client.wait_for_server()
        
        # Create the trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names
        
        # Add points to the trajectory
        for pos, time_from_start in zip(positions_list, time_from_start_list):
            point = JointTrajectoryPoint()
            point.positions = pos
            point.time_from_start = Duration(sec=time_from_start)
            trajectory.points.append(point)
        
        # Create the goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory
        
        # Send the goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Trajectory goal rejected')
            return

        self.get_logger().info('Trajectory goal accepted')
        self.goal_handle = goal_handle
        
        # Get result future
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Trajectory execution: {len(feedback.actual.positions)} joints active')

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Trajectory executed successfully')
        else:
            self.get_logger().info(f'Trajectory execution failed with error code: {result.error_code}')
        
        rclpy.shutdown()

    def cancel_goal(self):
        """Cancel the current trajectory goal."""
        if hasattr(self, 'goal_handle'):
            future = self.goal_handle.cancel_goal_async()
            future.add_done_callback(self.cancel_response_callback)

    def cancel_response_callback(self, future):
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().info('Goal failed to cancel')

def main(args=None):
    rclpy.init(args=args)
    action_client = TrajectoryActionClient()
    
    # Example trajectory: move 2 joints through 3 positions
    joint_names = ['joint1', 'joint2']
    positions = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]  # Start, mid, end
    times = [1, 2, 3]  # Execute at 1s, 2s, and 3s
    
    action_client.send_trajectory(joint_names, positions, times)
    
    # After 1.5 seconds, cancel the goal (for demonstration)
    def cancel_goal():
        time.sleep(1.5)
        action_client.cancel_goal()
    
    # Run cancel in a separate thread
    import threading
    cancel_thread = threading.Thread(target=cancel_goal)
    cancel_thread.start()
    
    # Spin to allow callbacks to be processed
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
```

## 4.5 Advanced Action Concepts

### 4.5.1 Action with Complex Feedback

For actions that need detailed feedback, create comprehensive feedback messages:

```python
# Custom action: ExecuteComplexTask.action
# Goal
string task_name
string[] parameters
---
# Feedback
string current_step
float64 progress_percentage
string status_message
int32 execution_count
---
# Result
bool success
string final_status
int32 error_code
float64[] final_values
```

### 4.5.2 Action Composition

Often, complex tasks require coordination of multiple actions:

```python
import asyncio
from rclpy.action import ActionClient
from rclpy.node import Node

class TaskCoordinator(Node):
    def __init__(self):
        super().__init__('task_coordinator')
        
        # Multiple action clients for different subtasks
        self.navigation_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manipulation_client = ActionClient(self, MoveGroup, 'move_group')
        self.perception_client = ActionClient(self, DetectObjects, 'detect_objects')

    async def execute_complex_task(self, target_pose, object_to_grab):
        """Execute a complex task involving navigation, perception, and manipulation."""
        # Step 1: Navigate to target
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = target_pose
        nav_future = await self.send_navigation_goal(nav_goal)
        
        if not nav_future.result().result.success:
            return False  # Navigation failed
        
        # Step 2: Detect objects
        detection_goal = DetectObjects.Goal()
        detection_goal.roi = self.get_roi_from_pose(target_pose)
        detection_future = await self.send_detection_goal(detection_goal)
        
        detected_objects = detection_future.result().result.objects
        target_object = self.find_target_object(detected_objects, object_to_grab)
        
        if not target_object:
            return False  # Object not found
        
        # Step 3: Manipulate object
        manipulation_goal = MoveGroup.Goal()
        manipulation_goal.target = self.calculate_grasp_pose(target_object)
        manipulation_future = await self.send_manipulation_goal(manipulation_goal)
        
        return manipulation_future.result().result.success
```

## 4.6 Action Design for Humanoid Robots

### 4.6.1 Walking Pattern Generation Action

```python
# GenerateWalkingPattern.action
# Goal
float64 step_length
float64 step_height  
float64 step_duration
int32 num_steps
bool is_turning
float64 turn_angle
---
# Feedback
int32 current_step
float64 progress_percentage
string status
bool is_balanced
---
# Result
bool success
string error_message
trajectory_msgs/JointTrajectory walking_trajectory
```

### 4.6.2 Whole-Body Motion Action

```python
# ExecuteWholeBodyMotion.action
# Goal
string motion_name
float64[] initial_positions
float64[] target_positions
float64 execution_time
bool avoid_obstacles
---
# Feedback
float64 progress_percentage
geometry_msgs/Pose current_pose
string status
bool is_stable
---
# Result
bool success
string error_message
string completion_status
```

### 4.6.3 Humanoid Action Server Implementation

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

class HumanoidMotionActionServer(Node):
    def __init__(self):
        super().__init__('humanoid_motion_action_server')
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'humanoid_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback)

    def goal_callback(self, goal_request):
        """Validate humanoid motion goals."""
        # Check joint limits and other constraints
        trajectory = goal_request.trajectory
        
        for point in trajectory.points:
            # Check each joint position against limits
            for i, pos in enumerate(point.positions):
                joint_name = trajectory.joint_names[i]
                # Implement joint limit checking
                if not self.is_within_joint_limits(joint_name, pos):
                    self.get_logger().warn(f'Joint {joint_name} position {pos} is out of limits')
                    return rclpy.action.GoalResponse.REJECT
        
        self.get_logger().info('Motion goal accepted')
        return rclpy.action.GoalResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute humanoid motion with balance control."""
        self.get_logger().info('Executing humanoid motion')
        
        # Initialize feedback
        feedback_msg = FollowJointTrajectory.Feedback()
        result_msg = FollowJointTrajectory.Result()
        
        # Get trajectory
        trajectory = goal_handle.request.trajectory
        joint_names = trajectory.joint_names
        
        # Initialize balance controller
        self.start_balance_control()
        
        try:
            # Execute trajectory point by point
            for i, point in enumerate(trajectory.points):
                # Check for cancellation
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                    goal_handle.canceled()
                    self.get_logger().info('Motion execution canceled')
                    return result_msg
                
                # Update feedback
                feedback_msg.desired = point
                feedback_msg.joint_names = joint_names
                
                # Publish feedback
                goal_handle.publish_feedback(feedback_msg)
                
                # Execute the motion while maintaining balance
                self.execute_motion_step(point, joint_names)
                
                # Check if robot is still balanced
                if not self.is_robot_balanced():
                    result_msg.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    goal_handle.abort()
                    self.get_logger().error('Robot lost balance during motion')
                    return result_msg
                
                self.get_logger().info(f'Completed motion step {i+1}/{len(trajectory.points)}')
        finally:
            # Stop balance control regardless of outcome
            self.stop_balance_control()
        
        # Check for final cancellation
        if goal_handle.is_cancel_requested:
            result_msg.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            goal_handle.canceled()
            return result_msg
        
        # Success
        result_msg.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        goal_handle.succeed()
        self.get_logger().info('Motion execution completed successfully')
        
        return result_msg

    def is_within_joint_limits(self, joint_name, position):
        """Check if joint position is within limits."""
        # Implementation depends on specific robot
        # Return True if within limits, False otherwise
        return True

    def start_balance_control(self):
        """Start balance control system."""
        # Implementation for humanoid balance control
        pass

    def stop_balance_control(self):
        """Stop balance control system."""
        # Implementation for humanoid balance control
        pass

    def is_robot_balanced(self):
        """Check if humanoid robot is currently balanced."""
        # Implementation for balance checking
        # Check ZMP, joint torques, etc.
        return True

    def execute_motion_step(self, point, joint_names):
        """Execute a single motion step."""
        # Send position commands to joints
        # Implementation depends on specific robot controller
        pass
```

## 4.7 Performance and Reliability Considerations

### 4.7.1 Resource Management

Actions should manage resources carefully to avoid blocking other operations:

```python
def execute_callback(self, goal_handle):
    # Acquire necessary resources
    resource_acquired = self.acquire_resources()
    if not resource_acquired:
        result_msg = MyAction.Result()
        result_msg.success = False
        result_msg.message = "Could not acquire required resources"
        goal_handle.abort()
        return result_msg
    
    try:
        # Execute the action
        result_msg = self.perform_action_with_resources(goal_handle)
        return result_msg
    finally:
        # Always release resources
        self.release_resources()
```

### 4.7.2 Error Recovery

Implement error recovery mechanisms for complex actions:

```python
def execute_callback(self, goal_handle):
    try:
        # Step 1: Initialize
        if not self.initialize_action():
            raise RuntimeError("Failed to initialize action")
        
        # Step 2: Execute main operation
        for step in self.get_execution_steps():
            if goal_handle.is_cancel_requested:
                return self.handle_cancel()
            
            try:
                self.execute_step(step)
            except StepExecutionError as e:
                # Try to recover from the error
                if self.can_recover_from_error(e):
                    self.recover_from_error(e)
                else:
                    raise e  # Cannot recover, re-raise
        
        # Step 3: Finalize
        return self.finalize_action()
    
    except Exception as e:
        # Handle any errors that occurred
        result_msg = MyAction.Result()
        result_msg.success = False
        result_msg.error_message = str(e)
        goal_handle.abort()
        return result_msg
```

## 4.8 Best Practices for Action Development

### 4.8.1 Action Interface Design

- Define clear, specific goals that can be broken into discrete steps
- Design feedback messages that provide meaningful progress information
- Create result messages that clearly indicate success or failure modes
- Use appropriate data types for all fields

### 4.8.2 Implementation Guidelines

- Always check for goal cancellation at appropriate points
- Provide regular feedback during long-running operations
- Implement proper error handling and recovery
- Use appropriate QoS settings for action communication
- Log important events for debugging and monitoring

### 4.8.3 Performance Considerations

- Minimize the size of feedback messages for high-frequency updates
- Use appropriate update rates for feedback
- Consider the computational complexity of your action execution
- Implement efficient algorithms for processing goals

## 4.9 Comparing Communication Patterns

| Feature | Topics | Services | Actions |
|---------|--------|----------|---------|
| Communication Type | Async | Sync | Async (with status) |
| Best For | Streaming data | One-shot queries | Long-running tasks |
| Blocking | No | Yes (sync) | No |
| Feedback | No | No | Yes |
| Preemption | N/A | N/A | Yes |
| Execution Time | N/A | Short | Long |
| Use Case Examples | Sensor data, robot state | Calibration, config | Navigation, manipulation |

## Chapter Summary

This chapter covered the fundamentals of action-based communication in ROS 2 for handling long-running tasks that require feedback and preemption capabilities. We explored implementation of action servers and clients, design patterns for humanoid robotics applications, and performance considerations. Actions represent a powerful communication pattern for complex robot operations that cannot be adequately handled by topics or services alone.

## Key Terms
- Action Server
- Action Client
- Goal-Feedback-Result Pattern
- Action Preemption
- Action Cancellation
- Long-Running Tasks
- Progress Feedback

## Exercises
1. Implement an action server for humanoid walking pattern generation
2. Create a complex action client that coordinates multiple robot subsystems
3. Design an action interface for whole-body motion planning
4. Implement error recovery mechanisms for action execution

## References
- ROS 2 Documentation: https://docs.ros.org/
- Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics.
- Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance.