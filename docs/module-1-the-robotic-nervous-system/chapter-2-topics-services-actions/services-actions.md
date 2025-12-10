---
id: module-1-chapter-2-services-actions
title: 'Module 1 — The Robotic Nervous System | Chapter 2 — Services and Actions'
sidebar_label: 'Chapter 2 — Services and Actions'
---

# Chapter 2 — Services and Actions

## Understanding Request-Response Communication in ROS 2

Besides the publish-subscribe pattern, ROS 2 provides synchronous and asynchronous request-response communication patterns through Services and Actions.

### Services

Services provide a synchronous request-response communication pattern between nodes. This is suitable for tasks that have a relatively quick response time.

#### Service Architecture

- **Service Server**: A node that provides a service
- **Service Client**: A node that uses a service
- **Service Request**: The data sent from client to server
- **Service Response**: The data sent from server to client

#### Creating Services

```cpp
// Example service interface definition (in srv directory)
// AddTwoInts.srv
# Request
int64 a
int64 b
---
# Response
int64 sum
```

```cpp
// C++ Example: Service Server
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class ServiceServer : public rclcpp::Node
{
public:
    ServiceServer() : Node("service_server")
    {
        service_ = this->create_service<example_interfaces::srv::AddTwoInts>(
            "add_two_ints",
            [this](const std::shared_ptr<rmw_request_id_t> request_header,
                   const std::shared_ptr<example_interfaces::srv::AddTwoInts::Request> request,
                   const std::shared_ptr<example_interfaces::srv::AddTwoInts::Response> response)
            {
                (void)request_header;
                response->sum = request->a + request->b;
                RCLCPP_INFO(this->get_logger(), "Incoming request: %ld + %ld = %ld",
                           request->a, request->b, response->sum);
            });
    }

private:
    rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};
```

```python
# Python Example: Service Server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class ServiceServer(Node):
    def __init__(self):
        super().__init__('service_server')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request: {request.a} + {request.b} = {response.sum}')
        return response
```

#### Creating Service Clients

```cpp
// C++ Example: Service Client
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class ServiceClient : public rclcpp::Node
{
public:
    ServiceClient() : Node("service_client")
    {
        client_ = this->create_client<example_interfaces::srv::AddTwoInts>("add_two_ints");
        
        while (!client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
        }
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&ServiceClient::send_request, this));
    }

private:
    void send_request()
    {
        auto request = std::make_shared<example_interfaces::srv::AddTwoInts::Request>();
        request->a = 2;
        request->b = 3;
        
        auto result_future = client_->async_send_request(request);
        // Handle the result when available
    }
    
    rclcpp::Client<example_interfaces::srv::AddTwoInts>::SharedPtr client_;
    rclcpp::TimerBase::SharedPtr timer_;
};
```

### Actions

Actions are designed for long-running tasks that require feedback during execution. They are ideal for navigation, manipulation, or other tasks that take a significant amount of time.

#### Action Architecture

- **Goal**: The request for the action to perform
- **Feedback**: Information sent periodically during execution
- **Result**: The final outcome of the action

#### Action States

Actions can be in one of several states:
- **PENDING**: Goal accepted but not yet executed
- **ACTIVE**: Currently executing
- **SUCCEEDED**: Execution completed successfully
- **ABORTED**: Execution failed
- **CANCELED**: Execution was canceled

#### Creating Actions

```cpp
// Example action interface definition (in action directory)
// Fibonacci.action
# Goal
int32 order
---
# Result
int32[] sequence
---
# Feedback
int32[] sequence
```

```cpp
// C++ Example: Action Server
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "example_interfaces/action/fibonacci.hpp"

class ActionServer : public rclcpp::Node
{
public:
    using Fibonacci = example_interfaces::action::Fibonacci;
    using GoalHandleFibonacci = rclcpp_action::ServerGoalHandle<Fibonacci>;

    ActionServer() : Node("action_server")
    {
        using namespace std::placeholders;
        
        action_server_ = rclcpp_action::create_server<Fibonacci>(
            this->get_node_base_interface(),
            this->get_node_clock_interface(),
            this->get_node_logging_interface(),
            this->get_node_waitables_interface(),
            "fibonacci",
            std::bind(&ActionServer::handle_goal, this, _1, _2),
            std::bind(&ActionServer::handle_cancel, this, _1),
            std::bind(&ActionServer::handle_accepted, this, _1));
    }

private:
    rclcpp_action::Server<Fibonacci>::SharedPtr action_server_;

    rclcpp_action::GoalResponse handle_goal(
        const rclcpp_action::GoalUUID & uuid,
        std::shared_ptr<const Fibonacci::Goal> goal)
    {
        RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->order);
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(
        const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    void handle_accepted(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        using namespace std::placeholders;
        // This needs to return quickly, so spin up a new thread
        std::thread{std::bind(&ActionServer::execute, this, _1), goal_handle}.detach();
    }

    void execute(const std::shared_ptr<GoalHandleFibonacci> goal_handle)
    {
        RCLCPP_INFO(this->get_logger(), "Executing goal");
        
        // Send feedback
        auto feedback = std::make_shared<Fibonacci::Feedback>();
        auto result = std::make_shared<Fibonacci::Result>();
        
        // Simulate execution
        auto sequence = std::vector<int32_t>{0, 1};
        auto order = goal_handle->get_goal()->order;
        
        for (int i = 1; i < order; ++i) {
            sequence.push_back(sequence[i] + sequence[i - 1]);
            
            // Send feedback
            feedback->sequence = sequence;
            goal_handle->publish_feedback(feedback);
            
            if (goal_handle->is_canceling()) {
                result->sequence = sequence;
                goal_handle->canceled(result);
                RCLCPP_INFO(this->get_logger(), "Goal canceled");
                return;
            }
            
            rclcpp::sleep_for(std::chrono::milliseconds(500));
        }
        
        result->sequence = sequence;
        goal_handle->succeed(result);
        RCLCPP_INFO(this->get_logger(), "Goal succeeded");
    }
};
```

### When to Use Each Pattern

#### Use Topics when:
- Communication is asynchronous
- Multiple publishers/subscribers are needed
- Data needs to be broadcast continuously
- Real-time performance is critical

#### Use Services when:
- Request-response pattern is needed
- Task completes quickly (under a few seconds)
- Synchronous communication is desired

#### Use Actions when:
- Task takes a long time to complete
- Feedback during execution is required
- Goal can be canceled during execution
- Task status needs to be monitored

Understanding these communication patterns is essential for designing efficient and responsive robotic systems.