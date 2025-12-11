---
id: module-4-chapter-3-quiz
title: 'Module 4 — Vision-Language-Action Systems | Chapter 3 — Quiz'
sidebar_label: 'Chapter 3 — Quiz'
sidebar_position: 6
---

# Chapter 3 — Quiz

## Deep Reinforcement Learning for Humanoid Control

### Instructions
- This quiz evaluates your understanding of deep reinforcement learning applications in humanoid robotics
- Choose the best answer for multiple-choice questions
- For short-answer questions, provide concise but complete responses
- Time limit: 45 minutes

---

### Section A: Multiple Choice Questions (5 points each)

**Question 1:** What is the primary advantage of using Actor-Critic methods in humanoid robotics control compared to value-based methods?

A) Simpler implementation and faster convergence
B) Direct policy optimization with value estimation for variance reduction
C) Guaranteed convergence to global optimum
D) Reduced computational complexity

**Question 2:** In Vision-Language-Action (VLA) systems, what is the main purpose of the attention mechanism?

A) To reduce the dimensionality of sensory data
B) To enable selective focus on relevant sensory inputs based on language instructions
C) To increase the speed of neural network computations
D) To simplify the network architecture

**Question 3:** What is the key benefit of Proximal Policy Optimization (PPO) over vanilla policy gradient methods?

A) Faster convergence with guaranteed optimality
B) Stable updates that prevent large policy changes and catastrophic forgetting
C) Elimination of the need for replay buffers
D) Reduction in the need for hyperparameter tuning

**Question 4:** Which technique is most effective for addressing the exploration challenge in continuous control tasks for humanoid robots?

A) Epsilon-greedy action selection
B) Entropy regularization or noise injection in action selection
C) Fixed action sequences
D) Supervised learning with demonstrations

**Question 5:** What distinguishes hierarchical reinforcement learning from standard RL in humanoid robotics applications?

A) Use of multiple reward functions
B) Decomposition of complex tasks into sub-tasks with different time scales
C) Increased network depth
D) Combination of multiple robot embodiments

---

### Section B: Short Answer Questions (10 points each)

**Question 6:** Explain the concept of "sim-to-real transfer" in the context of deep reinforcement learning for humanoid robots. What are the main challenges and how can domain randomization help address them?

<details>
<summary>Answer Guidance</summary>
The answer should cover: sim-to-real transfer definition, challenges like reality gap and dynamics mismatch, domain randomization as a solution, examples of randomization parameters, and validation methods.
</details>

**Question 7:** Describe the role of the reward function in training humanoid locomotion policies using deep RL. What are the important components that should be included in a reward function for bipedal walking?

<details>
<summary>Answer Guidance</summary>
Answer should include: reward function definition and importance, components like forward velocity, energy efficiency, stability, joint limits, contact consistency for walking, and how each component contributes to the learning objective.
</details>

**Question 8:** Compare and contrast model-based and model-free deep RL approaches for humanoid control. What are the advantages and disadvantages of each approach for complex humanoid tasks?

<details>
<summary>Answer Guidance</summary>
Answer should cover: definition of each approach, model-free advantages (no modeling required, handles complex dynamics), model-based advantages (sample efficiency, planning), disadvantages for each, and when to use each approach.
</details>

---

### Section C: Implementation Questions (20 points each)

**Question 9:** You are implementing a Vision-Language-Action system for a humanoid robot that needs to follow natural language instructions while navigating through a cluttered environment. Design the architecture and explain how you would handle the following challenges:

1. Fusing visual and linguistic inputs into a unified representation
2. Generating appropriate motor actions based on the fused representation
3. Ensuring robustness to variations in language formulations
4. Maintaining computational efficiency for real-time performance

Provide the neural network architecture and explain the key design choices.

<details>
<summary>Answer Guidance</summary>
Answer should include: multimodal fusion architecture (cross-attention, transformers), vision encoder (CNN/ViT), language encoder (BERT/GPT), fusion layers, action generation network, attention mechanisms between vision-language-action, and efficiency considerations (model compression, pruning).
</details>

```python
# Example architecture for Question 9
class VisionLanguageActionNetwork(nn.Module):
    def __init__(self, vision_encoder, language_encoder, action_dim, hidden_dim=512):
        super().__init__()
        
        self.vision_encoder = vision_encoder  # e.g., ResNet or ViT
        self.language_encoder = language_encoder  # e.g., BERT
        self.hidden_dim = hidden_dim
        
        # Cross-modal attention for fusion
        self.vision_language_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Action generation
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for concatenated features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # For humanoid control, might also have:
        self.value_head = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, vision_input, language_input, language_mask=None):
        # Encode vision and language separately
        vision_features = self.vision_encoder(vision_input)  # [B, patches, feat_dim]
        language_features = self.language_encoder(language_input, attention_mask=language_mask)  # [B, seq_len, feat_dim]
        
        # Cross-attention for fusion
        # Language attends to vision features
        attended_vision, _ = self.vision_language_attention(
            query=language_features,  # Query from language
            key=vision_features,      # Key from vision
            value=vision_features     # Value from vision
        )
        
        # Global pooling to get single representations
        vision_global = torch.mean(vision_features, dim=1)  # [B, feat_dim]
        language_global = torch.mean(attended_vision, dim=1)  # [B, feat_dim]
        
        # Concatenate and generate action
        combined_features = torch.cat([vision_global, language_global], dim=-1)
        
        action = self.action_head(combined_features)
        value = self.value_head(combined_features)
        
        return action, value
```

**Question 10:** You need to train a deep RL policy for humanoid manipulation tasks using only sparse rewards (success/failure). Design a training approach that addresses the following issues:

1. The sparse reward problem
2. Sample inefficiency in complex manipulation
3. Safety constraints during learning
4. Transfer from simulation to reality

Include specific techniques and explain how they work together.

<details>
<summary>Answer Guidance</summary>
Answer should cover: Hindsight Experience Replay (HER) for sparse rewards, curriculum learning, imitation learning, domain randomization, safety constraints (shaping, shielding), and how these components work together.
</details>

```python
# Example implementation for Question 10
class SparseRewardTrainingApproach:
    def __init__(self):
        # Use HER for sparse reward problems
        self.her_strategy = HindsightExperienceReplay()
        
        # Implement curriculum learning
        self.curriculum = CurriculumLearning()
        
        # Use demonstrations for initialization
        self.imitation_learning = BehavioralCloning()
        
        # Domain randomization for sim-to-real
        self.domain_randomizer = DomainRandomizer()
        
        # Safety constraints
        self.safety_layer = SafetyLayer()
    
    def train_policy(self, env, num_episodes=1000):
        """Training approach for sparse reward humanoid tasks"""
        
        # Phase 1: Initialize with demonstrations
        self.imitation_learning.pre_train_policy_with_demos()
        
        for episode in range(num_episodes):
            # Apply domain randomization each episode
            self.domain_randomizer.randomize_environment()
            
            # Apply curriculum to adjust task difficulty
            current_task_level = self.curriculum.get_current_level()
            
            # Collect trajectory
            trajectory = []
            state, _ = env.reset()
            
            for step in range(env.max_steps):
                # Apply safety constraints to actions
                raw_action = self.policy(state)
                safe_action = self.safety_layer.apply_constraints(raw_action, state)
                
                next_state, reward, done, truncated, info = env.step(safe_action)
                
                trajectory.append({
                    'state': state,
                    'action': safe_action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'info': info
                })
                
                state = next_state
                
                if done or truncated:
                    break
            
            # Apply HER to relabel goals in the trajectory
            her_trajectories = self.her_strategy.relabel_trajectory(trajectory)
            
            # Update policy with HER trajectories
            self.update_policy_with_her(her_trajectories)
            
            # Update curriculum based on performance
            self.curriculum.update_performance(episode, success=info.get('success', False))
```

### Section D: Scenario Analysis (25 points)

**Question 11:** A humanoid robot needs to learn to perform complex tasks like "Go to the kitchen, pick up the red mug, and place it on the table" using deep RL. Analyze the following challenges and propose solutions:

1. **Long-horizon task decomposition**: How would you break down this task into learnable components?
2. **Multimodal integration**: How would you handle the integration of vision, language, and proprioception?
3. **Credit assignment**: How would you assign credit for success/failure across the long sequence?
4. **Generalization**: How would you ensure the robot can perform similar tasks with different objects or locations?

Provide specific algorithmic approaches and implementation details for each challenge.

<details>
<summary>Answer Guidance</summary>
Answer should include: hierarchical RL with manager-worker decomposition, multimodal transformers with attention mechanisms, intrinsic motivation/reward shaping, generalization through domain randomization and data augmentation, and specific algorithms like HIRO, Feudal Networks, or DIAYN.
</details>

```python
# Complete system for Question 11
class HierarchicalVLA:
    def __init__(self):
        # High-level manager (handles language and task decomposition)
        self.manager = LanguageUnderstandingManager()
        
        # Low-level worker (handles vision, proprioception, and control)
        self.worker = VisionProprioControlWorker()
        
        # Goal conditioner (connects high-level goals to low-level actions)
        self.goal_conditioner = GoalConditioner()
        
        # Intrinsic reward generator (for credit assignment)
        self.intrinsic_reward_gen = IntrinsicRewardGenerator()
        
        # Generalization enhancer (for transfer learning)
        self.generalizer = DomainRandomizationModule()
    
    def execute_long_horizon_task(self, instruction):
        """Execute a long-horizon task like 'Go to kitchen, pick up red mug, place on table'"""
        
        # Phase 1: Task decomposition by manager
        task_plan = self.manager.decompose_task(instruction)
        # Output: [{'subtask': 'navigate_to_location', 'location': 'kitchen'}, 
        #          {'subtask': 'find_object', 'object': 'red mug'}, 
        #          {'subtask': 'grasp_object', 'object': 'red mug'},
        #          {'subtask': 'navigate_to_location', 'location': 'table'},
        #          {'subtask': 'place_object', 'location': 'table'}]
        
        success = True
        for subtask in task_plan:
            # Apply domain generalization for each subtask
            self.generalizer.randomize_domain_for_subtask(subtask)
            
            # Get goal for worker from manager
            goal = self.goal_conditioner.convert_to_low_level_goal(subtask)
            
            # Execute subtask with worker
            subtask_success = self.worker.execute_goal(goal, subtask)
            
            if not subtask_success:
                success = False
                break
        
        return success
```

---

### Section E: Critical Analysis (15 points)

**Question 12:** Discuss the limitations of current deep RL approaches for humanoid robotics and identify potential future directions for research. Consider aspects like sample efficiency, safety, generalization, and real-world deployment.

<details>
<summary>Answer Guidance</summary>
Answer should cover: current limitations (sample inefficiency, safety, sim-to-real gap, stability), potential solutions (meta-learning, learning from humans, better simulators), and future directions (embodied intelligence, lifelong learning, human-robot collaboration).
</details>

---

### Answer Key

#### Section A Answers:
1. B - Actor-Critic methods combine direct policy optimization with value estimation, which reduces variance
2. B - Attention mechanisms allow the system to focus on relevant visual regions based on language instructions
3. B - PPO uses a clipped objective to prevent excessively large policy updates
4. B - Entropy regularization adds noise to encourage exploration in continuous action spaces
5. B - Hierarchical RL breaks complex tasks into temporally-extended subtasks

#### Section B Sample Answers:

**Question 6:** Sim-to-real transfer refers to the challenge of deploying policies trained in simulation to real robots. The main challenges include:
- **Reality gap**: Differences in dynamics, actuator response, sensor noise
- **Dynamics mismatch**: Simulation may not perfectly model real physics
- **Visual perception differences**: Lighting, textures, visual artifacts

Domain randomization addresses these by training policies across diverse simulation conditions, helping the policy learn robust features that generalize across domains. Parameters like friction, mass, visual textures, lighting conditions, and sensor noise can be randomized.

**Question 7:** The reward function shapes the learning objective by providing feedback on action quality. For humanoid locomotion, key components include:
- **Forward velocity**: Encourages forward motion
- **Energy efficiency**: Penalizes excessive joint torques
- **Stability**: Rewards torso upright posture
- **Smoothness**: Penalizes jerky movements
- **Foot contact consistency**: Rewards proper walking gaits
- **Joint limit penalties**: Prevents dangerous configurations

**Question 8:** 
- **Model-free RL**: Learns directly from interaction; advantages include handling complex unknown dynamics, disadvantages include sample inefficiency
- **Model-based RL**: Learns environment dynamics model; advantages include sample efficiency and planning, disadvantages include model errors that can cause instability

Model-free is better for complex environments with unknown dynamics, while model-based is better when sample efficiency is crucial and dynamics can be reasonably modeled.

This quiz assesses your understanding of deep reinforcement learning techniques applied to humanoid robotics, including both theoretical concepts and practical implementation considerations.