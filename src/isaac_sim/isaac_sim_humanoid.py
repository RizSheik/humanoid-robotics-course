# Isaac Sim Humanoid Robot Script

import omni
from omni.isaac.core import World
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import carb

class IsaacSimHumanoid:
    """
    A Python class that interfaces with NVIDIA Isaac Sim to control a humanoid robot
    and generate synthetic data for AI model training.
    """
    
    def __init__(self):
        # Initialize the Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.humanoid = None
        self.assets_root_path = get_assets_root_path()
        
        # Set up the simulation environment
        self.setup_environment()
        
    def setup_environment(self):
        """Set up the simulation environment with a humanoid robot and scene"""
        # Add a ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add a simple humanoid robot (in a real scenario, you'd load a specific URDF or USD file)
        # For this example, we'll use a simple robot from the NVIDIA robotics library
        asset_path = self.assets_root_path + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
        
        # For humanoid, we might need to load a different asset
        # Using a placeholder - in real implementation would use actual humanoid model
        add_reference_to_stage(
            usd_path=asset_path,
            prim_path="/World/Robot"
        )
        
        # Get the robot as an Articulation object
        self.humanoid = self.world.scene.add(Articulation(prim_path="/World/Robot", name="humanoid_robot"))
        
        # Set initial camera view
        set_camera_view(eye=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.5])
        
        print("Isaac Sim environment set up with humanoid robot")
    
    def reset_robot_position(self):
        """Reset the robot to a default position"""
        self.humanoid.set_world_pose(position=np.array([0.0, 0.0, 0.5]))
        self.humanoid.set_velocities(np.zeros(self.humanoid.num_dof))
        
    def execute_action(self, joint_positions):
        """Execute an action by setting joint positions"""
        # Set the joint positions for the humanoid
        self.humanoid.set_joint_positions(joint_positions)
        
    def get_robot_state(self):
        """Get the current state of the robot"""
        positions = self.humanoid.get_joint_positions()
        velocities = self.humanoid.get_joint_velocities()
        pose, orientation = self.humanoid.get_world_pose()
        
        state = {
            'joint_positions': positions,
            'joint_velocities': velocities,
            'pose': pose,
            'orientation': orientation
        }
        
        return state
    
    def generate_synthetic_data(self, num_samples=100):
        """Generate synthetic sensor data for AI training"""
        print(f"Generating {num_samples} samples of synthetic sensor data...")
        
        synthetic_data = []
        
        for i in range(num_samples):
            # Simulate random movement
            random_positions = np.random.uniform(-0.5, 0.5, size=self.humanoid.num_dof)
            self.execute_action(random_positions)
            
            # Step the simulation
            self.world.step(render=True)
            
            # Get the resulting robot state
            state = self.get_robot_state()
            
            # Collect synthetic data (in a real implementation, this would include camera, LiDAR, etc.)
            sample = {
                'joint_commands': random_positions,
                'resulting_state': state,
                'timestamp': i
            }
            
            synthetic_data.append(sample)
            
            # Print progress
            if i % 20 == 0:
                print(f"Generated {i}/{num_samples} samples")
        
        print(f"Completed generating {num_samples} synthetic data samples")
        return synthetic_data
    
    def start_simulation(self):
        """Start the simulation and run the main loop"""
        print("Starting Isaac Sim simulation...")
        
        # Reset the robot to initial position
        self.reset_robot_position()
        
        # Play the simulation
        self.world.play()
        
        try:
            # Run for a number of steps
            for i in range(500):  # Run for 500 steps
                # Generate some synthetic data periodically
                if i % 100 == 0:
                    self.generate_synthetic_data(10)
                
                # Simple control loop - in real implementation, this would run control algorithms
                current_positions = self.humanoid.get_joint_positions()
                
                # Apply some simple control (e.g., move joints in a pattern)
                target_positions = current_positions + 0.01 * np.sin(i / 50.0)
                self.execute_action(target_positions)
                
                # Step the physics
                self.world.step(render=True)
                
                # Print status periodically
                if i % 100 == 0:
                    state = self.get_robot_state()
                    print(f"Step {i}: Robot position: {state['pose']}")
        
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        
        finally:
            self.world.stop()
            print("Simulation stopped")

def main():
    """Main function to run the Isaac Sim humanoid example"""
    print("Initializing Isaac Sim Humanoid Robot Control")
    
    # Initialize the simulation
    isaac_humanoid = IsaacSimHumanoid()
    
    # Run the simulation
    isaac_humanoid.start_simulation()
    
    print("Isaac Sim Humanoid example completed")

if __name__ == "__main__":
    main()