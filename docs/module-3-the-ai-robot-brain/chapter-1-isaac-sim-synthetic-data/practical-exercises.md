---
id: module-3-chapter-1-practical-exercises
title: 'Module 3 — The AI-Robot Brain | Chapter 1 — Practical Exercises'
sidebar_label: 'Chapter 1 — Practical Exercises'
---

# Chapter 1 — Practical Exercises

## Isaac Sim and Synthetic Data Generation: Hands-On Implementation

This practical lab focuses on implementing synthetic data generation pipelines using Isaac Sim. You'll learn to create, configure, and validate synthetic data generation systems for training AI models in robotics.

### Exercise 1: Basic Isaac Sim Environment Setup

#### Objective
Create a basic Isaac Sim environment and verify the installation and basic functionality.

#### Steps
1. Create a new Python script to initialize Isaac Sim
2. Set up a simple scene with objects and a camera
3. Capture basic RGB data from the camera
4. Verify the simulation runs correctly

```python
# basic_isaac_setup.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import cv2
import os

class BasicIsaacSetup:
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)
        self.assets_root_path = get_assets_root_path()
        
        # Create output directory
        self.output_dir = "isaac_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.camera = None
        self.setup_scene()
    
    def setup_scene(self):
        """Setup basic scene with ground plane and objects"""
        print("Setting up scene...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Set camera view
        set_camera_view(eye=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.5])
        
        # Add objects for data generation
        self.add_objects()
        
        # Add camera
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Camera",
                frequency=30,
                resolution=(640, 480)
            )
        )
        
        # Position camera
        self.camera.set_world_pose(
            position=np.array([1.5, 0.0, 1.0]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707])  # Looking down at 45 degrees
        )
        
        print("Scene setup complete")
    
    def add_objects(self):
        """Add basic objects to the scene"""
        # In a real setup, we would load specific objects
        # For this example, we'll just add them conceptually
        print("Added objects to scene")
    
    def capture_single_image(self):
        """Capture a single RGB image from the camera"""
        print("Capturing single image...")
        
        # Play the simulation
        self.world.play()
        
        # Step once to render
        self.world.step(render=True)
        
        # Capture RGB image
        rgb_image = self.camera.get_rgb()
        
        # Save the image
        image_path = os.path.join(self.output_dir, "test_image.png")
        # Convert from Isaac format to saveable format
        # Isaac Sim returns image in [height, width, channels] format
        if rgb_image is not None:
            # Convert from 0-1 float to 0-255 uint8
            img_uint8 = (rgb_image * 255).astype(np.uint8)
            cv2.imwrite(image_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
            print(f"Image saved to {image_path}")
        else:
            print("Failed to capture image - image is None")
        
        # Stop simulation
        self.world.stop()
        return rgb_image

def main():
    # Verify Isaac Sim is available
    try:
        app = BasicIsaacSetup()
        image = app.capture_single_image()
        print("Basic Isaac Sim setup completed successfully!")
        print(f"Captured image shape: {image.shape if image is not None else 'None'}")
    except Exception as e:
        print(f"Error in basic setup: {e}")

if __name__ == "__main__":
    main()
```

### Exercise 2: Domain Randomization Implementation

#### Objective
Implement domain randomization techniques to create diverse training data.

#### Steps
1. Create a domain randomizer class
2. Randomize lighting conditions
3. Randomize object appearances and positions
4. Validate diversity of generated data

```python
# domain_randomization.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.sensor import Camera
from pxr import UsdLux, Gf, Sdf
import numpy as np
import random
import os
import cv2

class DomainRandomizer:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.assets_root_path = get_assets_root_path()
        self.camera = None
        self.randomization_params = {
            'lighting': {
                'intensity_range': (0.5, 2.0),
                'position_range': ((-2, -2, 3), (2, 2, 5)),
                'color_temperature_range': (3000, 8000)
            },
            'objects': {
                'position_range': ((-1.5, -1.5, 0.1), (1.5, 1.5, 2.0)),
                'rotation_range': (0, 3.14159 * 2),  # 0 to 2π
                'colors': [
                    (1.0, 0.0, 0.0),    # Red
                    (0.0, 1.0, 0.0),    # Green
                    (0.0, 0.0, 1.0),    # Blue
                    (1.0, 1.0, 0.0),    # Yellow
                    (1.0, 0.0, 1.0),    # Magenta
                    (0.0, 1.0, 1.0),    # Cyan
                    (0.5, 0.5, 0.5),    # Gray
                    (1.0, 0.5, 0.0)     # Orange
                ]
            }
        }
        
        self.light_prim = None
        self.objects = []
        self.setup_scene()
    
    def setup_scene(self):
        """Setup scene with ground plane and basic objects"""
        print("Setting up domain randomization scene...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add a directional light source
        stage = omni.usd.get_context().get_stage()
        light_pos = Gf.Vec3f(0, 0, 5)
        self.light_prim = UsdLux.DistantLight.Define(stage, "/World/MyLight")
        self.light_prim.CreateIntensityAttr(600)
        self.light_prim.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        
        # Add camera
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Camera",
                frequency=30,
                resolution=(640, 480)
            )
        )
        
        self.camera.set_world_pose(
            position=np.array([2.0, 0.0, 1.5]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707])
        )
        
        # Add multiple objects that will be randomized
        self.add_objects()
    
    def add_objects(self):
        """Add objects to the scene"""
        # For this exercise, we'll conceptually track objects that will be randomized
        # In a real implementation, these would be USD objects added to the stage
        object_names = ["obj1", "obj2", "obj3", "obj4", "obj5"]
        for name in object_names:
            self.objects.append({
                'name': name,
                'position': np.array([0.0, 0.0, 0.5]),
                'color': (0.5, 0.5, 0.5),
                'rotation': 0.0
            })
    
    def randomize_lighting(self):
        """Randomize lighting conditions"""
        # Randomize light intensity
        intensity = random.uniform(
            self.randomization_params['lighting']['intensity_range'][0],
            self.randomization_params['lighting']['intensity_range'][1]
        )
        self.light_prim.GetIntensityAttr().Set(intensity * 600)  # Scale factor
        
        # Randomize light position
        pos_min = self.randomization_params['lighting']['position_range'][0]
        pos_max = self.randomization_params['lighting']['position_range'][1]
        pos = [
            random.uniform(pos_min[0], pos_max[0]),
            random.uniform(pos_min[1], pos_max[1]),
            random.uniform(pos_min[2], pos_max[2])
        ]
        # In a full implementation, we would update the light position
        
        # Randomize color temperature
        color_temp = random.uniform(
            self.randomization_params['lighting']['color_temperature_range'][0],
            self.randomization_params['lighting']['color_temperature_range'][1]
        )
        # Convert color temp to RGB (simplified)
        rgb_color = self.color_temperature_to_rgb(color_temp)
        self.light_prim.GetColorAttr().Set(Gf.Vec3f(*rgb_color))
    
    def color_temperature_to_rgb(self, color_temp):
        """Convert color temperature to RGB (simplified approximation)"""
        temp = color_temp / 100
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * np.log(green) - 161.1195681661
        else:
            red = temp - 60
            red = 329.698727446 * (red ** -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * (green ** -0.0755148492)
        
        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        
        # Clamp values to [0, 255] and normalize to [0, 1]
        red = np.clip(red, 0, 255) / 255.0
        green = np.clip(green, 0, 255) / 255.0
        blue = np.clip(blue, 0, 255) / 255.0
        
        return (red, green, blue)
    
    def randomize_objects(self):
        """Randomize object positions, colors, and rotations"""
        for obj in self.objects:
            # Randomize position
            pos_min = self.randomization_params['objects']['position_range'][0]
            pos_max = self.randomization_params['objects']['position_range'][1]
            obj['position'] = np.array([
                random.uniform(pos_min[0], pos_max[0]),
                random.uniform(pos_min[1], pos_max[1]),
                random.uniform(pos_min[2], pos_max[2])
            ])
            
            # Randomize color
            obj['color'] = random.choice(self.randomization_params['objects']['colors'])
            
            # Randomize rotation
            obj['rotation'] = random.uniform(0, self.randomization_params['objects']['rotation_range'])
            
            # In a real implementation, we would update the USD prims with these values
    
    def capture_randomized_data(self, num_samples=10):
        """Capture data with randomization applied"""
        print(f"Capturing {num_samples} randomized samples...")
        
        self.world.play()
        
        output_dir = "domain_randomization_output"
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(num_samples):
            # Apply randomization
            self.randomize_lighting()
            self.randomize_objects()
            
            # Step simulation to apply changes
            self.world.step(render=True)
            
            # Capture image
            rgb_image = self.camera.get_rgb()
            
            # Save image with randomization info
            if rgb_image is not None:
                img_uint8 = (rgb_image * 255).astype(np.uint8)
                img_path = os.path.join(output_dir, f"randomized_{i:03d}.png")
                cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                
                # Save randomization parameters
                params_path = os.path.join(output_dir, f"randomized_{i:03d}_params.txt")
                with open(params_path, 'w') as f:
                    f.write(f"Sample {i} - Randomization parameters:\n")
                    f.write(f"Light intensity: {self.light_prim.GetIntensityAttr().Get()}\n")
                    f.write(f"Light color: {self.light_prim.GetColorAttr().Get()}\n")
                    f.write(f"Object positions: {[obj['position'] for obj in self.objects]}\n")
                    f.write(f"Object colors: {[obj['color'] for obj in self.objects]}\n")
                
                print(f"Saved sample {i} with randomization")
        
        self.world.stop()
        print("Domain randomization data capture completed!")

def main():
    try:
        randomizer = DomainRandomizer()
        randomizer.capture_randomized_data(num_samples=5)
        print("Domain randomization exercise completed successfully!")
    except Exception as e:
        print(f"Error in domain randomization: {e}")

if __name__ == "__main__":
    main()
```

### Exercise 3: Multi-Modal Data Generation

#### Objective
Generate multi-modal data including RGB, depth, and segmentation information.

#### Steps
1. Set up multiple sensors (camera, depth sensor, segmentation)
2. Capture synchronized multi-modal data
3. Validate the correlation between modalities
4. Save data in a structured format

```python
# multi_modal_generation.py
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.viewports import set_camera_view
import numpy as np
import cv2
import os
import json

class MultiModalGenerator:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.setup_scene()
    
    def setup_scene(self):
        """Setup scene with objects for multi-modal capture"""
        print("Setting up multi-modal scene...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add objects of different colors and shapes
        self.add_multimodal_objects()
        
        # Add camera with multiple outputs
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Camera",
                frequency=30,
                resolution=(640, 480)
            )
        )
        
        # Position camera
        self.camera.set_world_pose(
            position=np.array([1.5, 0.0, 1.0]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707])
        )
        
        print("Multi-modal scene setup complete")
    
    def add_multimodal_objects(self):
        """Add objects that will have distinctive signatures in all modalities"""
        # In a real implementation, we would add USD objects
        # with different materials and properties that affect
        # RGB, depth, and segmentation differently
        pass
    
    def capture_multimodal_sample(self):
        """Capture a single multi-modal sample"""
        # Step the simulation
        self.world.step(render=True)
        
        # Capture different modalities
        rgb = self.camera.get_rgb()
        depth = self.camera.get_depth()
        # segmentation = self.camera.get_semantic_segmentation()  # If available
        
        return {
            'rgb': rgb,
            'depth': depth,
            # 'segmentation': segmentation,
            'timestamp': self.world.current_time_step_index
        }
    
    def capture_multimodal_dataset(self, num_samples=20):
        """Capture a multi-modal dataset"""
        print(f"Capturing {num_samples} multi-modal samples...")
        
        output_dir = "multi_modal_output"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
        # os.makedirs(os.path.join(output_dir, "segmentation"), exist_ok=True)
        
        self.world.play()
        
        samples = []
        
        for i in range(num_samples):
            # Capture multi-modal data
            sample = self.capture_multimodal_sample()
            
            # Save RGB image
            if sample['rgb'] is not None:
                rgb_uint8 = (sample['rgb'] * 255).astype(np.uint8)
                rgb_path = os.path.join(output_dir, "rgb", f"rgb_{i:04d}.png")
                cv2.imwrite(rgb_path, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
            
            # Save depth map
            if sample['depth'] is not None:
                # Normalize depth for visualization (0-1 range to 0-255)
                depth_normalized = (sample['depth'] - np.min(sample['depth'])) / (np.max(sample['depth']) - np.min(sample['depth']))
                depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                depth_path = os.path.join(output_dir, "depth", f"depth_{i:04d}.png")
                cv2.imwrite(depth_path, depth_uint8)
            
            # Save sample metadata
            metadata = {
                'sample_id': i,
                'timestamp': float(sample.get('timestamp', 0)),
                'rgb_shape': sample['rgb'].shape if sample['rgb'] is not None else None,
                'depth_shape': sample['depth'].shape if sample['depth'] is not None else None,
                'rgb_path': f"rgb/rgb_{i:04d}.png",
                'depth_path': f"depth/depth_{i:04d}.png"
            }
            
            meta_path = os.path.join(output_dir, f"metadata_{i:04d}.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            samples.append(metadata)
            
            if i % 5 == 0:
                print(f"Captured sample {i}")
        
        # Save dataset manifest
        manifest = {
            'total_samples': len(samples),
            'modalities': ['rgb', 'depth'],  # segmentation if available,
            'samples': samples
        }
        
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        self.world.stop()
        print(f"Multi-modal dataset completed! Saved {len(samples)} samples to {output_dir}")
        return manifest

def main():
    try:
        generator = MultiModalGenerator()
        manifest = generator.capture_multimodal_dataset(num_samples=10)
        print("Multi-modal generation exercise completed successfully!")
        print(f"Dataset contains {manifest['total_samples']} samples")
    except Exception as e:
        print(f"Error in multi-modal generation: {e}")

if __name__ == "__main__":
    main()
```

### Exercise 4: Synthetic Object Detection Dataset

#### Objective
Create a synthetic dataset for object detection with bounding box annotations.

#### Steps
1. Create a scene with multiple objects
2. Generate bounding box annotations
3. Apply domain randomization
4. Export in COCO format

```python
# detection_dataset.py
import omni
from omni.isaac.core import World
from omni.isaac.sensor import Camera
import numpy as np
import cv2
import os
import json
from typing import List, Dict, Tuple

class DetectionDatasetGenerator:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.camera = None
        self.objects = []
        self.categories = {
            'cube': 1,
            'cylinder': 2,
            'sphere': 3,
            'cone': 4
        }
        self.setup_scene()
    
    def setup_scene(self):
        """Setup scene with objects for detection"""
        print("Setting up object detection scene...")
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        
        # Add camera
        self.camera = self.world.scene.add(
            Camera(
                prim_path="/World/Camera",
                frequency=30,
                resolution=(640, 480)
            )
        )
        
        self.camera.set_world_pose(
            position=np.array([2.0, 0.0, 1.5]),
            orientation=np.array([0.707, 0.0, 0.0, 0.707])
        )
        
        # Define objects to add to scene
        self.object_configs = [
            {'type': 'cube', 'name': 'red_cube', 'position': [0.5, 0.3, 0.1], 'color': [1, 0, 0]},
            {'type': 'sphere', 'name': 'blue_sphere', 'position': [-0.4, -0.3, 0.2], 'color': [0, 0, 1]},
            {'type': 'cylinder', 'name': 'green_cylinder', 'position': [0.0, 0.5, 0.15], 'color': [0, 1, 0]},
        ]
        
        # Add objects to scene (conceptually - in real implementation they would be USD objects)
        for config in self.object_configs:
            self.objects.append({
                'name': config['name'],
                'type': config['type'],
                'position': np.array(config['position']),
                'color': config['color'],
                'bbox': None  # Will be calculated based on camera position
            })
        
        print("Object detection scene setup complete")
    
    def calculate_bounding_boxes(self) -> List[Dict]:
        """Calculate 2D bounding boxes for objects in camera view"""
        # In a real implementation, this would project 3D object bounds to 2D
        # For this exercise, we'll simulate the calculation
        
        bboxes = []
        for obj in self.objects:
            # Simulate projection of 3D object to 2D image
            # This is a simplified version - real implementation would use camera intrinsics
            img_width, img_height = 640, 480
            
            # Calculate approximate 2D position based on 3D position (simplified)
            # The further away, the smaller the object appears
            z = obj['position'][2]
            scale_factor = 200.0 / (z + 0.5)  # Simple perspective approximation
            
            # Calculate center in image coordinates
            # Convert 3D world coordinates to 2D image coordinates (simplified)
            x_2d = int((obj['position'][0] * scale_factor) + img_width // 2)
            y_2d = int((-obj['position'][1] * scale_factor) + img_height // 2)  # Flip Y
            
            # Calculate approximate width/height based on distance and object size
            obj_width = int(30 / (z + 0.5))  # Further = smaller
            obj_height = int(30 / (z + 0.5))
            
            # Create bounding box in COCO format [x, y, width, height]
            bbox = [
                max(0, x_2d - obj_width // 2),   # x
                max(0, y_2d - obj_height // 2),  # y
                min(img_width, obj_width),       # width
                min(img_height, obj_height)      # height
            ]
            
            bboxes.append({
                'bbox': bbox,
                'category_id': self.categories[obj['type']],
                'category_name': obj['type'],
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
        
        return bboxes
    
    def capture_detection_sample(self, sample_id: int) -> Dict:
        """Capture a single sample with detection annotations"""
        # Step the simulation
        self.world.step(render=True)
        
        # Capture RGB image
        rgb = self.camera.get_rgb()
        
        # Calculate bounding boxes
        bboxes = self.calculate_bounding_boxes()
        
        # Create sample record
        sample = {
            'id': sample_id,
            'image_filename': f"detection_{sample_id:06d}.jpg",
            'width': 640,
            'height': 480,
            'rgb_image': rgb,
            'annotations': bboxes
        }
        
        return sample
    
    def save_detection_sample(self, sample: Dict, output_dir: str):
        """Save detection sample with annotations"""
        # Save RGB image
        if sample['rgb_image'] is not None:
            img_uint8 = (sample['rgb_image'] * 255).astype(np.uint8)
            img_path = os.path.join(output_dir, "images", sample['image_filename'])
            os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
            cv2.imwrite(img_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        
        # For this exercise, we won't save annotation images, but in a real system
        # we might want to visualize the bounding boxes on the image
        
    def generate_detection_dataset(self, num_samples: int = 50) -> Dict:
        """Generate a complete detection dataset"""
        print(f"Generating detection dataset with {num_samples} samples...")
        
        output_dir = "detection_dataset"
        os.makedirs(output_dir, exist_ok=True)
        
        self.world.play()
        
        dataset_images = []
        dataset_annotations = []
        annotation_id = 0
        
        for i in range(num_samples):
            # Capture sample
            sample = self.capture_detection_sample(i)
            
            # Add image record to dataset
            image_record = {
                'id': i,
                'width': sample['width'],
                'height': sample['height'],
                'file_name': sample['image_filename']
            }
            dataset_images.append(image_record)
            
            # Add annotations
            for bbox_data in sample['annotations']:
                annotation_record = {
                    'id': annotation_id,
                    'image_id': i,
                    'category_id': bbox_data['category_id'],
                    'bbox': bbox_data['bbox'],
                    'area': bbox_data['area'],
                    'iscrowd': bbox_data['iscrowd']
                }
                dataset_annotations.append(annotation_record)
                annotation_id += 1
            
            # Save the sample
            self.save_detection_sample(sample, output_dir)
            
            if i % 10 == 0:
                print(f"Processed {i}/{num_samples} samples")
        
        self.world.stop()
        
        # Create COCO format dataset
        coco_dataset = {
            'info': {
                'description': 'Synthetic Object Detection Dataset',
                'version': '1.0',
                'year': 2023,
            },
            'licenses': [{'id': 1, 'name': 'MIT', 'url': ''}],
            'categories': [
                {'id': v, 'name': k, 'supercategory': 'object'} 
                for k, v in self.categories.items()
            ],
            'images': dataset_images,
            'annotations': dataset_annotations
        }
        
        # Save COCO format dataset
        coco_path = os.path.join(output_dir, "annotations.json")
        with open(coco_path, 'w') as f:
            json.dump(coco_dataset, f, indent=2)
        
        print(f"Detection dataset completed! {num_samples} samples saved to {output_dir}")
        return coco_dataset

def main():
    try:
        detector = DetectionDatasetGenerator()
        dataset = detector.generate_detection_dataset(num_samples=20)
        print("Detection dataset generation completed successfully!")
        print(f"Dataset contains {len(dataset['images'])} images and {len(dataset['annotations'])} annotations")
    except Exception as e:
        print(f"Error in detection dataset generation: {e}")

if __name__ == "__main__":
    main()
```

### Exercise 5: Validation and Quality Assessment

#### Objective
Validate the quality of synthetic data and assess its suitability for training.

#### Steps
1. Create validation metrics
2. Compare synthetic vs. real data characteristics
3. Assess quality of annotations
4. Generate validation reports

```python
# validation_assessment.py
import numpy as np
import cv2
from scipy import ndimage
from skimage import feature, measure
import matplotlib.pyplot as plt
import os

class DataValidator:
    def __init__(self):
        pass
    
    def calculate_image_quality_metrics(self, image):
        """Calculate various image quality metrics"""
        metrics = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = float(laplacian_var)
        
        # Entropy (measure of information content)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=[0,256])
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros to avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))
        metrics['entropy'] = float(entropy)
        
        # Contrast (standard deviation)
        contrast = float(np.std(gray))
        metrics['contrast'] = contrast
        
        # Mean brightness
        brightness = float(np.mean(gray))
        metrics['brightness'] = brightness
        
        return metrics
    
    def assess_annotation_quality(self, annotations, image_shape):
        """Assess quality of bounding box annotations"""
        quality_metrics = {}
        
        # Check if annotations are within image bounds
        valid_bboxes = []
        out_of_bounds = 0
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Check bounds
            if (x >= 0 and y >= 0 and 
                x + w <= image_shape[1] and y + h <= image_shape[0]):
                valid_bboxes.append(ann)
            else:
                out_of_bounds += 1
        
        quality_metrics['valid_annotations'] = len(valid_bboxes)
        quality_metrics['out_of_bounds_count'] = out_of_bounds
        quality_metrics['valid_ratio'] = len(valid_bboxes) / max(1, len(annotations))
        
        # Check for reasonable object sizes
        reasonable_size_count = 0
        for ann in valid_bboxes:
            bbox = ann['bbox']
            area = bbox[2] * bbox[3]  # width * height
            image_area = image_shape[0] * image_shape[1]
            area_ratio = area / image_area
            
            # Consider objects reasonable if they're between 0.1% and 50% of image area
            if 0.001 <= area_ratio <= 0.5:
                reasonable_size_count += 1
        
        quality_metrics['reasonable_size_ratio'] = reasonable_size_count / max(1, len(valid_bboxes))
        
        return quality_metrics
    
    def compare_with_real_data(self, synthetic_images, real_images):
        """Compare statistical properties of synthetic vs real data"""
        comparison_metrics = {}
        
        # Calculate basic statistics
        synth_means = [np.mean(img) for img in synthetic_images]
        real_means = [np.mean(img) for img in real_images]
        
        comparison_metrics['mean_similarity'] = {
            'synthetic_mean': float(np.mean(synth_means)),
            'real_mean': float(np.mean(real_means)),
            'difference': float(abs(np.mean(synth_means) - np.mean(real_means)))
        }
        
        synth_stds = [np.std(img) for img in synthetic_images]
        real_stds = [np.std(img) for img in real_images]
        
        comparison_metrics['std_similarity'] = {
            'synthetic_std': float(np.mean(synth_stds)),
            'real_std': float(np.mean(real_stds)),
            'difference': float(abs(np.mean(synth_stds) - np.mean(real_stds)))
        }
        
        return comparison_metrics
    
    def generate_validation_report(self, datasets_info):
        """Generate a comprehensive validation report"""
        report = {
            'summary': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        total_samples = sum(ds.get('sample_count', 0) for ds in datasets_info)
        avg_sharpness = np.mean([ds.get('avg_sharpness', 0) for ds in datasets_info])
        avg_entropy = np.mean([ds.get('avg_entropy', 0) for ds in datasets_info])
        
        report['summary'] = {
            'total_samples': total_samples,
            'average_sharpness': float(avg_sharpness),
            'average_entropy': float(avg_entropy),
            'datasets_validated': len(datasets_info)
        }
        
        # Generate recommendations based on metrics
        if avg_sharpness < 100:  # Threshold is example
            report['recommendations'].append(
                "Average image sharpness is low. Consider improving rendering quality or adding sharpening."
            )
        
        if avg_entropy < 5:  # Threshold is example
            report['recommendations'].append(
                "Dataset has low information content. Consider adding more diverse scenes/objects."
            )
        
        if total_samples < 1000:  # Threshold is example
            report['recommendations'].append(
                "Dataset size is small. Consider generating more samples for robust training."
            )
        
        return report

def main():
    # This is a validation framework - in practice it would be used with
    # actual synthetic datasets generated in previous exercises
    print("Validation and Quality Assessment framework initialized")
    print("This framework would be used to validate datasets generated in previous exercises.")
    
    # Example of how it would be used:
    validator = DataValidator()
    
    # Example metrics calculation
    sample_metrics = validator.calculate_image_quality_metrics(
        np.random.rand(480, 640, 3) * 255  # Random image for example
    )
    
    print("Sample validation metrics:", sample_metrics)
    print("Validation framework ready for use with actual datasets!")

if __name__ == "__main__":
    main()
```

### Assessment Criteria

Your implementation will be evaluated based on:

1. **Setup and Configuration**: Correct initialization of Isaac Sim environments
2. **Data Quality**: Generation of high-quality synthetic data with accurate annotations
3. **Diversity**: Effective use of domain randomization to create diverse data
4. **Multi-modal Integration**: Proper capture and correlation of different data modalities
5. **Validation**: Proper assessment of data quality and suitability for training

### Troubleshooting Tips

1. **Isaac Sim Not Starting**: Verify Omniverse system requirements and installation
2. **Camera Not Capturing**: Check camera positioning and rendering settings  
3. **Performance Issues**: Reduce scene complexity or lower resolution during development
4. **Annotation Accuracy**: Validate that 3D to 2D projections are calculated correctly
5. **Domain Randomization**: Ensure randomization doesn't make data too unrealistic

### Extensions for Advanced Students

- Implement physics-based data generation for dynamic scenarios
- Add more complex sensor models (LIDAR, thermal, multispectral)
- Create reinforcement learning environments using synthetic data
- Implement automatic quality assessment and filtering of generated data
- Develop dataset curation algorithms to select the most useful synthetic samples

This practical exercise provides comprehensive experience with synthetic data generation using Isaac Sim, covering the full pipeline from environment setup to dataset validation.