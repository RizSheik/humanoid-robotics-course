# Chapter 3: Unity - Visualization-Rich Simulation Environments


<div className="robotDiagram">
  <img src="/static/img/book-image/Illustration_explaining_Physical_AI_huma_1 (1).jpg" alt="Humanoid Robot" style={{borderRadius:"50px", width: '900px', height: '350px', margin: '10px auto', display: 'block'}} />
</div>


## Learning Objectives

After completing this chapter, students will be able to:
- Set up and configure Unity simulation environments for robotics
- Create realistic visual environments with accurate physics
- Implement sensor simulations with Unity's rendering pipeline
- Develop custom robotics assets and environments in Unity
- Integrate Unity simulations with ROS 2 communication systems
- Optimize Unity environments for real-time robotics simulation

## 3.1 Introduction to Unity for Robotics

Unity has emerged as a powerful platform for creating visualization-rich simulation environments for robotics, particularly for applications requiring high-fidelity graphics, photorealistic rendering, or complex environmental scenarios. Originally developed for game development, Unity's real-time rendering capabilities, extensive asset ecosystem, and physics engine make it suitable for simulating robotic perception, human-robot interaction, and complex environments.

### 3.1.1 Unity in the Robotics Context

Unity provides several advantages for robotics simulation:
- **Photorealistic Rendering**: Essential for training computer vision algorithms
- **Extensive Asset Library**: Thousands of pre-made models and environments
- **Flexible Scripting System**: Using C# for custom robot behaviors
- **Physics Engine**: Built-in physics simulation with configurable parameters
- **Cross-Platform Support**: Deploy to various hardware configurations
- **User Interaction**: Natural interfaces for human-in-the-loop scenarios

### 3.1.2 Unity vs Other Simulation Platforms

| Aspect | Unity | Gazebo | Other Platforms |
|--------|-------|--------|-----------------|
| Rendering Quality | High | Moderate | Varies |
| Physics Accuracy | Good | Excellent | Varies |
| Sensor Simulation | Good* | Excellent | Good |
| Realism | High | Moderate | Varies |
| Asset Creation | Easy | Complex | Varies |
| Development Speed | Fast | Moderate | Varies |

*Unity sensors require custom implementation but can achieve high fidelity

### 3.1.3 Unity Robotics Ecosystem

Unity provides several tools for robotics applications:
- **Unity Robotics Hub**: Centralized access to robotics packages and samples
- **Unity ML-Agents**: For reinforcement learning applications
- **ROS#**: For ROS/ROS 2 communication
- **Unity Perception**: For generating synthetic training data
- **Open Source Robotics (OSR) Integration**: For standardized interfaces

## 3.2 Setting Up Unity for Robotics Simulation

### 3.2.1 System Requirements

To run Unity effectively for robotics simulation:
- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **CPU**: 64-bit processor with SSE2 support
- **Memory**: 8GB+ RAM (16GB+ recommended for complex scenes)
- **GPU**: DirectX 10 or OpenGL 3.3+ compatible graphics card
- **Storage**: 20GB+ free space for Unity installation and projects

### 3.2.2 Installation Process

1. Download Unity Hub from Unity's website
2. Install Unity Hub and create an account
3. Use Unity Hub to install Unity version 2021.3 LTS or later (recommended for stability)
4. Install required modules: Windows/Linux/Mac Build Support, Visual Studio integration
5. Install Unity robotics packages via Package Manager

### 3.2.3 Recommended Packages

For robotics applications, install these packages:
- **ROS-TCP-Connector**: For ROS/ROS 2 communication
- **Unity Perception**: For synthetic data generation
- **Open Robotics Integration**: Standardized interfaces
- **Universal Render Pipeline (URP)**: For optimized rendering
- **XR Toolkit**: If developing VR/AR applications

## 3.3 Creating Robot Models in Unity

### 3.3.1 Importing Robot Models

Unity can import robot models from various formats:

```csharp
// Example of importing and setting up a robot model
using UnityEngine;

public class RobotModelSetup : MonoBehaviour
{
    [Header("Robot Configuration")]
    public string robotName;
    public Vector3 initialPosition = Vector3.zero;
    public Quaternion initialRotation = Quaternion.identity;

    [Header("Joint Configuration")]
    public Transform[] jointTransforms;
    public string[] jointNames;
    public float[] jointLimitsMin;
    public float[] jointLimitsMax;

    void Start()
    {
        SetupRobot();
    }

    void SetupRobot()
    {
        // Position and orient the robot
        transform.position = initialPosition;
        transform.rotation = initialRotation;

        // Configure joints (simplified example)
        ConfigureJoints();
    }

    void ConfigureJoints()
    {
        for (int i = 0; i < jointTransforms.Length; i++)
        {
            // Add joint constraints using configurable joints
            var joint = jointTransforms[i].GetComponent<ConfigurableJoint>();
            if (joint != null)
            {
                // Configure joint limits based on real robot specifications
                joint.angularXLimit = new SoftJointLimit
                {
                    limit = jointLimitsMax[i]
                };
                
                joint.angularXDrive = new JointDrive
                {
                    positionSpring = 10000f, // Stiffness
                    positionDamper = 100f   // Damping
                };
            }
        }
    }
}
```

### 3.3.2 Physics Configuration for Robots

Configuring physics properties for realistic robot behavior:

```csharp
// Physics configuration script for robot parts
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class RobotPhysicsConfig : MonoBehaviour
{
    [Header("Physics Properties")]
    public float mass = 1.0f;
    public PhysicMaterial physicMaterial;
    
    [Header("Drag Settings")]
    public float drag = 0.1f;
    public float angularDrag = 0.05f;
    
    [Header("Collision Settings")]
    public bool useInterpolation = true;
    public CollisionDetectionMode collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
    
    void Start()
    {
        SetupPhysics();
    }

    void SetupPhysics()
    {
        Rigidbody rb = GetComponent<Rigidbody>();
        rb.mass = mass;
        rb.drag = drag;
        rb.angularDrag = angularDrag;
        rb.interpolation = useInterpolation ? RigidbodyInterpolation.Interpolate : RigidbodyInterpolation.None;
        rb.collisionDetectionMode = collisionDetectionMode;
        
        // Assign physic material if specified
        if (physicMaterial != null && GetComponent<Collider>() != null)
        {
            GetComponent<Collider>().material = physicMaterial;
        }
    }
}
```

### 3.3.3 Animation and Joint Control

Creating systems for robot joint control and animation:

```csharp
// Joint controller for Unity robot
using UnityEngine;

public class RobotJointController : MonoBehaviour
{
    [System.Serializable]
    public class JointConfig
    {
        public string jointName;
        public Transform jointTransform;
        public float minAngle = -90f;
        public float maxAngle = 90f;
        public float currentAngle;
        public float targetAngle;
        public float maxSpeed = 100f; // degrees per second
    }

    public JointConfig[] joints;
    public bool usePhysics = false;

    void Update()
    {
        UpdateJoints();
    }

    void UpdateJoints()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            var joint = joints[i];
            
            // Move toward target angle
            joint.currentAngle = Mathf.MoveTowards(
                joint.currentAngle, 
                joint.targetAngle, 
                joint.maxSpeed * Time.deltaTime
            );
            
            // Apply rotation (assuming rotation around Y axis)
            Vector3 rotation = joint.jointTransform.localEulerAngles;
            rotation.y = joint.currentAngle;
            joint.jointTransform.localEulerAngles = rotation;
        }
    }

    // Method to set joint target angles
    public void SetJointTarget(int jointIndex, float angle)
    {
        if (jointIndex >= 0 && jointIndex < joints.Length)
        {
            joints[jointIndex].targetAngle = Mathf.Clamp(angle, 
                joints[jointIndex].minAngle, 
                joints[jointIndex].maxAngle);
        }
    }

    // Method to get current joint angles
    public float GetJointAngle(int jointIndex)
    {
        if (jointIndex >= 0 && jointIndex < joints.Length)
        {
            return joints[jointIndex].currentAngle;
        }
        return 0f;
    }
}
```

## 3.4 Environment Design and Physics

### 3.4.1 Creating Realistic Environments

Unity excels at creating photorealistic environments for robotics training:

```csharp
// Environment setup script
using UnityEngine;

public class EnvironmentSetup : MonoBehaviour
{
    [Header("Environment Configuration")]
    public Material[] environmentMaterials;
    public GameObject[] environmentalObjects;
    
    [Header("Lighting Setup")]
    public Light mainLight;
    public float lightIntensity = 1.0f;
    public Color lightColor = Color.white;
    
    [Header("Physics Environment")]
    public PhysicMaterial groundMaterial;
    public float gravityMultiplier = 1.0f;

    void Start()
    {
        SetupEnvironment();
    }

    void SetupEnvironment()
    {
        // Configure lighting
        if (mainLight != null)
        {
            mainLight.intensity = lightIntensity;
            mainLight.color = lightColor;
            
            // Add realistic shadows
            mainLight.shadows = LightShadows.Soft;
        }
        
        // Apply environment materials
        foreach (var material in environmentMaterials)
        {
            // Apply environmental properties
            if (material != null)
            {
                // Set up material properties for physics simulation
                // (This would be more complex in a real implementation)
            }
        }
        
        // Configure physics environment
        Physics.gravity *= gravityMultiplier;
        
        // Apply ground material to floor objects
        ApplyGroundMaterial();
    }

    void ApplyGroundMaterial()
    {
        GameObject[] floors = GameObject.FindGameObjectsWithTag("Floor");
        foreach (GameObject floor in floors)
        {
            Collider floorCollider = floor.GetComponent<Collider>();
            if (floorCollider != null && groundMaterial != null)
            {
                floorCollider.material = groundMaterial;
            }
        }
    }
}
```

### 3.4.2 Physics Simulation Parameters

Configuring Unity's physics system for robotics simulation:

```csharp
// Physics configuration for robotics simulation
using UnityEngine;

public class RoboticsPhysicsConfig : MonoBehaviour
{
    [Header("Physics Settings")]
    public int solverIterations = 6;           // Higher for stable joints
    public int solverVelocityIterations = 1;   // Higher for stable contacts
    public float sleepThreshold = 0.005f;      // Lower to detect small movements
    public float defaultContactOffset = 0.01f; // Small for accurate contacts
    public float bounceThreshold = 2f;         // Threshold for bouncing
    public float defaultSolverVelocityIterations = 1;

    void Start()
    {
        ConfigurePhysics();
    }

    void ConfigurePhysics()
    {
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;
        Physics.sleepThreshold = sleepThreshold;
        Physics.defaultContactOffset = defaultContactOffset;
        Physics.bounceThreshold = bounceThreshold;
    }
}
```

### 3.4.3 Terrain and Complex Environments

Creating complex outdoor environments:

```csharp
// Terrain configuration script
using UnityEngine;

[RequireComponent(typeof(Terrain))]
public class TerrainConfig : MonoBehaviour
{
    [Header("Terrain Properties")]
    public float terrainScale = 100f;
    public PhysicMaterial terrainMaterial;
    public Texture2D[] terrainTextures;
    public float[] textureScales;

    void Start()
    {
        ConfigureTerrain();
    }

    void ConfigureTerrain()
    {
        Terrain terrain = GetComponent<Terrain>();
        
        // Configure terrain properties
        terrain.terrainData.size = new Vector3(terrainScale, 20f, terrainScale);
        
        // Set up terrain collider with physics material
        TerrainCollider terrainCollider = GetComponent<TerrainCollider>();
        if (terrainCollider != null && terrainMaterial != null)
        {
            terrainCollider.material = terrainMaterial;
        }
        
        // Configure terrain textures
        SetupTerrainTextures(terrain);
    }

    void SetupTerrainTextures(Terrain terrain)
    {
        TerrainData terrainData = terrain.terrainData;
        
        // Create and assign terrain materials
        SplatPrototype[] splatPrototypes = new SplatPrototype[terrainTextures.Length];
        
        for (int i = 0; i < terrainTextures.Length; i++)
        {
            splatPrototypes[i] = new SplatPrototype();
            splatPrototypes[i].texture = terrainTextures[i];
            splatPrototypes[i].tileSize = new Vector2(textureScales[i], textureScales[i]);
        }
        
        terrainData.splatPrototypes = splatPrototypes;
    }
}
```

## 3.5 Sensor Simulation in Unity

### 3.5.1 Camera Sensor Simulation

Creating realistic camera sensors with Unity's rendering pipeline:

```csharp
// Unity camera sensor implementation
using UnityEngine;
using System.Collections;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera cameraComponent;
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float updateRate = 30f; // FPS
    public bool addNoise = true;
    
    [Header("Noise Configuration")]
    public float noiseIntensity = 0.01f;
    public float noiseScale = 0.1f;

    private RenderTexture renderTexture;
    private float updateInterval;
    private float lastUpdate;

    void Start()
    {
        SetupCamera();
        CreateRenderTexture();
        updateInterval = 1.0f / updateRate;
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            CaptureImage();
            lastUpdate = Time.time;
        }
    }

    void SetupCamera()
    {
        if (cameraComponent == null)
        {
            cameraComponent = GetComponent<Camera>();
        }
        
        if (cameraComponent == null)
        {
            cameraComponent = gameObject.AddComponent<Camera>();
        }
        
        cameraComponent.targetTexture = null; // Will be set to render texture
        cameraComponent.enabled = true;
    }

    void CreateRenderTexture()
    {
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        renderTexture.Create();
        cameraComponent.targetTexture = renderTexture;
    }

    void CaptureImage()
    {
        // Capture image to RenderTexture
        cameraComponent.Render();
        
        // Process image (in a real implementation, this would send to ROS)
        ProcessCapturedImage();
    }

    void ProcessCapturedImage()
    {
        // Create temporary texture to read pixels
        Texture2D imageTexture = new Texture2D(renderTexture.width, renderTexture.height, 
            TextureFormat.RGB24, false);
        
        // Keep current active render texture
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = renderTexture;
        
        // Read pixels from active render texture
        imageTexture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        imageTexture.Apply();
        
        // Restore active render texture
        RenderTexture.active = currentRT;
        
        // Add noise if enabled
        if (addNoise)
        {
            AddImageNoise(imageTexture);
        }
        
        // In a real implementation, convert to ROS message format and publish
        ConvertAndPublishImage(imageTexture);
        
        // Clean up temporary texture
        Destroy(imageTexture);
    }

    void AddImageNoise(Texture2D texture)
    {
        // Apply noise to image texture
        Color[] pixels = texture.GetPixels();
        
        for (int i = 0; i < pixels.Length; i++)
        {
            // Add random noise
            float noise = Random.Range(-noiseIntensity, noiseIntensity);
            pixels[i] = new Color(
                Mathf.Clamp01(pixels[i].r + noise),
                Mathf.Clamp01(pixels[i].g + noise),
                Mathf.Clamp01(pixels[i].b + noise)
            );
        }
        
        texture.SetPixels(pixels);
        texture.Apply();
    }

    void ConvertAndPublishImage(Texture2D image)
    {
        // Convert Unity image format to ROS format
        // In practice, this would use ROS-TCP-Connector to send image to ROS system
        // For this example, we'll just log that image conversion would happen
        Debug.Log("Image captured and ready for ROS conversion");
    }
}
```

### 3.5.2 LIDAR Sensor Simulation

Creating LIDAR sensors using Unity's raycast system:

```csharp
// LIDAR sensor simulation
using UnityEngine;
using System.Collections.Generic;

public class UnityLIDARSensor : MonoBehaviour
{
    [Header("LIDAR Configuration")]
    public float scanRange = 30f;
    public int horizontalResolution = 720;
    public int verticalResolution = 1;
    public float updateRate = 10f;
    public LayerMask detectionLayers = -1; // All layers by default
    
    [Header("Noise Configuration")]
    public float rangeNoise = 0.01f;
    
    private float updateInterval;
    private float lastUpdate;
    private float horizontalAngleIncrement;
    private List<float> scanData;

    void Start()
    {
        updateInterval = 1.0f / updateRate;
        horizontalAngleIncrement = 360.0f / horizontalResolution;
        scanData = new List<float>(horizontalResolution);
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            PerformScan();
            lastUpdate = Time.time;
        }
    }

    void PerformScan()
    {
        scanData.Clear();
        
        for (int i = 0; i < horizontalResolution; i++)
        {
            float angle = i * horizontalAngleIncrement * Mathf.Deg2Rad;
            float range = PerformRaycast(angle, 0); // 0 elevation for 2D LIDAR
            scanData.Add(range);
        }
        
        // Publish scan data (would connect to ROS in real implementation)
        PublishScanData();
    }

    float PerformRaycast(float horizontalAngle, float verticalAngle)
    {
        // Calculate ray direction based on angles
        Vector3 direction = new Vector3(
            Mathf.Cos(verticalAngle) * Mathf.Cos(horizontalAngle),
            Mathf.Sin(verticalAngle),
            Mathf.Cos(verticalAngle) * Mathf.Sin(horizontalAngle)
        );
        
        direction = transform.TransformDirection(direction);
        
        // Perform raycast
        if (Physics.Raycast(transform.position, direction, out RaycastHit hit, scanRange, detectionLayers))
        {
            float range = hit.distance;
            
            // Add noise to measurement
            range += Random.Range(-rangeNoise, rangeNoise);
            
            return Mathf.Clamp(range, 0f, scanRange);
        }
        else
        {
            // Return max range if no hit
            return scanRange;
        }
    }

    void PublishScanData()
    {
        // In a real implementation, convert scan data to ROS LaserScan message
        // and publish via ROS-TCP-Connector
        Debug.Log($"LIDAR scan completed with {scanData.Count} points");
    }

    // Visualization method for debugging
    void OnDrawGizmosSelected()
    {
        if (scanData != null && scanData.Count > 0)
        {
            for (int i = 0; i < scanData.Count; i++)
            {
                float angle = i * horizontalAngleIncrement * Mathf.Deg2Rad;
                Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
                direction = transform.TransformDirection(direction);
                
                float range = scanData[i];
                if (range < scanRange)
                {
                    Gizmos.color = Color.red;
                    Gizmos.DrawRay(transform.position, direction * range);
                }
                else
                {
                    Gizmos.color = Color.green;
                    Gizmos.DrawRay(transform.position, direction * scanRange);
                }
            }
        }
    }
}
```

### 3.5.3 IMU Sensor Simulation

Implementing IMU sensors using Unity's physics system:

```csharp
// IMU sensor simulation
using UnityEngine;

public class UnityIMUSensor : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float updateRate = 100f; // Hz
    public Vector3 noiseLinearAcceleration = Vector3.one * 0.017f; // m/s^2
    public Vector3 noiseAngularVelocity = Vector3.one * 0.0002f; // rad/s
    public Vector3 noiseOrientation = Vector3.one * 0.01f;         // rad
    
    [Header("Gravity Compensation")]
    public bool compensateGravity = true;

    private float updateInterval;
    private float lastUpdate;
    private Rigidbody attachedRigidbody;
    
    // These would be published as ROS messages in real implementation
    public Vector3 linearAcceleration;
    public Vector3 angularVelocity;
    public Quaternion orientation;

    void Start()
    {
        updateInterval = 1.0f / updateRate;
        attachedRigidbody = GetComponent<Rigidbody>();
        
        // If no rigidbody attached, try to find one in parent
        if (attachedRigidbody == null)
        {
            attachedRigidbody = GetComponentInParent<Rigidbody>();
        }
    }

    void Update()
    {
        if (Time.time - lastUpdate >= updateInterval)
        {
            UpdateIMUReading();
            lastUpdate = Time.time;
        }
    }

    void UpdateIMUReading()
    {
        if (attachedRigidbody != null)
        {
            // Get orientation (relative to world or other reference)
            orientation = transform.rotation;
            
            // Get angular velocity from rigidbody if attached
            if (attachedRigidbody != null)
            {
                // Angular velocity in the local frame
                angularVelocity = transform.InverseTransformDirection(attachedRigidbody.angularVelocity);
            }
            else
            {
                angularVelocity = Vector3.zero;
            }
            
            // Linear acceleration (calculated from change in velocity)
            if (attachedRigidbody != null)
            {
                // Get linear acceleration in local frame
                Vector3 worldAcc = attachedRigidbody.velocity / Time.deltaTime;
                linearAcceleration = transform.InverseTransformDirection(worldAcc);
                
                // Optionally remove gravity from linear acceleration
                if (compensateGravity)
                {
                    Vector3 gravityVector = Physics.gravity;
                    Vector3 gravityLocal = transform.InverseTransformDirection(gravityVector);
                    linearAcceleration -= gravityLocal;
                }
            }
            else
            {
                linearAcceleration = Physics.gravity;
                if (compensateGravity) linearAcceleration = Vector3.zero;
            }
        }
        else
        {
            // If no rigidbody, use transform changes (less accurate)
            orientation = transform.rotation;
            linearAcceleration = (transform.position - transform.position) / Time.deltaTime; // Simplified
            angularVelocity = Vector3.zero;
        }
        
        // Add noise to measurements
        AddNoiseToMeasurements();
        
        // Publish IMU data (would connect to ROS in real implementation)
        PublishIMUData();
    }

    void AddNoiseToMeasurements()
    {
        linearAcceleration += new Vector3(
            Random.Range(-noiseLinearAcceleration.x, noiseLinearAcceleration.x),
            Random.Range(-noiseLinearAcceleration.y, noiseLinearAcceleration.y),
            Random.Range(-noiseLinearAcceleration.z, noiseLinearAcceleration.z)
        );
        
        angularVelocity += new Vector3(
            Random.Range(-noiseAngularVelocity.x, noiseAngularVelocity.x),
            Random.Range(-noiseAngularVelocity.y, noiseAngularVelocity.y),
            Random.Range(-noiseAngularVelocity.z, noiseAngularVelocity.z)
        );
        
        // Add noise to orientation (simplified approach)
        orientation = Quaternion.Euler(
            orientation.eulerAngles.x + Random.Range(-noiseOrientation.x, noiseOrientation.x),
            orientation.eulerAngles.y + Random.Range(-noiseOrientation.y, noiseOrientation.y),
            orientation.eulerAngles.z + Random.Range(-noiseOrientation.z, noiseOrientation.z)
        );
    }

    void PublishIMUData()
    {
        // In a real implementation, convert to ROS sensor_msgs/Imu message
        // and publish via ROS-TCP-Connector
        Debug.Log($"IMU: Acc={linearAcceleration}, Vel={angularVelocity}");
    }
}
```

## 3.6 ROS Integration with Unity

### 3.6.1 ROS-TCP-Connector Setup

Integrating Unity with ROS systems:

```csharp
// ROS communication interface
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;

public class ROSCommunication : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeIP = "127.0.0.1";
    public int rosBridgePort = 9090;
    
    private TcpClient rosClient;
    private NetworkStream stream;
    private bool isConnected = false;

    void Start()
    {
        ConnectToROSBridge();
    }

    void ConnectToROSBridge()
    {
        try
        {
            rosClient = new TcpClient(rosBridgeIP, rosBridgePort);
            stream = rosClient.GetStream();
            isConnected = true;
            Debug.Log("Connected to ROS bridge");
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to connect to ROS bridge: " + e.Message);
        }
    }

    public void PublishToTopic(string topic, string messageType, object message)
    {
        if (!isConnected) return;

        // Create ROS bridge message
        var rosMessage = new
        {
            op = "publish",
            topic = topic,
            type = messageType,
            msg = message
        };

        string jsonMessage = JsonConvert.SerializeObject(rosMessage);
        byte[] data = Encoding.UTF8.GetBytes(jsonMessage + "\n");

        try
        {
            stream.Write(data, 0, data.Length);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to publish message: " + e.Message);
        }
    }

    public void SubscribeToTopic(string topic, string messageType)
    {
        if (!isConnected) return;

        var subscribeMessage = new
        {
            op = "subscribe",
            topic = topic,
            type = messageType
        };

        string jsonMessage = JsonConvert.SerializeObject(subscribeMessage);
        byte[] data = Encoding.UTF8.GetBytes(jsonMessage + "\n");

        try
        {
            stream.Write(data, 0, data.Length);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Failed to subscribe to topic: " + e.Message);
        }
    }

    void OnApplicationQuit()
    {
        if (rosClient != null)
        {
            rosClient.Close();
        }
    }
}
```

### 3.6.2 Message Conversion Utilities

Converting between Unity and ROS message formats:

```csharp
// Message conversion utilities
using UnityEngine;
using System.Collections.Generic;

public static class ROSMessageConverter
{
    // Convert Unity Vector3 to ROS geometry_msgs/Point
    public static Dictionary<string, object> Vector3ToROSPoint(Vector3 unityVector)
    {
        return new Dictionary<string, object>
        {
            {"x", unityVector.x},
            {"y", unityVector.z},  // Unity's z is ROS's y (assuming coordinate transformation)
            {"z", unityVector.y}   // Unity's y is ROS's z
        };
    }

    // Convert Unity Quaternion to ROS geometry_msgs/Quaternion
    public static Dictionary<string, object> QuaternionToROSQuaternion(Quaternion unityQuat)
    {
        return new Dictionary<string, object>
        {
            {"x", unityQuat.x},
            {"y", unityQuat.z},  // Coordinate transformation
            {"z", unityQuat.y},
            {"w", unityQuat.w}
        };
    }

    // Convert ROS timestamp to Unity
    public static double GetROSTimestamp()
    {
        System.DateTime epochStart = new System.DateTime(1970, 1, 1, 0, 0, 0, System.DateTimeKind.Utc);
        double timestamp = (System.DateTime.UtcNow - epochStart).TotalSeconds;
        return timestamp;
    }
}
```

## 3.7 Performance Optimization

### 3.7.1 Rendering Optimization

Optimizing Unity for real-time robotics simulation:

```csharp
// Rendering optimization settings
using UnityEngine;

public class RenderingOptimization : MonoBehaviour
{
    [Header("LOD Settings")]
    public float lodBias = 1.0f;
    public int maximumLODLevel = 0;
    
    [Header("Occlusion Culling")]
    public bool enableOcclusionCulling = true;
    
    [Header("Quality Settings")]
    public float shadowDistance = 50f;
    public int shadowResolution = 1024;
    public int shadowCascades = 2;
    
    [Header("Dynamic Batching")]
    public bool enableDynamicBatching = true;
    public bool enableStaticBatching = true;

    void Start()
    {
        ApplyOptimizations();
    }

    void ApplyOptimizations()
    {
        // Apply LOD bias
        QualitySettings.lodBias = lodBias;
        QualitySettings.maximumLODLevel = maximumLODLevel;
        
        // Configure shadow settings for performance
        QualitySettings.shadowDistance = shadowDistance;
        QualitySettings.shadowResolution = (ShadowResolution)shadowResolution;
        QualitySettings.shadowCascades = shadowCascades;
        
        // Note: Batching settings are project-wide and set in Quality Settings
    }
}
```

### 3.7.2 Physics Optimization

Optimizing physics simulation for real-time performance:

```csharp
// Physics optimization for robotics simulation
using UnityEngine;

public class PhysicsOptimization : MonoBehaviour
{
    [Header("Physics Settings")]
    public int solverIterations = 6; // Balance between stability and performance
    public int solverVelocityIterations = 1;
    public float sleepThreshold = 0.005f;
    public float contactOffset = 0.01f;
    
    [Header("Simulation Settings")]
    public int threadCount = 0; // 0 means use default
    public float fixedDeltaTime = 0.02f; // 50 Hz physics updates

    void Start()
    {
        OptimizePhysics();
    }

    void OptimizePhysics()
    {
        Physics.defaultSolverIterations = solverIterations;
        Physics.defaultSolverVelocityIterations = solverVelocityIterations;
        Physics.sleepThreshold = sleepThreshold;
        Physics.defaultContactOffset = contactOffset;
        
        // Set physics time step
        Time.fixedDeltaTime = fixedDeltaTime;
        
        // Adjust thread count if needed
        if (threadCount > 0)
        {
            Physics.defaultSolverThreadCount = threadCount;
        }
    }
}
```

## 3.8 Advanced Simulation Features

### 3.8.1 Domain Randomization

Implementing domain randomization to improve sim-to-real transfer:

```csharp
// Domain randomization implementation
using UnityEngine;
using System.Collections;

public class DomainRandomization : MonoBehaviour
{
    [Header("Material Randomization")]
    public Material[] baseMaterials;
    public Color[] randomColors;
    public float randomRoughnessRange = 0.5f;
    
    [Header("Lighting Randomization")]
    public Light[] lightsToRandomize;
    public float intensityRange = 0.5f;
    
    [Header("Object Randomization")]
    public GameObject[] objectsToRandomize;
    public float positionJitter = 0.1f;
    public float rotationJitter = 5f;
    
    [Header("Timing")]
    public float randomizationInterval = 10f; // seconds

    private float lastRandomization;

    void Start()
    {
        lastRandomization = Time.time;
        RandomizeEnvironment();
    }

    void Update()
    {
        if (Time.time - lastRandomization >= randomizationInterval)
        {
            RandomizeEnvironment();
            lastRandomization = Time.time;
        }
    }

    void RandomizeEnvironment()
    {
        RandomizeMaterials();
        RandomizeLighting();
        RandomizeObjects();
    }

    void RandomizeMaterials()
    {
        foreach (Material mat in baseMaterials)
        {
            if (mat != null)
            {
                // Randomize color
                if (randomColors.Length > 0)
                {
                    Color randomColor = randomColors[Random.Range(0, randomColors.Length)];
                    mat.color = randomColor;
                }
                
                // Randomize roughness
                float randomRoughness = Random.Range(0.5f, 0.5f + randomRoughnessRange);
                mat.SetFloat("_Metallic", 1.0f - randomRoughness); // Simplified
            }
        }
    }

    void RandomizeLighting()
    {
        foreach (Light light in lightsToRandomize)
        {
            if (light != null)
            {
                float randomIntensity = light.intensity * Random.Range(1f - intensityRange, 1f + intensityRange);
                light.intensity = Mathf.Clamp(randomIntensity, 0.1f, 5f);
                
                // Randomize color within reasonable ranges
                float hue, saturation, value;
                Color.RGBToHSV(light.color, out hue, out saturation, out value);
                hue = Mathf.Clamp01(hue + Random.Range(-0.1f, 0.1f));
                saturation = Mathf.Clamp01(saturation + Random.Range(-0.1f, 0.1f));
                light.color = Color.HSVToRGB(hue, saturation, value);
            }
        }
    }

    void RandomizeObjects()
    {
        foreach (GameObject obj in objectsToRandomize)
        {
            if (obj != null)
            {
                // Add small position jitter
                Vector3 randomPos = obj.transform.position;
                randomPos.x += Random.Range(-positionJitter, positionJitter);
                randomPos.y += Random.Range(-positionJitter, positionJitter);
                randomPos.z += Random.Range(-positionJitter, positionJitter);
                obj.transform.position = randomPos;
                
                // Add small rotation jitter
                Vector3 randomRot = obj.transform.eulerAngles;
                randomRot.x += Random.Range(-rotationJitter, rotationJitter);
                randomRot.y += Random.Range(-rotationJitter, rotationJitter);
                randomRot.z += Random.Range(-rotationJitter, rotationJitter);
                obj.transform.eulerAngles = randomRot;
            }
        }
    }
}
```

### 3.8.2 Synthetic Data Generation

Using Unity Perception for generating training data:

```csharp
// Synthetic data generation setup
using UnityEngine;
using Unity.Perception.GroundTruth;

[RequireComponent(typeof(PerceptionCamera))]
public class SyntheticDataGenerator : MonoBehaviour
{
    [Header("Dataset Configuration")]
    public string datasetName = "robotics_dataset";
    public int sequenceId = 0;
    public bool captureAtFixedIntervals = true;
    public float captureInterval = 0.5f; // seconds

    [Header("Annotation Types")]
    public bool captureBoundingBoxes = true;
    public bool captureSegmentation = true;
    public bool captureDepth = true;

    private PerceptionCamera perceptionCamera;
    private float lastCapture;
    private int frameCount = 0;

    void Start()
    {
        SetupPerceptionCamera();
        lastCapture = Time.time;
    }

    void Update()
    {
        if (captureAtFixedIntervals && Time.time - lastCapture >= captureInterval)
        {
            CaptureAnnotations();
            lastCapture = Time.time;
        }
    }

    void SetupPerceptionCamera()
    {
        perceptionCamera = GetComponent<PerceptionCamera>();
        if (perceptionCamera == null)
        {
            perceptionCamera = gameObject.AddComponent<PerceptionCamera>();
        }

        // Configure annotation types
        perceptionCamera.annotationCaptureSettings.boundingBox2D.enabled = captureBoundingBoxes;
        perceptionCamera.annotationCaptureSettings.instanceSegmentation.enabled = captureSegmentation;
        perceptionCamera.annotationCaptureSettings.depth.enabled = captureDepth;
    }

    void CaptureAnnotations()
    {
        // Capture annotations for this frame
        var captureFrame = new CaptureFrame
        {
            sequenceId = sequenceId,
            timestamp = Time.time,
            frameId = frameCount++
        };

        // In a real implementation, this would trigger annotation capture
        // and save to dataset in appropriate format
        perceptionCamera.CaptureFrame(captureFrame);

        Debug.Log($"Captured frame {captureFrame.frameId} for sequence {sequenceId}");
    }

    // Manual capture method
    public void ManualCapture()
    {
        CaptureAnnotations();
    }
}
```

## Chapter Summary

This chapter provided a comprehensive guide to using Unity for creating visualization-rich simulation environments for robotics applications. We covered Unity setup and configuration, robot model creation, environment design with realistic physics, sensor simulation implementations (camera, LIDAR, IMU), ROS integration using ROS-TCP-Connector, performance optimization techniques, and advanced features like domain randomization and synthetic data generation. Unity's strength lies in its ability to create photorealistic environments for perception-focused robotics applications.

## Key Terms
- Unity Robotics
- Photorealistic Rendering
- ROS-TCP-Connector
- Domain Randomization
- Synthetic Data Generation
- Physics Simulation
- Sensor Simulation
- Perception Training

## Exercises
1. Create a Unity scene with a simple robot model and export sensor data
2. Implement domain randomization techniques for a robotic perception task
3. Set up ROS communication between Unity and a ROS 2 system
4. Generate synthetic training data for a computer vision pipeline

## References
- Unity Robotics Hub: https://unity.com/solutions/robotics
- Unity Perception Package Documentation
- ROS-TCP-Connector GitHub Repository
- Unity ML-Agents Toolkit Documentation