"""
Setup configuration for the Physical AI & Humanoid Robotics project
"""
from setuptools import setup, find_packages

setup(
    name="humanoid-robotics-course",
    version="1.0.0",
    description="Code and tools for the Physical AI & Humanoid Robotics course",
    author="Physical AI & Humanoid Robotics Course Team",
    author_email="info@physicalai-humanoid-robotics.org",
    packages=find_packages(),
    install_requires=[
        "rclpy>=3.0.0",
        "opencv-python>=4.5.0",
        "cv-bridge>=3.0.0",
        "numpy>=1.21.0",
        "openai>=1.0.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "transformers>=4.20.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0"
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'humanoid-controller = src.ros2_nodes.humanoid_controller:main',
            'humanoid-perception = src.ros2_nodes.humanoid_perception:main',
            'vla-agent = src.vla_agents.vla_agent:main',
        ],
    },
)