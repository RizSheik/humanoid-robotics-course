import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  textbookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: Physical AI Foundations',
      link: {
        type: 'generated-index',
        title: 'Module 1: Physical AI Foundations',
        description: 'Covers the foundational elements of Physical AI and embodied intelligence.',
        slug: '/module-1-physical-ai-foundations',
      },
      items: [
        'module-1-physical-ai-foundations/overview',
        'module-1-physical-ai-foundations/weekly-breakdown',
        'module-1-physical-ai-foundations/deep-dive',
        'module-1-physical-ai-foundations/practical-lab',
        'module-1-physical-ai-foundations/simulation',
        'module-1-physical-ai-foundations/assignment',
        'module-1-physical-ai-foundations/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: ROS 2 Fundamentals',
      link: {
        type: 'generated-index',
        title: 'Module 2: ROS 2 Fundamentals',
        description: 'Covers the fundamentals of Robot Operating System 2 (ROS 2) including topics, services, actions, and distributed systems.',
        slug: '/module-2-ros-2-fundamentals',
      },
      items: [
        'module-2-ros-2-fundamentals/overview',
        'module-2-ros-2-fundamentals/weekly-breakdown',
        'module-2-ros-2-fundamentals/deep-dive',
        'module-2-ros-2-fundamentals/practical-lab',
        'module-2-ros-2-fundamentals/simulation',
        'module-2-ros-2-fundamentals/assignment',
        'module-2-ros-2-fundamentals/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Digital Twin Simulation',
      link: {
        type: 'generated-index',
        title: 'Module 3: Digital Twin Simulation',
        description: 'Focuses on simulation environments including Gazebo and Unity for robotics development.',
        slug: '/module-3-digital-twin-simulation',
      },
      items: [
        'module-3-digital-twin-simulation/overview',
        'module-3-digital-twin-simulation/weekly-breakdown',
        'module-3-digital-twin-simulation/deep-dive',
        'module-3-digital-twin-simulation/practical-lab',
        'module-3-digital-twin-simulation/simulation',
        'module-3-digital-twin-simulation/assignment',
        'module-3-digital-twin-simulation/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: AI Robot Brain',
      link: {
        type: 'generated-index',
        title: 'Module 4: AI Robot Brain',
        description: 'Explores AI algorithms and NVIDIA Isaac Platform for robot decision-making and intelligence.',
        slug: '/module-4-ai-robot-brain',
      },
      items: [
        'module-4-ai-robot-brain/overview',
        'module-4-ai-robot-brain/weekly-breakdown',
        'module-4-ai-robot-brain/deep-dive',
        'module-4-ai-robot-brain/practical-lab',
        'module-4-ai-robot-brain/simulation',
        'module-4-ai-robot-brain/assignment',
        'module-4-ai-robot-brain/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Capstone: The Autonomous Humanoid',
      link: {
        type: 'generated-index',
        title: 'Capstone: The Autonomous Humanoid',
        description: 'Integrates all concepts from previous modules into a comprehensive autonomous humanoid project with conversational AI.',
        slug: '/capstone-humanoid-with-conversational-ai',
      },
      items: [
        'capstone-humanoid-with-conversational-ai/overview',
        'capstone-humanoid-with-conversational-ai/weekly-breakdown',
        'capstone-humanoid-with-conversational-ai/deep-dive',
        'capstone-humanoid-with-conversational-ai/practical-lab',
        'capstone-humanoid-with-conversational-ai/simulation',
        'capstone-humanoid-with-conversational-ai/assignment',
        'capstone-humanoid-with-conversational-ai/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      link: {
        type: 'generated-index',
        title: 'Reference Materials',
        description: 'Supplementary information and reference materials for the course.',
        slug: '/appendices',
      },
      items: [
        'appendices/hardware-requirements',
        'appendices/lab-infrastructure',
        'appendices/safety-protocols',
      ],
    },
  ],
};

export default sidebars;