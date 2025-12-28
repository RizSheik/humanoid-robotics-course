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
        description: 'Covers the foundational elements of Physical AI and embodied intelligence.',
        slug: '/module-1-physical-ai-foundations',
      },
      items: [
        'module-1-physical-ai-foundations/overview',
        'module-1-physical-ai-foundations/chapter-1',
        'module-1-physical-ai-foundations/chapter-2',
        'module-1-physical-ai-foundations/chapter-3',
        'module-1-physical-ai-foundations/chapter-4',
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
        description: 'Covers the fundamentals of Robot Operating System 2 (ROS 2) including topics, services, actions, and distributed systems.',
        slug: '/module-2-ros-2-fundamentals',
      },
      items: [
        'module-2-ros-2-fundamentals/overview',
        'module-2-ros-2-fundamentals/chapter-1',
        'module-2-ros-2-fundamentals/chapter-2',
        'module-2-ros-2-fundamentals/chapter-3',
        'module-2-ros-2-fundamentals/chapter-4',
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
        description: 'Focuses on simulation environments including Gazebo and Unity for robotics development.',
        slug: '/module-3-digital-twin-simulation',
      },
      items: [
        'module-3-digital-twin-simulation/overview',
        'module-3-digital-twin-simulation/chapter-1',
        'module-3-digital-twin-simulation/chapter-2',
        'module-3-digital-twin-simulation/chapter-3',
        'module-3-digital-twin-simulation/chapter-4',
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
        description: 'Explores AI algorithms and NVIDIA Isaac Platform for robot decision-making and intelligence.',
        slug: '/module-4-ai-robot-brain',
      },
      items: [
        'module-4-ai-robot-brain/overview',
        'module-4-ai-robot-brain/chapter-1',
        'module-4-ai-robot-brain/chapter-2',
        'module-4-ai-robot-brain/chapter-3',
        'module-4-ai-robot-brain/chapter-4',
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
        description: 'Integrates all concepts from previous modules into a comprehensive autonomous humanoid project with conversational AI.',
        slug: '/capstone-the-autonomous-humanoid',
      },
      items: [
        'capstone-the-autonomous-humanoid/capstone-overview',
        'capstone-the-autonomous-humanoid/capstone-weekly-breakdown',
        'capstone-the-autonomous-humanoid/capstone-deep-dive',
        'capstone-the-autonomous-humanoid/capstone-practical-lab',
        'capstone-the-autonomous-humanoid/capstone-simulation',
        'capstone-the-autonomous-humanoid/capstone-assignment',
        'capstone-the-autonomous-humanoid/capstone-quiz',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      link: {
        type: 'generated-index',
        description: 'Supplementary information and reference materials for the course.',
        slug: '/appendices',
      },
      items: [
        'appendices/hardware-requirements',
        'appendices/lab-infrastructure',
        'appendices/safety-protocols',
        'appendices/cloud-vs-onprem',
      ],
    },
  ],
};

export default sidebars;