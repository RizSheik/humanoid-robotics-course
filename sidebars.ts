import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  textbookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System',
      link: {
        type: 'generated-index',
        title: 'Module 1: The Robotic Nervous System',
        description: 'Covers the foundational elements of robot control and communication.',
        slug: '/module-1-the-robotic-nervous-system',
      },
      items: [
        'module-1-the-robotic-nervous-system/module-1-intro',
        'module-1-the-robotic-nervous-system/module-1-overview',
        'module-1-the-robotic-nervous-system/module-1-weekly-breakdown',
        'module-1-the-robotic-nervous-system/module-1-deep-dive',
        'module-1-the-robotic-nervous-system/module-1-practical-lab',
        'module-1-the-robotic-nervous-system/module-1-simulation',
        'module-1-the-robotic-nervous-system/module-1-assignment',
        'module-1-the-robotic-nervous-system/module-1-quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin',
      link: {
        type: 'generated-index',
        title: 'Module 2: The Digital Twin',
        description: 'Focuses on simulation, modeling, and virtual representation of robots.',
        slug: '/module-2-the-digital-twin',
      },
      items: [
        'module-2-the-digital-twin/module-2-intro',
        'module-2-the-digital-twin/module-2-overview',
        'module-2-the-digital-twin/module-2-weekly-breakdown',
        'module-2-the-digital-twin/module-2-deep-dive',
        'module-2-the-digital-twin/module-2-practical-lab',
        'module-2-the-digital-twin/module-2-simulation',
        'module-2-the-digital-twin/module-2-assignment',
        'module-2-the-digital-twin/module-2-quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain',
      link: {
        type: 'generated-index',
        title: 'Module 3: The AI-Robot Brain',
        description: 'Explores AI algorithms for robot decision-making and intelligence.',
        slug: '/module-3-the-ai-robot-brain',
      },
      items: [
        'module-3-the-ai-robot-brain/module-3-intro',
        'module-3-the-ai-robot-brain/module-3-overview',
        'module-3-the-ai-robot-brain/module-3-weekly-breakdown',
        'module-3-the-ai-robot-brain/module-3-deep-dive',
        'module-3-the-ai-robot-brain/module-3-practical-lab',
        'module-3-the-ai-robot-brain/module-3-simulation',
        'module-3-the-ai-robot-brain/module-3-assignment',
        'module-3-the-ai-robot-brain/module-3-quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems',
      link: {
        type: 'generated-index',
        title: 'Module 4: Vision-Language-Action Systems',
        description: 'Integrates vision, language understanding, and physical actions for advanced robotics.',
        slug: '/module-4-vision-language-action-systems',
      },
      items: [
        'module-4-vision-language-action-systems/module-4-intro',
        'module-4-vision-language-action-systems/module-4-overview',
        'module-4-vision-language-action-systems/module-4-weekly-breakdown',
        'module-4-vision-language-action-systems/module-4-deep-dive',
        'module-4-vision-language-action-systems/module-4-practical-lab',
        'module-4-vision-language-action-systems/module-4-simulation',
        'module-4-vision-language-action-systems/module-4-assignment',
        'module-4-vision-language-action-systems/module-4-quiz',
      ],
    },
    {
      type: 'category',
      label: 'Capstone: The Autonomous Humanoid',
      link: {
        type: 'generated-index',
        title: 'Capstone: The Autonomous Humanoid',
        description: 'Integrates all concepts from previous modules into a comprehensive autonomous humanoid project.',
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
        title: 'Reference Materials',
        description: 'Supplementary information and reference materials for the course.',
        slug: '/appendices',
      },
      items: [
        'appendices/hardware-requirements',
        'appendices/lab-architecture',
        'appendices/cloud-vs-onprem',
      ],
    },
  ],
};

export default sidebars;