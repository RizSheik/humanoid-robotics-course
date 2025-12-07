import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System',
      link: {
        type: 'generated-index',
        title: 'Module 1 Overview',
        description: 'Covers the foundational elements of robot control and communication.',
        slug: '/category/module-1-the-robotic-nervous-system',
      },
      items: [
        'module-1-the-robotic-nervous-system/intro',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin',
      link: {
        type: 'generated-index',
        title: 'Module 2 Overview',
        description: 'Focuses on simulation, modeling, and virtual representation of robots.',
        slug: '/category/module-2-the-digital-twin',
      },
      items: [
        'module-2-the-digital-twin/intro',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI Robot Brain',
      link: {
        type: 'generated-index',
        title: 'Module 3 Overview',
        description: 'Explores AI algorithms for robot decision-making and intelligence.',
        slug: '/category/module-3-the-ai-robot-brain',
      },
      items: [
        'module-3-the-ai-robot-brain/intro',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems',
      link: {
        type: 'generated-index',
        title: 'Module 4 Overview',
        description: 'Integrates vision, language understanding, and physical actions for advanced robotics.',
        slug: '/category/module-4-vision-language-action-systems',
      },
      items: [
        'module-4-vision-language-action-systems/intro',
      ],
    },
  ],
};

export default sidebars;