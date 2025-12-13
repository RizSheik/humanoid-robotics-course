import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  textbookSidebar: [
    'introduction',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System',
      link: {
        type: 'doc',
        id: 'module-1-the-robotic-nervous-system/overview',
      },
      items: [
        'module-1-the-robotic-nervous-system/chapter-1',
        'module-1-the-robotic-nervous-system/chapter-2',
        'module-1-the-robotic-nervous-system/chapter-3',
        'module-1-the-robotic-nervous-system/chapter-4',
        'module-1-the-robotic-nervous-system/overview',
        'module-1-the-robotic-nervous-system/deep-dive',
        'module-1-the-robotic-nervous-system/practical-lab',
        'module-1-the-robotic-nervous-system/simulation',
        'module-1-the-robotic-nervous-system/assignment',
        'module-1-the-robotic-nervous-system/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin',
      link: {
        type: 'doc',
        id: 'module-2-the-digital-twin/overview',
      },
      items: [
        'module-2-the-digital-twin/chapter-1',
        'module-2-the-digital-twin/chapter-2',
        'module-2-the-digital-twin/chapter-3',
        'module-2-the-digital-twin/chapter-4',
        'module-2-the-digital-twin/overview',
        'module-2-the-digital-twin/deep-dive',
        'module-2-the-digital-twin/practical-lab',
        'module-2-the-digital-twin/simulation',
        'module-2-the-digital-twin/assignment',
        'module-2-the-digital-twin/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI Robot Brain',
      link: {
        type: 'doc',
        id: 'module-3-the-ai-robot-brain/overview',
      },
      items: [
        'module-3-the-ai-robot-brain/chapter-1',
        'module-3-the-ai-robot-brain/chapter-2',
        'module-3-the-ai-robot-brain/chapter-3',
        'module-3-the-ai-robot-brain/chapter-4',
        'module-3-the-ai-robot-brain/overview',
        'module-3-the-ai-robot-brain/deep-dive',
        'module-3-the-ai-robot-brain/practical-lab',
        'module-3-the-ai-robot-brain/simulation',
        'module-3-the-ai-robot-brain/assignment',
        'module-3-the-ai-robot-brain/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision Language Action Systems',
      link: {
        type: 'doc',
        id: 'module-4-vision-language-action-systems/overview',
      },
      items: [
        'module-4-vision-language-action-systems/chapter-1',
        'module-4-vision-language-action-systems/chapter-2',
        'module-4-vision-language-action-systems/chapter-3',
        'module-4-vision-language-action-systems/chapter-4',
        'module-4-vision-language-action-systems/overview',
        'module-4-vision-language-action-systems/deep-dive',
        'module-4-vision-language-action-systems/practical-lab',
        'module-4-vision-language-action-systems/simulation',
        'module-4-vision-language-action-systems/assignment',
        'module-4-vision-language-action-systems/quiz',
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'appendices/hardware-requirements',
        'appendices/lab-architecture',
        'appendices/cloud-vs-onprem',
      ],
    },
  ],
};

export default sidebars;