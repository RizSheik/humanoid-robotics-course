import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  textbookSidebar: [
    {
      type: 'category',
      label: 'Module 1: Foundational Robotics & AI Concepts',
      link: {
        type: 'generated-index',
        title: 'Module 1 Overview',
        description: 'Introduction to foundational concepts in robotics and AI.',
        slug: '/category/module-1-foundational',
      },
      items: [
        'module-1-foundational/chapter-1-intro',
        'module-1-foundational/chapter-2-math-fundamentals',
        'module-1-foundational/chapter-3-robot-kinematics',
        'module-1-foundational/chapter-4-robot-dynamics',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Robot Perception & State Estimation',
      link: {
        type: 'generated-index',
        title: 'Module 2 Overview',
        description: 'Explore how robots perceive their environment and estimate their state.',
        slug: '/category/module-2-perception',
      },
      items: [
        'module-2-perception/chapter-5-sensors',
        'module-2-perception/chapter-6-state-estimation',
        'module-2-perception/chapter-7-localization',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: Robot Manipulation & Control',
      link: {
        type: 'generated-index',
        title: 'Module 3 Overview',
        description: 'Understand the principles of robot manipulation and control.',
        slug: '/category/module-3-manipulation',
      },
      items: [
        'module-3-manipulation/chapter-8-motion-planning',
        'module-3-manipulation/chapter-9-manipulation-control',
        'module-3-manipulation/chapter-10-force-control',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Human-Robot Interaction & Advanced Topics',
      link: {
        type: 'generated-index',
        title: 'Module 4 Overview',
        description: 'Advanced topics in human-robot interaction and future trends.',
        slug: '/category/module-4-advanced',
      },
      items: [
        'module-4-advanced/chapter-11-human-robot-interaction',
        'module-4-advanced/chapter-12-collaborative-robotics',
        'module-4-advanced/chapter-13-ai-in-robotics',
      ],
    },
  ],
};

export default sidebars;