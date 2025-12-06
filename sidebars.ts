import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Module 1 - Foundational',
      link: {
        type: 'generated-index',
        title: 'Module 1 Overview',
        description: 'Foundational concepts for Humanoid Robotics.',
        slug: '/module-1-foundational',
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
      label: 'Module 2 - Perception',
      link: {
        type: 'generated-index',
        title: 'Module 2 Overview',
        description: 'Perception techniques for Humanoid Robotics.',
        slug: '/module-2-perception',
      },
      items: [
        'module-2-perception/chapter-5-sensors',
        'module-2-perception/chapter-6-state-estimation',
        'module-2-perception/chapter-7-localization',
      ],
    },
    {
      type: 'category',
      label: 'Module 3 - Manipulation',
      link: {
        type: 'generated-index',
        title: 'Module 3 Overview',
        description: 'Manipulation and control strategies for Humanoid Robotics.',
        slug: '/module-3-manipulation',
      },
      items: [
        'module-3-manipulation/chapter-8-motion-planning',
        'module-3-manipulation/chapter-9-manipulation-control',
        'module-3-manipulation/chapter-10-force-control',
      ],
    },
    {
      type: 'category',
      label: 'Module 4 - Advanced',
      link: {
        type: 'generated-index',
        title: 'Module 4 Overview',
        description: 'Advanced topics in Humanoid Robotics.',
        slug: '/module-4-advanced',
      },
      items: [
        'module-4-advanced/chapter-11-human-robot-interaction',
        'module-4-advanced/chapter-12-collaborative-robotics',
        'module-4-advanced/chapter-13-ai-in-robotics',
      ],
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
