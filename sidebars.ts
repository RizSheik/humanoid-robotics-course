import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  textbookSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Capstone: The Autonomous Humanoid',
      link: {
        type: 'generated-index',
        title: 'Capstone: The Autonomous Humanoid',
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
        title: 'Reference Materials',
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