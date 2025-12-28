import { themes as prismThemes } from 'prism-react-renderer';

// ✅ Detect GitHub Pages build
const isGithubPages = process.env.GITHUB_ACTIONS === 'true';

const config = {
  title: 'Physical AI & Humanoid Robotics — AI Systems in the Physical World',
  tagline: 'Advanced Robotics Textbook — From Theory to Practice',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // Production URL (GitHub Pages domain)
  url: 'https://rizsheik.github.io',

  // ✅ AUTO baseUrl (GitHub Pages vs Vercel)
  baseUrl: isGithubPages ? '/humanoid-robotics-course/' : '/',
  trailingSlash: false,

  // GitHub pages deployment configuration
  organizationName: 'RizSheik',
  projectName: 'humanoid-robotics-course',

  onBrokenLinks: 'warn',

  // ✅ Docusaurus v4 compatible markdown config
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.ts'),
          editUrl:
            'https://github.com/RizSheik/humanoid-robotics-course/edit/main/',
        },

        blog: false,

        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',

    colorMode: {
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Humanoid Robotics Logo',
        src: 'img/robot-head.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'textbookSidebar',
          position: 'left',
          label: 'Textbook Modules',
        },
        {
          href: 'https://github.com/RizSheik/humanoid-robotics-course',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Textbook',
          items: [
            {
              label: 'Module 1: Physical AI Foundations',
              to: '/docs/module-1-physical-ai-foundations/overview',
            },
            {
              label: 'Module 2: ROS 2 Fundamentals',
              to: '/docs/module-2-ros-2-fundamentals/overview',
            },
            {
              label: 'Module 3: Digital Twin Simulation',
              to: '/docs/module-3-digital-twin-simulation/overview',
            },
            {
              label: 'Module 4: AI Robot Brain',
              to: '/docs/module-4-ai-robot-brain/overview',
            },
            {
              label: 'Capstone: The Autonomous Humanoid',
              to: '/docs/capstone-the-autonomous-humanoid/capstone-overview',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Appendices',
              to: '/docs/appendices/hardware-requirements',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/RizSheik/humanoid-robotics-course',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics — AI Systems in the Physical World. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  },
};

export default config;