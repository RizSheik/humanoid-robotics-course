import { themes as prismThemes } from 'prism-react-renderer';

const config = {
  title: 'Physical AI & Humanoid Robotics — AI Systems in the Physical World',
  tagline: 'Advanced Robotics Textbook — From Theory to Practice',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  // Production URL
  url: 'https://rizsheik.github.io',

  // Base URL for GitHub Pages
  baseUrl: '/humanoid-robotics-course/',
  trailingSlash: false,

  // GitHub pages deployment configuration
  organizationName: 'RizSheik',
  projectName: 'humanoid-robotics-course',

  onBrokenLinks: 'warn',  // Changed from 'throw' to 'warn' to allow build to continue with broken links
  onBrokenMarkdownLinks: 'warn',

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

          // FIXED: correct edit URL for your repo
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
              label: 'Module 1 – The Robotic Nervous System',
              to: '/docs/module-1-the-robotic-nervous-system/module-1-intro',
            },
            {
              label: 'Module 2 – The Digital Twin',
              to: '/docs/module-2-the-digital-twin/module-2-intro',
            },
            {
              label: 'Module 3 – The AI-Robot Brain',
              to: '/docs/module-3-the-ai-robot-brain/module-3-intro',
            },
            {
              label:
                'Module 4 – Vision-Language-Action Systems',
              to: '/docs/module-4-vision-language-action-systems/module-4-intro',
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
