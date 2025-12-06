<<<<<<< HEAD
import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Humanoid Robotics Course',
  tagline: 'Advanced Robotics — Step by Step',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://rizsheik.github.io',
  baseUrl: '/humanoid-robotics-course/',
  trailingSlash: false,

  organizationName: 'RizSheik',
  projectName: 'humanoid-robotics-course',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

=======
import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'My Site',
  tagline: 'Dinosaurs are cool',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://anthropics.github.io/humanoid-robotics-course/',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/humanoid-robotics-course/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'anthropics', // Usually your GitHub org/user name.
  projectName: 'humanoid-robotics-course', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
<<<<<<< HEAD
          sidebarPath: require.resolve('./sidebars.ts'),
          editUrl: 'https://github.com/RizSheik/humanoid-robotics-course/tree/main/',
        },

=======
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/anthropics/humanoid-robotics-course/tree/main/my-book/',
        },
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
<<<<<<< HEAD
          editUrl: 'https://github.com/RizSheik/humanoid-robotics-course/tree/main/',
        },

        theme: {
          customCss: require.resolve('./src/css/custom.css'),
=======
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/anthropics/humanoid-robotics-course/tree/main/my-book/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
<<<<<<< HEAD
    image: 'img/docusaurus-social-card.jpg',

    colorMode: {
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Humanoid Robotics',
      logo: {
        alt: 'Humanoid Robotics Logo',
=======
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'My Site',
      logo: {
        alt: 'My Site Logo',
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
<<<<<<< HEAD
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Course Modules',
        },
        { to: '/blog', label: 'Blog', position: 'left' },
        {
          href: 'https://github.com/RizSheik/humanoid-robotics-course',
=======
          sidebarId: 'textbookSidebar',
          position: 'left',
          label: 'Tutorial',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/facebook/docusaurus',
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
<<<<<<< HEAD

=======
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
<<<<<<< HEAD
              label: 'Module 1 – Intro',
              to: '/docs/module-1-foundational/chapter-1-intro',
=======
              label: 'Tutorial',
              to: '/docs/intro',
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/docusaurus',
            },
            {
              label: 'Discord',
              href: 'https://discordapp.com/invite/docusaurus',
            },
            {
              label: 'X',
              href: 'https://x.com/docusaurus',
            },
          ],
        },
        {
          title: 'More',
          items: [
<<<<<<< HEAD
            { label: 'Blog', to: '/blog' },
            {
              label: 'GitHub',
              href: 'https://github.com/RizSheik/humanoid-robotics-course',
=======
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/facebook/docusaurus',
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
            },
          ],
        },
      ],
<<<<<<< HEAD
      copyright: `Copyright © ${new Date().getFullYear()} Humanoid Robotics Course. Built with Docusaurus.`,
    },

=======
      copyright: `Copyright © ${new Date().getFullYear()} My Project, Inc. Built with Docusaurus.`,
    },
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
<<<<<<< HEAD
  },
};

export default config;
=======
  } satisfies Preset.ThemeConfig,
};

export default config;
>>>>>>> 7569626419039b5d4323869afcce81ee346adf05
