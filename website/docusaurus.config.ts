import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Transcode',
  tagline: 'Memory-safe, high-performance media transcoding for Rust',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://transcode.dev',
  baseUrl: '/',

  organizationName: 'transcode',
  projectName: 'transcode',

  onBrokenLinks: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
    mermaid: true,
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
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/transcode/transcode/tree/main/website/',
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/transcode/transcode/tree/main/website/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        language: ['en'],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  themeConfig: {
    image: 'img/transcode-social-card.svg',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    announcementBar: {
      id: 'star_us',
      content:
        '⭐ If you like Transcode, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/transcode/transcode">GitHub</a>!',
      backgroundColor: '#3b82f6',
      textColor: '#ffffff',
      isCloseable: true,
    },
    navbar: {
      title: 'Transcode',
      logo: {
        alt: 'Transcode Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/reference/cli',
          label: 'CLI',
          position: 'left',
        },
        {
          to: '/docs/reference/api',
          label: 'API',
          position: 'left',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://docs.rs/transcode',
          label: 'docs.rs',
          position: 'right',
        },
        {
          href: 'https://crates.io/crates/transcode',
          label: 'crates.io',
          position: 'right',
        },
        {
          href: 'https://github.com/transcode/transcode',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting-started/installation',
            },
            {
              label: 'Core Concepts',
              to: '/docs/core-concepts/architecture',
            },
            {
              label: 'Guides',
              to: '/docs/guides/basic-transcoding',
            },
          ],
        },
        {
          title: 'API',
          items: [
            {
              label: 'CLI Reference',
              to: '/docs/reference/cli',
            },
            {
              label: 'Rust API',
              href: 'https://docs.rs/transcode',
            },
            {
              label: 'Python API',
              to: '/docs/integrations/python',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/transcode/transcode/discussions',
            },
            {
              label: 'Discord',
              href: 'https://discord.gg/transcode',
            },
            {
              label: 'Twitter',
              href: 'https://twitter.com/transcode_rs',
            },
            {
              label: 'Code of Conduct',
              href: 'https://github.com/transcode/transcode/blob/main/CODE_OF_CONDUCT.md',
            },
            {
              label: 'Contributing',
              to: '/docs/advanced/contributing',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/transcode/transcode',
            },
            {
              label: 'Changelog',
              to: '/docs/reference/changelog',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Transcode Contributors. Licensed under MIT/Apache-2.0.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml', 'bash', 'python', 'javascript', 'typescript', 'json', 'yaml', 'c', 'cpp'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
