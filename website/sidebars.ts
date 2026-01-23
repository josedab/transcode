import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/why-transcode',
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/first-transcode',
        'getting-started/faq',
      ],
    },
    {
      type: 'category',
      label: 'Core Concepts',
      items: [
        'core-concepts/architecture',
        'core-concepts/codecs-containers',
        'core-concepts/frames-packets',
        'core-concepts/pipeline',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        'guides/basic-transcoding',
        'guides/streaming-output',
        'guides/gpu-acceleration',
        'guides/ai-enhancement',
        'guides/filter-chains',
        'guides/quality-metrics',
        'guides/distributed-processing',
        'guides/error-handling',
        'guides/docker-deployment',
      ],
    },
    {
      type: 'category',
      label: 'Integrations',
      items: [
        'integrations/python',
        'integrations/nodejs',
        'integrations/webassembly',
        'integrations/c-api',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli',
        'reference/api',
        'reference/codec-parameters',
        'reference/codecs-matrix',
        'reference/configuration',
        'reference/benchmarks',
        'reference/changelog',
      ],
    },
    {
      type: 'category',
      label: 'Comparison',
      items: [
        'comparison/vs-ffmpeg',
        'comparison/vs-gstreamer',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      items: [
        'advanced/custom-codecs',
        'advanced/simd-optimization',
        'advanced/hardware-acceleration',
        'advanced/contributing',
      ],
    },
  ],
};

export default sidebars;
