{
  livedoor_news: {
    class_path: 'ClusteringEvaluator',
    init_args: {
      local task = '与えられたニュース記事の話題やテーマを特定してください。',
      prefix: 'Instruct: %s\nQuery: ' % task,
      val_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'livedoor_news',
        },
      },
      test_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'livedoor_news',
        },
      },
    },
  },
}
