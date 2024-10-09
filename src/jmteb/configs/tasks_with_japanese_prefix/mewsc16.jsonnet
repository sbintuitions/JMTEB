{
  mewsc16: {
    class_path: 'ClusteringEvaluator',
    init_args: {
      local task = '与えられた文章の話題やテーマを特定してください。',
      prefix: 'Instruct: %s\nQuery: ' % task,
      val_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'mewsc16_ja',
        },
      },
      test_dataset: {
        class_path: 'HfClusteringDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'mewsc16_ja',
        },
      },
    },
  },
}
