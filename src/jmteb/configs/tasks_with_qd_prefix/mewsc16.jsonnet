{
  mewsc16: {
    class_path: 'ClusteringEvaluator',
    init_args: {
      prefix: 'Query: ',
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
