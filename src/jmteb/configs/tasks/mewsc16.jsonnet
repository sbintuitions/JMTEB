{
  mewsc16: {
    class_path: 'ClusteringEvaluator',
    init_args: {
      dataset: {
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
