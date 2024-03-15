{
  paws_x_ja: {
    class_path: 'PairClassificationEvaluator',
    init_args: {
      dataset: {
        class_path: 'HfPairClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'paws_x_ja',
        },
      },
    },
  },
}
