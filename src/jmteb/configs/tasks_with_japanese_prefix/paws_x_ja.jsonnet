{
  paws_x_ja: {
    class_path: 'PairClassificationEvaluator',
    init_args: {
      local task = '類似した意味を持つテキストを検索してください。',
      sentence1_prefix: 'Instruct: %s\nQuery: ' % task,
      val_dataset: {
        class_path: 'HfPairClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'paws_x_ja',
        },
      },
      test_dataset: {
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
