{
  jsts: {
    class_path: 'STSEvaluator',
    init_args: {
      local task = '類似した意味を持つテキストを検索する',
      sentence1_prefix: 'Instruct: %s\nQuery: ' % task,
      val_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'jsts',
        },
      },
      test_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jsts',
        },
      },
    },
  },
}
