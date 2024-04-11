{
  jsts: {
    class_path: 'STSEvaluator',
    init_args: {
      test_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jsts',
        },
      },
      dev_dataset: {
        class_path: 'HfSTSDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'jsts',
        },
      },
    },
  },
}
