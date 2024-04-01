{
  jsts: {
    class_path: 'STSEvaluator',
    init_args: {
      dataset: {
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
