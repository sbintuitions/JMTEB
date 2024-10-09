{
  amazon_review_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      local task = '与えられたアマゾンのレビューを適切な評価カテゴリに分類してください。',
      prefix: 'Instruct: %s\nQuery: ' % task,
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'amazon_review_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'amazon_review_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'amazon_review_classification',
        },
      },
    },
  },
}
