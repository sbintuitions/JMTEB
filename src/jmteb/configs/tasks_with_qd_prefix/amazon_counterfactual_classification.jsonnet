{
  amazon_counterfactual_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      local task = '与えられたアマゾンのお客様レビューのテキストを反事実的か反事実的でないかに分類してください',
      prefix: 'Query: ',
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'amazon_counterfactual_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'amazon_counterfactual_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'amazon_counterfactual_classification',
        },
      },
    },
  },
}