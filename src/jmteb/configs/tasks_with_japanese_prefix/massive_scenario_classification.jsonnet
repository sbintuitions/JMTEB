{
  massive_scenario_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      local task = 'クエリとしてユーザーの発話が与えられたとき、ユーザーのシナリオを見つけてください。',
      prefix: 'Instruct: %s\nQuery: ' % task,
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'massive_scenario_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'massive_scenario_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'massive_scenario_classification',
        },
      },
    },
  },
}
