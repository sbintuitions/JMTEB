{
  massive_intent_classification: {
    class_path: 'ClassificationEvaluator',
    init_args: {
      local task = 'クエリとしてユーザー発話が与えられたとき、ユーザーの意図を見つ毛てください。',
      prefix: 'Instruct: %s\nQuery: ' % task,
      train_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'train',
          name: 'massive_intent_classification',
        },
      },
      val_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'massive_intent_classification',
        },
      },
      test_dataset: {
        class_path: 'HfClassificationDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'massive_intent_classification',
        },
      },
    },
  },
}
