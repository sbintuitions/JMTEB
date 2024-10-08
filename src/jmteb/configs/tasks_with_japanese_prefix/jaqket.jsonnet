{
  jaqket: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      local task = '与えられたクイズに対する回答として、Wikipedia の中から関連する文章を検索する',
      query_prefix: 'Instruct: %s\nQuery: ' % task,
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'jaqket-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jaqket-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'jaqket-corpus',
        },
      },
    },
  },
}
