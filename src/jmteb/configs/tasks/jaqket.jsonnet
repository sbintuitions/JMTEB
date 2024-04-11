{
  jaqket: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jaqket-query',
        },
      },
      dev_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
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
