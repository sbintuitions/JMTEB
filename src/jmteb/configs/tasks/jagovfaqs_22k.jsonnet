{
  jagovfaqs_22k: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'jagovfaqs_22k-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'jagovfaqs_22k-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'jagovfaqs_22k-corpus',
        },
      },
    },
  },
}
