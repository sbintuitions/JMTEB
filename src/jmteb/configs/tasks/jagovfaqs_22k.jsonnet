{
  jagovfaqs_22k: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_dataset: {
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
