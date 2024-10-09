{
  jagovfaqs_22k: {
    class_path: 'RetrievalEvaluator',
    local task = '与えられた質問に対する回答として、日本の行政機関のFAQサイトから回答となる文章を検索してください。',
    init_args: {
      query_prefix: 'Instruct: %s\nQuery: ' % task,
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
