{
  mrtydi: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      local task = '与えられたクエリに対する回答として、Wikipedia の中から関連する文章を検索する',
      query_prefix: 'Instruct: %s\nQuery: ' % task,
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'mrtydi-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'mrtydi-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'mrtydi-corpus',
        },
      },
      doc_chunk_size: 10000,
    },
  },
}
