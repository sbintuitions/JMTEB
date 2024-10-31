{
  mrtydi: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      query_prefix: 'Query: ' ,
      doc_prefix: 'Document: ',
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
