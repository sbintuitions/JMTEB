{
  esci: {
    class_path: 'RerankingEvaluator',
    init_args: {
      query_dataset: {
        class_path: 'HfRerankingQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'esci-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRerankingDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'esci-corpus',
        },
      },
    },
  },
}
