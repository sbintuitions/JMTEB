{
  nlp_journal_abs_intro: {
    class_path: 'RetrievalEvaluator',
    init_args: {
      local task = '与えられた自然言語処理学会の論文のアブストラクトから、その論文のタイトルを検索する',
      query_prefix: 'Instruct: %s\nQuery: ' % task,
      val_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'validation',
          name: 'nlp_journal_abs_intro-query',
        },
      },
      test_query_dataset: {
        class_path: 'HfRetrievalQueryDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'test',
          name: 'nlp_journal_abs_intro-query',
        },
      },
      doc_dataset: {
        class_path: 'HfRetrievalDocDataset',
        init_args: {
          path: 'sbintuitions/JMTEB',
          split: 'corpus',
          name: 'nlp_journal_abs_intro-corpus',
        },
      },
    },
  },
}
