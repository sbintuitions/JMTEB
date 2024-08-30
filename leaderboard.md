# Leaderboard
This leaderboard shows the results stored under `docs/results`. The scores are all multiplied by 100.

## Summary

The summary shows the average scores within each task.

| Model                                         | Avg.      | Retrieval   | STS       | Classification   | Reranking   | Clustering   | PairClassification   |
|:----------------------------------------------|:----------|:------------|:----------|:-----------------|:------------|:-------------|:---------------------|
| OpenAI/text-embedding-3-large                 | **73.97** | **74.48**   | 82.52     | **77.58**        | **93.58**   | **53.32**    | 62.35                |
| intfloat/multilingual-e5-large                | 71.65     | 70.98       | 79.70     | 72.89            | 92.96       | 51.24        | 62.15                |
| OpenAI/text-embedding-3-small                 | 70.86     | 66.39       | 79.46     | 73.06            | 92.92       | 51.06        | 62.27                |
| pkshatech/GLuCoSE-base-ja                     | 70.44     | 59.02       | 78.71     | 76.82            | 91.90       | 49.78        | **66.39**            |
| intfloat/multilingual-e5-base                 | 70.12     | 68.21       | 79.84     | 69.30            | 92.85       | 48.26        | 62.26                |
| intfloat/multilingual-e5-small                | 69.52     | 67.27       | 80.07     | 67.62            | 93.03       | 46.91        | 62.19                |
| OpenAI/text-embedding-ada-002                 | 69.48     | 64.38       | 79.02     | 69.75            | 93.04       | 48.30        | 62.40                |
| cl-nagoya/sup-simcse-ja-base                  | 68.56     | 49.64       | 82.05     | 73.47            | 91.83       | 51.79        | 62.57                |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup    | 66.89     | 47.38       | 78.99     | 73.13            | 91.30       | 48.25        | 62.27                |
| oshizo/sbert-jsnli-luke-japanese-base-lite    | 66.75     | 43.00       | 76.60     | 76.61            | 91.56       | 50.33        | 62.38                |
| cl-nagoya/sup-simcse-ja-large                 | 66.51     | 37.62       | **83.18** | 73.73            | 91.48       | 50.56        | 62.51                |
| cl-nagoya/unsup-simcse-ja-large               | 66.27     | 40.53       | 80.56     | 74.66            | 90.95       | 48.41        | 62.49                |
| MU-Kindai/Japanese-SimCSE-BERT-base-unsup     | 66.23     | 46.36       | 77.49     | 73.30            | 91.16       | 46.68        | 62.38                |
| MU-Kindai/Japanese-SimCSE-BERT-large-sup      | 65.28     | 40.82       | 78.28     | 73.47            | 90.95       | 45.81        | 62.35                |
| MU-Kindai/Japanese-MixCSE-BERT-base           | 65.14     | 42.59       | 77.05     | 72.90            | 91.01       | 44.95        | 62.33                |
| cl-nagoya/unsup-simcse-ja-base                | 65.07     | 40.23       | 78.72     | 73.07            | 91.16       | 44.77        | 62.44                |
| MU-Kindai/Japanese-DiffCSE-BERT-base          | 64.77     | 41.79       | 75.50     | 73.77            | 90.95       | 44.22        | 62.38                |
| sentence-transformers/LaBSE                   | 64.70     | 40.12       | 76.56     | 72.66            | 91.63       | 44.88        | 62.33                |
| pkshatech/simcse-ja-bert-base-clcmlp          | 64.42     | 37.00       | 76.80     | 71.30            | 91.49       | 47.53        | 62.40                |
| MU-Kindai/Japanese-SimCSE-BERT-base-sup       | 64.15     | 41.32       | 74.66     | 72.76            | 90.66       | 43.11        | 62.37                |
| colorfulscoop/sbert-base-ja                   | 58.85     | 16.52       | 70.42     | 69.07            | 89.97       | 44.81        | 62.31                |
| sentence-transformers/stsb-xlm-r-multilingual | 58.01     | 21.00       | 75.40     | 71.84            | 90.20       | 27.46        | 62.20                |

## Retrieval
| Model                                         | Avg.      | jagovfaqs_22k<br>(ndcg@10)   | jaqket<br>(ndcg@10)   | mrtydi<br>(ndcg@10)   | nlp_journal_abs_intro<br>(ndcg@10)   | nlp_journal_title_abs<br>(ndcg@10)   | nlp_journal_title_intro<br>(ndcg@10)   |
|:----------------------------------------------|:----------|:-----------------------------|:----------------------|:----------------------|:-------------------------------------|:-------------------------------------|:---------------------------------------|
| OpenAI/text-embedding-3-large                 | **74.48** | **72.41**                    | 48.21                 | 34.88                 | **99.33**                            | **96.55**                            | **95.47**                              |
| intfloat/multilingual-e5-large                | 70.98     | 70.30                        | **58.78**             | **43.63**             | 86.00                                | 94.70                                | 72.48                                  |
| intfloat/multilingual-e5-base                 | 68.21     | 65.34                        | 50.67                 | 38.38                 | 87.10                                | 94.73                                | 73.05                                  |
| intfloat/multilingual-e5-small                | 67.27     | 64.11                        | 49.97                 | 36.05                 | 85.21                                | 95.26                                | 72.99                                  |
| OpenAI/text-embedding-3-small                 | 66.39     | 64.02                        | 33.94                 | 20.03                 | 98.47                                | 91.70                                | 90.17                                  |
| OpenAI/text-embedding-ada-002                 | 64.38     | 61.02                        | 42.56                 | 14.51                 | 94.99                                | 91.23                                | 81.98                                  |
| pkshatech/GLuCoSE-base-ja                     | 59.02     | 63.88                        | 39.82                 | 30.28                 | 78.26                                | 82.06                                | 59.82                                  |
| cl-nagoya/sup-simcse-ja-base                  | 49.64     | 51.62                        | 50.25                 | 13.98                 | 68.08                                | 65.71                                | 48.22                                  |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup    | 47.38     | 50.14                        | 45.84                 | 13.00                 | 55.09                                | 74.97                                | 45.24                                  |
| MU-Kindai/Japanese-SimCSE-BERT-base-unsup     | 46.36     | 47.39                        | 39.57                 | 11.44                 | 64.16                                | 70.23                                | 45.37                                  |
| oshizo/sbert-jsnli-luke-japanese-base-lite    | 43.00     | 51.99                        | 42.07                 | 10.12                 | 49.30                                | 71.94                                | 32.59                                  |
| MU-Kindai/Japanese-MixCSE-BERT-base           | 42.59     | 42.37                        | 37.72                 | 7.88                  | 63.70                                | 64.13                                | 39.73                                  |
| MU-Kindai/Japanese-DiffCSE-BERT-base          | 41.79     | 42.31                        | 36.20                 | 7.81                  | 60.77                                | 64.34                                | 39.32                                  |
| MU-Kindai/Japanese-SimCSE-BERT-base-sup       | 41.32     | 44.11                        | 39.61                 | 8.15                  | 62.76                                | 58.39                                | 34.89                                  |
| MU-Kindai/Japanese-SimCSE-BERT-large-sup      | 40.82     | 47.04                        | 44.10                 | 11.43                 | 43.43                                | 62.41                                | 36.52                                  |
| cl-nagoya/unsup-simcse-ja-large               | 40.53     | 45.09                        | 34.60                 | 5.75                  | 55.07                                | 63.07                                | 39.61                                  |
| cl-nagoya/unsup-simcse-ja-base                | 40.23     | 46.00                        | 39.46                 | 5.55                  | 60.26                                | 55.63                                | 34.49                                  |
| sentence-transformers/LaBSE                   | 40.12     | 43.10                        | 34.25                 | 4.24                  | 48.92                                | 75.13                                | 35.09                                  |
| cl-nagoya/sup-simcse-ja-large                 | 37.62     | 46.84                        | 39.88                 | 11.83                 | 63.43                                | 37.93                                | 25.79                                  |
| pkshatech/simcse-ja-bert-base-clcmlp          | 37.00     | 41.50                        | 46.00                 | 10.19                 | 40.14                                | 59.63                                | 24.53                                  |
| sentence-transformers/stsb-xlm-r-multilingual | 21.00     | 25.11                        | 21.61                 | 2.76                  | 28.49                                | 36.47                                | 11.55                                  |
| colorfulscoop/sbert-base-ja                   | 16.52     | 21.50                        | 13.16                 | 0.44                  | 28.78                                | 22.40                                | 12.82                                  |

## STS
| Model                                         | Avg.      | jsick<br>(spearman)   | jsts<br>(spearman)   |
|:----------------------------------------------|:----------|:----------------------|:---------------------|
| cl-nagoya/sup-simcse-ja-large                 | **83.18** | **83.80**             | 82.57                |
| OpenAI/text-embedding-3-large                 | 82.52     | 81.27                 | **83.77**            |
| cl-nagoya/sup-simcse-ja-base                  | 82.05     | 82.83                 | 81.27                |
| cl-nagoya/unsup-simcse-ja-large               | 80.56     | 80.15                 | 80.98                |
| intfloat/multilingual-e5-small                | 80.07     | 81.50                 | 78.65                |
| intfloat/multilingual-e5-base                 | 79.84     | 81.28                 | 78.39                |
| intfloat/multilingual-e5-large                | 79.70     | 78.40                 | 80.99                |
| OpenAI/text-embedding-3-small                 | 79.46     | 80.83                 | 78.08                |
| OpenAI/text-embedding-ada-002                 | 79.02     | 79.09                 | 78.94                |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup    | 78.99     | 79.84                 | 78.14                |
| cl-nagoya/unsup-simcse-ja-base                | 78.72     | 78.49                 | 78.95                |
| pkshatech/GLuCoSE-base-ja                     | 78.71     | 74.97                 | 82.46                |
| MU-Kindai/Japanese-SimCSE-BERT-large-sup      | 78.28     | 78.75                 | 77.81                |
| MU-Kindai/Japanese-SimCSE-BERT-base-unsup     | 77.49     | 78.18                 | 76.81                |
| MU-Kindai/Japanese-MixCSE-BERT-base           | 77.05     | 77.57                 | 76.53                |
| pkshatech/simcse-ja-bert-base-clcmlp          | 76.80     | 73.08                 | 80.52                |
| oshizo/sbert-jsnli-luke-japanese-base-lite    | 76.60     | 72.11                 | 81.09                |
| sentence-transformers/LaBSE                   | 76.56     | 76.99                 | 76.12                |
| MU-Kindai/Japanese-DiffCSE-BERT-base          | 75.50     | 75.42                 | 75.58                |
| sentence-transformers/stsb-xlm-r-multilingual | 75.40     | 72.36                 | 78.44                |
| MU-Kindai/Japanese-SimCSE-BERT-base-sup       | 74.66     | 74.64                 | 74.68                |
| colorfulscoop/sbert-base-ja                   | 70.42     | 66.59                 | 74.24                |

## Classification
| Model                                         | Avg.      | amazon_counterfactual<br>(macro_f1)   | amazon_review<br>(macro_f1)   | massive_intent<br>(macro_f1)   | massive_scenario<br>(macro_f1)   |
|:----------------------------------------------|:----------|:--------------------------------------|:------------------------------|:-------------------------------|:---------------------------------|
| OpenAI/text-embedding-3-large                 | **77.58** | 77.90                                 | **60.44**                     | **80.91**                      | **91.08**                        |
| pkshatech/GLuCoSE-base-ja                     | 76.82     | **82.44**                             | 58.07                         | 78.85                          | 87.94                            |
| oshizo/sbert-jsnli-luke-japanese-base-lite    | 76.61     | 79.95                                 | 57.48                         | 80.26                          | 88.75                            |
| cl-nagoya/unsup-simcse-ja-large               | 74.66     | 76.79                                 | 55.37                         | 79.13                          | 87.36                            |
| MU-Kindai/Japanese-DiffCSE-BERT-base          | 73.77     | 78.10                                 | 51.56                         | 78.79                          | 86.63                            |
| cl-nagoya/sup-simcse-ja-large                 | 73.73     | 73.21                                 | 54.76                         | 79.23                          | 87.72                            |
| MU-Kindai/Japanese-SimCSE-BERT-large-sup      | 73.47     | 77.25                                 | 53.42                         | 76.83                          | 86.39                            |
| cl-nagoya/sup-simcse-ja-base                  | 73.47     | 72.34                                 | 54.41                         | 79.52                          | 87.60                            |
| MU-Kindai/Japanese-SimCSE-BERT-base-unsup     | 73.30     | 76.20                                 | 51.52                         | 78.95                          | 86.54                            |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup    | 73.13     | 76.36                                 | 52.75                         | 76.88                          | 86.51                            |
| cl-nagoya/unsup-simcse-ja-base                | 73.07     | 73.30                                 | 53.93                         | 79.07                          | 85.97                            |
| OpenAI/text-embedding-3-small                 | 73.06     | 70.01                                 | 55.92                         | 77.66                          | 88.67                            |
| MU-Kindai/Japanese-MixCSE-BERT-base           | 72.90     | 77.62                                 | 50.86                         | 77.19                          | 85.93                            |
| intfloat/multilingual-e5-large                | 72.89     | 70.66                                 | 56.54                         | 75.78                          | 88.59                            |
| MU-Kindai/Japanese-SimCSE-BERT-base-sup       | 72.76     | 76.20                                 | 52.06                         | 77.89                          | 84.90                            |
| sentence-transformers/LaBSE                   | 72.66     | 73.61                                 | 51.70                         | 76.99                          | 88.35                            |
| sentence-transformers/stsb-xlm-r-multilingual | 71.84     | 75.65                                 | 51.32                         | 74.28                          | 86.10                            |
| pkshatech/simcse-ja-bert-base-clcmlp          | 71.30     | 67.49                                 | 50.85                         | 79.67                          | 87.20                            |
| OpenAI/text-embedding-ada-002                 | 69.75     | 64.42                                 | 53.13                         | 74.57                          | 86.89                            |
| intfloat/multilingual-e5-base                 | 69.30     | 63.67                                 | 54.24                         | 72.78                          | 86.53                            |
| colorfulscoop/sbert-base-ja                   | 69.07     | 72.21                                 | 47.95                         | 72.52                          | 83.62                            |
| intfloat/multilingual-e5-small                | 67.62     | 62.14                                 | 51.27                         | 70.85                          | 86.22                            |

## Reranking
| Model                                         | Avg.      | esci<br>(ndcg@10)   |
|:----------------------------------------------|:----------|:--------------------|
| OpenAI/text-embedding-3-large                 | **93.58** | **93.58**           |
| OpenAI/text-embedding-ada-002                 | 93.04     | 93.04               |
| intfloat/multilingual-e5-small                | 93.03     | 93.03               |
| intfloat/multilingual-e5-large                | 92.96     | 92.96               |
| OpenAI/text-embedding-3-small                 | 92.92     | 92.92               |
| intfloat/multilingual-e5-base                 | 92.85     | 92.85               |
| pkshatech/GLuCoSE-base-ja                     | 91.90     | 91.90               |
| cl-nagoya/sup-simcse-ja-base                  | 91.83     | 91.83               |
| sentence-transformers/LaBSE                   | 91.63     | 91.63               |
| oshizo/sbert-jsnli-luke-japanese-base-lite    | 91.56     | 91.56               |
| pkshatech/simcse-ja-bert-base-clcmlp          | 91.49     | 91.49               |
| cl-nagoya/sup-simcse-ja-large                 | 91.48     | 91.48               |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup    | 91.30     | 91.30               |
| MU-Kindai/Japanese-SimCSE-BERT-base-unsup     | 91.16     | 91.16               |
| cl-nagoya/unsup-simcse-ja-base                | 91.16     | 91.16               |
| MU-Kindai/Japanese-MixCSE-BERT-base           | 91.01     | 91.01               |
| cl-nagoya/unsup-simcse-ja-large               | 90.95     | 90.95               |
| MU-Kindai/Japanese-DiffCSE-BERT-base          | 90.95     | 90.95               |
| MU-Kindai/Japanese-SimCSE-BERT-large-sup      | 90.95     | 90.95               |
| MU-Kindai/Japanese-SimCSE-BERT-base-sup       | 90.66     | 90.66               |
| sentence-transformers/stsb-xlm-r-multilingual | 90.20     | 90.20               |
| colorfulscoop/sbert-base-ja                   | 89.97     | 89.97               |

## Clustering
| Model                                         | Avg.      | livedoor_news<br>(v_measure_score)   | mewsc16<br>(v_measure_score)   |
|:----------------------------------------------|:----------|:-------------------------------------|:-------------------------------|
| OpenAI/text-embedding-3-large                 | **53.32** | 57.09                                | 49.55                          |
| cl-nagoya/sup-simcse-ja-base                  | 51.79     | 52.67                                | 50.91                          |
| intfloat/multilingual-e5-large                | 51.24     | **57.13**                            | 45.34                          |
| OpenAI/text-embedding-3-small                 | 51.06     | 54.57                                | 47.55                          |
| cl-nagoya/sup-simcse-ja-large                 | 50.56     | 50.75                                | 50.38                          |
| oshizo/sbert-jsnli-luke-japanese-base-lite    | 50.33     | 46.77                                | **53.89**                      |
| pkshatech/GLuCoSE-base-ja                     | 49.78     | 49.89                                | 49.68                          |
| cl-nagoya/unsup-simcse-ja-large               | 48.41     | 50.90                                | 45.92                          |
| OpenAI/text-embedding-ada-002                 | 48.30     | 49.67                                | 46.92                          |
| intfloat/multilingual-e5-base                 | 48.26     | 55.03                                | 41.49                          |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup    | 48.25     | 53.20                                | 43.31                          |
| pkshatech/simcse-ja-bert-base-clcmlp          | 47.53     | 44.77                                | 50.30                          |
| intfloat/multilingual-e5-small                | 46.91     | 54.70                                | 39.12                          |
| MU-Kindai/Japanese-SimCSE-BERT-base-unsup     | 46.68     | 53.02                                | 40.35                          |
| MU-Kindai/Japanese-SimCSE-BERT-large-sup      | 45.81     | 48.45                                | 43.17                          |
| MU-Kindai/Japanese-MixCSE-BERT-base           | 44.95     | 52.62                                | 37.28                          |
| sentence-transformers/LaBSE                   | 44.88     | 48.29                                | 41.47                          |
| colorfulscoop/sbert-base-ja                   | 44.81     | 42.99                                | 46.64                          |
| cl-nagoya/unsup-simcse-ja-base                | 44.77     | 52.23                                | 37.31                          |
| MU-Kindai/Japanese-DiffCSE-BERT-base          | 44.22     | 49.67                                | 38.77                          |
| MU-Kindai/Japanese-SimCSE-BERT-base-sup       | 43.11     | 41.04                                | 45.18                          |
| sentence-transformers/stsb-xlm-r-multilingual | 27.46     | 24.49                                | 30.43                          |

## PairClassification
| Model                                         | Avg.      | paws_x_ja<br>(binary_f1)   |
|:----------------------------------------------|:----------|:---------------------------|
| pkshatech/GLuCoSE-base-ja                     | **66.39** | **66.39**                  |
| cl-nagoya/sup-simcse-ja-base                  | 62.57     | 62.57                      |
| cl-nagoya/sup-simcse-ja-large                 | 62.51     | 62.51                      |
| cl-nagoya/unsup-simcse-ja-large               | 62.49     | 62.49                      |
| cl-nagoya/unsup-simcse-ja-base                | 62.44     | 62.44                      |
| pkshatech/simcse-ja-bert-base-clcmlp          | 62.40     | 62.40                      |
| OpenAI/text-embedding-ada-002                 | 62.40     | 62.40                      |
| MU-Kindai/Japanese-SimCSE-BERT-base-unsup     | 62.38     | 62.38                      |
| oshizo/sbert-jsnli-luke-japanese-base-lite    | 62.38     | 62.38                      |
| MU-Kindai/Japanese-DiffCSE-BERT-base          | 62.38     | 62.38                      |
| MU-Kindai/Japanese-SimCSE-BERT-base-sup       | 62.37     | 62.37                      |
| MU-Kindai/Japanese-SimCSE-BERT-large-sup      | 62.35     | 62.35                      |
| OpenAI/text-embedding-3-large                 | 62.35     | 62.35                      |
| MU-Kindai/Japanese-MixCSE-BERT-base           | 62.33     | 62.33                      |
| sentence-transformers/LaBSE                   | 62.33     | 62.33                      |
| colorfulscoop/sbert-base-ja                   | 62.31     | 62.31                      |
| OpenAI/text-embedding-3-small                 | 62.27     | 62.27                      |
| MU-Kindai/Japanese-SimCSE-BERT-large-unsup    | 62.27     | 62.27                      |
| intfloat/multilingual-e5-base                 | 62.26     | 62.26                      |
| sentence-transformers/stsb-xlm-r-multilingual | 62.20     | 62.20                      |
| intfloat/multilingual-e5-small                | 62.19     | 62.19                      |
| intfloat/multilingual-e5-large                | 62.15     | 62.15                      |

