# Submission Guideline
A guideline for developers who would like to register their own models to the [JMTEB leaderboard](leaderboard.md).

## Submit the evaluation results
Developers shall open a pull request in regards to each model they would like to add to the leaderboard. Please make the PR following the steps below.

1. Train your embedding model, and evaluate with JMTEB.

2. Your results shall be added to [docs/results](docs/results/). The result file (name should be `summary.json`) should be put in a directory named as `owner/model_name`, that results in a two-layer folder, for example, [docs/results/OpenAI/text-embedding-3-large](docs/results/OpenAI/text-embedding-3-large).

3. Run `pip install tabulate && python make_leaderboard.py`, and you will get a new `leaderboard.md` which contains the results of your model.

4. Push your update to a new branch `leaderboard/<your_model_name>`, and write the PR description (follow the [template](.github/PULL_REQUEST_TEMPLATE/leaderboard_submission.md)) about your model with details as much as possible, including the type, size and structure of your model, and if possible, how it is trained and what training datasets are used. We **strongly recommend** to include information about seen/unseen information of the training dataset, that is, whether a JMTEB evaluation dataset was used in the training of your model. For example, your model used `JAQKET`'s training set in the training stage, so mark `JAQKET` as `seen`, and other datasets as `unseen`. Also, please include **an instruction to reproduce** the evaluation results (e.g., evaluation scripts, special settings needed to fit your model's setting) as possible as you can.

## Submit your model
For developers who are reluctant to run all the evaluations due to the limits of computing resources, we enable the evaluation with some of [our](https://www.sbintuitions.co.jp/) resources when it is available. Please follow the instructions below if you want us to help you evaluate your model.

1. Train your embedding model.

2. Upload your model to somewhere publicly accessible. We recommend [Hugging Face Hub](https://huggingface.co/), as it is the de facto standard to make your models publicly available.

3. Add an issue to request evaluation. Please refer to step 4 of the last chapter (Submit the evaluation results) as well as the [issue template](.github/ISSUE_TEMPLATE/evaluation_request.md) for the contents.

4. We may respond within a few business days, if it is available for us to run the evaluation.

Please note:

* Please understand that we might be not able to cover all evaluation requests, as our computing resource is also limited.

* If possible, please include a script for your model, as incorrect settings may result in performance deterioration of your model. At least you need to figure out what special settings are needed for your model.
