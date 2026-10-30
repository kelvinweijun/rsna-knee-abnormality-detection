## Efficiency Prize

We are hosting a second track that focuses on model efficiency, because highly accurate models are often computationally heavy.

For the Efficiency Prize, we will evaluate submissions on both runtime and predictive performance.

To be eligible for an Efficiency Prize, a submission:
- Must be among the submissions selected by a team for the Leaderboard Prize, or else among those submissions automatically selected under the conditions described in the [**My Submissions**](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/submissions) tab.
- Must be ranked on the Private Leaderboard higher than the `sample_submission.csv` benchmark.

All submissions meeting these conditions will be considered for the Efficiency Prize. A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize.

An Efficiency Prize will be awarded to eligible submissions according to how they are ranked by the following evaluation metric on the private test data. See the [**Prizes**](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/overview/prizes) tab for the prize awarded to each rank. More details may be posted via discussion forum updates.

## Efficiency Score

We compute a submission's **efficiency score** by:
\\[ \text{Efficiency} = \frac{ \text{AUC} }{ \text{Benchmark} - \max\text{AUC} } + \frac{ \text{RuntimeSeconds} }{ 32400 } \\]

where \\(\text{AUC}\\) is the submission's score on the [main competition metric](https://www.kaggle.com/competitions/rsna-knee-abnormalities-detection/overview/evaluation), \\(\text{Benchmark}\\) is the score of the benchmark `sample_submission.csv`, \\(\max\text{AUC}\\) is the maximum \\(\text{AUC}\\) of all submissions on the Private Leaderboard, and \\(\text{RuntimeSeconds}\\) is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score.

During the training period of the competition, you may see a leaderboard for the *public* test data in the following notebook, updated daily: **[Efficiency
Leaderboard](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)**. After the competition ends, we will update this leaderboard with efficiency scores on the *private* data. During the training period, this leaderboard will show only the rank of each team, but not the complete score.