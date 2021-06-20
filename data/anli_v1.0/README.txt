This is Adversarial NLI, rounds 1-3, the 1.0 version. Date: June 30, 2020

If you have any questions, comments or suggestions, contact
<dkiela@fb.com> and <yixin1@cs.unc.edu>.

If you use this dataset in your own work, please cite the paper.

Github: https://github.com/facebookresearch/anli

Demo: https://adversarialnli.com

== Rules ==

When using this dataset, we ask that you obey some very simple rules:

1. We want to make it easy for people to provide ablations on test sets without
being rate limited, so we release labeled test sets with this distribution. We
trust that you will act in good faith, and will not tune on the test set (this
should really go without saying)! We may release unlabeled test sets later.

2. Training data is for training, development data is for development, and test
data is for reporting test numbers. This means that you should not e.g. train
on the train+dev data from rounds 1 and 2 and then report an increase in per-
formance on the test set of round 3.

3. We will host a leaderboard on the Github page (for now). If you want to be
added to the leaderboard, please contact us and/or submit a PR with a link to
your paper, a link to your code in a public repository (e.g. Github), together
with the following information: number of parameters in your model, data used
for (pre-)training, and your dev and test results for *each* round.
