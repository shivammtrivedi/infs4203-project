# Demo script: credit-card default classifier (INFS4203/7203)

Timings assume a ~5 minute presentation (scale the middle sections if yours is shorter). Text in *italics* is a cue for what to show on screen; say the rest out loud. Numbers are from `README.md` / `development.ipynb`.

---

## 1. The dataset (about 0:45)

*Show: figure 02 (target distribution), then figure 01 (missingness).*

"I built a classifier that predicts whether a credit-card client will default next month. I had 24,000 labelled clients to develop on and 6,000 unlabelled clients to predict. Each client has 23 predictors: 14 numeric ones, which are credit limit, age, and six months of bill and payment amounts; three categorical ones, sex, education and marital status; and six monthly repayment-status labels.

There were three main challenges.

First, **class imbalance**: only 22.1% of clients default. A model that says 'no default' for everyone is 78% accurate and useless, so I optimised F1 on the default class instead of accuracy, and used stratified cross-validation so every fold keeps that 22% rate.

Second, **missing data**: every one of the 23 predictors is missing in exactly 40% of rows, but in different rows. Only 17 rows are missing everything, while 98% of rows are missing at least one value, so deleting incomplete rows would throw away almost the whole dataset.

Third, **skew and outliers**: payments are extremely right-skewed, bills are strongly skewed and can be negative, and the six bill columns are almost copies of each other, with correlations up to 0.95."

---

## 2. Pre-processing: strategy and results (about 1:30)

*Show: figure 05 (default rate by repayment status), then the Section 6.2 imputation table.*

"All preprocessing sits inside one scikit-learn pipeline, so it is refit on each training fold and never sees validation data.

**Repayment status.** I pooled the six months and looked at default rate by label. The three 'not delayed' labels, paid on time, revolving credit and no balance, are all about 16 to 18%, so they carry no order. But default risk jumps to 35% at a one-month delay, 55% at two months, and 63 to 79% beyond that. So I encoded it ordinally: not delayed is 0, otherwise the number of months late. Compared with one-hot, that was worth about 0.07 F1 for k-NN and 0.01 for the tree models. The other categoricals have no order, so they are one-hot encoded.

**Scaling.** I standardised the numeric columns for k-NN only, because distance is dominated by the dollar columns. That took k-NN from 0.18 to 0.39 F1. Trees and Naive Bayes don't need it.

**Missing values.** Baseline is the Lecture 2 method: mean for numbers, most-frequent for categories. I excluded class-specific imputation because it needs the label, which the test set doesn't have. Then I noticed that because the gaps fall in different rows, a client usually still has other months observed, and bills are correlated at 0.95. So I added **per-client imputation**: a missing bill is replaced by the mean of that same client's other bills, and the same for payments, falling back to the column mean. It uses no labels, adds no columns, and is fitted inside each fold. Over three different fold shuffles it raised F1 from 0.506 to 0.514 and won on every shuffle. The gain is small, but consistent.

I also tried cluster-based and neighbour-based imputation, and trimming outliers with an Isolation Forest, but none helped, so they are not in the final pipeline."

---

## 3. Hyperparameter tuning (about 1:00)

*Show: the grids in README Section 4, then the class-weight/threshold rows of Table A.*

"I tuned with `GridSearchCV`, scored on positive-class F1, using the same five folds for everything.

For **k-NN** I searched k from 1 to 41, odd values only to avoid ties, and Manhattan versus Euclidean distance. Best was k = 25 with Euclidean.

For the **decision tree** I searched depth, minimum leaf size, and gini versus entropy. For the **random forest**, depth, minimum leaf size and criterion with 200 trees. Naive Bayes has nothing to tune.

The biggest finding was not in those grids. Without class weighting every model plateaus around 0.41 to 0.42, because with only 22% defaulters a 0.5 cut-off finds only about 30% of them. I tested two fixes: `class_weight='balanced'`, which makes defaulters count 3.5 times more, and tuning the decision threshold, picked inside each training fold so the score isn't inflated. Both lifted the forest to about 0.51, and doing both added nothing, so I kept the simpler one: class weights with the default cut-off. With weights on, the forest preferred a minimum leaf size of 10."

---

## 4. Model selection and ensembles (about 1:00)

*Show: figure 07 (validation performance chart) and Table A.*

"Here are all the approaches on held-out folds, mean and standard deviation over five folds. For context, a model that just predicts 'default' for everybody already scores F1 0.362, and predicting 'no default' for everybody scores zero, so 0.362 is the floor to beat.

- **Naive Bayes** is worst, F1 0.384. It labels 88% of clients as defaulters, because the correlated bills break its independence assumption and the skew breaks its normality assumption.
- **k-NN** reaches 0.407. It's precise but finds only 30% of defaulters.
- A tuned **decision tree** gets 0.419, or 0.484 with class weights.
- The **random forest**, which is the ensemble from the course, gets 0.421, or **0.509 with class weights**. So yes, an ensemble helps: forest versus single tree is 0.509 against 0.484 under the same preprocessing.

I also tried blending the forest with k-NN and the tree, and it did not beat the forest alone.

My final model is the random forest with balanced class weights and per-client imputation: **F1 of 0.514 plus or minus 0.008**, macro-F1 0.687, precision 0.51, recall 0.52."

---

## 5. Findings and next steps (about 0:45)

"Main findings: the class imbalance matters more than any model choice, since class weighting was worth about 0.09 F1; the repayment-status delay length is the strongest signal; and per-client imputation was the one preprocessing change that helped.

Limitations: the score plateaus around 0.51 because 40% of every column is missing, and nothing I tried moved it beyond fold-to-fold noise, which is about 0.008. I also selected and reported on the same folds, so I expect the hidden-test F1 to be around 0.50 to 0.51.

Next steps: a more principled model-based imputation, better use of the repayment history across months, choosing the decision threshold from the real cost of a missed default, and, most valuably, less-missing data."

---

## Code interview prep (3 minutes: likely questions, short answers)

| Question | Short answer |
|---|---|
| Walk me through `main.py`. | Load both CSVs and map repayment status to numbers. Build the pipeline (per-client imputation, one-hot, forest). Cross-validate for the five metrics. Refit on all development rows, predict the test set in order, write the `.infs4203` file. |
| Why a `Pipeline`? | So imputation, encoding and the model are refit inside each training fold; nothing fits on validation rows. |
| What does `PerClientImputer` do? | For each client it fills a missing bill or payment with the mean of that client's other months. Leftovers get the training-fold column mean. `fit` only stores that fallback. |
| Why `class_weight='balanced'`? | Defaulters are 22%, so it weights them about 2.26 versus 0.64 for non-defaulters, which raises recall and F1 on the positive class. |
| Why `min_samples_leaf=10`? | It smooths the forest's predictions; the grid preferred it once class weights were on. |
| Why `handle_unknown='ignore'`? | A rare category can be absent from a training fold; this avoids an error. |
| Why `shuffle=True`, seed 42? | Stratified shuffled folds, fixed seed so results are reproducible. |
| What if you used a different k / depth / weight? | Too small a k overfits; too large drifts to the majority class. Unlimited depth overfits. Heavier weights trade precision for recall. |
| How do you know there's no leakage? | All fitted steps are inside the pipeline. The ordinal mapping is a fixed dictionary, and `test.csv` is only used to predict. |
| Why is your F1 only about 0.51? | 40% missing per column. Imputation and tuning gains are within a couple of thousandths; the data limits the score. |
