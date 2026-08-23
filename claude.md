# Project Context: INFS4203/7203 Credit Default Classifier

## Assignment
Individual project (Track 1: Data-oriented). Build a binary classifier predicting
`defaulted_next_month` (yes/no) for credit-card clients. Deadline: 13:00, 19 Oct 2026 (Brisbane).

Full spec is in `Project_Spec_Sheet` — read it if anything below is ambiguous.

## Required deliverables (submission root)
```
sXXXXXXX.zip
├── sXXXXXXX.infs4203      # 6,001 rows: 6000 predictions + 1 metrics row
├── main.py                # final pipeline only, no EDA/tuning
├── development.ipynb      # full EDA, preprocessing trials, model comparison, tuning
├── README.md              # experimental report
└── (optional extra source/dependency files)
```

## Hard constraints — do not violate
- **Only these 4 classifier families are permitted**: Naive Bayes, k-NN, decision tree,
  random forest (and ensembles built from them). No other model families.
- **Only use techniques taught in the Week 2-6 coding guides** (below): the specific
  imputation/encoding/scaling approaches, `cross_val_score`, `GridSearchCV`, and the four
  classifier classes shown there. If unsure whether a technique is permitted, ask on Ed
  Discussion rather than assuming — don't reach for sklearn features (e.g. class_weight
  tricks, SMOTE, custom scorers) not demonstrated in the guides without checking first.
- **client_id is an identifier only** — never a predictor.
- **All preprocessing (imputation, encoding, scaling) must be fit inside each CV training
  fold only.** Never fit on validation-fold data or on the full development set before CV.
- **Optimize for F1 on the positive class (`yes` = default)**, not accuracy. Also report
  macro-F1, accuracy, precision, recall (positive class) — mean + std over folds.
- **Never evaluate on rows the model was fit on.** Reported validation metrics must come
  from held-out folds/split only.
- **Fix random seeds everywhere** (train/test split, KFold, model init) — reproducibility
  is graded. Use seed = 42 unless changed here.
- `main.py` run from the submission root, given `development.csv` + `test.csv`, must:
  1. reproduce the 5 validation metrics reported in the README,
  2. refit the final pipeline on all of development.csv,
  3. reproduce the exact 6,000 test predictions in original order,
  4. write `sXXXXXXX.infs4203` in the required format.
- `.infs4203` file format: rows 1–6000 = `yes,` or `no,` (comma, no header). Row 6001 =
  `accuracy,precision,recall,f1,macro_f1,` each rounded to 3 decimals, trailing comma.

## Dataset
- `development.csv`: 24,000 labelled rows, 25 columns (includes `defaulted_next_month`).
- `test.csv`: 6,000 unlabelled rows, 24 columns (no target).
- Predictors: `credit_limit`, `age` (numeric); `sex`, `education`, `marital_status`
  (categorical); `repayment_status_{oct..mar}` (string labels — ordinal vs categorical is
  an open decision, see below); `bill_amount_{oct..mar}`, `payment_amount_{oct..mar}`
  (numeric, 6 months each).
- Target: `defaulted_next_month` — `yes` (default) / `no`. Class imbalance expected.

## Key open decisions / working notes
- `repayment_status_*` values (`no_balance`, `paid_on_time`, `revolving_credit`,
  `payment_delayed_1_month`, `payment_delayed_2_months`, ...) — need to decide ordinal
  encoding vs one-hot. Check via EDA whether delay severity is monotonic w.r.t. default risk.
- The 6 `bill_amount_*` and 6 `payment_amount_*` columns are expected to be highly
  correlated month-to-month — likely to hurt Naive Bayes' independence assumption.
  Consider engineered features (avg bill, trend, utilization ratio) as an alternative
  representation to test.
- GaussianNB assumes per-class normality of numeric features — bill/payment amounts are
  likely right-skewed; may need log-transform or a non-Gaussian NB variant.
- Numeric feature scaling matters for k-NN, irrelevant for trees/forest.
- Naive Bayes is being tested as one of the 4 required comparisons, not assumed to be the
  final model — expect random forest to be a strong baseline given correlated/mixed
  features; final choice will be decided empirically via CV, not in advance.

## README must include (don't let main.py/notebook drift from this)
- Initial data audit (dimensions, feature types, target balance, missingness).
- Preprocessing alternatives tested (imputation, encoding, scaling, thresholds).
- Classifiers/ensembles tested, hyperparameter ranges, validation procedure.
- Experimental table: mean ± std for F1 (positive class) and macro-F1 (and balanced
  accuracy if included) per approach.
- Comparison + justification for final model choice.
- Final pipeline/model/hyperparameters used for the submitted predictions.
- Limitations / things that didn't work.
- Required plots (with titles, axis labels, legends, and written interpretation):
  missingness by feature, target class distribution, numeric/categorical feature
  distributions, validation-performance comparison across models.
- Environment: OS, Python version, package versions, seeds, exact run command.

## Conventions for this repo
- Python, scikit-learn as primary ML library, pandas for data handling.
- Random seed: 42 (update here if changed).
- Stratified K-fold CV (update fold count here once decided).
- Keep `main.py` minimal: load data → apply final fitted pipeline → write metrics +
  predictions. No exploratory code, no hyperparameter search loops.

## Actual Assignment specification
# INFS4203/7203 Project (20 marks) - 2026
 
**Deadline: 13:00 on 19 October 2026 (Brisbane time).**
 
This is an individual project. Choose one of the following two tracks:
 
1. **Data-oriented project:** Apply data-mining techniques from the course to the provided dataset. Train and
    tune a classifier or ensemble, report cross-validation results, and predict the hidden test labels.
2. **Competition-oriented challenge:** Participate in an approved external data-mining or machine-learning
    competition and document your Public Leader Board performance.
 
For either track, submit all source code and a README containing enough information to reproduce your
results. The requirements and marking schemes differ between the two tracks, so read the relevant section
carefully.
 
## Common submission guide
 
These requirements apply to both project tracks.
 
## Files and submission
 
```
Submit all required files exclusively through the specified UQ Blackboard Turnitin links. Email
submissions are not accepted.
Submit the source code and README as one ZIP file named sXXXXXXX.zip, where sXXXXXXX is your
student username.
The Turnitin submission title must exactly match the filename sXXXXXXX.zip.
Submit the ZIP through Blackboard → Assessment → Project → Readme and code submission →
Turnitin.
The ZIP file must be smaller than 100 MB. If it is larger, contact infs4203@eecs.uq.edu.au before the
deadline.
Only the last submitted version will be marked.
Track-specific files, layouts, and any additional submission links are described in the relevant track
below.
```
## Late-submission penalties
 
```
Submit all required files by 13:00 on 19 October 2026 (Brisbane time).
A penalty of 10% of the maximum available mark is deducted per 24 hours after the deadline, for up to
seven days, according to the ECP.
A submission received more than seven days after the deadline receives a mark of zero.
```
## Academic integrity and use of AI tools
 
You may use generative AI or machine-translation tools, but you must appropriately acknowledge their use and
remain able to demonstrate clear, independent understanding of your submission. Failure to disclose such use
appropriately may constitute student misconduct under the Student Code of Conduct.
 
 
## Track 1: Data-oriented project
 
### 1. Overview
 
In this project, you will build a binary classifier that predicts whether a credit-card client will default in the next
month. Your solution should demonstrate a complete data-mining workflow: data exploration, preprocessing,
model selection, hyperparameter tuning, cross-validation, final training, and prediction on unseen test data.
 
### 2. Dataset
 
Two CSV files are provided:
 
```
development.csv: 24,000 labelled clients and 25 columns.
test.csv: 6,000 unlabelled clients and 24 columns.
```
The development file contains client_id, 23 predictors, and the target column defaulted_next_month. The
test file contains the same identifier and predictors but does not contain the target.
 
#### 2.1 Columns
 
```
Role Columns Description
```
```
Identifier client_id Unique client identifier. Do not use it as a predictor.
```
```
Numeric credit_limit, age Credit limit and client age.
```
```
Numeric bill_amount_oct to bill_amount_mar Monthly statement amounts from October through March.
```
```
Numeric payment_amount_oct to
payment_amount_mar
```
```
Monthly payment amounts from October through March.
```
```
Categorical sex, education, marital_status Client demographic information represented by readable
string labels.
```
```
Repayment
status
```
```
repayment_status_oct to
repayment_status_mar
```
```
Monthly repayment behaviour from October through
March, represented by string labels.
```
```
Target defaulted_next_month yes means default; no means no default.
```
Repayment-status values include no_balance, paid_on_time, revolving_credit, and increasing payment
delays such as payment_delayed_1_month and payment_delayed_2_months. You should investigate whether
these values are best treated as ordinal or categorical.
 
### 3. Objective
 
Develop a classifier, or an ensemble of classifiers, that achieves a strong F1 score on the positive class yes in
the hidden test labels.
 
Your work should consider:
 
 
```
suitable methods for handling missing values;
encoding categorical and repayment-status features;
whether numerical features require scaling;
the four classification methods covered in the course: naïve Bayes, k-nearest neighbours, decision tree,
and random forest;
reasonable hyperparameter ranges and a reproducible tuning procedure; and
whether an ensemble improves upon individual classifiers.
```
Use only techniques permitted within the relevant course content. If you are unsure whether a particular
technique is permitted, please ask on the Ed Discussion board.
 
### 4. Model selection and evaluation
 
The test labels are reserved for marking. All preprocessing choices, feature decisions, model comparisons,
hyperparameters, and classification thresholds must therefore be selected using development.csv only.
 
Use stratified cross-validation so that each fold contains a similar proportion of default cases. Optimise and
select models using F1 for the positive class yes. For each model, report the mean and standard deviation of:
 
```
accuracy;
precision for the positive class yes;
recall for the positive class yes;
F1 for the positive class yes; and
macro-F1.
```
Balanced accuracy may be included as an additional diagnostic, but it is not required.
 
All preprocessing must be fitted inside each cross-validation training fold. For example, an imputer must not
calculate replacement values from the fold currently being used for validation. After selecting the final
approach, refit it using all of development.csv before predicting test.csv.
 
Fix all random seeds so that your results can be reproduced.
 
### 5. Required results
 
#### 5.1 Test predictions
 
Submit one file named sXXXXXXX.infs4203, where sXXXXXXX is your student username. It must follow the same
row-oriented format as the provided result_report_example.infs4203 file and contain no header.
 
The example demonstrates the required row structure. For this project, the final row contains the five
development validation metrics listed below.
 
The file must contain exactly 6,001 rows:
 
```
Rows 1–6,000: one predicted label for each row of test.csv, in the original test order. Use only yes or
no, followed by a comma.
Row 6,001: development validation accuracy, positive-class precision, positive-class recall, positive-
class F1, and macro-F1, in that order. Report all five as floating-point values rounded to three decimal
places, and end the row with a comma.
```
 
Example structure:
 
```
yes,
no,
yes,
...,
no,
no,
yes,
0.800,0.600,0.500,0.545,0.700,
```
If cross-validation is used, place the mean of each metric across folds in this final row. Report the
corresponding standard deviations in the README.
 
Only the final model selected from training cross-validation should be used to generate these predictions.
 
#### 5.2 Validation performance
 
In your submitted README, report the validation performance of the final model used to generate
sXXXXXXX.infs4203. Include:
 
```
accuracy;
precision for the positive class yes;
recall for the positive class yes;
F1 for the positive class yes; and
macro-F1.
```
Balanced accuracy may be reported as an additional diagnostic.
 
State how the validation data was obtained. If you use cross-validation, report the mean and standard deviation
across the folds. These values must be calculated using only labelled training data, with all preprocessing fitted
inside the corresponding training fold.
 
Do not report metrics calculated by predicting the same development rows used to fit the model. The reported
values must come from held-out validation predictions.
 
**The reported validation F1 is not the F1 used to grade test performance.** It documents how you selected
the final model. Your performance mark is calculated separately by the teaching team using your submitted
predictions and the hidden test labels.
 
### 6. Development notebook, final code, and reproducibility
 
#### 6.1 README experimental report
 
The README is part of the assessment and must explain the experimental process that led to the final
submitted model. It must include enough evidence and analysis for the teaching team to understand and
reproduce your decisions.
 
**At minimum, include** :
 
```
an initial data audit covering dimensions, feature types, target balance, and missing values;
```
 
```
the preprocessing alternatives you tested, such as imputation, categorical or ordinal encoding, scaling,
feature handling, and any classification-threshold choices;
the classifiers and ensembles tested;
the hyperparameter ranges and validation procedure used;
experimental tables showing mean and standard deviation for F1 and balanced accuracy;
a comparison of the tested approaches and an explanation of why the final model was selected;
the final preprocessing pipeline, model, and hyperparameters used to create sXXXXXXX.infs4203; and
limitations, unexpected results, or approaches that did not work well.
```
Include and discuss plots that support your analysis. **At minimum, provide** :
 
```
a missingness plot by feature;
a target-class distribution plot;
relevant numerical or categorical feature-distribution plots; and
a validation-performance plot comparing the various tested models or pipelines.
```
Plots must have readable titles, axis labels, and legends where appropriate. Embed or link them in the
README and comment on what each plot shows, why it matters, and how it influenced your modelling
decisions. A collection of plots without interpretation is not sufficient.
 
Also document the operating system, Python version, package versions, random seeds, and exact command
required to run the final solution.
 
#### 6.2 Jupyter notebook and README
 
Submit a Jupyter notebook named development.ipynb containing the work used to develop your solution. The
notebook should include your exploratory data analysis and plots, preprocessing and imputation trials, model
comparisons, hyperparameter tuning, validation results, and final model selection. It must be well commented
so that your process and decisions are clear.
 
The README should summarise the final approach and results and provide the command required to run the
final solution. The notebook and README will also be used later in the course as the basis for an oral
examination.
 
#### 6.3 Source code and main.py
 
The root-level main.py must contain the final preprocessing pipeline and model, using the parameters selected
in development.ipynb. It does not need to repeat the exploratory analysis or hyperparameter search.
 
Running main.py from the provided development.csv and test.csv must:
 
1. reproduce the five development validation metrics reported in the README;
2. fit the documented final pipeline and model using the labelled development data;
3. reproduce the exact 6,000 submitted test labels in the original test order; and
4. write sXXXXXXX.infs4203 in the required 6,001-row format, including the same rounded development
    metrics in its final row.
 
The teaching team will run python main.py from the submission root and compare the generated test
predictions and validation scores with the submitted files.
 
 
Do not include trained models, the provided datasets, or other large generated files unless requested. Your
code may assume that the teaching team places development.csv and test.csv in the documented input
location before running it.
 
### 7. Data-oriented package layout
 
Use the common sXXXXXXX.zip submission rules above. The required root-level layout for the data-oriented
track is:
 
```
sXXXXXXX.zip
├── sXXXXXXX.infs
├── main.py
├── development.ipynb
├── README.md
└── additional source or dependency files, if required
```
The submitted main.py must reproduce both the final validation results and sXXXXXXX.infs4203.
 
### 8. Test-performance marking ladder
 
Performance marks are based on F1 for the positive class yes, calculated using the hidden test labels. Marks
are rounded to the nearest 0.1 and capped at 20.
 
```
Hidden test F1 interval Performance-mark interval
```
```
F1 < 0.40 0
```
```
0.40 ≤ F1 < 0.42 0.0 ≤ mark < 2.
```
```
0.42 ≤ F1 < 0.44 2.0 ≤ mark < 4.
```
```
0.44 ≤ F1 < 0.46 4.0 ≤ mark < 6.
```
```
0.46 ≤ F1 < 0.48 6.0 ≤ mark < 8.
```
```
0.48 ≤ F1 < 0.50 8.0 ≤ mark < 10.
```
```
0.50 ≤ F1 < 0.51 10.0 ≤ mark < 12.
```
```
0.51 ≤ F1 < 0.52 12.0 ≤ mark < 14.
```
```
0.52 ≤ F1 < 0.53 14.0 ≤ mark < 16.
```
```
0.53 ≤ F1 < 0.54 16.0 ≤ mark < 18.
```
```
0.54 ≤ F1 < 0.55 18.0 ≤ mark < 20.
```
```
F1 ≥ 0.55 20
```
Within each interval, calculate the exact mark using:
 
```
If F1 < 0.40: Mark = 0
If 0.40 ≤ F1 < 0.50: Mark = (F1 - 0.40) / 0.
```
 
```
If 0.50 ≤ F1 < 0.55: Mark = 10 + (F1 - 0.50) / 0.
If F1 ≥ 0.55: Mark = 20
```
A valid performance mark also requires a reproducible submission, predictions generated by the submitted
code, and compliance with the permitted-techniques requirement. Additional grading criteria may be specified
separately.
 
## Track 2: Competition-oriented challenge
 
Select and participate in an external data-mining or machine-learning competition that aligns with the learning
objectives of the course.
 
### 1. Competition eligibility
 
The competition must:
 
```
offer a monetary reward, ensuring that it is a genuine competitive challenge;
conclude no later than 1 October 2026; and
have at least 10 competitors.
```
Entry-level Kaggle competitions labelled **Getting Started** , **Playground** , or **Community** are not eligible.
 
This track is limited to five students. Places are allocated on a first-come-first-served basis, subject to approval
of the competition by the teaching team. To apply for this track, first complete the Expression of Interest (EOI)
form (link).
 
Unlike the data-oriented track, you may use any appropriate data-mining or machine-learning techniques,
including techniques outside the scope of INFS4203/7203. Any programming language may be used.
 
### 2. Project requirements
 
#### 2.1 Result report
 
Submit a brief report documenting your Public Leader Board result. It must include:
 
```
a screenshot showing your rank on the Public Leader Board; and
the URL of the Public Leader Board page.
```
Your Public Leader Board username must be your student username in the form sXXXXXXX, where each X is a
digit.
 
#### 2.2 Code and README
 
The README must include:
 
```
an environment description, including operating system, hardware, programming language and version,
and additional packages; and
```
 
```
instructions for reproducing your competition submission.
```
The submitted code must include:
 
```
all tuning, training, and testing code used to generate the final competition submission; and
a main function in a main file, such as main.py, that executes the complete process.
```
Include the README and all source code in the common sXXXXXXX.zip submission described at the top of this
specification.
 
### 3. Challenge result-report format
 
```
Name the result report sXXXXXXX.pdf or sXXXXXXX.docx.
The result-report Turnitin submission title must exactly match its filename.
Result reports submitted with other names or formats will not be accepted or marked.
```
### 4. Challenge result-report submission
 
```
Submit the result report through Blackboard → Assessment → Project → Report submission → Turnitin.
Submit the code and README ZIP through the common link specified at the top of this document.
The common deadline, file-size limit, and late-submission penalties apply to both files.
```
### 5. Marking standard
 
The following standard applies unless the teaching team agrees to an alternative for an exceptionally
challenging competition.
 
Evidence of your Public Leader Board achievement must be submitted by the project deadline, and your Public
Leader Board username must be your student username.
 
If the selected competition finishes before the project deadline, you may instead use cross-validation to
demonstrate that you achieved performance comparable to a particular competitor on the Public Leader Board
before the deadline. Your mark may then be based on that competitor's corresponding rank percentage.
 
Let the **Public Leader Board ranking index** be your rank divided by the total number of competitors. The
performance mark is:
 
```
Public Leader Board position Ranking-index interval Performance-mark interval
```
```
Top 30% or better ranking index ≤ 0.30 20
```
```
Below top 30%, up to top 40% 0.30 < ranking index ≤ 0.40 18 ≤ mark < 20
```
```
Below top 40%, up to top 50% 0.40 < ranking index ≤ 0.50 16 ≤ mark < 18
```
```
Below top 50%, up to top 60% 0.50 < ranking index ≤ 0.60 14 ≤ mark < 16
```
```
Below top 60%, up to top 70% 0.60 < ranking index ≤ 0.70 12 ≤ mark < 14
```
```
Below top 70%, up to top 80% 0.70 < ranking index ≤ 0.80 10 ≤ mark < 12
```
```
Below top 80%, up to top 90% 0.80 < ranking index ≤ 0.90 8 ≤ mark < 10
```
 
```
Public Leader Board position Ranking-index interval Performance-mark interval
```
```
Below top 90%, up to 100% 0.90 < ranking index ≤ 1.00 6 ≤ mark < 8
```
Calculate the exact mark within each interval using:
 
```
Mark = max(20 - max(ranking_index - 0.30, 0) * 20, 0)
```
A Public Leader Board ranking within the top 30% earns 20 marks.

## Coding Guides
# Week 4 Coding Guide
 
## August, 2026
 
```
The main purpose of this document is to introduce how to apply two classifiers, kNN and Naive
Bayes , implemented byscikit-learn. We will use the modified Iris dataset introduced in previous
weeks, and assume we have already completed the imputation, normalization and one-hot encoding.
```
## 0.1 1.Data Preparation
 
```
We first import the packages that will be used in this document.
```
1. Pandas: Pandas is an open-source Python library widely used for data manipulation, anal-
    ysis, and cleaning tasks. The central data structure in Pandas is the DataFramewhich
    provides methods to facilitate the preliminary examination of essential properties, statistical
    summaries, and a select number of rows for a cursory exploration of the data.
2. Numpy: Numpy is a powerful Python library for numerical and array-based computing. It
    provides support for large, multi-dimensional arrays and matrices, along with a wide range
    of mathematical functions to operate on these arrays eﬀiciently.
3. sklearn.neighbors.KNeighborsClassifier: KNeighborsClassifier is a class provided by scikit-
    learn, used to create a k-NN classification model.
4. sklearn.naive_bayes.GaussianNB: GaussianNB is a class provided by scikit-learn, which im-
    plements the Gaussian Naive Bayes algorithm.
5. sklearn.metrics: sklearn.metrics includes performance metrics functions used to evaluate a
    classifier’s performance.
6. sklearn.model_selection.cross_val_score: is a function provided by scikit-learn that facili-
    tates performing cross-validation on a given model to evaluate its performance.
7. sklearn.model_selection.GridSearchCV: GridSearchCV is a class provided by scikit-learn that
    facilitates hyperparameter tuning for models using a technique called grid search with cross-
    validation.
These packages will be utilized in following tasks for data loading, classification and evaluation.
 
[1]: **import pandas as pd
import numpyas np
from sklearn.neighbors import** KNeighborsClassifier
**from sklearn.naive_bayes import** GaussianNB
**from sklearnimport** metrics
**from sklearn.model_selection import** cross_val_score
**from sklearn.model_selection import** GridSearchCV
 
 
```
First, we load the training and test data.
```
[2]: df = pd.read_csv('iris_modified_train.csv')
df_test =pd.read_csv('iris_modified_test.csv')
 
[3]: X_train =df.iloc[:,:- 1 ].values
y_train =df.iloc[:,- 1 ].values
X_test = df_test.iloc[:,:- 1 ].values
y_test = df_test.iloc[:,- 1 ].values
 
## 0.2 2. Classifiers
 
```
0.2.1 2.1 kNN
For conducting kNN classification, we use theKNeighborsClassifierprovided by scikit-learn.
The parametern_neighborsrepresents the important hyperparameter of kNN, the number of
neighbours k. By default, it is set to 5. Changing the value of it would control the learning process.
We will introduce later on tuning this hyperparameter.
The parameter metric shows the method of distance computation. By default, it is set to
minkowski, which results in the standard Euclidean distance when the other parameterpis set to
2.
The parameterpis the power parameter for the Minkowski metric. By default, it is set to 2. When
p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
We will use standard Euclidean distance to build the model in this file. Feel free to explore other
distance metrics to better understand their effects on kNN classification in our analysis.
First, let’s have a look at how the kNN classifier performs on the defaultn_neighbors.
```
[4]: kNN_5= KNeighborsClassifier()
 
```
Thefit()method is a fundamental function in scikit-learn’s machine learning models used for
training the model on the provided training data.
However, thefit()method for kNN is different from those discussed last week. It simply stores the
training data for reference when performing predictions instead of creating a generalisable model
because the kNN classifier is a lazy learning approach as introduced in the lecture.
```
[5]: kNN_5.fit(X_train, y_train)
 
[5]: KNeighborsClassifier()
 
```
Then, we classify the test data bypredict().
```
[6]: y_pred_kNN =kNN_5.predict(X_test)
 
```
Using the method introduced before, we can evaluate the test performance.
```
[7]: acc_kNN =metrics.accuracy_score(y_test, y_pred_kNN)
print("The test accuracy of decision tree on the dataset is:", acc_kNN)
 
 
```
The test accuracy of decision tree on the dataset is: 0.
```
```
[8]: f1_kNN = metrics.f1_score(y_test, y_pred_kNN, average='macro')
print("The test macro f1-score of decision tree on the dataset is: ", f1_kNN)
```
```
The test macro f1-score of decision tree on the dataset is: 0.
```
```
2.1.1 Cross Validation for Evaluation However, in practical scenarios, acquiring the ground
truth for the test dataset is often unfeasible. To overcome this, we employ cross-validation, which
provides a more reliable estimate of our model’s accuracy when test data is not available.We will
use thecross_val_score()to evaluate a classifier by cross validation.
First, we employ 5-fold cross-validation on the model with k = 5 to evaluate its performance.
```
```
[9]: acc_kNN_5= cross_val_score(kNN_5, X_train, y_train, cv= 5 , scoring=('accuracy'))
f1_kNN_5 =cross_val_score(kNN_5, X_train, y_train, cv= 5 , scoring=('f1_macro'))
print("The cross-validation accuracy is: {:}\n The cross-validation f1-score is:␣
↪ {:} ".format(acc_kNN_5.mean(), f1_kNN_5.mean()))
```
```
The cross-validation accuracy is: 0.
The cross-validation f1-score is: 0.
Then let’s have a look at the performance of the model with k = 3 by 5-fold cross-validation.
```
[10]: kNN_3= KNeighborsClassifier(n_neighbors = 3 )
acc_kNN_3= cross_val_score(kNN_3, X_train, y_train, cv= 5 , scoring=('accuracy'))
f1_kNN_3 =cross_val_score(kNN_3, X_train, y_train, cv= 5 , scoring=('f1_macro'))
print("The cross-validation accuracy is: **{:}\n** The cross-validation f1-score is:␣
↪ **{:}** ".format(acc_kNN_3.mean(), f1_kNN_3.mean()))
 
```
The cross-validation accuracy is: 0.
The cross-validation f1-score is: 0.
Obviously, thekNNclassifier’s performance varies with different hyperparameter values fork.
To improve the score we can consider tuning some hyperparameters.
```
```
2.1.2 Cross Validation for Tuning Hyperparameter Scikit-learnprovides a function called
GridSearchCVwhich will perform a grid search across the parameter space you define.
The parameter determines the parameter space ofGridSearchCVisparam_grid. It should be in
the form of a dictionary or a list of dictionaries.
We define parameter search space forkas an array of odd numbers between 1 and 21, and make it
into the the required form.
```
[11]: parameters =[{'n_neighbors': [int(x) **for** x **in** np.arange( 1 , 22 , 2 )]}]
 
```
Then we create kNN models, evaluating their performance using 5-fold cross validation, and then
returning the best performing model on macro-F1 and the output.
```
 
[12]: kNN =KNeighborsClassifier()
clf_best_kNN= GridSearchCV(kNN, parameters, cv= 5 , scoring='f1_macro')
clf_best_kNN.fit(X_train, y_train)
print(clf_best_kNN.best_params_)
 
```
{'n_neighbors': 11}
Empolying the kNN classifier with the best parameter selected, we can further see its cross-
validation performance by accuracy and F1 score.
```
[13]: kNN_11 = KNeighborsClassifier(n_neighbors = 11 )
acc_kNN_11 =cross_val_score(kNN_11, X_train, y_train, cv= 5 ,␣
↪scoring=('accuracy'))
f1_kNN_11= cross_val_score(kNN_11, X_train, y_train, cv= 5 ,␣
↪scoring=('f1_macro'))
print("The cross-validation accuracy is: **{:}\n** The cross-validation f1-score is:␣
↪ **{:}** ".format(acc_kNN_11.mean(), f1_kNN_11.mean()))
 
```
The cross-validation accuracy is: 0.
The cross-validation f1-score is: 0.
Fit the kNN classifier from the training dataset.
```
[14]: kNN_11.fit(X_train, y_train)
 
[14]: KNeighborsClassifier(n_neighbors=11)
 
```
Then, we classify the test data bypredict().
```
[15]: y_pred_kNN_11 = kNN_11.predict(X_test)
 
```
Using the method introduced before, we can evaluate the test performance.
```
[16]: acc_kNN_11 =metrics.accuracy_score(y_test, y_pred_kNN_11)
print("The test accuracy of kNN on the dataset is: ", acc_kNN_11)
 
```
The test accuracy of kNN on the dataset is: 0.
```
[17]: f1_kNN_11= metrics.f1_score(y_test, y_pred_kNN_11, average='macro')
print("The test macro f1-score of kNN on the dataset is:", f1_kNN_11)
 
```
The test macro f1-score of kNN on the dataset is: 0.
We can observe that compared to the case settingn_neighbors=5, the test performance have
increased when settingn_neighborsto 11, a hyperparameter selected by cross-validation.
```
```
0.2.2 2.2 Naive Bayes
We utilize theGaussianNBprovided in scikit-learn for our classification tasks for Naive Bayes.
```
[18]: gnb =GaussianNB()
 
 
```
Given that no particular hyperparameter necessitates tuning in this model, we proceed to directly
fit it using the training dataset byfit().
```
[19]: gnb.fit(X_train, y_train)
 
[19]: GaussianNB()
 
```
We employ 5-fold cross-validation on the model to evaluate its performance.
```
[20]: acc_gnb =cross_val_score(gnb, X_train, y_train, cv= 5 , scoring=('accuracy'))
f1_gnb = cross_val_score(gnb, X_train, y_train, cv= 5 , scoring=('f1_macro'))
print("The cross-validation accuracy is: **{:}\n** The cross-validation f1-score is:␣
↪ **{:}** ".format(acc_gnb.mean(), f1_gnb.mean()))
 
```
The cross-validation accuracy is: 0.
The cross-validation f1-score is: 0.
We proceed with the process to observe the outcome. We classify the test data bypredict().
```
[21]: y_pred_gnb =gnb.predict(X_test)
 
```
And we evaluate by
```
[22]: acc_gnb =metrics.accuracy_score(y_test, y_pred_gnb)
print("The test accuracy of naive bayes on the dataset is: ", acc_gnb)
 
```
The test accuracy of naive bayes on the dataset is: 0.
```
[23]: f1_gnb = metrics.f1_score(y_test, y_pred_gnb, average='macro')
print("The test macro f1-score of naive bayes on the dataset is: ", f1_gnb)
 
```
The test macro f1-score of naive bayes on the dataset is: 0.
Author: Kaki Zhou 8/8/
```

# Week 3 Coding Guide
 
### August 10, 2026
 
```
The main purpose of this document is to introduce how to apply two classifiers, decision tree and
random forest, implemented byscikit-learn. We will use the modified Iris dataset introduced in
Week 2, and assume we have already completed the imputation, so there are no null values.
```
## 1 Data preprocessing
 
```
We first import the packages that will be used in this document.
```
1. Pandas: Pandas is an open-source Python library widely used for data manipulation, anal-
    ysis, and cleaning tasks. The central data structure in Pandas is the DataFramewhich
    provides methods to facilitate the preliminary examination of essential properties, statistical
    summaries, and a select number of rows for a cursory exploration of the data.
2. Numpy: Numpy is a powerful Python library for numerical and array-based computing. It
    provides support for large, multi-dimensional arrays and matrices, along with a wide range
    of mathematical functions to operate on these arrays eﬀiciently.
3. sklearn.preprocessing.OneHotEncoder: OneHotEncoder is a package in the scikit-learn library
    (sklearn) used for one-hot encoding categorical (nominal) features. One-hot encoding is a
    process of converting categorical data into binary vectors, where each category is represented
    by a unique binary vector with a 1 in the position corresponding to the category and 0s
    everywhere else.
4. sklearn.preprocessing.StandardScaler: StandardScaler is a package from scikit-learn (sklearn)
    used for normalization. It scales the data to have a mean of 0 and a standard deviation of 1.
5. sklearn.model_selection.train_test_split: train_test_split() is used to split a dataset into
    training and testing subsets, allowing users to evaluate the performance of machine learning
    models on unseen data.
6. sklearn.metrics: sklearn.metrics includes performance metrics functions used to evaluate a
    classifier’s performance.
7. sklearn.tree.DecisionTreeClassifier: A decision tree classifier, limited to handling only numer-
    ical values.
8. sklearn.ensemble.RandomForestClassifier: A random forest classifier, limited to handling only
    numerical values.
These packages will be utilized in following tasks for data preprocessing, classification and evalua-
tion.
 
[1]: **import pandas as pd
import numpyas np
from sklearn.preprocessing import** OneHotEncoder, StandardScaler
**from sklearn.model_selection import** train_test_split
 
 
```
from sklearnimport metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```
```
First, we load the data as introduced last week.
```
[2]: df = pd.read_csv('iris_modified.csv')
 
### 1.1 One-hot Encoding
 
```
The scikit-learn’s implementation of decision tree and random forest models does not natively
handle nominal features. Instead, it requires all features to be numerical or ordinal. If your dataset
contains nominal features, you’ll need to preprocess them into numerical representations, such as
using one-hot encoding, before fitting the models.
We preprocess the nominal values using theOneHotEncoder()from scikit-learn. First, we apply
thefit_transform()method of the OneHotEncoder to the data, which both fit the encoder to the
dataset and transforms it into a one-hot encoded array. We then store this resulting array as
feature_array for further use in our analysis
```
[3]: ohe =OneHotEncoder()
feature_array = ohe.fit_transform(df.iloc[:, 3 ].to_frame()).toarray()
 
```
The attribute categories_ in the OneHotEncoder object holds the categories of each feature deter-
mined during the fitting process. To observe the newly created features, we print the categories_
attribute, and then we store it as feature_labels for future reference in our analysis
```
[4]: print(ohe.categories_)
 
```
[array(['petal_width_0', 'petal_width_1', 'petal_width_2', 'petal_width_3',
'petal_width_4'], dtype=object)]
```
[5]: feature_labels =ohe.categories_[ 0 ]
 
```
Then we utilize the pandas library to construct aDataFramecalled features. It takes two essential
components:
```
1. feature_array: This is an array containing the transformed data after performing one-hot
    encoding on the nominal features of the original dataset.
2. columns = feature_labels: This specifies the column names for the DataFrame. fea-
    ture_labels is a list (or an array-like object) containing the names of the new features created
    as a result of the one-hot encoding process.
 
[6]: features =pd.DataFrame(feature_array, columns = feature_labels)
 
```
When executed, this code creates a structured DataFrame where each column corresponds to one
of the original nominal features, and each row represents an individual data point. In summary,
the code effectively converts the transformed one-hot encoded data, along with their corresponding
feature labels, into a structured DataFrame, enabling easier handling, analysis, and manipulation
of the data in tabular form.
```
 
```
We then merge the original numerical features with the transformed one-hot features by
```
[7]: df_new = pd.concat([df.iloc[:,: 3 ],features,df.iloc[:, 4 ]],axis = 1 )
 
```
We can then get some statistical information on each feature of the new dataset df_new byde-
scribe()andinfo().
```
[8]: print(df_new.describe())
 
```
sepal length in cm sepal width in cm petal length in cm \
count 150.000000 150.000000 150.
mean 5.843333 3.054000 3.
std 0.828066 0.433594 1.
min 4.300000 2.000000 1.
25% 5.100000 2.800000 1.
50% 5.800000 3.000000 4.
75% 6.400000 3.300000 5.
max 7.900000 4.400000 6.
```
```
petal_width_0 petal_width_1 petal_width_2 petal_width_3 \
count 150.000000 150.000000 150.000000 150.
mean 0.333333 0.100000 0.340000 0.
std 0.472984 0.301005 0.475296 0.
min 0.000000 0.000000 0.000000 0.
25% 0.000000 0.000000 0.000000 0.
50% 0.000000 0.000000 0.000000 0.
75% 1.000000 0.000000 1.000000 0.
max 1.000000 1.000000 1.000000 1.
```
```
petal_width_
count 150.
mean 0.
std 0.
min 0.
25% 0.
50% 0.
75% 0.
max 1.
```
[9]: print(df_new.info())
 
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 9 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 sepal length in cm 150 non-null float
1 sepal width in cm 150 non-null float
```
 
```
2 petal length in cm 150 non-null float
3 petal_width_0 150 non-null float
4 petal_width_1 150 non-null float
5 petal_width_2 150 non-null float
6 petal_width_3 150 non-null float
7 petal_width_4 150 non-null float
8 class 150 non-null object
dtypes: float64(8), object(1)
memory usage: 10.7+ KB
None
As we can observe, all the features are numerical now.
```
### 1.2 Train/test split and normalization
 
```
Last week, we briefly introduced thetrain_test_split()function. By utilizing classifiers, we can
gain a deeper understanding of the data processing sequence.
When working with machine learning models, it’s crucial to avoid data leakage from the test set into
the training process. Therefore, normalization across instances should be performed after splitting
the data into training and test sets. This ensures that the normalization process relies solely on
the training data, avoiding any data leakage from the test set.
Furthermore, there is no need to normalize the features obtained through one-hot encoding, as they
already represent binary sets with values 0 or 1.
As a result, we follow this sequence:
```
1. First, we usetrain_test_split()to split the dataset into training and test sets. By the default
    settings, it will create a test set consisting of 25% of the original data. For example, in the
    current dataset with 150 instances, it will lead to 38 test instances.
2. After splitting the data, we then proceed with the normalization process, applying it only
    to the numerical features present in the training set. This ensures that the normalization is
    based on the training data and avoids any data leakage from the test set.
3. Since the one-hot encoded features already represent binary sets, there is no need to perform
    any further normalization on them.
Following this data processing sequence ensures that we have a proper setup for training and
evaluating our machine learning models.
 
[10]: y= df_new.iloc[:,- 1 ].values
X_num= df_new.iloc[:,: 3 ].values _#numerical features_
X_nom= df_new.iloc[:, 3 :- 1 ].values _#nominal features_
X_num_train, X_num_test, X_nom_train, X_nom_test, y_train, y_test =␣
↪train_test_split(X_num, X_nom, y, random_state = 0 )
 
```
We then apply theStandardScaler()to achieve the standardization (or z-score normalization). We
normalize both the training and testing sets, based on the statistics calculated from the training
set.
```
[11]: scaler = StandardScaler()
 
 
[12]: scaler.fit(X_num_train)
X_num_train =scaler.transform(X_num_train)
X_num_test =scaler.transform(X_num_test)
 
```
Afterward, we usenumpy.concatenate()to merge the standardized features with the one-hot en-
coded ones to create the complete feature set for further analysis or model training.
```
[13]: X_train =np.concatenate((X_num_train, X_nom_train), axis = 1 )
X_test = np.concatenate((X_num_test, X_nom_test), axis= 1 )
 
```
You may try other normalisation method by yourself.
```
## 2 Classifiers
 
### 2.1 Decision Tree
 
```
For conducting decision tree classification, we use theDecisionTreeClassifierprovided by scikit-
learn.
By default, the parameter criterion is set to'gini', which employs the Gini impurity as the
criterion for splitting nodes in the decision tree. However, it can also be set to'entropy', using
information gain based on entropy.
As we have already introduced information gain based on entropy in class, it might be worthy
to experiment by setting the criterion parameter to'entropy'and observe the differences in the
results compared to the default'gini'criterion. We will leave that as a self-practice.
Switching the criterion allows us to assess how the decision tree’s performance varies with different
impurity measures and helps us gain insights into the impact of criterion selection on the model’s
decision boundaries and accuracy.
Feel free to explore both'gini' and'entropy'criteria to better understand their effects on
decision tree classification in our analysis.
Note that scikit-learn’s DecisionTreeClassifier does not have built-in support for splitting nodes
using the'gain ratio'criterion. The'gain ratio'is an alternative to information gain that
takes into account the number of categories (branches) for each attribute when making splits.
While the'gain ratio'is not directly available in scikit-learn’s DecisionTreeClassifier, you can
create a custom implementation of a decision tree using the gain ratio criterion by extending the
DecisionTreeClassifier class or using other libraries that offer this feature. We will leave that as an
optional extension for those interested in further experimentation.
Concretely, We first initialize a DecisionTreeClassifier model object by
```
[14]: dt = DecisionTreeClassifier(random_state = 0 )
 
```
Thefit()method is a fundamental function in scikit-learn’s machine learning models used for
training the model on the provided training data.
We then instruct the decision tree model dt to learn from the provided training data. During this
process, the model will analyze the features and target labels to create a decision tree that can
```
 
```
make predictions based on the patterns and relationships it identifies in the data.
After the fit() method completes its execution, the dt model will be trained and ready to make
predictions on new, unseen data based on the knowledge it has acquired from the training dataset.
```
[15]: dt = dt.fit(X_train, y_train)
 
```
Then, we classify the test data bypredict().
```
[16]: y_pred_dt= dt.predict(X_test)
 
```
Using the method introduces last week, we can evaluate the test performance.
```
[17]: acc_dt = metrics.accuracy_score(y_test, y_pred_dt)
print("The test accuracy of decision tree on the dataset is:", acc_dt)
 
```
The test accuracy of decision tree on the dataset is: 0.
```
[18]: f1_dt= metrics.f1_score(y_test, y_pred_dt, average='macro')
print("The test macro f1-score of decision tree on the dataset is: ", f1_dt)
 
```
The test macro f1-score of decision tree on the dataset is: 0.
```
### 2.2 Random Forest
 
```
Similar to the process described above, we utilize theRandomForestClassifierprovided in scikit-
learn for our classification tasks.
The RandomForestClassifier is an ensemble learning method that constructs multiple decision trees
and combines their predictions to improve overall accuracy and reduce overfitting. By default, the
parameter criterion in the RandomForestClassifier is set to'gini', which uses the Gini impurity
as the criterion for splitting nodes in the individual decision trees.
Feel free to explore both'gini'and'entropy'criteria in your random forest classification to
further enhance your comprehension of their effects on the model’s performance.
```
[19]: rf = RandomForestClassifier(random_state= 0 )
rf = rf.fit(X_train,y_train)
 
[20]: rf = rf.fit(X_train,y_train)
 
```
Then, we do test bypredict().
```
[21]: y_pred_rf= rf.predict(X_test)
 
```
And we evaluate by
```
[22]: acc_rf = metrics.accuracy_score(y_test, y_pred_rf)
print("The test accuracy of random forest on the dataset is:", acc_rf)
 
```
The test accuracy of random forest on the dataset is: 0.
```
 
[23]: f1_rf= metrics.f1_score(y_test, y_pred_rf, average='macro')
print("The test macro f1-score of random forest on the dataset is: ", f1_rf)
 
```
The test macro f1-score of random forest on the dataset is: 0.
We can see the performance of random tree is better than decision tree on this dataset.
Author: Kaki Zhou 1/8/

# Week 2 Coding Guide
 
### Aug 2026
 
In this document, we introduce a modified iris dataset. This dataset has undergone some changes,
including the deletion of certain values and the addition of a nominal feature named petal width.
Our primary objective is to demonstrate the processes of data loading, handling missing values,
normalizing data, and evaluating predictions.
 
## 1 Data Preparation
 
Access theIrisdataset from https://archive.ics.uci.edu/, the University of California, Irvine (UCI)
Machine Learning Repository. UCI is a well-known online repository that hosts various datasets
for machine learning research. This toy dataset is widely used for testing out machine learning
algorithms and visualizations.
 
**More Information on the Modified Iris Dataset** : A dataset with 150 instances of iris flowers,
each described by four features (sepal length, sepal width, petal length, and petal width) and is
assigned to one of three iris species. The first three features are **numerical** and the last one petal
width is nominal among a value from {0,1,2,3,4}.
 
## 2 Data Loading
 
We first import the packages that will be used in this document.
 
1. Pandas: Pandas is an open-source Python library widely used for data manipulation, anal-
    ysis, and cleaning tasks. The central data structure in Pandas is the DataFramewhich
    provides methods to facilitate the preliminary examination of essential properties, statistical
    summaries, and a select number of rows for a cursory exploration of the data.
2. sklearn.preprocessing.MinMaxScaler: MinMaxScaler is a package from scikit-learn (sklearn)
    used for normalization. It scales the data to a specific range (usually between 0 and 1).
3. sklearn.preprocessing.StandardScaler: StandardScaler is another package from scikit-learn
    (sklearn) used for normalization. It scales the data to have a mean of 0 and a standard
    deviation of 1.
4. train_test_split: train_test_split() is used to split a dataset into training and testing subsets,
    allowing users to evaluate the performance of machine learning models on unseen data.
5. sklearn.metrics: sklearn.metrics includes performance metrics functions used to evaluate a
    classifier’s performance.
 
These packages will be utilized later in the following tasks for data preprocessing and evaluating
the performance of a classifier.
 
 
[1]: **import pandas as pd
from sklearn.preprocessing import** MinMaxScaler, StandardScaler
**from sklearn.model_selection import** train_test_split
**from sklearnimport** metrics
 
```
After importing these packages, we can begin to load the dataset. The dataset is stored in the
iris_modified.csv file. We will use theread_csv()for loading this file and useinfo()to get an initial
overview of the dataset.
```
[2]: df = pd.read_csv('iris_modified.csv')
print(df.info())
 
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 sepal length in cm 148 non-null float
1 sepal width in cm 148 non-null float
2 petal length in cm 150 non-null float
3 petal width 147 non-null float
4 class 150 non-null object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
In this dataset, we observe a total of 150 instances, with each instance containing 5 columns. The
first three columns represent the numerical values of sepal length, sepal width, and petal length,
respectively. The fourth column denotes the nominal feature petal width of each instance whose
values are among {0,1,2,3,4}. Lastly, the fifth column provides the class label for each instance.
We can then get some statistical information on each feature of the dataset bydescribe():
```
[3]: print(df.describe())
 
```
sepal length in cm sepal width in cm petal length in cm petal width
count 148.000000 148.000000 150.000000 147.
mean 5.831757 3.053378 3.758667 1.
std 0.819328 0.435096 1.764420 1.
min 4.300000 2.000000 1.000000 0.
25% 5.100000 2.800000 1.600000 0.
50% 5.800000 3.000000 4.350000 2.
75% 6.400000 3.300000 5.100000 2.
max 7.900000 4.400000 6.900000 4.
Despite there being 150 instances, we only have 148 or 147 values for some features. There definitely
exist some missing values. Moreover, the value ranges of features are also different from each other.
That means we may normalize the data to achieve a better performance.
Let’s have a look at the first 10 rows of the dataset byhead().
```
 
[4]: df.head( 10 )
 
[4]: sepal length in cm sepal width in cm petal length in cm petal width \
0 5.1 3.5 1.4 0.
1 4.9 3.0 1.4 0.
2 4.7 3.2 1.3 0.
3 4.6 3.1 1.5 0.
4 5.0 3.6 1.4 0.
5 5.4 3.9 1.7 0.
6 4.6 3.4 1.4 0.
7 5.0 3.4 1.5 0.
8 4.4 2.9 1.4 0.
9 4.9 3.1 1.5 0.
 
```
class
0 Iris-setosa
1 Iris-setosa
2 Iris-setosa
3 Iris-setosa
4 Iris-setosa
5 Iris-setosa
6 Iris-setosa
7 Iris-setosa
8 Iris-setosa
9 Iris-setosa
```
```
We can see all the detailed information of the first 10 instances, and they all belong to the Iris-setora
class. Note that checking the dataset and understanding the properties of it are always necessary
before we conduct further steps!
```
## 3 Missing Value
 
```
As expounded in the lecture, the handling of missing values in the dataset entails two primary
approaches: the former entails the removal of all data instances containing missing values, while
the latter involves the process of imputation. Regarding the imputation process, it encompasses
two distinct strategies: one based on the utilization of aggregate values for all instances, and the
other employing class-specific values for imputing missing data.
```
### 3.1 Removal
 
```
This can be conducted by employing thePandas.DataFramesfunctiondropna().
```
[5]: df_deleting =df.dropna()
 
[6]: print(df_deleting.info())
 
```
<class 'pandas.core.frame.DataFrame'>
Index: 143 entries, 0 to 149
```
 
```
Data columns (total 5 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 sepal length in cm 143 non-null float
1 sepal width in cm 143 non-null float
2 petal length in cm 143 non-null float
3 petal width 143 non-null float
4 class 143 non-null object
dtypes: float64(4), object(1)
memory usage: 6.7+ KB
None
We can see there are 143 instances left after the removal of any instance with a missing value.
```
### 3.2 Imputation by all values
 
```
Utilize thefillna()method to impute missing or NaN values within the DataFrame by replacing
them with the respective feature’s average computed from all available non-null values.
Firstly, we apply theDataFrame.copy()method to create a new DataFrame, ensuring that any
modifications made will not affect the original one. This precautionary step allows us to work with
a separate copy, preserving the integrity of the original data for further analysis or comparison.
```
[7]: df_impu_all =df.copy()
 
```
3.2.1 Impute the numerical features
We then impute the numerical values by the average of the feature’s all values. We use the
DataFrame.iloc[]method with thefillna()to impute the first three numerical columns.
```
[8]: df_impu_all.iloc[:,: 3 ] =df_impu_all.iloc[:,: 3 ].fillna(df_impu_all.iloc[:,: 3 ].
↪mean())
 
[9]: print(df_impu_all.info())
 
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 sepal length in cm 150 non-null float
1 sepal width in cm 150 non-null float
2 petal length in cm 150 non-null float
3 petal width 147 non-null float
4 class 150 non-null object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
After printing the information and observing that all the numerical values have been imputed (the
```
 
```
Non-Null values increase from 148 to 150), we proceed to handle the missing values in the nominal
feature.
```
```
3.2.2 Impute the nominal features
For this task, we will use theDataFrame.mode()method, which allows us to impute the missing
categorical values with the most common one in the corresponding column.
```
[10]: df_impu_all.iloc[:, 3 ]=df_impu_all.iloc[:, 3 ].fillna(df_impu_all.iloc[:, 3 ].
↪mode().iloc[ 0 ])
 
[11]: print(df_impu_all.info())
 
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 sepal length in cm 150 non-null float
1 sepal width in cm 150 non-null float
2 petal length in cm 150 non-null float
3 petal width 150 non-null float
4 class 150 non-null object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
After conducting the imputations for both numerical and nominal features, we can verify the
successful completion of this process by printing the information.
```
### 3.3 Imputation by class-specific values
 
```
In this part, we will employ a similar approach to addressing missing or NaN values in another
way. Specifically, the missing or NaN values are replaced with the mean or most common of the
corresponding feature values within their respective class.
To obtain the list of unique classes, we employ theunique()method. In the following codes, we
use a for loop to iterate through each class, and perform the imputation process. We will utilize
DataFrame.loc()to confirm conditions and facilitate the imputation process.
```
[12]: df_impu_class = df.copy()
cat_list =df_impu_class.iloc[:, 4 ].unique()
**for** cat **in** cat_list:
_#imputate numerical values_
df_impu_class.loc[df_impu_class.iloc[:, 4 ]==cat,df_impu_class.columns[: 3 ]] =␣
↪df_impu_class.loc[df_impu_class.iloc[:, 4 ]==cat,df_impu_class.columns[: 3 ]].
↪fillna(
df_impu_class.loc[df_impu_class.iloc[:, 4 ]==cat,df_impu_class.columns[:
↪ 3 ]].mean()
)
 
 
```
#imputate categorical values
df_impu_class.loc[df_impu_class.iloc[:, 4 ]==cat,df_impu_class.columns[ 3 ]]=␣
↪df_impu_class.loc[df_impu_class.iloc[:, 4 ]==cat,df_impu_class.columns[ 3 ]].
↪fillna(
df_impu_class.loc[df_impu_class.iloc[:, 4 ]==cat,df_impu_class.
↪columns[ 3 ]].mode().iloc[ 0 ]
)
```
[13]: print(df_impu_class.info())
 
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
# Column Non-Null Count Dtype
--- ------ -------------- -----
0 sepal length in cm 150 non-null float
1 sepal width in cm 150 non-null float
2 petal length in cm 150 non-null float
3 petal width 150 non-null float
4 class 150 non-null object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
```
## 4 Normalization
 
```
After successfully performing the imputation using class-specific values on the dataset, we will
proceed to normalization. Note that normalization can be only applied to numerical features so it
is essential to select those columns.
Before doing normalization, let us have a look again at the dataset bydescribe().
```
[14]: print(df_impu_class.describe())
 
```
sepal length in cm sepal width in cm petal length in cm petal width
count 150.000000 150.000000 150.000000 150.
mean 5.837374 3.053918 3.758667 1.
std 0.816055 0.433823 1.764420 1.
min 4.300000 2.000000 1.000000 0.
25% 5.100000 2.800000 1.600000 0.
50% 5.800000 3.000000 4.350000 2.
75% 6.400000 3.300000 5.100000 2.
max 7.900000 4.400000 6.900000 4.
```
### 4.1 Max-min normalization
 
```
We apply theMinMaxScaler()to achieve the max-min normalization.
```
 
[15]: df_class_minmax =df_impu_class.copy()
scaler = MinMaxScaler()
scaler.fit(df_class_minmax.loc[:,df_impu_class.columns[: 3 ]])
df_class_minmax.loc[:,df_impu_class.columns[: 3 ]] =scaler.
↪transform(df_class_minmax.loc[:,df_impu_class.columns[: 3 ]])
 
```
Let us have a look at the data again
```
[16]: print(df_class_minmax.describe())
 
```
sepal length in cm sepal width in cm petal length in cm petal width
count 150.000000 150.000000 150.000000 150.
mean 0.427048 0.439133 0.467571 1.
std 0.226682 0.180760 0.299054 1.
min 0.000000 0.000000 0.000000 0.
25% 0.222222 0.333333 0.101695 0.
50% 0.416667 0.416667 0.567797 2.
75% 0.583333 0.541667 0.694915 2.
max 1.000000 1.000000 1.000000 4.
All three numeraical features are now normalized to the range of [0, 1].
```
### 4.2 Standardization
 
```
We then apply theStandardScaler()to achieve the standardization (or z-score normalization).
```
[17]: df_class_standard = df_impu_class.copy()
scaler = StandardScaler()
scaler.fit(df_class_standard.loc[:,df_impu_class.columns[: 3 ]])
df_class_standard.loc[:,df_impu_class.columns[: 3 ]] = scaler.
↪transform(df_class_standard.loc[:,df_impu_class.columns[: 3 ]])
 
```
We can also have a brief look at the dataset after standardization.
```
[18]: print(df_class_standard.describe())
 
```
sepal length in cm sepal width in cm petal length in cm petal width
count 1.500000e+02 1.500000e+02 1.500000e+02 150.
mean 1.421085e-15 -1.586879e-15 3.315866e-16 1.
std 1.003350e+00 1.003350e+00 1.003350e+00 1.
min -1.890220e+00 -2.437514e+00 -1.568735e+00 0.
25% -9.066106e-01 -5.872651e-01 -1.227541e+00 0.
50% -4.595198e-02 -1.247030e-01 3.362659e-01 2.
75% 6.917554e-01 5.691402e-01 7.627586e-01 2.
max 2.536024e+00 3.113232e+00 1.786341e+00 4.
Ideally, the mean should be 0 with std to be 1 after z-score normalization. We can round the results
to avoid those floating point precision deviations.
```
[19]: print(df_class_standard.describe().round( 2 ))
 
 
```
sepal length in cm sepal width in cm petal length in cm petal width
count 150.00 150.00 150.00 150.
mean 0.00 -0.00 0.00 1.
std 1.00 1.00 1.00 1.
min -1.89 -2.44 -1.57 0.
25% -0.91 -0.59 -1.23 0.
50% -0.05 -0.12 0.34 2.
75% 0.69 0.57 0.76 2.
max 2.54 3.11 1.79 4.
```
## 5 Evaluation
 
```
Since we don’t have a test file, we split the dataset into two subsets: a training set (used for
training) and a testing set (used for validating or testing) bytrain_test_split()method. By the
default setting, it will create a test set consisting of 25% of the original data. In the current dataset
consisting of 150 instances, it will lead to 38 test instances.
```
[20]: y= df_class_standard.iloc[:,- 1 ].values
X= df_class_standard.iloc[:, 0 : 4 ].values
X_train, X_test, y_train, y_test =train_test_split(X, y, random_state = 0 )
 
```
Since we will introduce constructing classifiers from Week 3, we can proceed directly to the evalua-
tion phase using the test results provided. Classifier_A provides a prediction on the test data which
is stored in prediction_1.csv, and Classifier B provides one in prediction_2.csv. In the following,
we will evaluate Classifier_B and leave evaluating Classifier_A as optional homework.
First, we load the test csv file prediction_2.csv and have a look at them.
```
[21]: df_prediction_1 =pd.read_csv('prediction_2.csv')
df_prediction_1.head( 38 )
 
[21]: predicted class
0 Iris-virginica
1 Iris-versicolor
2 Iris-setosa
3 Iris-virginica
4 Iris-setosa
5 Iris-virginica
6 Iris-setosa
7 Iris-versicolor
8 Iris-virginica
9 Iris-versicolor
10 Iris-virginica
11 Iris-versicolor
12 Iris-versicolor
13 Iris-versicolor
14 Iris-versicolor
15 Iris-setosa
 
 
```
16 Iris-versicolor
17 Iris-versicolor
18 Iris-setosa
19 Iris-setosa
20 Iris-virginica
21 Iris-versicolor
22 Iris-setosa
23 Iris-setosa
24 Iris-virginica
25 Iris-setosa
26 Iris-setosa
27 Iris-versicolor
28 Iris-versicolor
29 Iris-setosa
30 Iris-virginica
31 Iris-versicolor
32 Iris-setosa
33 Iris-virginica
34 Iris-virginica
35 Iris-versicolor
36 Iris-setosa
37 Iris-virginica
```
```
We can observe that the file provides the predicted class, which we will use for evaluation so we
select those columns.
```
[22]: y_predict= df_prediction_1.iloc[:,- 1 ].values
 
```
We can then show the accuracy of the classifier byaccuracy_score()
```
[23]: acc =metrics.accuracy_score(y_test, y_predict)
print("The prediction accuracy is:", acc)
 
```
The prediction accuracy is: 0.
Also, we can get the macro f1-score bymetrics.f1_score()
```
[24]: f1 = metrics.f1_score(y_test, y_predict, average='macro')
print("The prediction macro f1-score is: ", f1)
 
```
The prediction macro f1-score is: 0.
Please try to evaluate Classifier_A by yourself:)
```

# Week 5 Coding Guide

### August, 2026

```
The main purpose of this document is to introduce how to apply the clustering,𝑘 -means , im-
plemented byscikit-learn. This document encompasses three scenarios. In the first case, we will
generate a mock dataset forming spherical clusters. Then in the second case, we will employ the
normalized iris dataset. And the third scenario involves a dataset lacking spherical clusters.
```
## 1 1. Simulated data for clustering

### 1.1 1.1 Data generation

```
We first import the packages that will be used in this case.
```
1. Numpy: Numpy is a powerful Python library for numerical and array-based computing. It
    provides support for large, multi-dimensional arrays and matrices, along with a wide range
    of mathematical functions to operate on these arrays eﬀiciently.
2. sklearn.datasets.make_blobs: Make_blobs is a function provided by scikit-learn, used to
    generate isotropic Gaussian blobs for clustering.
3. sklearn.cluster.KMeans: KMeans is a class provided by scikit-learn, used to create a𝑘-means
    clustering.
4. matplotlib.pyplot: matplotlib.pyplot is a state-based interface to matplotlib. It provides an
    implicit, MATLAB-like, way of plotting. It also opens figures on your screen, and acts as the
    figure GUI manager.
5. seaborn: Seaborn is a Python data visualization library based on matplotlib. It provides a
    high-level interface for drawing attractive and informative statistical graphics.
These packages will be utilized in following tasks for data generation, clustering, hyperparameter
tuning and evaluation.

[1]: **import numpyas np
from sklearn.datasets import** make_blobs
**from sklearn.cluster import** KMeans
**import matplotlib.pyplot as plt
import seaborn as sns**

```
First, we generate the data bymake_blobs.
The parametern_samplesindicates the number of data points;n_featuresrepresents the number
of features for each sample;centers(int) specifies the number of centers to generate;cluster_std
```

```
controls the standard deviation of the clusters;random_statedetermines random number genera-
tion for dataset creation, ensuring consistent output when supplied with an integer across multiple
function calls.
In this instance, we generate a dataset featuring 2 attributes, allowing for straightforward visual-
ization. The dataset comprises 100 data points and encompasses 3 centers. The standard deviation
of these clusters is defined as 1.5, and an integer is employed as the random state to ensure repro-
ducibility.
```
[2]: X, y =make_blobs(n_samples= 100 , n_features= 2 , centers= 3 , cluster_std =1.5,␣
↪random_state= 42 )

```
Visualize the generated dataset bysns.scatterplot()to gain a concise overview of its distribution.
```
[3]: sns.scatterplot(x = X[:, 0 ],y = X[:, 1 ])

[3]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3accd0088>

### 1.2 1.2 Clustering: 𝑘 -Means

```
For conducting a𝑘-means clustering, we use theKMeansprovided by scikit-learn.
The parametern_clustersrepresents the important hyperparameter of𝑘-means, the number of
clusters𝑘. By default, it is set to 8. Later on, we will delve into optimizing this hyperparameter.
The attributelabels_represents the labels of each point.
The attributeinertia_indicates the sum of squared distances of samples to their closest cluster
centre.
```

```
First, let’s have a look at how the𝑘-means performs on the defaultn_clusters, withrandom_state
set to 0.
```
[4]: kmeans_model= KMeans(random_state= 0 )

```
Thefit()method is a fundamental function in scikit-learn’s machine learning models used for
training the model on the provided training data.
Thefit()here computes𝑘-means clustering.
```
[5]: kmeans_model.fit(X)

[5]: KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
n_clusters=8, n_init=10, n_jobs=None, precompute_distances='auto',
random_state=0, tol=0.0001, verbose=0)

```
We utilize thepredict()function to determine the nearest cluster for each sample in X.
This function yields the index of the corresponding cluster for each sample.
```
[6]: labels = kmeans_model.predict(X)

```
Next, we employsns.color_palette()to generate a range of colors, allowing us to designate distinct
colors for each cluster.
```
[7]: colors = sns.color_palette("bright", 8 )

```
Then we usesns.scatterplot()to create a scatter plot of the dataset with clustered labels.
```
[8]: sns.scatterplot(x = X[:, 0 ],y = X[:, 1 ], hue = labels, palette= colors)

[8]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3ad444b48>


```
When executing𝑘-means clustering, the first step is to randomly choose several data points to serve
as the initial center points. Varying starting centroids can yield divergent clustering outcomes,
converging to local minima instead of the global one. Thus, employing a distinct random seed
regulated by therandom_stateparameter can engender disparate results.
We try a differentrandom_statevalue then, while keeping all other parameters unchanged.
```
[9]: kmeans_model= KMeans(random_state= 42 )
kmeans_model.fit(X)
labels = kmeans_model.predict(X)
colors = sns.color_palette("bright", 8 )
sns.scatterplot(x = X[:, 0 ],y = X[:, 1 ], hue = labels, palette= colors)

[9]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3ad943fc8>

```
Obviously, we have obtained different clustering results.
```
```
1.2.1 1.2.1 Tuning hyperparameter
First, we calculate SSE for different values of k.
As introduced in the lecture, k is usually selected from 2 to
```
#### √

```
𝑛where𝑛is the number of all data
points. Thus, for this dataset, we can have a look at the range from 2 to 10.
Then for each k, computes𝑘-means clustering and calculate the SSE by the attributeinertia_. You
could also calculate and plot according the mean SSE as introduced in the lecture, by dividing the
```

```
SSE by the number of data points. We will include a plot illustrating the average Sum of Squared
Errors (SSE) for the purpose of comparison. It’s evident that employing the elbow method to
determine the optimal value of k using either individual SSE values or the mean SSE will yield the
same selected value of k.
```
[10]: sse =[]
mean_sse =[]
divisor =X.shape[ 0 ]
**for** k **in** range( 2 , 11 ):
kmeans =KMeans(n_clusters=k)
kmeans.fit(X)
sse.append(kmeans.inertia_)
mean_sse.append(kmeans.inertia_/divisor)

```
Then we plot SSE vs. k byplt.
plt.figure()is used to create a new figure and we set the figure size by the parameterfigsize.
plt.plot()is used to plot y versus x as lines and markers as we set parametermarkertoo.
The title and the names for x-axis and y-axis are set byplt.title(),plt.xlabel(),plt.ylabel(), respec-
tively.
plt.xticks()is employed to designate labels for the x-axis.
The layout of grid lines is configured withplt.grid().
To render all active figures,plt.show()is called.
```
[11]: plt.figure(figsize=( 8 , 6 ))
plt.plot(range( 2 , 11 ), sse, marker='o')
plt.title('SSE vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(np.arange( 2 , 11 ))
plt.grid( **True** )
plt.show()

```
plt.figure(figsize=( 8 , 6 ))
plt.plot(range( 2 , 11 ), mean_sse, marker='o')
plt.title('Mean SSE vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Mean of the Sum of Squared Errors (SSE)')
plt.xticks(np.arange( 2 , 11 ))
plt.grid( True )
plt.show()
```


```
Following theElbowmethod outlined in the lecture, the optimal value appears to be k = 3.
Subsequently, we proceed to execute𝑘-means clustering with k = 3 and generate a corresponding
plot.
```
[12]: kmeans_model_3 =KMeans(n_clusters = 3 )
kmeans_model_3.fit(X)
labels_3 =kmeans_model_3.predict(X)
colors = sns.color_palette("bright", 3 )
sns.scatterplot(x = X[:, 0 ],y = X[:, 1 ], hue = labels_3, palette =colors)

[12]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3adad0388>


```
Visual inspection reveals that the model exhibits good performance.
```
## 2 2. Real dataset

### 2.1 2.1 Data Preparation

```
We import some additional packages for this case.
```
1. Pandas: Pandas is an open-source Python library widely used for data manipulation, anal-
    ysis, and cleaning tasks. The central data structure in Pandas is the DataFramewhich
    provides methods to facilitate the preliminary examination of essential properties, statistical
    summaries, and a select number of rows for a cursory exploration of the data.
2. sklearn.manifold.TSNE: TSNE is a class provided by scikit-learn, which implement T-SNE(T-
    distributed Stochastic Neighbor Embedding), a tool to visualize high-dimensional data.
These packages and those imported in the first case will be utilized in following tasks for data
loading, clustering, hyperparameter tuning and evaluation.

[13]: **import pandas as pd
from sklearn.manifold import** TSNE

```
First, we load the data for clustering.
```
[14]: df = pd.read_csv('iris_normalized.csv')

### 2.2 2.2 Clustering: 𝑘 -Means

```
Apply𝑘-means clustering to the dataset using the same method as described earlier.
```

[15]: kmeans_8 =KMeans()
kmeans_8.fit(df)

[15]: KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
n_clusters=8, n_init=10, n_jobs=None, precompute_distances='auto',
random_state=None, tol=0.0001, verbose=0)

```
Calculate the SSE by the attributeinertia_.
```
[16]: SSE_8= kmeans_8.inertia_
print('The SSE is:', SSE_8)

```
The SSE is: 63.
Save the predicted labels by the attributelabels_for evaluation.
```
[17]: labels_8 =kmeans_8.labels_

```
Plotting the data and clusters can give us a qualitative sense of how well the algorithm has separated
the data.
However, this dataset has 4 features so we cannot plot directly. We use the toolTSNEfrom
scikit-learn here. T-SNE converts similarities between data points to joint probabilities and tries
to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional
embedding and the high-dimensional data.
Note: T-SNE falls outside the scope of this course, rendering this section supplemen-
tary.
First, we initialize T-SNE and transform the dataset to 2D. We set the parameterrandom_state
to 0 for reproducibility.
```
[18]: tsne =TSNE(random_state= 0 )
tsne_results= tsne.fit_transform(df)

```
Then we usesns.color_palette()to generate colors for specifying the colors for different clusters.
```
[19]: colors = sns.color_palette("bright", 8 )

```
After that, we usesns.scatterplot()to plot the transformed dataset in the scatter plot with clustered
labels.
```
[20]: sns.scatterplot(x = tsne_results[:, 0 ],y = tsne_results[:, 1 ], hue = labels_8,␣
↪palette =colors)

[20]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3adb58848>


```
2.2.1 2.2.1 Tuning hyperparameter
We tune the hyperparameter using the same method as described earlier.
```
[21]: _# Calculate SSE for different values of k_
sse =[]
**for** k **in** range( 1 , 13 ):
kmeans =KMeans(n_clusters=k)
kmeans.fit(df)
sse.append(kmeans.inertia_)

```
# Plot SSE vs. k
plt.figure(figsize=( 8 , 6 ))
plt.plot(range( 1 , 13 ), sse, marker='o')
plt.title('SSE vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(np.arange( 1 , 13 ))
plt.grid( True )
plt.show()
```

```
According to theElbowmethod introduced in the lecture, k = 3 can be considered as the best
value.
Then we implement a𝑘-means clustering with k = 3.
```
[22]: kmeans_3 =KMeans(n_clusters = 3 )
kmeans_3.fit(df)

[22]: KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
random_state=None, tol=0.0001, verbose=0)

```
Calculate the SSE by the attributeinertia_.
```
[23]: SSE_3= kmeans_3.inertia_
print('The SSE is:', SSE_3)

```
The SSE is: 140.
Save the predicted labels by the attributelabels_for evaluation.
```
[24]: labels_3 =kmeans_3.labels_


```
Plot it for visual evaluation as introduced above.
```
[25]: _# Initialize TSNE and transform the dataset to 2D_
tsne =TSNE(random_state= 0 )
tsne_results= tsne.fit_transform(df)

```
# Specify the colors for different clusters
colors = sns.color_palette("bright", 3 )
```
```
# Plot the transformed dataset in the scatter plot with clustered labels
sns.scatterplot(x = tsne_results[:, 0 ],y = tsne_results[:, 1 ], hue = labels_3,␣
↪palette =colors)
```
[25]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3aa96f848>

```
While the performance of the upper left section remains suboptimal, the visualization for k = 3
significantly outperforms that of k = 8.
```
## 3 3. Data not forming spherical clusters

### 3.1 3.1 Data preparation

```
First, we load the data for clustering.
```
[26]: df_spiral= pd.read_csv('spiral.csv')
X_spiral =df_spiral.iloc[:,:- 1 ].values
labels_spiral = df_spiral.iloc[:,- 1 ].values


```
Then we plot the data for visual inspection.
```
[27]: sns.scatterplot(x = X_spiral[:, 0 ],y = X_spiral[:, 1 ])

[27]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3b0194888>

```
We can observe that the data do not coalesce into spherical clusters, contrasting with the patterns
observed in the previous two cases.
Try to apply𝑘-means to it and assess its performance.
By visual inspection, the best k should be 2, so set then_clustersto 2 forKMeans().
```
[28]: kmeans_spiral = KMeans(n_clusters = 2 )
kmeans_spiral.fit(X_spiral)
labels_spiral = kmeans_spiral.labels_

[29]: colors = sns.color_palette("bright", 2 )

[30]: sns.scatterplot(x = X_spiral[:, 0 ],y = X_spiral[:, 1 ], hue = labels_spiral,␣
↪palette =colors)

[30]: <matplotlib.axes._subplots.AxesSubplot at 0x1f3b0135708>


The dataset indicates the limitation𝑘-means. In the upcoming week, we will introduce alternative
clustering models that exhibit improved results on this data.

Author: _Kaki Zhou_ 15/8/


