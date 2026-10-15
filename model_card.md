# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model was developed by James Bundrick in October 2026 as part of the Udacity Machine Learning DevOps Engineer Nanodegree (Project 2: Deploying a Scalable ML Pipeline with FastAPI). It is a binary classifier that predicts whether a person's annual income is above or below $50,000 based on census attributes. The model is a scikit-learn `RandomForestClassifier` with 100 trees, default depth and split settings, and `random_state=42` so that results are reproducible. Categorical features are one-hot encoded with a scikit-learn `OneHotEncoder` (with `handle_unknown="ignore"`), and the label is encoded with a `LabelBinarizer`. The trained model and encoder are saved as `model/model.pkl` and `model/encoder.pkl` and are served through a FastAPI application.

## Intended Use

The model is intended for educational purposes: to demonstrate how to train, evaluate, test, and deploy a machine learning model through a reproducible pipeline with continuous integration and a REST API. Its intended users are students and reviewers of the project. It is not intended to make or support real decisions about individuals, such as hiring, lending, insurance, housing, or eligibility for services, because it is trained on old data and shows uneven performance across demographic groups.

## Training Data

The model was trained on the Census Income dataset (also known as the "Adult" dataset) from the UCI Machine Learning Repository, which was extracted from the 1994 U.S. Census database. The provided file `data/census.csv` contains 32,561 rows and 15 columns: six numeric features (age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week), eight categorical features (workclass, education, marital-status, occupation, relationship, race, sex, and native-country), and the label, salary. The data contains no blank values, but unknown values are recorded as "?" in workclass (1,836 rows), occupation (1,843 rows), and native-country (583 rows); these were kept as their own category. The label is imbalanced: 75.9% of rows are "<=50K" and 24.1% are ">50K".

The data was split into 80% training data (26,048 rows) and 20% test data (6,513 rows) using `train_test_split` with `random_state=42`, stratified on the salary column so both sets keep the same class balance. The categorical features were one-hot encoded and the numeric features were passed through unchanged, giving 108 input columns.

## Evaluation Data

The model was evaluated on the 20% held-out test set (6,513 rows), which was not used during training. The test set was processed with the encoder and label binarizer fitted on the training data, so it has exactly the same feature columns as the training data. In addition to overall performance, the model was evaluated on slices of the test set: one slice for every unique value of every categorical feature (101 slices in total). The slice results are saved in `slice_output.txt`.

## Metrics

Because the classes are imbalanced, accuracy alone would be misleading, so the model was evaluated with three metrics. Precision is the share of people predicted to earn ">50K" who actually do. Recall is the share of people who actually earn ">50K" whom the model correctly identifies. F1 is the harmonic mean of precision and recall, which balances the two.

On the test set, the model achieved a precision of 0.7353, a recall of 0.6378, and an F1 score of 0.6831. In other words, when the model predicts that someone earns more than $50,000 it is right about 74% of the time, but it finds only about 64% of the people who actually earn more than $50,000.

Performance varies across data slices. Some examples from `slice_output.txt`:

| Slice | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| sex: Female | 2,158 | 0.7680 | 0.6082 | 0.6788 |
| sex: Male | 4,355 | 0.7298 | 0.6432 | 0.6838 |
| race: White | 5,533 | 0.7340 | 0.6440 | 0.6861 |
| race: Black | 662 | 0.8060 | 0.5806 | 0.6750 |
| race: Asian-Pac-Islander | 200 | 0.6596 | 0.5962 | 0.6263 |
| race: Amer-Indian-Eskimo | 73 | 0.6667 | 0.4444 | 0.5333 |
| education: HS-grad | 2,120 | 0.5907 | 0.4130 | 0.4861 |
| education: Bachelors | 1,096 | 0.7483 | 0.7757 | 0.7618 |
| education: Doctorate | 84 | 0.8769 | 0.9048 | 0.8906 |
| workclass: ? | 363 | 0.5517 | 0.4444 | 0.4923 |

## Ethical Considerations

The dataset includes sensitive attributes such as race, sex, and native country, and the model uses them as input features. Because the data reflects income patterns from 1994, the model can learn and repeat historical inequalities. For example, only 10.9% of women in the dataset earn more than $50,000 compared with 30.6% of men, and the model may associate sex with income for that reason rather than for any causal one.

The slice metrics show that the model does not perform equally well for all groups. Recall is lower for women (0.6082) than for men (0.6432), and lower for Black (0.5806) and American Indian or Eskimo (0.4444) individuals than for White individuals (0.6440). This means that high earners in these groups are more likely to be missed by the model. Using this model to make decisions about people could therefore produce unfair outcomes, and it should not be used for that purpose.

## Caveats and Recommendations

The data is more than 30 years old, and incomes, inflation, occupations, and the makeup of the workforce have changed a great deal since 1994, so a $50,000 threshold does not mean the same thing today. The model should not be applied to current populations without retraining on recent data. Some slices are very small (for example, 45 people with race "Other" and 73 with race "Amer-Indian-Eskimo"), so their metrics are unreliable and can change a lot with a different random split. Unknown values recorded as "?" were treated as their own category rather than imputed, and the model performs noticeably worse for these rows.

Recommendations for future work include tuning hyperparameters with cross-validation, adjusting the classification threshold or class weights to improve recall on the minority ">50K" class, evaluating fairness metrics directly, considering whether sensitive attributes such as race and sex should be used as features at all, and retraining on more recent census data before any real-world use.