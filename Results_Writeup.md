# Results and Limitations: Titanic Survival Classifier

**Prepared by:** Your Name | **Project 04 | Data Science & AI | InternCircle**

## The prediction
Predict whether a passenger survived the Titanic from their ticket class, sex, age and fare. `survived` is the target; `pclass`, `sex`, `age` and `fare` are the features.

## Method
- Data: Titanic dataset from seaborn, 714 passengers after dropping 177 rows with a missing age
- Split: 80% training (571 passengers) and 20% test (143), `random_state=42`, stratified so both sets have a 40.6% survival rate
- Model: `DecisionTreeClassifier(max_depth=4, random_state=42)`, trained on the training set only
- Baseline: always guess "did not survive"

## Results
| Measure | Value |
|---|---|
| Baseline accuracy | 59.4% |
| **Test accuracy** | **79.7%** (+20.3 points over baseline) |
| Training accuracy | 84.8% |
| MSE (test) | 0.203 (equals the error rate for a 0/1 target) |
| Precision (survived) | 0.76 |
| Recall (survived) | 0.72 |
| 5-fold cross-validated accuracy (depth 4) | about 79% |

**Confusion matrix** (143 test passengers)

| | Predicted: did not survive | Predicted: survived |
|---|---|---|
| Actual: did not survive | 72 | 13 |
| Actual: survived | 16 | 42 |

**Most important features:** sex (0.55), ticket class (0.24), age (0.14), fare (0.08).

**Overfitting:** training accuracy rises from 77.9% (depth 1) to 96.8% (depth 12), while test accuracy stays between 76% and 81% (77.6% at depth 12). Very deep trees memorise the training passengers instead of learning the general pattern.

## Limitations
1. **Small dataset.** Only 143 test passengers, so one passenger changes accuracy by 0.7 points. Cross-validation shows depths 3 to 6 perform about the same.
2. **Dropped rows.** The 177 passengers with missing age were mostly 3rd class (77% vs 55% overall) and less likely to survive (29% vs 41%), so dropping them may bias the model.
3. **Few features.** Family size, title, port and cabin are not used.
4. **Misses survivors.** Recall for survivors is 72%, so about 28% of survivors are predicted not to survive.
5. **Patterns, not causes.** The model finds links in historical data and does not prove why people survived.
