# Telco Customer Churn Prediction

While learning machine learning, I decided to learn and implement concepts together on a single real dataset - the Telco Customer Churn dataset - rather than jumping between datasets and tutorials. This project documents that process: from first exploring the data to comparing multiple models and picking a final one, with reasoning at each step.

## Dataset Overview

Using basic pandas exploration, I first got an overview of what I was working with:

- **7,043 rows, 21 columns**
- Mainly `object` (text) data type columns
- **No null values and no duplicate rows** in the raw data
- Target column: `Churn` - **5,174 "No"** vs **1,869 "Yes"** (imbalanced, ~73%/27%)

I also ran a **pandas profiling report** (`ydata-profiling`) to get a fuller picture of each column - value distributions, unique counts, and how individual columns related to each other - before starting to clean anything.

## Data Cleaning

- Dropped `customerID` - a unique identifier with no predictive value.
- Used **OrdinalEncoder** to encode binary Yes/No columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) as 0/1.
- For other categorical columns that aren't ordered/level-wise (like `Contract`, `InternetService`, `PaymentMethod`), used **one-hot encoding** instead - since these categories have no natural rank, one-hot avoids forcing a false order between them. This produced boolean (True/False) columns, which I then converted to integers (0/1).
- `TotalCharges` was stored as text (`dtype('O')`) due to a small number of blank entries (new customers with 0 tenure). Converted it to numeric with `pd.to_numeric(errors='coerce')` and dropped the resulting null rows.

After this, the dataset was fully numeric and ready for analysis.

## Exploratory Data Analysis

**Finding relationships with churn:**
- Grouped by `tenure` to see how churn rate changes over a customer's lifetime.
- Used `.corr()['Churn']` to get a single, sortable view of how every column relates to churn — and understood clearly what a positive vs. negative correlation actually means in this context.

**Strongest correlations found:**
| Feature | Correlation with Churn |
|---|---|
| `tenure` | -0.354 |
| `InternetService_Fiber optic` | +0.307 |
| `Contract_Two year` | -0.302 |
| `PaymentMethod_Electronic check` | +0.301 |
| `MonthlyCharges` | +0.193 |
| `PaperlessBilling` | +0.191 |
| `SeniorCitizen` | +0.151 |

**Visualizations** (matplotlib + seaborn):
- Churn rate by tenure
- Monthly charges by churn
- Churn by two-year contract

**Key insights from visualization:**
- **Churn rate decreases with tenure** - new customers churn far more than long-term ones.
- **The spread for churned customers is narrower** - churned customers cluster in the $55–95 monthly-charge range, while retained customers are spread more widely from $25–90. This kind of detail (spread, not just average) is something a correlation number alone can't show - only the plot reveals it.
- **Two-year contracts strongly reduce churn.** Customers on two-year contracts almost never churn, while customers without long-term contracts churn at a much higher rate - contract length appears to be one of the strongest retention factors.

## Modeling

Trained and compared three classifiers - Logistic Regression, Decision Tree, and Random Forest - each with a default version and a `class_weight='balanced'` version, to address the class imbalance in the target.

Also explored hyperparameter tuning:
- Manually tested Decision Tree across `max_depth` values (3, 5, 7, 9, 10, None) to observe the underfitting -> overfitting curve firsthand.
- Used **GridSearchCV** to search Random Forest hyperparameters (`n_estimators`, `max_features`, `max_depth`, `max_samples`, `bootstrap`, `min_samples_split`, `min_samples_leaf`) rather than tuning by hand.
- Used **OOB (out-of-bag) score** on Random Forest as a built-in validation check, without needing a separate validation split.

### Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.793 | 0.592 | 0.503 | 0.544 |
| **Logistic Regression (balanced)** | 0.732 | 0.470 | **0.734** | **0.573** |
| Decision Tree | 0.787 | 0.612 | 0.364 | 0.457 |
| Decision Tree (balanced) | 0.636 | 0.389 | **0.838** | 0.531 |
| Random Forest | 0.808 | 0.676 | 0.422 | 0.520 |
| Random Forest (balanced) | 0.719 | 0.457 | 0.769 | 0.573 |

### Key takeaways

- **Accuracy alone is misleading on this dataset.** Every default model scored close to or above 0.79 accuracy while catching under 51% of actual churners - a model can look strong overall by favoring the majority "stayed" class.
- **`class_weight='balanced'` trades accuracy for recall, deliberately.** Every balanced version showed a large recall jump (up to 0.838 for Decision Tree) at the cost of accuracy and precision — a direct, explainable result of reweighting how much the minority class's mistakes cost during training.
- **Logistic Regression (balanced) and Random Forest (balanced) tied almost exactly on F1** (0.573 vs 0.573) - Random Forest was chosen as the final model, since it also has slightly higher recall (0.769 vs 0.734) and benefits from ensemble stability across many trees.

## Final Model: Random Forest (balanced)

Chosen for its strong recall (0.769) and best-tied F1 score (0.573), with the added benefit of ensemble stability (built from many trees) and a built-in OOB score for validation without a separate holdout set. Feature importance from this model (plotted in the notebook) also gives a clear view of which columns drove predictions.

**Why recall matters most here:** missing an actual churner (false negative) costs the business a lost customer, while a false alarm (false positive) only costs an unnecessary retention offer. Since missing a churner is the more expensive mistake, this model was deliberately chosen to prioritize recall over raw accuracy.

## What I Learned

This project was as much about learning ML fundamentals as it was about the dataset itself. Concepts covered, in the order I learned them:

**Data handling**
- Reading, inspecting, and cleaning data with pandas (`.info()`, `.isnull()`, `.duplicated()`)
- Handling columns stored in the wrong data type (`TotalCharges` as text due to blank values)
- Ordinal Encoding vs. One-Hot Encoding — and *why* the choice matters (avoiding false ordinal relationships between unordered categories)
- Why feature scaling matters for some models (Logistic Regression, KNN, SVM) but not others (Decision Tree, Random Forest) - because trees split on order/thresholds, not distance or gradients
- Data leakage - why you must split before scaling/encoding, never after, so test data never influences training

**Exploratory Data Analysis**
- Using `groupby()` to uncover relationships (e.g., churn rate by tenure)
- Reading correlation values - what positive vs. negative actually means, and their limits (only linear relationships, doesn't imply causation)
- Multicollinearity - noticing redundant features (e.g., the `_No internet service` columns) that carry near-identical information

**Models**
- Logistic Regression - how it estimates probabilities and makes decisions
- Decision Trees - how splits work (Gini Impurity, Entropy), and how `max_depth` and other hyperparameters control complexity
- Random Forest - bagging (bootstrap + aggregating), why randomness in row *and* column sampling reduces variance, and how it differs from plain bagging
- The Bias–Variance tradeoff - why a single decision tree is low-bias/high-variance, and how Random Forest reduces variance while keeping bias low

**Evaluation**
- Why accuracy is misleading on imbalanced data
- Precision, Recall, and F1 Score -> what each means and when each matters more, based on the real-world cost of false positives vs. false negatives
- Confusion matrices -> reading true/false positives and negatives directly
- `class_weight='balanced'` -> how reweighting the minority class's training penalty trades accuracy for recall, and why that trade-off is worth it for churn prediction
- Out-of-Bag (OOB) score -> using leftover bootstrap samples as a built-in validation check
- Hyperparameter tuning with `GridSearchCV` -> systematically searching combinations instead of guessing values by hand

**Overall workflow**
- How to structure an ML project end-to-end: clean → explore → model → tune → evaluate → compare → conclude
- The value of comparing multiple models honestly, including reporting when a "supposedly stronger" model (Random Forest) initially underperformed a simpler one (Logistic Regression) before tuning
- Why the "best" model isn't always the one with the highest accuracy -> it depends on which mistake is more costly for the specific problem

## Tools & Techniques Used

**Language & libraries:** Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, ydata-profiling

**Models:** Logistic Regression, Decision Tree Classifier, Random Forest Classifier (all from scikit-learn)

**Techniques:** OrdinalEncoder, One-Hot Encoding, StandardScaler, train/test split, class-weight balancing, GridSearchCV for hyperparameter tuning, OOB scoring

