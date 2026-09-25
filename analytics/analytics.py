import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
# I keep a local CSV as a fallback so the analysis can still run offline.
try:
    df = sns.load_dataset("titanic")
    df.to_csv("titanic.csv", index=False)
    print("Titanic dataset loaded from Seaborn and saved locally.")
except Exception:
    df = pd.read_csv("titanic.csv")
    print("Could not load from Seaborn, so I used the local titanic.csv copy.")

print("Dataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nDataset summary:")
print(df.describe(include="all"))

# Missing value percentages
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

print("\nMissing value percentages:")
print(missing_percent[missing_percent > 0])

# ---------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------

print("\nMissing value handling decisions:")

# Under 5% missing -> drop affected rows
df = df.dropna(subset=["embarked", "embark_town"])

# 5% to 30% missing -> impute
age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)

# Very high missing rate -> drop column
df = df.drop(columns=["deck"])

print("Dropped rows with missing embarked/embark_town")
print("Filled missing age values with median:", age_median)
print("Dropped deck because its missing rate is very high")

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ---------------------------------------------------------
# Univariate analysis: age and fare
# ---------------------------------------------------------

def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) | (series > upper_bound)
    ]

    return len(outliers), lower_bound, upper_bound


age_outliers, age_lower, age_upper = iqr_outlier_count(df["age"])
fare_outliers, fare_lower, fare_upper = iqr_outlier_count(df["fare"])

print("\nAge statistics:")
print("Mean:", df["age"].mean())
print("Median:", df["age"].median())
print("Mode:", df["age"].mode()[0])
print("IQR outliers:", age_outliers)

print("\nFare statistics:")
print("Mean:", df["fare"].mean())
print("Median:", df["fare"].median())
print("Mode:", df["fare"].mode()[0])
print("IQR outliers:", fare_outliers)

fare_skewness = df["fare"].skew()

print("\nFare skewness:", fare_skewness)

if fare_skewness > 0:
    print("Fare distribution is right-skewed.")
elif fare_skewness < 0:
    print("Fare distribution is left-skewed.")
else:
    print("Fare distribution is approximately symmetric.")


# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("age_histogram.png")
plt.close()


# Age boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["age"])
plt.title("Age Boxplot")
plt.tight_layout()
plt.savefig("age_boxplot.png")
plt.close()


# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fare_histogram.png")
plt.close()


# Fare boxplot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["fare"])
plt.title("Fare Boxplot")
plt.tight_layout()
plt.savefig("fare_boxplot.png")
plt.close()

print("\nCharts saved successfully.")

# ---------------------------------------------------------
# Bivariate analysis
# ---------------------------------------------------------

print("\nSurvival rate by sex:")
survival_by_sex = df.groupby("sex")["survived"].mean()
print(survival_by_sex)

print("\nSurvival rate by passenger class:")
survival_by_pclass = df.groupby("pclass")["survived"].mean()
print(survival_by_pclass)

print("\nSurvival rate by sex and passenger class:")
survival_by_sex_pclass = df.groupby(["sex", "pclass"])["survived"].mean()
print(survival_by_sex_pclass)


# ---------------------------------------------------------
# Correlation matrix - exact required columns
# ---------------------------------------------------------

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df[corr_columns].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

# Find strongest off-diagonal correlations
corr_abs = corr_matrix.abs()

pairs = []

for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):
        pairs.append(
            (
                corr_columns[i],
                corr_columns[j],
                corr_abs.iloc[i, j],
                corr_matrix.iloc[i, j]
            )
        )

pairs = sorted(
    pairs,
    key=lambda x: x[2],
    reverse=True
)

print("\nTwo strongest off-diagonal correlations:")

for pair in pairs[:2]:
    print(
        pair[0],
        "and",
        pair[1],
        "- correlation:",
        pair[3]
    )


# Correlation heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

print("\nCorrelation heatmap saved successfully.")


# ---------------------------------------------------------
# Task 5 - Multivariate data story
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="sex", y="survived", hue="pclass")
plt.title("Survival Rate by Sex and Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("survival_by_sex_class.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="pclass", y="age", hue="survived")
plt.title("Age by Passenger Class and Survival")
plt.tight_layout()
plt.savefig("age_class_survival.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived")
plt.title("Age vs Fare by Survival")
plt.tight_layout()
plt.savefig("age_fare_survival.png")
plt.close()


plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Survival by Passenger Class and Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("class_sex_survival.png")
plt.close()

print("\nFour multivariate charts saved successfully.")

print("""
Multivariate interpretation:
Female passengers had much higher survival rates than male passengers.
First-class passengers also had higher survival rates than second- and third-class passengers.
The combination of sex and passenger class shows that female passengers in higher classes had the strongest survival outcomes.
Fare is also related to passenger class, so passengers paying higher fares were often in the classes with better survival outcomes.
""")


# ---------------------------------------------------------
# Task 6 - Exploratory standardization
# ---------------------------------------------------------

eda_scaled = df[["age", "fare"]].copy()

print("\nBefore standardization:")
print(eda_scaled.agg(["mean", "std"]))

eda_scaled["age"] = (
    eda_scaled["age"] - eda_scaled["age"].mean()
) / eda_scaled["age"].std()

eda_scaled["fare"] = (
    eda_scaled["fare"] - eda_scaled["fare"].mean()
) / eda_scaled["fare"].std()

print("\nAfter standardization:")
print(eda_scaled.agg(["mean", "std"]))

print("""
The standardized age and fare columns have means close to 0
and standard deviations close to 1.
This standardization is only for exploratory analysis.
The modeling pipeline will perform its own scaling using training data only.
""")
# ---------------------------------------------------------
# Part B - Predictive Modeling
# Task 7 - Stratified train/test split
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Use survived as the classification target
X = df[
    [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]
].copy()

y = df["survived"].copy()

print("\nOverall class balance:")
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining class balance:")
print(y_train.value_counts(normalize=True))

print("\nTest class balance:")
print(y_test.value_counts(normalize=True))

print("""
A stratified split is used because the survived and not-survived
classes are not perfectly balanced. Stratification keeps approximately
the same class proportions in both the training and test datasets.
""")


# ---------------------------------------------------------
# Task 8 - Preprocessing
# Fit only on training data
# ---------------------------------------------------------

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training-fitted preprocessing
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

print("""
Preprocessing was fitted only on the training split.
The test set was transformed using the already-fitted preprocessing
pipeline, which avoids leaking test-set information into training.
""")

# ---------------------------------------------------------
# Task 9 - Train three classifiers
# Task 10 - Evaluate all classifiers
# ---------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


model_results = {}

plt.figure(figsize=(8, 6))

for name, model in models.items():

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    model_results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", auc)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix
    plt_cm = plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{safe_name}_confusion_matrix.png")
    plt.close(plt_cm)

    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(1)
    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )


# ROC curve for all three models
plt.figure(1)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Classification Models")
plt.legend()
plt.tight_layout()
plt.savefig("classification_roc_curves.png")
plt.close()


# ---------------------------------------------------------
# Decision Tree visualization
# ---------------------------------------------------------

decision_tree = models["Decision Tree"]

feature_names = preprocessor.get_feature_names_out()

plt.figure(figsize=(20, 10))

plot_tree(
    decision_tree,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree.png")
plt.close()


# ---------------------------------------------------------
# Comparison table
# ---------------------------------------------------------

classification_results = pd.DataFrame(model_results).T

print("\nClassification model comparison:")
print(classification_results)

classification_results.to_csv(
    "classification_model_results.csv"
)

print("\nClassifier evaluation completed successfully.")
# ---------------------------------------------------------
# Task 11 - Imbalance handling comparison
# Baseline vs class_weight='balanced' vs SMOTE
# ---------------------------------------------------------

from imblearn.over_sampling import SMOTE

print("\nClass balance before SMOTE:")
print(y_train.value_counts())

imbalance_results = {}

# 1. Baseline Logistic Regression
baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

baseline_model.fit(X_train_processed, y_train)

baseline_pred = baseline_model.predict(X_test_processed)

imbalance_results["Baseline"] = {
    "precision": precision_score(y_test, baseline_pred),
    "recall": recall_score(y_test, baseline_pred),
    "f1": f1_score(y_test, baseline_pred)
}


# 2. Logistic Regression with balanced class weights
balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

balanced_model.fit(X_train_processed, y_train)

balanced_pred = balanced_model.predict(X_test_processed)

imbalance_results["Class Weight Balanced"] = {
    "precision": precision_score(y_test, balanced_pred),
    "recall": recall_score(y_test, balanced_pred),
    "f1": f1_score(y_test, balanced_pred)
}


# 3. SMOTE - training data only
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nClass balance after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

smote_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

smote_model.fit(X_train_smote, y_train_smote)

smote_pred = smote_model.predict(X_test_processed)

imbalance_results["SMOTE"] = {
    "precision": precision_score(y_test, smote_pred),
    "recall": recall_score(y_test, smote_pred),
    "f1": f1_score(y_test, smote_pred)
}


imbalance_comparison = pd.DataFrame(imbalance_results).T

print("\nImbalance handling comparison:")
print(imbalance_comparison)

imbalance_comparison.to_csv(
    "imbalance_comparison.csv"
)

best_method = imbalance_comparison["f1"].idxmax()

print(
    "\nBased on F1 score, the strongest imbalance-handling "
    f"approach in this comparison is: {best_method}."
)

print("""
SMOTE was applied only to the training data.
The original test set was left unchanged so that the evaluation
continues to represent unseen real-world data.
""")
# ---------------------------------------------------------
# Task 12 - Random Forest hyperparameter tuning
# GridSearchCV + OOB score
# ---------------------------------------------------------

from sklearn.model_selection import GridSearchCV

rf_for_tuning = RandomForestClassifier(
    random_state=42,
    oob_score=True
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    estimator=rf_for_tuning,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train_processed,
    y_train
)

best_rf = grid_search.best_estimator_

print("\nRandom Forest GridSearchCV results:")
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation F1:", grid_search.best_score_)
print("OOB score:", best_rf.oob_score_)

best_rf_predictions = best_rf.predict(X_test_processed)
best_rf_probabilities = best_rf.predict_proba(X_test_processed)[:, 1]

print("\nTuned Random Forest test metrics:")
print("Accuracy:", accuracy_score(y_test, best_rf_predictions))
print("Precision:", precision_score(y_test, best_rf_predictions))
print("Recall:", recall_score(y_test, best_rf_predictions))
print("F1 Score:", f1_score(y_test, best_rf_predictions))
print("ROC AUC:", roc_auc_score(y_test, best_rf_probabilities))
# ---------------------------------------------------------
# Task 13 - Regression side-task
# Predict fare using multivariate linear regression
# ---------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Use a reasonable set of predictors for fare
regression_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

X_reg = df[regression_features].copy()
y_reg = df["fare"].copy()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

reg_categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            reg_numeric_pipeline,
            reg_numeric_features
        ),
        (
            "cat",
            reg_categorical_pipeline,
            reg_categorical_features
        )
    ]
)

regression_pipeline = Pipeline(
    steps=[
        ("preprocessor", reg_preprocessor),
        ("model", LinearRegression())
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

fare_predictions = regression_pipeline.predict(
    X_reg_test
)

mae = mean_absolute_error(
    y_reg_test,
    fare_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        fare_predictions
    )
)

r2 = r2_score(
    y_reg_test,
    fare_predictions
)

n = len(y_reg_test)

# Use the actual number of predictors after preprocessing.
X_reg_test_processed = regression_pipeline.named_steps["preprocessor"].transform(X_reg_test)
p = X_reg_test_processed.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\nRegression metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)


# ---------------------------------------------------------
# Residual plot
# ---------------------------------------------------------

residuals = y_reg_test - fare_predictions

plt.figure(figsize=(8, 5))

plt.scatter(
    fare_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Plot - Fare Regression")
plt.tight_layout()

plt.savefig("fare_regression_residuals.png")
plt.close()


# Simple heteroscedasticity interpretation
residual_correlation = np.corrcoef(
    np.abs(residuals),
    fare_predictions
)[0, 1]

print(
    "\nCorrelation between absolute residuals "
    "and predicted fare:",
    residual_correlation
)

if abs(residual_correlation) > 0.3:
    print(
        "The residual spread changes noticeably "
        "with predicted fare, suggesting heteroscedasticity."
    )
else:
    print(
        "The residual spread does not show a strong "
        "systematic change, so heteroscedasticity is not obvious."
    )
# ---------------------------------------------------------
# Task 14 - Final model comparison
# ---------------------------------------------------------

final_comparison = classification_results.copy()

final_comparison["MAE"] = np.nan
final_comparison["RMSE"] = np.nan
final_comparison["R2"] = np.nan
final_comparison["Adjusted_R2"] = np.nan

final_comparison.loc["Linear Regression"] = {
    "accuracy": np.nan,
    "precision": np.nan,
    "recall": np.nan,
    "f1": np.nan,
    "auc": np.nan,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}

print("\nFinal model comparison table:")
print(final_comparison)

final_comparison.to_csv(
    "final_model_comparison.csv"
)

best_classifier_name = classification_results["f1"].idxmax()
best_classifier_f1 = classification_results.loc[
    best_classifier_name,
    "f1"
]

print(f"""
Final recommendation:
Among the three classifiers, {best_classifier_name} produced the
highest F1 score in this run, with an F1 score of
{best_classifier_f1:.3f}. I would use this classifier as the preferred
classification model because it gave the best balance between
precision and recall on the test set.

The regression model is evaluated separately because regression
metrics are on a different scale from classification metrics.
Its R2 was {r2:.3f}, with an adjusted R2 of {adjusted_r2:.3f}.
""")


# ---------------------------------------------------------
# Task 15 - Save complete fitted pipeline
# ---------------------------------------------------------

import joblib

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

final_pipeline.fit(
    X_train,
    y_train
)

joblib.dump(
    final_pipeline,
    "final_classification_pipeline.joblib"
)

print("\nSaved complete fitted pipeline successfully.")

# Reload and test on raw data
loaded_pipeline = joblib.load(
    "final_classification_pipeline.joblib"
)

reload_predictions = loaded_pipeline.predict(
    X_test.head(5)
)

original_predictions = final_pipeline.predict(
    X_test.head(5)
)

print("\nOriginal pipeline predictions:")
print(original_predictions)

print("\nReloaded pipeline predictions:")
print(reload_predictions)

print(
    "\nReloaded pipeline matches original:",
    np.array_equal(
        original_predictions,
        reload_predictions
    )
)
# ---------------------------------------------------------
# Correct final saved pipeline to use the selected model
# ---------------------------------------------------------

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=grid_search.best_params_["n_estimators"],
                max_depth=grid_search.best_params_["max_depth"],
                max_features=grid_search.best_params_["max_features"],
                random_state=42,
                oob_score=True
            )
        )
    ]
)

final_pipeline.fit(X_train, y_train)

joblib.dump(
    final_pipeline,
    "final_classification_pipeline.joblib"
)

loaded_pipeline = joblib.load(
    "final_classification_pipeline.joblib"
)

original_predictions = final_pipeline.predict(
    X_test.head(5)
)

reload_predictions = loaded_pipeline.predict(
    X_test.head(5)
)

print("\nCorrected final pipeline saved using tuned Random Forest.")

print("Original predictions:")
print(original_predictions)

print("Reloaded predictions:")
print(reload_predictions)

print(
    "Reloaded pipeline matches original:",
    np.array_equal(
        original_predictions,
        reload_predictions
    )
)
