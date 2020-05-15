```python
# dropping columns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test =  train_test_split(X_, y, stratify=y)
nan_columns = np.any(np.isnan(X_train), axis=0)
X_drop_columns = X_train[:, ~nan_columns]
scores = cross_val_score(LogisticRegressionCV(v=5), X_drop_columns, y_train, cv=10)
np.mean(scores)


# SimpleImputer
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy=median).fit(X_train)
X_median_imp = imp.transform(X_train)

# drop column + pipline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
nan_columns = np.any(np.isnan(X_train), axis=0)
X_drop_columns = X_train[:, ~nan_columns]
log_reg = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(log_reg, X_drop_columns, y_train, cv=10)
np.mean(scores)
#0.794

# simple imputer + LogisticRegression
mean_pipe = make_pipeline(SimpleImputer(strategy='median'),
                          StandardScalar(),
                          LogisticRegression()
                         )
scores = cross_val_score(mean_pipe, X_train, y_train, cv=10)
np.mean(scores)
#0.849

# KNN Imputation
knn_pipe = make_pipeline(KNNImputer(), StandardScaler(), LogisticRegression())
np.mean(scores)

# Model driven Imputation
rf_imp = IterativeImputer(predictor=RandomForestRegressor())
rf_pipe = make_pipeline(rf_imp, StandardScaler(), LogisticRegression())
scores = cross_val_score(rf_pipe, X_rf_imp, y_train, cv=10)
np.mean(scores)
# 0.845

# Linear Regression

# Ridge Regression





                        

```