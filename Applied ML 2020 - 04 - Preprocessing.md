```python

#    if np.mean(cross_val_score(knn, X, y, cv=kfold)) > bestAcc:

# standard scaler
from sklearn.linear import Ridge
#Back to King Country house prices
X_train, X_test, y_train, y_test = train_test_split(X, y, randon_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
# transform will subtract from the mean and divide by std. dev.
X_train_scaled = scaler.transform(X_train)

ridge = Ridge().fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
ridge.score(X_test_scaled, y_test)


### RidgeCV()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
scores = cross_val_score(RidgeCV(), X_train, y_train, cv=10)
np.mean(scores), np.std(scores)
# (0.694, 0.027)

scores = cross_val_score(RidgeCV(), X_train_scaled, y_train, cv=10)
np.mean(scores), np.std(scores)
# (0.694, 0.027)

from sklearn.neighbors import KNeighborsRegressor
scores = cross_val_score(KNeighborsRegressor(), X_train, y_train, cv=10)
np.mean(scores), np.std(scores)
# (0.500, 0.039)

from sklearn.neighbors import KNeighborsRegressor
scores = cross_val_score(KNeighborsRegressor(), X_train_scaled, y_train, cv=10)
np.mean(scores), np.std(scores)
# (0.786, 0.030)


# preprocesssing and pipelines
# common error ( portion of data to reduce feature selection)
print(X.shape)
# (100, 10000)

# select most informative 5% of features
from sklearn.feature_selection import SelectPercentile, f_regression
select = SelectPercentile(score_func=f_regression, percentile=5)
select.fit(X, y)
X_selected = select.transform(X)
print(X_selected.shape)
# (100, 500)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
np.mean(cross_val_score(Ridge(), X_selected, y))
#0.9

ridge = Ridge().fit(X_selected, y)
X_test_selected = select.transform(X_test)
ridge.score(X_test_selected, y_test)
#-0.18

# Leaking information
#BAD!
select.fit(X, y) # include cv test parts
X_sel = select.transform(X)
scores = []
for train, test in cv.split(X, y):
    ridge = Ridge().fit(X_sel[train], y[train])
    score = ridge.score(X_sel[test], y[test])
    scores.append(score)
# same as
select.fit(X, y)
X_sel = select.transform(X, y)
np.mean(cross_val_score(Ridge(), X_sel, y))
# 0.9

# GOOD
scores = []
for train, test in cv.split(X, y):
    select.fit(X[train])
    X_sel_train = select.transform(X[train])
    ridge = Ridge().fit(X_sel_train, y[train])
    X_sel_test = select.transform(X[test])
    score = ridge.score(X_sel_test, y[test])
    scores.append(score)
# same as
pipe = Pipeline([("select", select),
                 ("ridge", Ridge())])
np.mean(cross_val_score(pipe, X, y))
# -0.79

# housing data example
from sklearn.linear_model import Ridge
X, y = df, target

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scalar.transform(X_train)
ridge = Ridge().fit(X_train_scaled, y_train)

X_test_scaled = scalar.transform(X_test)
ridge = Ridge().fit(X_test_scaled, y_test)
# 0.684

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
# 0.684

# Naming steps
from sklearn.pipeline import make_pipeline
knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
print(knn_pipe.steps)
[('standardscaler', StandardScaler(with_mean=True, with_std=True)),
('kneighborsregressor', KNeighborsRegressor(algorithm='auto',...))]

from sklearn.pipeline import Pipeline
pipe = Pipeline((("scaler", StandardScaler()),
                 ("regressor", KNeighborsRegressor) ))

# Pipeline and GridSearchCV
from sklearn.model_selection import GridSearchCV

knn_pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
param_grid = {'kNeighborsregressor_n_neighbors' : range(1,10)}
grid = GridSearchCV(knn_pipe, param_grid, cv=10)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))

#{'kNeighborsregressor_n_neighbors': 7}
#0.60

# going wild with pipelines
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=0)

from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), KNeighborsRegressor())
param_grid = {'polynomialfeatures_degree' : [1, 2, 3],
              'ridge_alpha' : [0.001, 0.01, 0.1, 1, 10, 100] }
grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, return_train_score=True)
grid.fit(X_train, y_train)

# going wild with pipelines2
pipe = Pipeline([('scaler', StandardScaler()),
                 ('regressor', Ridge())])
param_grid = {'scaler' : [StandardScaler(), MinMaxScaler(), 'passthrough'],
              'regressor' : [Ridge(), Lasso()],
              'regressor__alpha' : np.logspace(-3, 3, 7)}
grid = GridSearchCV(pipe, param_grid=param_grid)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)

# going wild with pipelines3
from sklearn.tree import DecisionTreeRegressor

pipe = Pipeline([('scaler', StandardScaler()),
                 ('regressor', Ridge())])
param_grid = [{'regressor' : [DecisionTreeRegressor()],
              'regressor__max_depth' : [2, 3, 4],
              'scaler' : ['passthrough']
              },
              {'regressor' : [Ridge()],
              'regressor__alpha' : [0.1, 1],
              'scaler' : [StandardScaler(), MinMaxScaler(), 'passthrough']
              },


              ]
grid = GridSearchCV(pipe, param_grid=param_grid)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)

### categorical variables

#boro -> column name
df["boro_ordinal"] = df.boro.astype("category").cat.codes

# pandas one hot ( dummy ) encoding
pd.get_dummies(df) # for all possible columns

pd.get_dummies(df, columns=['boro']) # only for the boro column

## more nice way, just in case of more category will be added in the future
df["boro"] = pd.Categorical(df.boro, categories=["Manhattan",
                                                 "Queens",
                                                 "Brooklyn",
                                                 "Bronx",
                                                 "Staten Island"])
pd.get_dummies(df, columns=['bro'])

# OneHotEncoder ( scikitlearn)
ce = OneHotEncoder().fit(df)
ce.transform(df).toarray()

# OneHotEncoder + ColumnTransformer
categorical = df.types == object

preprocess = make_column_transformer(
    (StandardScaler(), ~categorical),
    (OneHotEncoder(), categorical)
    )

model = make_pipeline(preprocess, LogisticRegression())


# target encoding
te = TargetEncoder(cols='zipcode').fit(X_train, y_train)
te.transform(X_train).head()

X = data.frame.drop(["date", "price", "zipcode"], axis=1)
scores = cross_val_score(Ridge(), X, target)
np.mean(scores)
#0.69

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
X = data.frame.drop(["date", "price"], axis=1)

ct = make_column_transformer((OneHotEncoder(), ['zipcode']), remainder='passthrough')
pipe_ohe = make_pipeline(ct, Ridge())
scores = cross_val_score(pipe_ohe, X, target)
np.mean(scores)
# 0.52

from category_encoders import TargetEncoder
X = data.frame.drop(['date', 'price'], axis=1)
pipe_target = make_pipe_line(TargetEncoder(cols='zipcode'), Ridge())
scores = cross_val_score(pipe_target, X, target)
np.mean(scores)
# 0.78

```
