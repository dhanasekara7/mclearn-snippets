```python
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data = pd.read_csv("../../datasets/titanic_train.csv")
data.head()

data = data[['Pclass', 'Sex', 'Age', 'Survived']]

gender_cloumns = {"male": 1, "female" : 0}

data['Sex'] = data['Sex'].map(gender_cloumns)

data.describe()
data.count()

len(data[data.isnull().any(axis=1)])

data = data.dropna()

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, randomsize=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, predictions)
print("Accuracy score %.2f%%" % (acc * 100))

```
