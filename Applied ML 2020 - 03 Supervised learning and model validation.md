```python
### KNN with scikit learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassififer
knn = KNeighborsClassififer(n_neighbors=1)
knn.fit(X_train, y_train)
print()"accuracy: ", knn.score(X_test, y_test)
y_pred = knn.predict(X_test)


```