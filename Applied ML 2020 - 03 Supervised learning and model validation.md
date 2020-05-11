```python
### KNN with scikit learn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neighbors import KNeighborsClassififer
knn = KNeighborsClassififer(n_neighbors=1)
knn.fit(X_train, y_train)
print()"accuracy: ", knn.score(X_test, y_test)
y_pred = knn.predict(X_test)

### 3 fold split 
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval)
val_scores=[]
neighbors = np.arange(1,15, 2)
for i in neighbors:
    knn = KneighborsClassifier(i)
    knn.fit(X_train, y_train)
    val_scores.append(knn.score(X_val, y_val)
print("best validation score : {:.3f}".format(np.max(val_scores)))
best_n_neighbors = neighbors[np.argmax(val_scores)]

knn = KneighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_trainval, y_trainval)
print("test-set score : {:.3f}".format(knn.score(X_test, y_test)))

### cross val score
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y)

cross_val_scores=[]
neighbors = np.arange(1,15, 2)
for i in neighbors:
    knn = KneighborsClassifier(i)
    scores = cross_val_score(knn, X_train, y_train, cv=10)
    cross_val_scores.append(np.mean(scores))

print("best cross validation score : {:.3f}".format(np.max(cross_val_scores)))
best_n_neighbors = neighbors[np.argmax(cross_val_scores)]

knn = KneighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)
print("test-set score : {:.3f}".format(knn.score(X_test, y_test)))


# GridSearchCV ( brute force parameter search )
from sklearn.model_selection import GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y)

param_grid = {'n_neighbors' : np.arange(1,15,2)}
grid = GridSearchCV(KneighborsClassifier(), param_grid=param_grid, )
                    cv=10, reurn_train_score=True)
grid.fit(X_train. y_train)



```