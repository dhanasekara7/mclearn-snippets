```python
wine = pd.read_csv("csv file", separator="")

wine.info()

wine.describe()

wine.isnull().sum()

bins=(1,6.5,9)
labels=['bad', 'good']
wine["quality"] = pd.cut(wine["quality"], bins=bins, labels=labels)
wine["quality"].unique()

l_encoder = LabelEncoder()
wine["quality"] = l_encoder.fit_transform(wine["quality"])

wine["quality"].value_counts()

sns.countplot(wine["quality"])

X = wine.drop('quality', axis=1)
y = wine["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## https://datascience.stackexchange.com/questions/12321/difference-between-fit-and-fit-transform-in-scikit-learn-models
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)

rfc = RandomForestClassifier(n_estimators=200)
svc = svm.SVC()

# Multi layer perceptron Classifier
mlpc = MLPClassifier(hidden_layer_sizes(11,11,11), max_iter=500)
mplc.fit(X_train, y_train)
y_pred_mlpc = mlpc.predict(X_test)

Xnew = [[1,2,3,4,5,6,7,8,9]]
Xnew = sc.transform(Xnew)
Ypred = rfc.predict(Xnew)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)

```