def end_of_import():
    return 0

def end_of_data():
    return 0

def end_of_fitting():
    return 0

def end_of_predicting():
    return 0

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
end_of_import()

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.25, shuffle=False
)
end_of_data()

clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)
end_of_fitting()

predicted = clf.predict(X_test)
end_of_predicting()