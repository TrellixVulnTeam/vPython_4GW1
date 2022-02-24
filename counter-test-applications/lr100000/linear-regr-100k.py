def end_of_import():
    return 0

def end_of_init():
    return 0

def end_of_computing():
    return 0

import numpy as np
from sklearn.linear_model import LinearRegression
end_of_import()

X = np.array(range(0,100000)).reshape(-1, 1)
# y = 2x + 3
y = np.dot(X, 2) + 3
end_of_init()

reg = LinearRegression().fit(X, y)
end_of_computing()