import numpy as np
import time
from using_pyseal import homenc as he
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def sigmoid_original(x):
    return 1 / (1 + np.exp(-x))


# sigmoid approximations

# degree 3, interval [-5,5]
# see Hao Chen, Ran Gilad-Bachrach, Kyoohyung Han, Zhicong Huang, Amir Jalali, Kim Laine, and Kristin Lauter:
#  Logistic regression over encrypted data from fully homomorphic encryption. BMC medical genomics, 11(4):81, 2018.
def sigmoid_minimax(x):
    return (-0.004 * x * x * x) + (0.197 * x) + 0.5


# degree 3, interval [-6,6]
def sigmoid_polyfit(x):
    return (-0.002564 * (x**3)) - (-0.001435 * (x**2)) + (0.1523 * x) + 0.4576


# degree 14, interval [-1,1]
def sigmoid_chebyshev(x):
    return -3.33481e-12 * x ** 14 + 1.52005e-07 * x ** 13 + 1.14899e-11 * x ** 12 - 2.07025e-06 * x ** 11 - 1.58991e-11 * x ** 10 + 2.12879e-05 * x ** 9 + 1.12891e-11 * x ** 8 - 0.000210786 * x ** 7 - 4.31714e-12 * x ** 6 + 0.00208333 * x ** 5 + 8.32342e-13 * x ** 4 - 0.0208333 * x ** 3 - 6.29348e-14 * x ** 2 + 0.25 * x + 0.5 * 1


# see https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
# and https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html#gradient-descent
class BinLogReg:

    def __init__(
            self,
            iterations=10000,
            learning_rate=0.02,
            sigmoid=sigmoid_minimax,
            encrypted=False
    ):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.sigmoid = sigmoid
        self.encrypted = encrypted
        self.weights = None

        if self.sigmoid == sigmoid_minimax:
            self.sigmoid_interval_from = -5
            self.sigmoid_interval_to = 5
        else:
            self.sigmoid_interval_from = None
            self.sigmoid_interval_to = None

        # print params
        str_fit = "Fitting encrypted model ..." if self.encrypted else "Fitting plain model ..."
        print(str_fit)
        print(f"  Iterations: {self.iterations}")
        print(f"  Learning rate: {self.learning_rate}")

    def fit(self, X, y):
        t = time.process_time()

        # add intercept
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)

        # initialize weights
        np.random.seed(1204)
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=X.shape[1])

        mean = 1 / y.size

        for i in range(self.iterations):
            # predict
            z = np.dot(X, self.weights)

            y_predictions = self.sigmoid(z)

            if self.encrypted:
                # recrypt y_predictions, this simulates bootstrapping and should be replaced
                y_predictions = he.recrypt_ndarray(y_predictions)

            # calculate average loss gradient
            gradient = np.dot(X.T, y_predictions - y) * mean

            # update weights
            self.weights = self.weights - (self.learning_rate * gradient)

            self.print_current_result(i, z, y, y_predictions)

        # measure time
        elapsed_time = time.process_time() - t
        print(f"Fit-Time: {format_seconds(elapsed_time)}")

    def print_current_result(self, i, z, y, y_predictions):
        if not self.encrypted:
            if self.sigmoid_interval_from is not None and self.sigmoid_interval_to is not None:
                if np.any(z > self.sigmoid_interval_to) or np.any(z < self.sigmoid_interval_from):
                    raise ValueError('Sigmoid function interval overflow')

        if self.encrypted:
            # recrypt weights, this simulates bootstrapping and should be replaced
            self.weights = he.decrypt_ndarray(self.weights)

        if (i == self.iterations - 1) or (i % int(max(self.iterations, 10) / 10) == 0):
            print(f"Epoche {i + 1} ...")
            print(f"  Weights: {self.weights}")

        if self.encrypted:
            self.weights = he.encrypt_ndarray(self.weights)

    def predict_probability(self, X):
        t = time.process_time()

        # add intercept
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)

        y_predictions = self.sigmoid(np.dot(X, self.weights))

        elapsed_time = time.process_time() - t
        print(f"Prediction-Time: {format_seconds(elapsed_time)}")

        return y_predictions

    def predict_class(self, X):
        y_predictions = self.predict_probability(X)

        if self.encrypted:
            y_predictions = he.decrypt_ndarray(y_predictions)

        y_predictions = np.where(y_predictions > 0.5, 1, 0)
        return y_predictions

    def print_class_report(self, X, y):
        print_example_banner("Class Report", ch="-")
        target_names = ['negatives', 'positives']
        y_predictions = self.predict_class(X)
        print(classification_report(y, y_predictions, target_names=target_names))

    def print_confusion_matrix(self, X, y):
        print_example_banner("Confusion Matrix", ch="-")
        y_predictions = self.predict_class(X)
        tn, fp, fn, tp = confusion_matrix(y, y_predictions).ravel()
        print(f" True Negatives: {tn}")
        print(f" False Positives: {fp}")
        print(f" False Negatives: {fn}")
        print(f" True Positives: {tp}")


def example_binlogreg_iris_bivariate():
    print_example_banner("Example: Binary Logistic Regression", ch="#")

    print_example_banner("Dataset: Iris")

    iris = datasets.load_iris()

    # choose features
    X = iris.data[:, :2]

    # learn class 0
    y = np.where(iris.target == 0, 1, 0)

    # shuffle and split the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1204)

    # scale
    min_max_scaler = MinMaxScaler(feature_range=(0.25, 0.75), copy=False)
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # prepare for encoding
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    # encrypt
    X_train_enc = he.encrypt_ndarray(X_train)
    X_test_enc = he.encrypt_ndarray(X_test)
    y_train_enc = he.encrypt_ndarray(y_train)

    # print shapes
    positives_train = np.count_nonzero(y_train == 1)
    positives_test = np.count_nonzero(y_test == 1)
    print(f" Features: {X_train.shape[1]}")
    print(f" Training set size: {X_train.shape[0]} (Positives: {positives_train})")
    print(f" Test set size: {X_test.shape[0]} (Positives: {positives_test})")

    # choose fit params
    iterations = 1000
    learning_rate = 0.5

    print_example_banner("Plain")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=False
    )

    model.fit(X_train, y_train)

    model.print_confusion_matrix(X_test, y_test)
    model.print_class_report(X_test, y_test)

    print_example_banner("Encrypted")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=True
    )

    model.fit(X_train_enc, y_train_enc)

    model.print_confusion_matrix(X_test_enc, y_test)
    model.print_class_report(X_test_enc, y_test)


def example_binlogreg_iris_trivariate():
    print_example_banner("Example: Binary Logistic Regression", ch="#")

    print_example_banner("Dataset: Iris")

    iris = datasets.load_iris()

    # choose features
    X = iris.data[:, :3]

    # learn class 2
    y = np.where(iris.target == 0, 1, 0)

    # shuffle and split the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1204)

    # scale
    min_max_scaler = MinMaxScaler(feature_range=(0.25, 0.75), copy=False)
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # prepare for encoding
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    # encrypt
    X_train_enc = he.encrypt_ndarray(X_train)
    X_test_enc = he.encrypt_ndarray(X_test)
    y_train_enc = he.encrypt_ndarray(y_train)

    # print shapes
    positives_train = np.count_nonzero(y_train == 1)
    positives_test = np.count_nonzero(y_test == 1)
    print(f" Features: {X_train.shape[1]}")
    print(f" Training set size: {X_train.shape[0]} (Positives: {positives_train})")
    print(f" Test set size: {X_test.shape[0]} (Positives: {positives_test})")

    # choose fit params
    iterations = 1000
    learning_rate = 0.2

    print_example_banner("Plain")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=False
    )

    model.fit(X_train, y_train)

    model.print_confusion_matrix(X_test, y_test)
    model.print_class_report(X_test, y_test)

    print_example_banner("Encrypted")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=True
    )

    model.fit(X_train_enc, y_train_enc)

    model.print_confusion_matrix(X_test_enc, y_test)
    model.print_class_report(X_test_enc, y_test)


def example_binlogreg_iris_quadvariate():
    print_example_banner("Example: Binary Logistic Regression", ch="#")

    print_example_banner("Dataset: Iris")

    iris = datasets.load_iris()

    # choose features
    X = iris.data

    # learn class 2
    y = np.where(iris.target == 0, 1, 0)

    # shuffle and split the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1204)

    # scale
    min_max_scaler = MinMaxScaler(feature_range=(0.25, 0.75), copy=False)
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # prepare for encoding
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    # encrypt
    X_train_enc = he.encrypt_ndarray(X_train)
    X_test_enc = he.encrypt_ndarray(X_test)
    y_train_enc = he.encrypt_ndarray(y_train)

    # print shapes
    positives_train = np.count_nonzero(y_train == 1)
    positives_test = np.count_nonzero(y_test == 1)
    print(f" Features: {X_train.shape[1]}")
    print(f" Training set size: {X_train.shape[0]} (Positives: {positives_train})")
    print(f" Test set size: {X_test.shape[0]} (Positives: {positives_test})")

    # choose fit params
    iterations = 1000
    learning_rate = 0.2

    print_example_banner("Plain")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=False
    )

    model.fit(X_train, y_train)

    model.print_confusion_matrix(X_test, y_test)
    model.print_class_report(X_test, y_test)

    print_example_banner("Encrypted")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=True
    )

    model.fit(X_train_enc, y_train_enc)

    model.print_confusion_matrix(X_test_enc, y_test)
    model.print_class_report(X_test_enc, y_test)


def example_binlogreg_digits():
    print_example_banner("Example: Binary Logistic Regression", ch="#")

    print_example_banner("Dataset: Digits")

    digits = datasets.load_digits()

    # choose features
    X = digits.data

    # learn digit 0
    y = np.where(digits.target == 0, 1, 0)

    # shuffle and split the dataset into the training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1204)

    # scale
    min_max_scaler = MinMaxScaler(feature_range=(0.25, 0.75), copy=False)
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # prepare for encoding
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)

    # encrypt
    X_train_enc = he.encrypt_ndarray(X_train)
    X_test_enc = he.encrypt_ndarray(X_test)
    y_train_enc = he.encrypt_ndarray(y_train)

    # print shapes
    positives_train = np.count_nonzero(y_train == 1)
    positives_test = np.count_nonzero(y_test == 1)
    print(f" Features: {X_train.shape[1]}")
    print(f" Training set size: {X_train.shape[0]} (Positives: {positives_train})")
    print(f" Test set size: {X_test.shape[0]} (Positives: {positives_test})")

    # choose fit params
    iterations = 425
    learning_rate = 0.2

    print_example_banner("Plain")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=False
    )

    model.fit(X_train, y_train)

    model.print_confusion_matrix(X_test, y_test)
    model.print_class_report(X_test, y_test)

    print_example_banner("Encrypted")

    model = BinLogReg(
        learning_rate=learning_rate,
        iterations=iterations,
        encrypted=True
    )

    model.fit(X_train_enc, y_train_enc)

    model.print_confusion_matrix(X_test_enc, y_test)
    model.print_class_report(X_test_enc, y_test)


def main():
    # "exact" up to roughly 4 digits
    he.initialize_fractional()

    # "exact"
    #he.initialize_fractional(poly_modulus_degree=8192, plain_modulus_power_of_two=30, encoder_integral_coefficients=2048, encoder_fractional_coefficients=6144)

    example_binlogreg_iris_bivariate()
    #example_binlogreg_iris_trivariate()
    #example_binlogreg_iris_quadvariate()

    #example_binlogreg_iris_multivariate()
    #example_binlogreg_digits()


def print_example_banner(title, ch='*', length=50):
    spaced_text = ' %s ' % title
    print(spaced_text.center(length, ch))


def format_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    str_seconds = "{:0>2d}h:{:.0f}m:{:.10f}s".format(int(h), m, s)
    return str_seconds


if __name__ == '__main__':
    main()
