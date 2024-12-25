from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import numpy as np
from model.base_learner import LogisticRegressionClassifier, DecisionTreeClassifier, \
                KNearestNeighborClassifier, NaiveBayesClassifier, MLPClassifier
from model.meta_learner import StackingClassifier


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! # 
# Do not modify any thing below this line to prevent the error
def get_linear_data():

    np.random.seed(42)
    n_samples_per_class = 50
    n_features = 77
    
    # class 1
    X_class1 = np.random.randn(n_samples_per_class, n_features) + np.array([2] * n_features)  
    y_class1 = np.zeros(n_samples_per_class, dtype=int)
    
    # class 2
    X_class2 = np.random.randn(n_samples_per_class, n_features) + np.array([-2] * n_features)  
    y_class2 = np.ones(n_samples_per_class, dtype=int)

    X = np.vstack((X_class1, X_class2))
    y = np.hstack((y_class1, y_class2))
    print(f"The shape of X is: {X.shape}")
    print(f"The shape of y is: {y.shape}")
    return X, y

# generate XOR data
# def get_xor_data():
#     X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#     y = np.array([0, 1, 1, 0])  # XOR labels
#     return X, y

def test_model(model_class, X, y, test_name=""):
    print(f"Running test: {test_name} with {model_class.__name__}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    model = model_class()
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    f1 = f1_score(y_pred=y_pred, y_true=y_test)
    mcc = matthews_corrcoef(y_pred=y_pred, y_true=y_test)
    scoring = acc * 0.3 + f1 * 0.35 + mcc * 0.35
    print(f"The scoring of {model_class.__name__} is: {scoring}")

    if scoring >= 0.9:
        print(f"{model_class.__name__} passed the test!")
        print("-" * 50)
        return True
    else:
        print(f"{model_class.__name__} failed the test! Expected scoring >= 0.9.\n")
        print("-" * 50)
        return False


def test_base_learner():
    X, y = get_linear_data()

    passed_tests = 0
    if test_model(LogisticRegressionClassifier, X, y, "Linear Data Test"):
        passed_tests += 1
    if test_model(DecisionTreeClassifier, X, y, "Linear Data Test"):
        passed_tests += 1
    if test_model(KNearestNeighborClassifier, X, y, "Linear Data Test"):
        passed_tests += 1
    if test_model(NaiveBayesClassifier, X, y, "Linear Data Test"):
        passed_tests += 1
    if test_model(MLPClassifier, X, y, "Linear Data Test"):
        passed_tests += 1
    
    return passed_tests
    # print(f"\nTotal Base learner passed: {passed_tests} / 5")


def test_meta_learner():
    X, y = get_linear_data()
    if test_model(StackingClassifier, X, y, "Linear Data Test"):
        return 1
    else: 
        return 0
        # print(f"\nMeta learner passed!")


# Do not modify anything above this line to prevent error
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! # 

if __name__ == "__main__":
    print("Starting model tests...\n")
    
    print("Testing with linearly separable data...")
    base_learner_pass = test_base_learner()
    meta_learner_pass = test_meta_learner()
    

    print(f"\nTotal # Base learner passed: {base_learner_pass} / 5")

    if meta_learner_pass:
        print(f"\nMeta learner passed!")
    else:
        print(f"\nMeta learner failed.")
    
    print("\nAll tests completed.")