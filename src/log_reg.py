"""
IMPORTS
"""
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import pandas as pd
from sklearn.metrics import classification_report

"""
#### FUNCTIONS ####
"""
# Function for training the model and getting predictions
def classification_rog_reg (X_train_dataset, y_train, X_test_dataset, tol = 0.1, verbose = True, solver="saga", multi_class="multinomial"):
    """
    Function for training the model and getting predictions
    """
    # Calling classifier
    classifier = LogisticRegression(random_state=42,
                                    tol = tol,
                                    verbose = verbose,
                                    solver=solver,
                                    multi_class=multi_class).fit(X_train_dataset, y_train)
    # get predictions from the test data
    y_pred = classifier.predict(X_test_dataset)
    return y_pred

# Getting prediction metrics
def logreg_classification_report(y_test, y_pred, labels):
    """
    : y_test: numpy array 
    : y_pred: numpy array 
    """
    report = classification_report(y_test, 
                               y_pred, 
                               target_names=labels,
                               output_dict=True)
    df = pd.DataFrame(report).transpose()
    # Round the values in df to 2 decimals
    df = df.round(2)
    df.to_csv(os.path.join('out', 'classification_report_log_reg.csv'), index=True)