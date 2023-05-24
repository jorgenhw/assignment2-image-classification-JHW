"""
### IMPORTS ###
"""
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import os
import pandas as pd

"""
#### FUNCTIONS ####
"""

# Function for training the model and getting predictions
def train_model(X_train_dataset, y_train, X_test_dataset, hidden_layer_sizes=(64, 10), learning_rate="adaptive", early_stopping=True, nn_verbose=True, max_iter=100):
    """
    : X_train_feats: numpy array of shape (n_train, d)
    : y_train: numpy array of shape (n_train,)
    : X_test_feats: numpy array of shape (n_test, d)
    """
    nn_classifier = MLPClassifier(random_state=42,
                    hidden_layer_sizes=hidden_layer_sizes, # 64 neurons in the first layer, 10 neurons in the second layer
                    learning_rate=learning_rate, # adaptive learning rate:
                    early_stopping=early_stopping, # set to true to stop training when the validation score stops improving for x consecutive epochs
                    verbose=nn_verbose,
                    max_iter=max_iter).fit(X_train_dataset, y_train)
    # Fit the model to the training data
    y_pred = nn_classifier.predict(X_test_dataset)
    return y_pred

# Function for evaluating the model on the test set, printing the classification report and saving it to a csv file in out
def nn_classification_report (y_test, y_pred, labels):
    """
    : y_test: numpy array of shape (n_test,)
    : y_pred: numpy array of shape (n_test,)
    """
    report = classification_report(y_test, 
                            y_pred, 
                            target_names=labels,
                            output_dict=True)
    # get predictions from the test data
    df = pd.DataFrame(report).transpose()
    # Round the values in df to 2 decimals
    df = df.round(2)
    df.to_csv(os.path.join('out', 'classification_report_nn.csv'), index=True)