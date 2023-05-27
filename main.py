import argparse
import src.log_reg as log_reg
import src.neural_network as nn
import src.preprocessing as pre

def main(args):
    # Import images
    print("PREPROCESSING: Importing cifar10 dataset...")
    X_train, y_train, X_test, y_test, labels = pre.load_cifar10_data()

    # Convert images to gray scale
    print("PREPROCESSING: Converting images to gray scale...")
    X_train_scaled, X_test_scaled = pre.convert_to_grey_scale(X_train, X_test)

    # Flattening the images (both train and test) using reshape
    print("PREPROCESSING: Flattening images...")
    X_train_dataset, X_test_dataset = pre.flatten_data(X_train_scaled, X_test_scaled)

    """
    ### Logistic Regression ###
    """
    # Doing logistic regression for classification on the image data
    print("LOGISTIC REGRESSION: Training model...")
    y_pred = log_reg.classification_rog_reg(X_train_dataset, y_train, X_test_dataset, 
                                            tol=args.tol, verbose=args.verbose, solver=args.solver, multi_class=args.multi_class)

    # Saving classification report in folder 'out'
    print("LOGISTIC REGRESSION: Saving classification report...")
    log_reg.logreg_classification_report(y_test, y_pred, labels)

    """
    ### Neural network ###
    """
    # Running data through a neural network to classify images
    print("NEURAL NETWORK: Training model...")
    y_pred = nn.train_model(X_train_dataset, y_train, X_test_dataset, hidden_layer_sizes=args.hidden_layer_sizes, learning_rate=args.learning_rate, early_stopping=args.early_stopping, nn_verbose=args.nn_verbose, max_iter=args.max_iter)

    # Saving classification report in folder 'out'
    print("NEURAL NETWORK: Saving classification report...")
    nn.nn_classification_report (y_test, y_pred, labels)

    print("[MESSAGE] Script finished successfully!")

# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for training and evaluating a machine learning model for fake news detection.')
    # Optional arguments for logstic regression
    parser.add_argument('--tol', type=float, default=0.1, help='specify the size of the test/train split, default is 0.2')
    parser.add_argument('--verbose', type=bool, default=True, help='whether to print the progress of the model, default is True')
    parser.add_argument('--solver', type=str, default="saga", help='specify the solver for the model, default is saga')
    parser.add_argument('--multi_class', type=str, default="multinomial", help='specify if it is a multi_class problem, default is multinomial')
    # Arguments for neural network
    parser.add_argument('--hidden_layer_sizes', type=tuple, nargs=2, default=(64, 10), help='specify the size of the hidden layers, default is (64, 10)')
    parser.add_argument('--learning_rate', type=str, default="adaptive", help='specify the learning rate, default is adaptive (meaning that the learning rate is adapts according to the loss function)')
    parser.add_argument('--early_stopping', type=bool, default=True, help='specify if early stopping should be used, default is True')
    parser.add_argument('--nn_verbose', type=bool, default=True, help='whether to print the progress of the model, default is True')
    parser.add_argument('--max_iter', type=int, default=20, help='specify the maximum number of iterations, default is 20')
    args = parser.parse_args()
    main(args)