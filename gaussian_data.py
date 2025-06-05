import numpy as np

def generate_gaussian_data(n_train_samples, n_test_samples, rho, noise_degree, n_class=10):
    """
    Generates Gaussian training and testing datasets.

    Args:
        n_train_samples (int): Number of samples for the training set.
        n_test_samples (int): Number of samples for the testing set.
        rho (float): Correlation coefficient for generating X_2.
        noise_degree (float): Standard deviation of the noise.
        n_class (int, optional): Number of classes. Defaults to 10.

    Returns:
        dict: A dictionary containing:
            'train_data': A list [train_features, train_labels].
                            train_features is a numpy array of shape (n_train_samples, 2 * n_class).
                            train_labels is a numpy array of shape (n_train_samples,).
            'test_data': A list [test_features, test_labels].
                            test_features is a numpy array of shape (n_test_samples, 2 * n_class).
                            test_labels is a numpy array of shape (n_test_samples,).
            'dims': A list [X_1_dim, X_2_dim, Y_dim].
    """
    
    def _generate_single_dataset(num_samples_in_set):
        
        # Generate Y - integer labels
        y_labels = np.random.randint(0, n_class, size=[num_samples_in_set])
        
        # Create one-hot encoded Y
        rows = np.arange(num_samples_in_set)
        onehot_y = np.zeros((num_samples_in_set, n_class))
        onehot_y[rows, y_labels] = 1
        
        # Generate noise components u and v
        u = np.random.randn(num_samples_in_set, n_class) * noise_degree
        v = np.random.randn(num_samples_in_set, n_class) * noise_degree
        
        # Generate modalities X_1 and X_2
        x_1 = onehot_y + u
        x_2 = onehot_y + rho * u + np.sqrt(1 - rho ** 2) * v
        
        # Define dimensions
        x_1_dim = n_class  # Dimension of X_1 features per sample
        x_2_dim = n_class  # Dimension of X_2 features per sample
        y_dim = 1          # Dimension of the original label Y (scalar)
        
        data_payload_list = [x_1.reshape(-1, x_1_dim), x_2.reshape(-1, x_2_dim), y_labels]
        
        dims_payload_list = [x_1_dim, x_2_dim, y_dim]
        
        return data_payload_list, dims_payload_list

    train_data_list, dims_list = _generate_single_dataset(n_train_samples)
    test_data_list, _ = _generate_single_dataset(n_test_samples)
    
    return {
        'train_data': train_data_list,
        'test_data': test_data_list,
        'dims': dims_list
    }
