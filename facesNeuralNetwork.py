import numpy as np
import time
import matplotlib.pyplot as plt

def load_labels_from_file(label_file):
    """Load labels from the provided label file."""
    with open(label_file) as f:
        lines = [int(line.strip()) for line in f.readlines()]
    sample_num = len(lines)
    return lines, sample_num

def load_samples_from_file(sample_file, sample_num, pool_size):
    """Load samples from the given sample file and downscale them by a pooling factor."""
    with open(sample_file) as f:
        lines = f.readlines()
    file_length = len(lines)
    width = len(lines[0])
    length = int(file_length / sample_num)
    all_images = []

    for i in range(sample_num):
        single_image = np.zeros((length, width))
        count = 0
        for j in range(length * i, length * (i + 1)):
            single_line = lines[j]
            for k in range(len(single_line)):
                if single_line[k] in ('+', '#'):
                    single_image[count, k] = 1
            count += 1
        all_images.append(single_image)

    new_row = int(length / pool_size)
    new_col = int(width / pool_size)
    pooled_images = np.zeros((sample_num, new_row, new_col))
    
    for i in range(sample_num):
        for j in range(new_row):
            for k in range(new_col):
                pooled_pixel = 0
                for row in range(pool_size * j, pool_size * (j + 1)):
                    for col in range(pool_size * k, pool_size * (k + 1)):
                        pooled_pixel += all_images[i][row, col]
                pooled_images[i, j, k] = pooled_pixel
    return pooled_images

def process_and_shuffle_data(data_file, label_file, pool_size):
    """Process and shuffle the data and labels."""
    labels, sample_num = load_labels_from_file(label_file)
    data = load_samples_from_file(data_file, sample_num, pool_size)
    flattened_data = [image.flatten() for image in data]
    indices = np.arange(len(flattened_data))
    np.random.shuffle(indices)
    return np.array(flattened_data)[indices], np.array(labels)[indices]

def gradient_descent_optimization(weights, bias, x, y, iterations, learning_rate):
    """Optimize weights and bias using gradient descent."""
    for _ in range(iterations):
        weight_gradient, bias_gradient, _ = compute_gradients(weights, bias, x, y)
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
    return weights, bias

def compute_gradients(weights, bias, x, y):
    """Calculate gradients and cost for the given weights, bias, and data. (LOSS FUNCTION)"""
    m = x.shape[0]
    activation = np.squeeze(sigmoid(np.dot(x, weights) + bias))
    y = np.array([int(item) for item in y])
    cost = -(1 / m) * np.sum(y * np.log(activation) + (1 - y) * np.log(1 - activation))
    weight_gradient = (1 / m) * np.dot(x.T, (activation - y)).reshape(weights.shape[0], 1)
    bias_gradient = (1 / m) * np.sum(activation - y)
    return weight_gradient, bias_gradient, cost

def sigmoid(z):
    """Apply sigmoid activation."""
    return 1 / (1 + np.exp(-z))

def make_predictions(weights, bias, x):
    """Predict the labels for the given input data."""
    weights = weights.reshape(x.shape[1], 1)
    predictions = sigmoid(np.dot(x, weights) + bias)
    predictions = np.where(predictions > 0.5, 1, 0)
    return predictions

def train_model(x_train, y_train, iterations=2000, learning_rate=0.5):
    """Train the model using training data."""
    weights = np.zeros((x_train.shape[1], 1))
    bias = 0
    weights, bias = gradient_descent_optimization(weights, bias, x_train, y_train, iterations, learning_rate)
    return weights, bias

def plot_graph(var, title, color, ylabel):
    """Plot the specified graph with the given data."""
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label='time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def calculate_accuracy(pred, label):
    """Calculate the accuracy of the predictions."""
    return 1 - np.mean(np.abs(pred - label))

def main():
    pooling_factor = 3
    train_data_file = "data/facedata/facedatatrain"
    train_label_file = "data/facedata/facedatatrainlabels"
    test_data_file = "data/facedata/facedatatest"
    test_label_file = "data/facedata/facedatatestlabels"
    x_train, y_train = process_and_shuffle_data(train_data_file, train_label_file, pooling_factor)
    x_test, y_test = process_and_shuffle_data(test_data_file, test_label_file, pooling_factor)
    sample_size = int(x_train.shape[0] / 10)
    time_taken = []
    test_accuracies = []

    for i in range(10):
        start = time.time()
        print('Training using', sample_size * (i + 1))
        weights, bias = train_model(x_train[:sample_size * (i + 1)], y_train[:sample_size * (i + 1)])
        end = time.time()
        y_pred_test = make_predictions(weights, bias, x_test)
        test_accuracy = calculate_accuracy(np.squeeze(y_pred_test), y_test)
        print(f"Test accuracy: {round(test_accuracy, 3)}")
        time_taken.append(end - start)
        test_accuracies.append(test_accuracy)

    plot_graph(time_taken, title='Neural Network Classifier for Faces (Time)', color='blue', ylabel="Time (s)")
    plot_graph(test_accuracies, title='Neural Network Classifier for Faces (Accuracy)', color='green', ylabel='Accuracy')

if __name__ == "__main__":
    main()