import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import readdata as rd


def training_neural(x_train, y_train, epoch, label):
    features = len(x_train[0])
    weights = np.random.rand(features, label)
    epoch_count = []
    accuracy_each_epoch = []
    for epo in range(0, epoch):
        print("Epoch %d/%d" % (epo + 1, epoch))
        epoch_count.append(epo + 1)
        error = 0

        for i in range(0, len(x_train)):
            current_label = y_train[i]
            individual_image_pixels = np.zeros((features))

            individual_image_pixels[:] = x_train[i][:]
            individual_image_pixels[:] = x_train[i][:]

            dot_product = np.dot(weights.T, individual_image_pixels)
            predicted_label = np.argmax(dot_product)

            if predicted_label != current_label:
                error = error + 1
                weights[:, current_label] = weights[:,
                                                    current_label] + individual_image_pixels[:]
                weights[:, predicted_label] = weights[:,
                                                      predicted_label] - individual_image_pixels[:]

        accuracy = 100 - ((error / len(x_train)) * 100)
        accuracy_each_epoch.append(accuracy)
        print("Accuracy: ", accuracy)
        if (accuracy == 100.0):
            break

    return weights, epoch_count, accuracy_each_epoch


def testing_neural(x_test, y_test, weights_learned):
    features = len(x_test[0])
    error = 0
    for i in range(0, len(x_test)):
        current_label = y_test[i]
        individual_image_pixels = np.zeros((features, 1))
        for j in range(0, features):
            individual_image_pixels[j] = x_test[i][j]

        dot_product = np.dot(weights_learned.T, individual_image_pixels)
        predicted_label = np.argmax(dot_product)
        if predicted_label != current_label:
            error += 1

    return 100 - error/len(y_test)*100


def perceptron(training_images_file, training_labels_file, test_images_file, test_labels_file, type):

    if (type == "Digits"):
        classes = 10
        image_height = 28
        image_width = 28
        training_images = 5000
        test_images = 1000

    elif (type == "Faces"):
        classes = 2
        image_height = 70
        image_width = 60
        training_images = 451
        test_images = 150

    fetch_data_train = rd.load_data(
        training_images_file, training_images,  image_height, image_width)
    fetch_data_test = rd.load_data(
        test_images_file, test_images, image_height, image_width)
    X_train = rd.matrix_transformation(
        fetch_data_train, image_height, image_width)
    X_test = rd.matrix_transformation(
        fetch_data_test, image_height, image_width)

    Y_train_labels = rd.load_label(training_labels_file, training_images)
    Y_test_labels = rd.load_label(test_labels_file, test_images)

    data_percentange = 1
    accuracies = []
    percent_training_amounts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    total_training_time = 0
    start_time = time.time()

    for i in range(0, len(X_test)):
        X_train[i] = X_train[i].flatten()
        X_test[i] = X_test[i].flatten()

    Y_train_labels = list(map(int, Y_train_labels))
    Y_test_labels = list(map(int, Y_test_labels))

    for i in range(0, 10):
        train_start_time = time.time()
        data_percentange -= 0.10
        if data_percentange <= 0:
            data_percentange = 0.001
        x_train, placeholder1, y_train, placeholder2 = train_test_split(
            X_train, Y_train_labels, test_size=data_percentange, random_state=45)

        for j in range(0, len(x_train)):
            x_train[j] = x_train[j].flatten()

        weights_learned, epoch_count, counter = training_neural(
            x_train, y_train, 150, classes)
        total_training_time += time.time() - train_start_time
        pre = testing_neural(X_test, Y_test_labels, weights_learned)
        accuracies.append(pre)

    fig, axs = plt.subplots(2)
    plt.title(type)
    axs[0].plot(percent_training_amounts, accuracies,
                '-ko', linewidth=1, markersize=3)
    axs[0].set_xlabel("Partition Percentage")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title(type)

    axs[1].plot(epoch_count, counter, '-b')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    total_time = time.time() - start_time
    plt.show()

    print("Testing accuracies", accuracies)
    print("Total training time: ", total_training_time, " seconds")
    print("Total time taken: ", total_time, " seconds")


digits_file_path = "./data/digitdata/"
perceptron(digits_file_path+"trainingimages", digits_file_path+"traininglabels",
           digits_file_path+"testimages", digits_file_path+"testlabels", "Digits")

faces_file_path = "./data/facedata/facedata"
perceptron(faces_file_path+"train", faces_file_path+"trainlabels",
           faces_file_path+"test", digits_file_path+"testlabels", "Faces")