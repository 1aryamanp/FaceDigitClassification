'''
import readdata as rd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# 28 rows of an image, pixel count has the number of pixels for each row for each number image (1-9)
def NaiveBayes(pixel_count, prior_prob, test_image, image_count, type):
    if type == "Digits":
        likelihood = [1] * 10
    elif type == "Faces":
        likelihood = [1] * 2

    for j in range(0, len(likelihood)):
        prior_value = prior_prob[j]
        for i in range(0, len(test_image)):
            test_value = test_image[i]
            if pixel_count[j, i, test_value] == 0:
                likelihood[j] = likelihood[j] * (1 / image_count[j])
            else:
                num_of_occ = pixel_count[j, i, test_value]
                likelihood[j] = likelihood[j] * (num_of_occ / image_count[j])

        likelihood[j] = likelihood[j] * prior_value
    return likelihood.index(max(likelihood))


def algo(training_images_file, training_labels_file, test_images_file, test_labels_file, type):

    if (type == "Digits"):
        classes = 10
        image_height = 28
        image_width = 28
        training_images = 5000
        test_images = 1000
        ran_state = 30

    elif (type == "Faces"):
        classes = 2
        image_height = 70
        image_width = 60
        training_images = 451
        test_images = 150
        ran_state = 10

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

    tem = 1
    accuracy_array = []
    percent_training_labels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    total_training_time = 0
    total_start = time.time()

    for i in range(0, 10):  # Numbers range from 0 to 9
        start = time.time()
        tem -= 0.10
        if tem < 0:
            tem = 0.0001
        x_train, placeholder1, y_train, placeholder2 = train_test_split(
            X_train, Y_train_labels, test_size=tem, random_state=ran_state)

        pixel_count = np.zeros((classes, image_height, image_width))
        image_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for w in range(0, len(x_train)):

            # Checks what number image is via labels
            k = int(y_train[w])
            image_count[k] = image_count[k] + 1
            it = -1
            # x_train[w] is 28 by 28 and i of x_train[w] is 28 rows (Goes through each image and populates the pixel matrix)
            for i in x_train[w]:
                it = it + 1
                count = 0
                for j in i:
                    if j == 1:
                        count = count + 1
                pixel_count[k, it, count] += 1

    #  print(image_count) Image is a 1 by 10 matrix

        prior = classes * [0]
        for i in range(0, classes):
            num_of_labels = len(y_train)
            prior[i] = image_count[i] / num_of_labels

        end = time.time()
        total_training_time = total_training_time + (end - start)
        img_di = image_height * [0]
        predicted_values = []
        for w in range(0, test_images):  # Number of images in test set is always 1000
            count = 0
            it = 0
            for i in X_test[w]:
                count = 0
                for j in i:
                    if j == 1:
                        count = count + 1
                img_di[it] = count
                it = it + 1

            # img_di Tells how many pixels each row of an image has

            pred_lab = NaiveBayes(pixel_count, prior,
                                  img_di, image_count, type)
            predicted_values.append(pred_lab)

        final_count = 0
        for i in range(0, test_images):  # Number of labels in test will always be 1000
            if int(Y_test_labels[i]) == predicted_values[i]:
                final_count += 1

        print("Accuracy with "+str(round(1-tem, 2)*100) +
              "% training data: " + str(final_count/len(Y_test_labels)*100))
        accuracy_array.append(final_count / len(Y_test_labels) * 100)

    total_end = time.time()
    plt.plot(percent_training_labels, accuracy_array,
             '-ko', linewidth=1, markersize=3)
    plt.title(type)
    plt.xlabel("Partition Percentage")
    plt.ylabel("Accuracy")
    plt.show()

    print("Total training time: ", total_training_time, " seconds")
    print("Total time taken: ", total_end - total_start, " seconds")


digits_file_path = "./data/digitdata/"
algo(digits_file_path+"trainingimages", digits_file_path+"traininglabels",
     digits_file_path+"testimages", digits_file_path+"testlabels", "Digits")

faces_file_path = "./data/facedata/facedata"
algo(faces_file_path+"train", faces_file_path+"trainlabels",
     faces_file_path+"test", faces_file_path+"testlabels", "Faces")
'''