from matplotlib import pyplot as plt
import numpy as np


# Predicts 0 or 1
def predict(inputs, weights):
    activation = 0.0
    for i, w in zip(inputs, weights):
        activation += i * w
    return 1.0 if activation >= 0.0 else 0.0


# Visualization
def plot(matrix, weights=None, title="Prediction Matrix"):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("i1")
    ax.set_ylabel("i2")

    if weights is not None:
        map_min = 0.0
        map_max = 1.1
        y_resolution = 0.001
        x_resolution = 0.001
        ys = np.arange(map_min, map_max, y_resolution)
        xs = np.arange(map_min, map_max, x_resolution)
        zs = []
        for cur_y in np.arange(map_min, map_max, y_resolution):
            for cur_x in np.arange(map_min, map_max, x_resolution):
                zs.append(predict([1.0, cur_x, cur_y], weights))
        xs, ys = np.meshgrid(xs, ys)
        zs = np.array(zs)
        zs = zs.reshape(xs.shape)
        plt.contourf(xs, ys, zs, levels=[-1, -0.0001, 0, 1], colors=('blue', 'red'), alpha=0.1)  # Background color

    c1_data = [[], []]
    c0_data = [[], []]
    for i in range(len(matrix)):
        cur_i1 = matrix[i][1]
        cur_i2 = matrix[i][2]
        cur_y = matrix[i][-1]
        if cur_y == 1:
            c1_data[0].append(cur_i1)
            c1_data[1].append(cur_i2)
        else:
            c0_data[0].append(cur_i1)
            c0_data[1].append(cur_i2)

    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)

    plt.scatter(c0_data[0], c0_data[1], s=40.0, c='red')  # Red dots
    plt.scatter(c1_data[0], c1_data[1], s=40.0, c='blue')  # Blue dots

    plt.show()
    return


# Calculates accuracy: (correct predictions / total predictions)
def accuracy(matrix, weights):
    correct_count = 0.0
    predictions = []
    for i in range(len(matrix)):
        prediction = predict(matrix[i][:-1], weights)  # get predicted classification
        predictions.append(prediction)
        if prediction == matrix[i][-1]: correct_count += 1.0
    print("Predictions:", predictions)
    return correct_count / float(len(matrix))


# Updates weights based on error from predictions
def train_weights(matrix, weights, epochs_count=10, threshold=1.0):
    for epoch in range(epochs_count):
        current_accuracy = accuracy(matrix, weights)
        print("\nEpoch %d \nWeights: " % epoch, weights)
        print("Accuracy: ", current_accuracy)

        if current_accuracy >= threshold:
            print("Threshold (%0.1f) is fulfilled, stop training" % threshold)
            break

        plot(matrix, weights, title="Epoch %d" % epoch)

        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1], weights)  # get predicted classificaion
            error = matrix[i][-1] - prediction  # get error from real classification
            print("Training on data at index %d...\n" % (i))
            for j in range(len(weights)):  # calculate new weight for each node
                print("\tWeight[%d]: %0.5f --> " % (j, weights[j]))
                weights[j] = weights[j] + (error * matrix[i][j])
                print("%0.5f\n" % (weights[j]))

    plot(matrix, weights, title="Final Epoch")
    return weights


if __name__ == '__main__':
    print("Starting")

    # Bias, i1, i2,  y
    dataset = [[1.00, 1.0, 1.0, 1.0],
               [1.00, 1.0, 0.0, 0.0],
               [1.00, 0.0, 1.0, 0.0],
               [1.00, 0.0, 0.0, 0.0]]  # AND

    # dataset = [[1.00, 1.0, 1.0, 1.0],
    #           [1.00, 1.0, 0.0, 1.0],
    #           [1.00, 0.0, 1.0, 1.0],
    #           [1.00, 0.0, 0.0, 0.0]]  # OR

    initial_weights = [0.40, -1.00, 1.00]  # initial weights specified in problem
    initial_threshold = 1

    train_weights(dataset, weights=initial_weights, threshold=initial_threshold)
