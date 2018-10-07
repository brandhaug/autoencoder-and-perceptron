from matplotlib import pyplot as plt
import numpy as np


def predict(inputs, weights):
    activation = 0.0
    for i, w in zip(inputs, weights):
        activation += i * w
    return 1.0 if activation >= 0.0 else 0.0


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

    plt.scatter(c0_data[0], c0_data[1], s=40.0, c='red') # Red dots
    plt.scatter(c1_data[0], c1_data[1], s=40.0, c='blue') # Blue dots

    plt.legend(fontsize=10, loc=1)
    plt.show()
    return


def accuracy(matrix, weights):
    correct_count = 0.0
    predictions = []
    for i in range(len(matrix)):
        prediction = predict(matrix[i][:-1], weights)  # get predicted classification
        predictions.append(prediction)
        if prediction == matrix[i][-1]: correct_count += 1.0
    print("Predictions:", predictions)
    return correct_count / float(len(matrix))


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


def main():
    print("Starting")

    # Bias, i1, i2,  y
    matrix = [[1.00, 0.08, 0.72, 1.0],
              [1.00, 0.10, 1.00, 0.0],
              [1.00, 0.26, 0.58, 1.0],
              [1.00, 0.35, 0.95, 0.0],
              [1.00, 0.45, 0.15, 1.0],
              [1.00, 0.60, 0.30, 1.0],
              [1.00, 0.70, 0.65, 0.0],
              [1.00, 0.42, 0.45, 1.0],
              [1.00, 0.92, 0.95, 0.0],
              [1.00, 0.22, 0.11, 1.0],
              [1.00, 0.32, 0.75, 0.0],
              [1.00, 0.92, 0.95, 0.0]]
    weights = [0.40, -1.00, 1.00]  # initial weights specified in problem
    threshold = 1

    weights = train_weights(matrix, weights=weights, threshold=threshold)


main()
