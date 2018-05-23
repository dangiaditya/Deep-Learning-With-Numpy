import numpy as np
import pickle
import os
import convnet as conv
from loss import optimize
import matplotlib.pyplot as plt


IMAGE_PICKLE = "image_dict.pickle"
steps = 300
lr = 0.1
batch_size = 64


def load_data(preprocess):

    image_dict = {}
    if not IMAGE_PICKLE in os.listdir("."):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        x_train, y_train = mnist.train.images, mnist.train.labels
        x_val, y_val = mnist.validation.images, mnist.validation.labels
        x_test, y_test = mnist.test.images, mnist.test.labels

        if preprocess:
            mean = np.mean(x_train)
            x_train -= mean
            x_val -= mean
            x_test -= mean

        x_train = x_train.reshape(-1, 1, 28, 28)
        x_val = x_val.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)

        with open("image_dict.pickle", 'wb') as f:
            image_dict = dict(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test
            )

            pickle.dump(image_dict, f, pickle.HIGHEST_PROTOCOL)
        return image_dict

    with open(IMAGE_PICKLE, 'rb') as f:
        image_dict = pickle.load(f)

    return image_dict


if __name__ == "__main__":

    image_dict = load_data(preprocess=True)

    x_train, y_train = image_dict["x_train"], image_dict["y_train"]
    x_val, y_val = image_dict["x_val"], image_dict["y_val"]
    x_test, y_test = image_dict["x_test"], image_dict["y_test"]

    classes = 10
    hidden_unit = 128
    filters = 16
    print("\nParameters to the conv net \n")
    print('''
    classes = 10
    hidden_unit = 128
    filters = 16\n''')
    print("""
    x_train shape {}
    y_train shape {}
    x_val shape {}
    y_val shape {}\n""".format(x_train.shape, y_train.shape, x_val.shape, y_val.shape))

    model = conv.Conv(num_filters=filters, classes=classes, hidden_units=hidden_unit)
    loss_list = optimize(model, x_train, y_train, x_val, y_val, batch_size=batch_size, lr=lr, steps=steps)

    y_pred = model.predict(x_test[:2000])
    acc = np.mean(y_pred == y_test[:2000])
    print("Accuracy ", acc)
    plt.plot(range(len(loss_list)), loss_list, label="loss")
    plt.xlabel("Number of steps")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.show()
