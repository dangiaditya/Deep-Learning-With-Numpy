import numpy as np


def softmax(x):

    reg = (x.T - np.max(x, axis=1)).T
    ex = np.exp(reg)
    ex = ex / np.sum(ex, axis=1).reshape(-1, 1)

    return ex


def cross_entropy(y_pred, y_train):

    n = y_pred.shape[0]
    prob = softmax(y_pred)
    log_likelihood = -np.log(prob[range(n), y_train])
    loss = np.sum(log_likelihood) / n

    return loss


def dcross_entropy(y_pred, y_train):

    n = y_pred.shape[0]
    grad = softmax(y_pred)
    grad[range(n), y_train] -= 1
    grad /= n

    return grad


def get_batches(x, y, batch_size, steps):

    batches = []
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = y[i:i + batch_size]

        batches.append((x_batch, y_batch))
    perm = np.random.permutation(len(batches))
    batches = np.array(batches)[perm][:steps]

    return batches


def optimize(model, x_train, y_train, x_val, y_val, lr=0.001, batch_size=256, steps=500):

    batches = get_batches(x_train, y_train, batch_size, steps)
    count = 0
    loss_list = []
    for x, y in batches:

        grad, loss = model.train(x, y)
        loss_list.append(loss)
        if count % 100 == 0:
            val_acc = np.mean(y_val == model.predict(x_val))
            print("""
    Step {}
    loss {}
    validation accuracy {}
            """.format(count, loss, val_acc))
        count += 1
        for layer in grad:
            model.params[layer] -= lr * grad[layer]

    return loss_list
