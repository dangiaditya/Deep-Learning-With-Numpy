import numpy as np
from loss import cross_entropy, dcross_entropy
from steps import Steps


class Conv(object):

    def __init__(self, num_filters, classes, hidden_units=256, batch_norm=True, eps=1e-8):

        self.params = {}
        self.num_filters = num_filters
        self.classes = classes
        self.hidden_units = hidden_units
        self.step = Steps()
        self.batch_norm = batch_norm
        self.eps = eps
        self.initialize_model()

    def initialize_model(self):

        if self.batch_norm:
            self.params = dict(
                w1=np.random.randn(self.num_filters, 1, 3, 3) / np.sqrt(self.num_filters / 2.0),
                w2=np.random.randn(self.num_filters * 14 * 14, self.hidden_units) / np.sqrt(self.num_filters * 14 * 14 / 2.0),
                w3=np.random.randn(self.hidden_units, self.classes) / np.sqrt(self.hidden_units / 2.0),
                b1=np.zeros((self.num_filters, 1)) + 0.002,
                b2=np.zeros((1, self.hidden_units)) + 0.001,
                b3=np.zeros((1, self.classes)) + 0.06,
                gamma=0.2,
                beta=0.3
            )
        else:
            self.params = dict(
                w1=np.random.randn(self.num_filters, 1, 3, 3) / np.sqrt(self.num_filters / 2.0),
                w2=np.random.randn(self.num_filters * 14 * 14, self.hidden_units) / np.sqrt(
                    self.num_filters * 14 * 14 / 2.0),
                w3=np.random.randn(self.hidden_units, self.classes) / np.sqrt(self.hidden_units / 2.0),
                b1=np.zeros((self.num_filters, 1)) + 0.002,
                b2=np.zeros((1, self.hidden_units)) + 0.001,
                b3=np.zeros((1, self.classes)) + 0.06
            )

        print("Model Params")
        for i, j in self.params.items():
            if i in ["gamma", "beta"]:
                print(i, ": ", j)
            else:
                print(i, ": ", j.shape)

    def train(self, x_train, y_train):

        y_pred, cache = self.forward(x_train)
        loss = cross_entropy(y_pred, y_train)
        grad = self.backward(y_pred, y_train, cache)

        return grad, loss

    def forward(self, x):

        num_samples = x.shape[0]
        h_1, h_1_cache = self.step.conv_forward(x, self.params["w1"], self.params["b1"], stride=1, padding=1)
        h_1, relu_cache_1 = self.step.relu_forward(h_1)
        pool, pool_cache = self.step.maxpool_forward(h_1, size=2, stride=2)

        h_2 = pool.ravel().reshape(num_samples, -1)

        fc_1, fc_cache_1 = self.step.fc_forward(h_2, self.params["w2"], self.params["b2"])
        if self.batch_norm:

            fc_1, bn_cache = self.step.batchnorm_forward(fc_1, self.params["gamma"], self.params["beta"], self.eps)

        fc_1, relu_cache_2 = self.step.relu_forward(fc_1)

        fc_2, fc_cache_2 = self.step.fc_forward(fc_1, self.params["w3"], self.params["b3"])
        if self.batch_norm:
            cache = (x, h_1_cache, relu_cache_1, pool, pool_cache, fc_cache_1, relu_cache_2, fc_cache_2, bn_cache)
        else:
            cache = (x, h_1_cache, relu_cache_1, pool, pool_cache, fc_cache_1, relu_cache_2, fc_cache_2)
        return fc_2, cache

    def backward(self, y_pred, y_train, cache):

        if self.batch_norm:
            x, h_1_cache, relu_cache_1, pool, pool_cache, fc_cache_1, relu_cache_2, fc_cache_2, bn_cache = cache
        else:
            x, h_1_cache, relu_cache_1, pool, pool_cache, fc_cache_1, relu_cache_2, fc_cache_2 = cache

        grad = dcross_entropy(y_pred, y_train)

        dh3, dw3, db3 = self.step.fc_backward(grad, fc_cache_2)
        dh3 = self.step.relu_backward(dh3, relu_cache_2)

        if self.batch_norm:
            dh3, dgamma, dbeta = self.step.batchnorm_backward(dh3, bn_cache)

        dh2, dw2, db2 = self.step.fc_backward(dh3, fc_cache_1)

        dh2 = dh2.ravel().reshape(pool.shape)

        dpool = self.step.maxpool_backward(dh2, pool_cache)

        dh1 = self.step.relu_backward(dpool, relu_cache_1)

        dh, dw1, db1 = self.step.conv_backward(dh1, h_1_cache)

        if self.batch_norm:
            grads = dict(
                w1=dw1,
                w2=dw2,
                w3=dw3,
                b1=db1,
                b2=db2,
                b3=db3,
                gamma=dgamma,
                beta=dbeta
            )
        else:
            grads = dict(
                w1=dw1,
                w2=dw2,
                w3=dw3,
                b1=db1,
                b2=db2,
                b3=db3
            )

        return grads

    def predict(self, x):
        pred, _ = self.forward(x)
        return np.argmax(self.softmax(pred), axis=1)

    def softmax(self, x):

        reg = (x.T - np.max(x, axis=1)).T
        ex = np.exp(reg)
        ex = ex / np.sum(ex, axis=1).reshape(-1, 1)
        return ex

