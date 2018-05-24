import numpy as np
from im2col import *


class Steps(object):

    def fc_forward(self, x, w, b):

        layer_out = np.dot(x, w) + b
        cache = (w, x)

        return layer_out, cache

    def fc_backward(self, dout, cache):

        w, h = cache
        dw = np.dot(h.T, dout)
        db = np.sum(dout, axis=0)
        dh = np.dot(dout, w.T)

        return dh, dw, db

    def relu_forward(self, x):

        return np.maximum(x, 0), x

    def relu_backward(self, dout, cache):

        dout[cache<0] = 0

        return dout

    def conv_forward(self, x, w, b, padding, stride):

        n_filter, d_filter, h_filter, w_filter = w.shape
        n_x, d_x, h_x, w_x = x.shape

        h_out = int((h_x - h_filter + 2 * padding) / stride + 1)
        w_out = int((w_x - w_filter + 2 * padding) / stride + 1)

        x_col = im2col_indices(x, h_filter, h_filter, padding, stride)
        w_col = w.reshape(n_filter, -1)

        layer_out = np.dot(w_col, x_col) + b
        layer_out = layer_out.reshape(n_filter, h_out, w_out, n_x)
        # layer_out = layer_out.reshape(n_filter, h_out, w_out, n_x)
        layer_out = layer_out.transpose(3, 0, 1, 2)

        cache = (x, w, b, stride, padding, x_col)

        return layer_out, cache

    def conv_backward(self, dout, cache):

        x, w, b, stride, padding, x_col = cache
        n_filter, d_filter, h_filter, w_filter = w.shape

        db = np.sum(dout, axis=(0, 2, 3)).reshape(n_filter, -1)
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
        dw = np.dot(dout_reshaped, x_col.T)
        dw = dw.reshape(w.shape)

        w_reshape = w.reshape(n_filter, -1)
        dx_col = np.dot(w_reshape.T, dout_reshaped)
        dx = col2im_indices(dx_col, x.shape, h_filter, w_filter, padding, stride)

        return dx, dw, db

    def maxpool_forward(self, x, size=2, stride=2):

        n_x, d_x, h_x, w_x = x.shape
        h_out = int((h_x - size) / stride + 1)
        w_out = int((w_x - size) / stride + 1)

        x_reshaped = x.reshape(n_x * d_x, 1, h_x, w_x)
        x_col = im2col_indices(x_reshaped, size, size, 0, 2)

        max_idx = np.argmax(x_col, axis=0)
        layer_out = x_col[max_idx, range(max_idx.size)]
        layer_out = layer_out.reshape(h_out, w_out, n_x, d_x)
        layer_out = layer_out.transpose(2, 3, 0, 1)

        cache = (x, size, stride, x_col, max_idx)

        return layer_out, cache

    def maxpool_backward(self, dout, cache):

        x, size, stride, x_col, max_idx = cache
        n_x, d_x, h_x, w_x = x.shape

        dx_col = np.zeros_like(x_col)
        dout_col = dout.transpose(2, 3, 0, 1).ravel()

        dx_col[max_idx, range(dout_col.size)] = dout_col
        dx = col2im_indices(dx_col, (n_x*d_x, 1, h_x, w_x), size, size, 0, stride)

        return dx.reshape(x.shape)

    def batchnorm_forward(self, x, gamma, beta, eps):

        n_x = x.shape[0]
        mu = 1. / n_x * np.sum(x, axis=0)
        xmu = x - mu
        sq = xmu ** 2
        var = 1. / n_x * np.sum(sq, axis=0)
        sqrtvar = np.sqrt(var + eps)
        ivar = 1. / sqrtvar
        xhat = xmu * ivar
        gammax = gamma * xhat
        out = gammax + beta
        cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

        return out, cache

    def batchnorm_backward(self, dout, cache):

        xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
        n_x, d_x = dout.shape
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * xhat, axis=0)
        dxhat = dout * gamma
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. / (sqrtvar ** 2) * divar
        dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
        dsq = 1. / n_x * np.ones((n_x, d_x)) * dvar
        dxmu2 = 2 * xmu * dsq
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = 1. / n_x * np.ones((n_x, d_x)) * dmu
        dx = dx1 + dx2

        return dx, dgamma, dbeta
