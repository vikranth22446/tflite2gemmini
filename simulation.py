import numpy as np
from scipy.special import softmax
import tensorflow as tf

def pad2d(X, pad, kernel_shape, stride=None):
    p = pad
    if isinstance(p, int):
        p = (p, p, p, p)
    if isinstance(p, tuple):
        if len(p) == 2:
            p = (p[0], p[0], p[1], p[1])

        X_pad = np.pad(
            X,
            pad_width=((p[0], p[1]), (p[2], p[3]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    if p == "same" and kernel_shape and stride is not None:
        p = compute_pad(X, kernel_shape, stride)
        X_pad, p = pad2d(X, p, kernel_shape, stride)
    return X_pad, p

def compute_pad(X, kernel_shape, stride):
    in_channels, in_rows, in_cols = X.shape
    s = stride

    p_rows = int(((s - 1) * in_rows - s + kernel_shape[0]) / 2)
    p_cols = int(((s - 1) * in_cols - s + kernel_shape[1]) / 2)

    return (p_rows, p_rows, p_cols, p_cols)

def saturating_add_int8(a, b):
    added = a + b
    if added < -128:
        return -128
    elif added > 127:
        return 127
    else:
        return added

def conv2d(X: np.ndarray, W, b, scales, stride, padding, in_zero_point=-128, out_zero_point=-128):
    out_channels, kernel_height, kernel_width, in_channels = W.shape
    in_rows, in_cols, in_channels = X.shape
    kernel_shape = (kernel_height, kernel_width)
    if stride == 2 and padding == 1:
        padding = (0, 1, 0, 1)
    X_pad, p = pad2d(X.astype(np.int32) - in_zero_point, padding, kernel_shape, stride)
    out_rows = int((in_rows + p[0] + p[1] - kernel_height) / stride + 1)
    out_cols = int((in_cols + p[2] + p[3] - kernel_width) / stride + 1)
    Z = np.empty((out_rows, out_cols, out_channels), dtype=np.int8)
    for r in range(out_rows):
        for c in range(out_cols):
            for oc in range(out_channels):
                filter_applied = np.sum(X_pad[r * stride: r * stride + kernel_height,
                                              c * stride: c * stride + kernel_width, :] * W[oc, :, :, :],
                                axis=(0, 1, 2))
                Z[r, c, oc] = saturating_add_int8((filter_applied + b[oc]) * scales[oc], out_zero_point)
    assert out_zero_point == -128
    # Z[Z < out_zero_point] = 0 # relu not needed because saturating add
    return Z

def conv2d_dw(X: np.ndarray, W, b, scales, stride=1, in_zero_point=-128, out_zero_point=-128):
    _, kernel_height, kernel_width, out_channels = W.shape
    in_rows, in_cols, in_channels = X.shape
    kernel_shape = (kernel_height, kernel_width)
    padding = 1
    if stride == 2 and padding == 1:
        padding = (0, 1, 0, 1)
    X_pad, p = pad2d(X.astype(np.int32) - in_zero_point, padding, kernel_shape, stride)
    out_rows = int((in_rows + p[0] + p[1] - kernel_height) / stride + 1)
    out_cols = int((in_cols + p[2] + p[3] - kernel_width) / stride + 1)
    Z = np.empty((out_rows, out_cols, out_channels), dtype=np.int8)
    for r in range(out_rows):
        for c in range(out_cols):
            for oc in range(out_channels):
                filter_applied = np.sum(X_pad[r * stride: r * stride + kernel_height,
                                              c * stride: c * stride + kernel_width, oc] * W[0, :, :, oc],
                                        axis=(0, 1))
                Z[r, c, oc] = saturating_add_int8((filter_applied + b[oc]) * scales[oc], out_zero_point)
    assert out_zero_point == -128
    # Z[Z < out_zero_point] = 0 # relu not needed because saturating add
    return Z

def avg_pool2d(X, pooling_shape=(3,3)):
    return np.average(X, axis=(0, 1)).astype(np.int8).reshape((1, 1, X.shape[-1]))

def fc(X, W, b, output_scale, in_zero_point=-128, out_zero_point=-128, weight_zero_point=0):
    Z = ((W.astype(np.int32) - weight_zero_point) @ (X.astype(np.int32) - in_zero_point) + b) * output_scale
    Z += out_zero_point
    Z[Z < -128] = -128
    Z[Z > 127] = 127
    return Z.astype(np.int8)

def residual_add(X1, X2):
    return X1 + X2

def softmax_layer(X, in_zero_point, in_scale, out_zero_point, out_scale):
    Z = (softmax((X.astype(np.float32) - in_zero_point) * in_scale) / out_scale) + out_zero_point
    Z[Z < -128] = -128
    Z[Z > 127] = 127
    return Z.astype(np.int8)
