import numpy as np
import cupy as cp

from project.scratch1.data.mnist_fashion import mnist_fashion
from project.scratch1.model import TrainingModel, u

import matplotlib.pyplot as plt


def tr(a):
    # Take transpose of a matrix
    return xp.einsum('...ij->...ji', a)


def matmul(*matrices, sum=False):
    # Perform matrix multiplication over many matrices efficiently
    sum_axis = 'z' if sum else ''
    einsum_str = '...{0}ab'.format(sum_axis)
    for i in range(1, len(matrices)):
        sum_axis_ = sum_axis if len(matrices[i].shape) >= 3 else ''
        einsum_str += ',...{0}{1}{2}'.format(sum_axis_, einsum_str[-1], chr(ord(einsum_str[-1]) + 1))
    einsum_str += '->...a{0}'.format(einsum_str[-1])
    res = xp.einsum(einsum_str, *matrices)
    return res


def batches(size, stop):
    n_batches = int(np.ceil(stop / size))
    tmp = np.round(np.linspace(0, stop, n_batches + 1)).astype(int)
    tmp2 = [slice(tmp[x], tmp[x + 1]) for x in range(n_batches)]
    return tmp2


def calc_Q(t, mdl, gamma):
    num_outs = mdl.u.d
    Q = xp.zeros((num_outs, num_outs))
    W = mdl.W
    index = 0
    for n in range(len(mdl.layers)):
        layer_d = mdl.layers[n]
        Wn = xp.zeros((layer_d, num_outs), dtype=float)
        if n > 0:
            Wn_vals = W[t, n]
            Wn[:, :Wn_vals.shape[1]] = Wn_vals
        Pn = xp.zeros_like(Wn)
        Pn[:, index:index + layer_d] = xp.eye(layer_d)
        tmp = Pn - Wn
        Q += gamma[n] * tmp.T @ tmp
        index += layer_d
    return Q


def calc_u_star(t, mdl, x, y, gamma, theta_t):
    W = mdl.W
    V = mdl.V[t - 1, 0]
    u_pr_prev = mdl.u[t - 1][:, mdl.n_inputs:]
    _, _d, _ = u_pr_prev.shape
    P = xp.eye(_d)[-mdl.layers[-1]:]
    tmp = (1 - theta_t)**2
    tmp1 = matmul(tr(P), tr(V), V, P) + (tmp + 1) * xp.eye(_d)
    tmp2 = matmul(tr(P), tr(V), y) + tmp * u_pr_prev

    for n in range(1, mdl.n_layers):
        Wn_pr = W[t-1, n][:, mdl.n_inputs:]
        _d1 = np.sum(mdl.layers[1:n+1])
        eye_d = xp.eye(_d)[:_d1]
        A, B = eye_d[-mdl.layers[n]:], eye_d[:-mdl.layers[n]]
        tmp3 = (A - Wn_pr @ B)
        tmp1 += gamma[n] * tr(tmp3) @ tmp3

        tmp2 += matmul(tr(tmp3), W[t-1, n, 0], x)

    u_pr = matmul(xp.linalg.inv(tmp1), tmp2)

    if gradient_check:
        # check whether the objective gradient is zero at u_pr
        grad = matmul(tr(u_pr), tr(P), tr(V), V, P) - matmul(tr(y), V, P) + (tmp + 1) * tr(u_pr) - tmp * tr(u_pr_prev)
        for n in range(1, mdl.n_layers):
            Wn_pr = W[t-1, n][:, mdl.n_inputs:]
            _d1 = np.sum(mdl.layers[1:n+1])
            eye_d = xp.eye(_d)[:_d1]
            A, B = eye_d[-mdl.layers[n]:], eye_d[:-mdl.layers[n]]
            tmp3 = (A - Wn_pr @ B)
            grad += gamma[n] * matmul(tr(u_pr), tr(tmp3), tmp3) - matmul(gamma[n] * tr(x), tr(W[t-1, n, 0]), tmp3)

        #assert xp.all(grad ** 2 < 0.001)
        if not xp.all(grad ** 2 < 0.001):
            print("Objective gradient not zero at Wn_star (norm: {})".format(xp.linalg.norm(grad)))

    u_star = xp.concatenate([x, u_pr], axis=1)
    return u_star


def calc_V_star(t, mdl, P, y, theta_t):
    n, _, _ = y.shape
    _d, _du = P.shape
    tmp = (1 - theta_t) ** 2
    A = matmul(y, tr(mdl.u[t]), sum=True)
    B = matmul(mdl.u[t], tr(mdl.u[t]), sum=True)
    tmp1 = A @ tr(P) + tmp * mdl.V[t - 1]
    tmp2 = (tmp + 1) * xp.eye(_d) + P @ B @ tr(P)

    V_star = tmp1 @ xp.linalg.inv(tmp2)

    if gradient_check:
        # check whether the objective gradient is zero at V_star
        grad = -P @ tr(A) + matmul(P, B, tr(P), tr(V_star)) + (tmp + 1) * tr(V_star) - tmp * tr(mdl.V[t - 1])
        # assert xp.all(grad ** 2 < 0.001)
        if not xp.all(grad ** 2 < 0.001):
            print("Objective gradient not zero at V_star (norm: {})".format(xp.linalg.norm(grad)))

    return V_star


def calc_Wn_star(t, n, mdl, gamma, theta_t):
    u = mdl.u
    W = mdl.W
    gamma = gamma[n]
    dn, sum_dm = W[t, n].shape
    sum_dm_I = xp.eye(sum_dm+dn)
    A, P = sum_dm_I[:sum_dm], sum_dm_I[sum_dm:]
    tmp = (1 - theta_t)**2

    u_n = u[t][:, :sum_dm+dn]
    _n, d, _ = u_n.shape
    op = matmul(u_n, tr(u_n), sum=True)
    tmp1 = gamma * matmul(P, op, tr(A)) + tmp * W[t - 1, n]
    tmp2 = gamma * matmul(A, op, tr(A)) + (tmp + 1) * xp.eye(sum_dm)
    Wn_star = tmp1 @ xp.linalg.inv(tmp2)

    if gradient_check:
        # check whether the objective gradient is zero at Wn_star
        grad = gamma * (A @ op @ tr(A) @ tr(Wn_star) - A @ op @ tr(P)) + (tmp + 1) * tr(Wn_star) - tmp * tr(W[t - 1, n])
        # assert xp.all(grad**2 < 0.001)
        if not xp.all(grad ** 2 < 0.001):
            print("Objective gradient not zero at Wn_star (norm: {})".format(xp.linalg.norm(grad)))

    return Wn_star


def foreword(t, mdl, x):
    uf = [x]
    num, _, _ = x.shape
    W = mdl.W
    V = mdl.V
    for n in range(1, mdl.n_layers):
        tmp = xp.zeros((num, mdl.layers[n], 1))
        for m in range(0, n):
            # tmp += matmul(W[t, n, m], uf[m])
            tmp += matmul(W[t, n, m], uf[m])  # not sure why, but this avoids a memory overflow
        tmp = xp.maximum(tmp, 0)
        uf.append(tmp)
    val = V[t, 0] @ uf[-1]
    return val


def mse(y1, y2):
    e = y1 - y2
    return tr(e) @ e


def objective1(t, t_u, mdl, P, y, gamma, theta_t):
    u = mdl.u
    term1 = mse(y, matmul(mdl.V[t - 1, 0], P, u[t_u]))
    term2 = 1 / 2 * matmul(tr(u[t_u]), calc_Q(t - 1, mdl, gamma), u[t_u])
    term3 = 1 / 2 * matmul((1 - theta_t) ** 2 * tr(u[t_u] - u[t - 1]), (u[t_u] - u[t - 1]))
    val = cp.asnumpy(term1 + term2 + term3)
    return val


def objective2(t, t_v, mdl, P, y, theta_t):
    u = mdl.u
    V = mdl.V
    term1 = xp.sum(mse(y, matmul(V[t_v, 0], P, u[t])))
    term2 = 1 / 2 * (1 - theta_t) ** 2 * xp.linalg.norm(V[t_v, 0] - V[t - 1, 0], 'fro') ** 2
    val = term1 + term2
    return val


def objective3(t, t_w, mdl, gamma, theta_t):
    u = mdl.u
    W = mdl.W
    val = 0
    for n in range(1, mdl.n_layers):
        gamma_n = gamma[n]
        dn, sum_dm = W[t, n].shape
        sum_dm_I = xp.eye(sum_dm + dn)
        A, P = sum_dm_I[:sum_dm], sum_dm_I[sum_dm:]

        u_n = u[t][:, :sum_dm + dn]

        tmp1 = P - W[t_w, n] @ A
        # term1 = xp.sum(tr(u_n) @ (gamma_n / 2 * tr(tmp1) @ tmp1) @ u_n, axis=0)
        term1 = gamma_n / 2 * matmul(tr(u_n), tr(tmp1), tmp1, u_n, sum=True)
        term2 = 1 / 2 * (1 - theta_t)**2 * xp.linalg.norm(W[t_w, n] - W[t - 1, n], 'fro')
        val += (term1 + term2).item()
    return val


def algorithm(mdl, x, y, gamma):
    n_layers = mdl.n_layers
    n, d, _ = x.shape
    n, do, _ = y.shape
    dn = mdl.layers[-1]
    mdl.u.size = n
    d_u = mdl.u.d

    P = xp.zeros((dn, d_u), dtype=float)
    P[:, -dn:] = xp.eye(dn)
    u = mdl.u
    W = mdl.W
    V = mdl.V
    mdl.randomize()
    loss_history = np.zeros(mdl.n_its)
    # calculate loss
    loss = xp.sum(mse(y, foreword(0, mdl, x)))
    print(loss)
    loss_history[0] = loss

    objective1_hist = np.zeros((mdl.n_its * 2, n))
    objective2_hist = np.zeros((mdl.n_its * 2,))
    objective3_hist = np.zeros((mdl.n_its * 2,))

    for t in range(1, mdl.n_its):
        theta_t = 1 / t ** 2
        objective1_hist[t * 2] = objective1(t, t - 1, mdl, P, y, gamma, theta_t)[:, 0, 0]
        # u update
        u_i_star = calc_u_star(t, mdl, x, y, gamma, theta_t)
        u[t] = u[t - 1] + theta_t * (u_i_star - u[t - 1])

        objective1_hist[t * 2 + 1] = objective1(t, t, mdl, P, y, gamma, theta_t)[:, 0, 0]

        objective2_hist[t*2] = objective2(t, t-1, mdl, P, y, theta_t)
        # V update
        V_star = calc_V_star(t, mdl, P, y, theta_t)
        V[t] = V[t - 1] + theta_t * (V_star - V[t - 1])

        objective2_hist[t*2+1] = objective2(t, t, mdl, P, y, theta_t)

        objective3_hist[t * 2] = objective3(t, t - 1, mdl, gamma, theta_t)

        # W update
        for n in range(1, n_layers):
            Wn_star = calc_Wn_star(t, n, mdl, gamma, theta_t)
            W[t, n] = W[t - 1, n] + theta_t * (Wn_star - W[t - 1, n])

        objective3_hist[t * 2 + 1] = objective3(t, t, mdl, gamma, theta_t)

        # calculate loss
        loss = xp.sum(mse(y, foreword(t, mdl, x)))
        print(loss)
        loss_history[t] = loss

    fig, axs = plt.subplots(2, 2)

    ax = axs[0, 0]
    ax.semilogy(np.arange(mdl.n_its), loss_history)
    ax.set_title("Overall loss")

    ax = axs[0, 1]
    for i in range(objective1_hist.shape[1]):
        ax.semilogy(np.arange(len(objective2_hist))[2:], objective1_hist[2:, i])
    ax.set_title("u objective")

    ax = axs[1, 0]
    ax.semilogy(np.arange(len(objective2_hist))[2:], objective2_hist[2:])
    ax.set_title("V objective")

    ax = axs[1, 1]
    ax.semilogy(np.arange(len(objective3_hist))[2:], objective3_hist[2:])
    ax.set_title("Wn objective")
    plt.show(block=False)


if __name__ == '__main__':
    gradient_check = True
    xp = cp
    n_its = 5
    x_, y_ = mnist_fashion()
    x = xp.asarray(x_[:10000])
    y = xp.asarray(y_[:10000])
    n, dx, _ = x.shape
    _, dy, _ = y.shape
    mdl = TrainingModel(n_its, [dx, 784, 784, 784], dy, xp=xp)
    gamma = xp.ones(4, dtype=float)
    algorithm(mdl, x, y, gamma)

    # test model
    y_pred = foreword(n_its - 1, mdl, x)
    y_pred = np.argmax(y_pred[..., 0], axis=1)
    y = np.argmax(y[..., 0], axis=1)
    acc = np.count_nonzero(y_pred == y) / len(y)
    print("Final train accuracy: {0}".format(acc))

    x = xp.asarray(x_[1000:1200])
    y = xp.asarray(y_[1000:1200])
    y_pred = foreword(n_its - 1, mdl, x)
    y_pred = np.argmax(y_pred[..., 0], axis=1)
    y = np.argmax(y[..., 0], axis=1)
    acc = np.count_nonzero(y_pred == y) / len(y)
    print("Final test accuracy: {0}".format(acc))
    print("Done")
