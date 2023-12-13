from typing import List, Union

import numpy as np


class Model(object):
    def __init__(self, W, V):
        self._W = W
        self._V = V
        self._u = None

    def foreword(self, x):
        u = self._u
        u.size = x.shape[0]
        W = self._W
        V = self._V
        u[0][:] = x
        for m in range(1, self._n):
            for n in range(0, m):
                u[:, m] += W[m, n] @ u[:, n]
            u[:, m] = self._xp.maximum(u[:, m], 0)
        return V @ u[-1]


class TrainingModel(object):
    def __init__(self, n_its, layers: List[int], num_outs, xp=np):
        self._u = u(layers, xp=xp)
        self._n = len(layers)
        self._W = W(layers, xp=xp)
        self._V = xp.empty((n_its, 1, num_outs, layers[-1]), dtype=float)
        self._layers = np.asarray(layers)
        self.n_its = n_its
        self._xp = xp

    @property
    def u(self):
        return self._u

    @property
    def W(self):
        return self._W

    @property
    def V(self):
        return self._V

    @property
    def n_layers(self):
        return self._W.num_layers()

    @property
    def n_inputs(self):
        return self._layers[0]

    @property
    def layers(self):
        return self._layers

    def randomize(self):
        self._W.randomize()
        self._u.randomize()
        self._V[:] = self._xp.random.rand(*self._V.shape) * 2 - 1


class W(object):
    """
    indexing convention: (iteration [t], input index [i], matrix row, matrix col)
    """
    def __init__(self, layers: Union[List, np.ndarray], xp=np):
        layers = np.asarray(layers)
        self._n = len(layers)
        self._d = np.sum(layers)

        self._index = []
        start = 0
        for i in range(self._n):
            stop = start + layers[i]
            self._index.append(slice(start, stop))
            start = stop
        self._buf = {}
        self._xp = xp

    def _gen_index(self, item):
        if isinstance(item, tuple):
            t, n, m = None, None, None
            if len(item) == 2:
                t, n = item
                n = self._index[n]
                m = slice(0, n.start)
            elif len(item) == 3:
                t, n, m = item
                n, m = self._index[n], self._index[m]
        else:
            t = item
            n = slice(None, None, None)
            m = slice(None, None, None)
        return t, n, m

    def __getitem__(self, item):
        t, n, m = self._gen_index(item)
        self._expand(t)
        return self._buf[t][n, m]

    def __setitem__(self, item, value):
        t, n, m = self._gen_index(item)
        self._expand(t)
        self._buf[t][n, m] = value

    def randomize(self):
        self._buf = {}
        self._expand(0)
        self._xp.random.default_rng().random(out=self._buf[0])
        self._buf[0] *= 2
        self._buf[0] -= 1

    def num_layers(self):
        return self._n

    def _expand(self, t):
        keys = list(self._buf.keys())
        if t in keys:
            return
        if len(keys) >= 2:
            lowest_key = min(*keys)
            del self._buf[lowest_key]
        self._buf[t] = self._xp.empty((self._d, self._d), dtype=float)


class u(object):
    """
    indexing convention: (iteration [t], input index [i], matrix row, matrix col)
    """
    def __init__(self, layers: Union[List, np.ndarray], xp=np):
        layers = np.asarray(layers)
        self._n = 0
        self._d = np.sum(layers)
        self._index = []
        start = 0
        for layer in layers:
            stop = start + layer
            self._index.append(slice(start, stop))
            start = stop
        self._buf = {}
        self._xp = xp

    def randomize(self):
        self._buf = {}
        self._expand(0)
        self._xp.random.default_rng().random(out=self._buf[0])
        self._buf[0] *= 2
        self._buf[0] -= 1

    def _gen_index(self, item):
        if isinstance(item, tuple):
            t = item[0]
            if len(item) >= 3:
                item = list(item)
                item[2] = self._index[item[2]]
                item = tuple(item[1:])
        else:
            t = item
            item = slice(None, None, None)
        return t, item

    def __getitem__(self, item):
        t, item = self._gen_index(item)
        self._expand(t)
        return self._buf[t][item]

    def __setitem__(self, item, value):
        t, item = self._gen_index(item)
        self._expand(t)
        self._buf[t][item] = value

    def __len__(self):
        return self._d

    @property
    def d(self):
        return self._d

    @property
    def size(self):
        """Number of instances."""
        return self._n

    @size.setter
    def size(self, n):
        self._n = n

    def _expand(self, t):
        keys = list(self._buf.keys())
        if t in keys:
            return
        if len(keys) >= 2:
            lowest_key = min(*keys)
            del self._buf[lowest_key]
        self._buf[t] = self._xp.empty((self._n, self._d, 1), dtype=float)