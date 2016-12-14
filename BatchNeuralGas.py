import numpy as np
import matplotlib.pylab as plt
import os
import shutil
import cPickle


class BatchNeuralGas:
    def __init__(self, num_node=10, lamb=0.5, t_max=5000):
        self.lamb_i = lamb
        self.lamb_f = 0.1
        self.t_max = t_max
        self.num_node = num_node
        self.node = None
        self.k = None
        self.x = None

    def hl(self, t):
        # h_lambda(k^p_n, t)
        l = self.lamb_i * ((self.lamb_f / self.lamb_i)**(float(t) / self.t_max))
        return np.exp(-self.k / l)

    def rank(self):
        # rank k^p_n
        self.k = np.empty((self.num_node, self.x.shape[0]))
        for p, x_ in enumerate(self.x):
            dist = [[np.dot(x_ - n, x_ - n), i] for i, n in enumerate(self.node)]
            dist.sort()
            for i, d in enumerate(dist):
                self.k[d[1], p] = float(i)
            del dist

    def execute(self, inputs, path_dir='img'):
        # initialize
        self.x = inputs
        self.node = [np.average(self.x, axis=0)for _ in xrange(self.num_node)]

        # make output directory
        if os.path.exists(path_dir):
            shutil.rmtree(path_dir)
        os.mkdir(path_dir)

        for t in xrange(self.t_max):

            # ranking
            self.rank()

            # calculate h_lambda
            h_lambda = self.hl(t)

            # update weight
            sum_hl = np.sum(h_lambda, axis=1)
            self.node = [np.sum(h_lambda[i][:, np.newaxis] * self.x, axis=0) / sum_hl[i] for i in range(self.num_node)]

            if t % 50 == 0:
                print t, self.error()

            # save current state to image
            w = np.array(self.node)
            fig = plt.figure(figsize=(3.5, 3.5), dpi=30)
            plt.scatter(self.x[:, 0], self.x[:, 1], marker='o', color='b', s=10)
            plt.scatter(w[:, 0], w[:, 1], marker='*', color='r', s=60)
            plt.savefig(path_dir + '/' + '%03d' % t + '.png')
            plt.close(fig)

    def error(self):
        return np.sum([np.amin([np.dot(x_ - n, x_ - n) for n in self.node]) for x_ in self.x])

    def labeling(self):
        label = {}
        for j in xrange(len(self.node)):
            label[j] = []

        for p, x_ in enumerate(self.x):
            dist = [[np.dot(x_ - n, x_ - n), i] for i, n in enumerate(self.node)]
            dist.sort()
            label[int(dist[0][1])].append(p)
            del dist
        return label

if __name__ == '__main__':
    # make demo-data
    data = 0.75 * np.random.randn(250, 2) + np.array([-1., 1.])
    data = np.vstack((data, 0.75 * np.random.randn(250, 2) + np.array([5., 5.])))
    data = np.vstack((data, 0.75 * np.random.randn(250, 2) + np.array([4., -3.])))

    # Neural Gas
    ng = BatchNeuralGas(num_node=50, t_max=100, lamb=5.)
    ng.execute(data)

    print ng.labeling()
    cPickle.dump(data, open('input_data', 'wb'))
