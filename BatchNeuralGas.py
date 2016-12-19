import numpy as np
import matplotlib.pylab as plt
import os
import shutil
from make_data import Circle, Rect, Gauss
import sys


class BatchNeuralGas:
    def __init__(self, num_node=10, lamb_i=0.5, lamb_f=0.05, t_max=5000):
        self.lamb_i = lamb_i
        self.lamb_f = lamb_f
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

    def execute(self, inputs, path_dir=None):
        # initialize
        self.x = inputs
        self.node = [np.average(self.x, axis=0)for _ in range(self.num_node)]
        width = float(max(inputs[:, 0]) - min(inputs[:, 0]))
        height = float(max(inputs[:, 1]) - min(inputs[:, 1]))
        aspect = height / width

        # make output directory
        if path_dir is not None:
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
            if path_dir is not None:
                w = np.array(self.node)
                fig = plt.figure(figsize=(5., 5. * aspect), dpi=30)
                plt.xlim(min(self.x[:, 0]) - 0.5, max(self.x[:, 0]) + 0.5)
                plt.ylim(min(self.x[:, 1]) - 0.5, max(self.x[:, 1]) + 0.5)
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
    if len(sys.argv) != 2:
        print "usage: " + sys.argv[0] + " 'data-shape'"
        print "'data-shape' is chosen from: "
        print "     'gaussian', 'rings', 'rectangle', '2circles'"
        exit(0)

    d_shape = sys.argv[1]
    if d_shape not in ['gaussian', 'rings', 'rectangle', '2circles']:
        print "'data-shape' must be chosen from: "
        print "     'gaussian', 'rings', 'rectangle', '2circles'"
        exit(0)

    # make demo-data
    data = None

    if d_shape == 'gaussian':
        # 3 gaussian
        c1 = Gauss(sigma=0.75, mu_x=-1., mu_y=1.)
        c2 = Gauss(sigma=0.75, mu_x=5., mu_y=5.)
        c3 = Gauss(sigma=0.75, mu_x=4., mu_y=-3.)
        data = np.vstack((c1.set_data(250), c2.set_data(250), c3.set_data(250)))

    elif d_shape == 'rings':
        # 2 rings
        r1 = Circle(radius=2., x=0., y=0.)
        r2 = Circle(radius=4., x=0., y=0.)
        data = np.vstack((r1.set_data(250, is_fill=False), r2.set_data(250, is_fill=False)))

    elif d_shape == 'rectangle':
        # 1 rectangle
        f1 = Rect(edge=2., x=0, y=0.)
        data = f1.set_data(500, is_fill=True)

    elif d_shape == '2circles':
        # 2-size circles
        c1 = Circle(radius=1., x=-1., y=0.)
        c2 = Circle(radius=2., x=4., y=0.)
        data = np.vstack((c1.set_data(250, is_fill=True), c2.set_data(250, is_fill=True)))

    # dump demo-data
    import cPickle
    cPickle.dump(data, open('input_data', 'wb'))

    # Neural Gas
    ng = BatchNeuralGas(num_node=50, t_max=300, lamb_i=7.)
    ng.execute(data, path_dir=d_shape)
