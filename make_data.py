import random
import numpy as np 


class Shape:
    def __init__(self, z, x, y):
        self.l = z
        self.c = [x, y]


class Circle(Shape):
    # circle
    def __init__(self, radius, x, y):
        Shape.__init__(self, radius, x, y)

    def set_data(self, num, is_fill):
        if is_fill:
            data = []
            for i in range(num):
                rad = random.triangular(0., self.l, self.l)
                th = 2.0 * np.pi * random.random()
                data.append(np.array([rad * np.cos(th) + self.c[0], rad * np.sin(th) + self.c[1]]))
        else:
            data = [np.array([self.l * np.cos(2.0 * np.pi * j / num) + self.c[0],
                              self.l * np.sin(2.0 * np.pi * j / num) + self.c[1]]) for j in range(num)]
        return np.array(data)


class Gauss(Shape, Circle):
    # gauss
    def __init__(self, sigma, mu_x, mu_y):
        Shape.__init__(self, sigma, mu_x, mu_y)

    def set_data(self, num, is_fill=None):
        return np.random.multivariate_normal(self.c, np.diag([self.l, self.l]), size=num)


class Rect(Shape, Circle):
    # rectangle
    def __init__(self, edge, x, y):
        Shape.__init__(self, edge, x, y)

    def set_data(self, num, is_fill):
        if is_fill:
            data = [np.array([random.uniform(self.c[0] - self.l / 2., self.c[0] + self.l / 2.),
                             random.uniform(self.c[1] - self.l / 2., self.c[1] + self.l / 2.)]) for _ in range(num)]
        else:
            data = [np.array([self.c[0] + self.l * (8. * i / num - 1.) / 2., self.c[1] - self.l / 2.])
                    for i in range(num / 4 + 1)]
            data.extend([np.array([self.c[0] + self.l * (8. * i / num - 1.) / 2., self.c[1] + self.l / 2.])
                         for i in range(num / 4 + 1)])
            data.extend([np.array([self.c[0] - self.l / 2., self.c[1] + self.l * (8. * i / num - 1.) / 2.])
                         for i in range(num / 4)])
            data.extend([np.array([self.c[0] + self.l / 2., self.c[1] + self.l * (8. * i / num - 1.) / 2.])
                         for i in range(num / 4)])
        return np.array(data)


class Tri(Shape, Circle):
    # triangle
    def __init__(self, edge, x, y):
        Shape.__init__(self, edge, x, y)

    def set_data(self, num, is_fill):
        p1 = [self.c[0], self.c[1] + np.sqrt(3.) * self.l / 3.]
        p2 = [self.c[0] - self.l / 2., self.c[1] - np.sqrt(3.) * self.l / 6.]
        p3 = [self.c[0] + self.l / 2., self.c[1] - np.sqrt(3.) * self.l / 6.]

        if is_fill:
            data = []
            for i in range(num):
                y_p = random.triangular(0, self.l * np.sqrt(3.) / 2., 0)
                x_p = random.uniform(p2[0] + y_p / np.sqrt(3.), p3[0] - y_p / np.sqrt(3.))
                data.append(np.array([x_p, y_p + p2[0]]))

        else:
            data = [np.array([p2[0] + 3. * self.l * i / num, p2[1]])
                    for i in range(num / 3 + 1)]
            data.extend([np.array([p3[0] + (p1[0] - p3[0]) * 3. * i / num, p3[1] + (p1[1] - p3[1]) * 3. * i / num])
                         for i in range(num / 3 + 1)])
            data.extend([np.array([p1[0] + (p2[0] - p1[0]) * 3. * i / num, p1[1] + (p2[1] - p1[1]) * 3. * i / num])
                         for i in range(num / 3)])
        return np.array(data)
