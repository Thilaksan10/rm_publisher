import numpy as np

def go_to_start(x, y, randomize=False):
    if randomize:
        rnd_pos = np.random.uniform(-0.5, 0.5, (2,))
    else:
        rnd_pos = [0, 0]
    xdot = 0.6 * -x  + rnd_pos[0]
    ydot = 0.6 * -y  + rnd_pos[1]
                                
    return xdot, ydot

def f01(x, y, c):
    xdot = 10 * c[1] * (y / ((x + 1) ** 2 + x ** 2 + 1e-6) - x / ((x - 1) ** 2 + y ** 2 + 1e-6))
    ydot = -10 * c[0] * ((x + 1) / (((x + 1) ** 2 + y ** 2) + 1e-6) + (x - 1) / (((x - 1) ** 2 + y ** 2) + 1e-6))

    return xdot, ydot


def f02(x, y, c):
    xdot = 10 * c[0] * -y / (np.sqrt(x ** 2 + y ** 2) + 1e-6)
    ydot = 10 * c[1] * x / (np.sqrt(x ** 2 + y ** 2) + 1e-6)
    return xdot, ydot


def f03(x, y, c):

    ydot = 4 * c[0] * (np.sin(x) + np.sin(y))
    xdot = 10 * c[0] * np.sin(2 * y)

    return xdot, ydot


def f04(x, y, c):
    xdot = 6 * c[0] * np.sin(c[1] * x)
    ydot = 3 * c[0] * np.sin(c[1] * y)
    return xdot, ydot


def f05(x, y, c):
    xdot = 4 * c[0] * (np.sin(x) + np.sin(y))
    ydot = 4 * c[1] * (np.sin(x) - np.sin(y))
    return xdot, ydot


def f06(x, y, c):
    xdot = 11 * c[0] * (x ** 2 + y ** 2) / (2 * x + 1) * (np.cos(2 * c[1] * x) + 1)
    ydot = -11 * c[0] * (x ** 2 + y ** 2) / (2 * x + 1) * (np.sin(2 * c[1] * x))
    return xdot, ydot


def f07(x, y, c):
    xdot = -6 * c[0] * np.cos(x / (y + 1e-6))
    ydot = 6 * c[0] * np.sin(x * y)

    return xdot, ydot


def f08(x, y, c):
    rnd_pos = np.random.uniform(-8, 8, (2,))

    xdot = (x - y - (x * (x ** 2 + y ** 2))) / (c[0] + 1e-6) + rnd_pos[0]
    ydot = (x + y - (y * (x ** 2 + y ** 2))) / (c[1] + 1e-6) + rnd_pos[1]

    return xdot, ydot


def f09(x, y, c):
    xdot = -40 * c[1] * (y + y) / (x ** 2 + y ** 2 + 10)
    ydot = -40 * c[0] * (y - x) / (x ** 2 + y ** 2 + 10)

    return xdot, ydot


def f10(x, y, c):
    xdot = 5 * c[0] * np.sin((x - y) / 3)
    ydot = -5 * c[1] * np.cos((x + y) / 3)
    return xdot, ydot


def f11(x, y, c):
    xdot = 5 * c[0] * np.sin(x) * np.cos(y)
    ydot = -5 * c[1] * np.cos(x) * np.sin(y)
    return xdot, ydot


def f12(x, y, scale):
    rnd_pos = np.random.uniform(-8, 8, (2,))
    xdot = -scale[0] * x + rnd_pos[0]
    ydot = -scale[1] * y + rnd_pos[1]
    return xdot, ydot

def f13(x, y, c):
    xdot = x + (3 * c[0] * y)
    ydot = -3 * c[1] * x
    return xdot, ydot


def f14(x, y, c):
    xdot = 3 * c[1] * (1 / (x + (x / (abs(x) + 1e-6)))) * (y - x)
    ydot = 3 * c[0] * (4 + ((x + y)/(x**2 + y**2 + 1)))

    return xdot, ydot


def f15(x, y, c):
    xdot = c[0] * (x - y)
    ydot = c[1] * (x + y)
    return xdot, ydot


def f16(x, y, c):
    xdot = c[0] * 2 * x
    ydot = np.sin(1 * c[1] * x) * np.exp(np.log(2) * x)
    return xdot, ydot


def f17(x, y, c):
    xdot = c[0] * y
    ydot = c[1] * (-x + (2 * y))
    return xdot, ydot


def f18(x, y, c):
    xdot = c[0] * y
    ydot = c[1] * ((-0.5 * (1 - x**2) * y) - x)
    return xdot, ydot


def f19(x, y, c):
    xdot = y
    ydot = (0.4 * c[0] * y) + ((2 * c[1] * x) * (1 - x**2))
    return xdot, ydot


class VelocityField:

    def __init__(self, num_rm, visualization=True, reset=False):
        self._velocity_fields = {}

        self.num_rm = num_rm
        self.generate_field_functions()
        self.vis = visualization
        self.reset = reset
        # self.p = 0
        if self.vis:
            self.last_shuffle = ["0" for _ in range(self.num_rm)]

    def generate_field_functions(self):
        self._velocity_fields["Go to start"] = go_to_start
        self._velocity_fields["f01"] = f01
        self._velocity_fields["f02"] = f02
        self._velocity_fields["f03"] = f03
        self._velocity_fields["f04"] = f04
        self._velocity_fields["f05"] = f05
        self._velocity_fields["f06"] = f06
        self._velocity_fields["f07"] = f07
        self._velocity_fields["f08"] = f08
        self._velocity_fields["f09"] = f09
        self._velocity_fields["f10"] = f10
        self._velocity_fields["f11"] = f11
        self._velocity_fields["f12"] = f12
        self._velocity_fields["f13"] = f13
        self._velocity_fields["f14"] = f14
        self._velocity_fields["f15"] = f15
        self._velocity_fields["f16"] = f16
        self._velocity_fields["f17"] = f17
        self._velocity_fields["f18"] = f18
        self._velocity_fields["f19"] = f19

    def get_last_shuffle(self):
        if self.vis:
            return self.last_shuffle

    def __getitem__(self, env):
        rnd_idx = np.random.randint(1, len(self))
        # rnd_idx = self.p
        if self.reset:
            rnd_key = [*self._velocity_fields][0]
        else:
            rnd_key = [*self._velocity_fields][rnd_idx]
        if self.vis:
            self.last_shuffle[env] = rnd_key
        # self.p = (self.p + 1) % self.num_rm
        return self._velocity_fields[rnd_key]

    def __len__(self):
        return len(self._velocity_fields)
