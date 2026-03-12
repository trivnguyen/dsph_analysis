
import scipy.stats as stats

class UniformPrior():
    def __init__(self, name, low, high):
        self.name = name
        self.low = low
        self.high = high
        self.dist = stats.uniform(loc=low, scale=high - low)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def rvs(self, size=1):
        return self.dist.rvs(size=size)


class GaussianPrior():
    def __init__(self, name, mean, std):
        self.name = name
        self.mean = mean
        self.std = std
        self.dist = stats.norm(loc=mean, scale=std)

    def logpdf(self, x):
        return self.dist.logpdf(x)

    def rvs(self, size=1):
        return self.dist.rvs(size=size)


class ConstantPrior():
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def logpdf(self, x):
        return 0.0

    def rvs(self, size=1):
        return np.full(size, self.value)