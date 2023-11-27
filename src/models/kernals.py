
import numpy as np

class Kernel:
    def __add__(self, other):
        return SumKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)

class SumKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def compute(self, x1, x2):
        return self.kernel1.compute(x1, x2) + self.kernel2.compute(x1, x2)

class ProductKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def compute(self, x1, x2):
        return self.kernel1.compute(x1, x2) * self.kernel2.compute(x1, x2)

class GaussianKernel(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def compute(self, x1, x2):
        return np.exp(-0.5 * np.sum((x1 - x2) ** 2) / self.length_scale ** 2)

class RBFKernel(Kernel):
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def compute(self, x1, x2):
        return np.exp(-0.5 * np.sum((x1 - x2) ** 2) / (2 * self.length_scale ** 2))

class RationalQuadraticKernel(Kernel):
    def __init__(self, alpha=1.0, length_scale=1.0):
        self.alpha = alpha
        self.length_scale = length_scale

    def compute(self, x1, x2):
        dist = np.sum((x1 - x2) ** 2)
        return (1 + dist / (2 * self.alpha * self.length_scale ** 2)) ** (-self.alpha)
