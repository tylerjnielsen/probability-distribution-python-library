"""
Distributions Supported:
    UniformDistribution
    BinomialDistribution
    GeometricDistribution
    HypergeometricDistribution
    PoissonDistribution
"""

import math
import matplotlib.pyplot as plt
from typing import List
from abc import ABC, abstractmethod


class DiscreteDistribution(ABC):
    def __init__(self):
        self._mean = None
        self._variance = None
        self._low = None
        self._high = None

    @abstractmethod
    def probability(self, x: int) -> float:
        """
        Args:
            x (int): The value to get the probability of observing in the discrete distribution.

        Returns:
            The probability of observing exactly x in this distribution.
        """
        pass

    def probability_of_range(self, low: int, high: int) -> float:
        """
        Args:
            low (int): The lower bound of the range to calculate the probability of (inclusive).
            high (int): The upper bound of the range to calculate the probability of (inclusive).

        Returns:
            The probability of observing a value within the range provided on the distribution.
        """
        if low > high:
            raise ValueError('The lower bound must be less than or equal to the upper bound.')
        if low < 0:
            raise ValueError('The lower bound must be greater than or equal to 0.')

        probability = 0
        for x in range(int(low), int(high + 1), 1):
            probability += self.probability(x)
        return probability

    def probability_greater_than(self, low: int) -> float:
        """
        Args:
            low (int): The lower bound of the range to calculate the probability of (exclusive).

        Returns:
            The probability of observing a value greater than the lower bound provided.
        """
        return 1 - self.probability_less_than(low + 1)

    def probability_less_than(self, high: int) -> float:
        """
        Args:
            high (int): The lower bound of the range to calculate the probability of (exclusive).

        Returns:
            The probability of observing a value less than the lower bound provided.
        """
        return self.probability_of_range(self._low, high - 1)

    def mean(self) -> float:
        """
        Returns:
            The expected value of the distribution.
        """
        return self._mean

    def variance(self) -> float:
        """
        Returns:
            The variance of the distribution.
        """
        return self._variance

    def low(self) -> int:
        """
        Returns:
            The lower bound of where the distribution is defined.
        """
        return self._low

    def high(self) -> int:
        """
        Returns:
            The upper bound of where the distribution is defined.
        """
        return self._high

    def standard_deviation(self) -> float:
        """
        Returns:
            The value of 1 standard deviation of the discrete distribution.
        """
        return math.sqrt(self.variance())

    def display_graph(self, plot_title: str = None, low: int = None, high: int = None,
                      save_fig: bool = False):
        """
        Displays a graph of the discrete distribution using Matplotlib.

        Args:
            plot_title (string): The title of the plot.
            low (int): The lower bound of where to display the graph (inclusive).
            high (int): The upper bound of where to display the graph (inclusive).
            save_fig (bool): True to save fig as a file, false to not. Default is false.
        """
        if plot_title is None:
            plot_title = "Probability Distribution"
        if low is None:
            low = self._low
        if high is None:
            high = self._high

        if math.isinf(high):
            high = int(self._mean + 3 * math.sqrt(self._variance))

        numbers = []
        probabilities = []

        for x in range(int(low), int(high + 1), 1):
            numbers.append(x)
            probabilities.append(self.probability(x))

        plt.bar(numbers, probabilities)

        plt.title(plot_title)
        plt.xlabel('x')
        plt.ylabel('Probability')

        if save_fig:
            plt.savefig(plot_title)

        plt.show()


class CustomDistribution(DiscreteDistribution):
    _values: List[int]
    _probabilities: List[float]

    def __init__(self, values: List[int], probabilities: List[float]):
        if len(values) != len(probabilities):
            raise ValueError('There must be a value corresponding to every probability')
        super().__init__()

        self._low = values[0]
        self._high = values[len(values) - 1]

        self._mean = 0
        for x in values:
            self._mean += x * probabilities[x - self._low]

        self._variance = 0

        for i, x in enumerate(values):
            if i > 0 and x != values[i - 1] + 1:
                raise ValueError('Values list must be sorted and consecutive.')
            if probabilities[x - self._low] < 0:
                raise ValueError('Probabilities must not be negative. Probability at index ' + str(x - self._low) +
                                 ' of the probabilities list is negative.')
            self._variance += x ** 2 * probabilities[x - self._low]
        self._variance -= self._mean**2

        self._values = values
        self._probabilities = probabilities

    def probability(self, x: int) -> float:
        if not (self._low <= x <= self._high):
            return 0
        low = self._values[0]
        return self._probabilities[x - low]


class UniformDistribution(DiscreteDistribution):
    def __init__(self, low: int, high: int):
        """
        Initializes an object to handle a discrete uniform distribution.

        Args:
            low (int): The lower bound of where the uniform distribution is defined (inclusive).
            high (int): The upper bound of where the uniform distribution is defined (inclusive).
        """
        if low > high:
            raise ValueError('The lower bound must be less than or equal to the upper bound.')
        super().__init__()
        self._low = low
        self._high = high

        self._mean = uniform_mean(low, high)
        self._variance = uniform_variance(low, high)

    def probability(self, x: int) -> float:
        if x < self._low or x > self._high:
            return 0
        if self._high == self._low:
            return 1
        return 1 / (self._high - self._low + 1)


class BinomialDistribution(DiscreteDistribution):
    def __init__(self, n: int, p: float):
        """
        Initializes an object for handling a binomial distribution.

        Args:
            n (int): The number of trials for the binomial distribution.
            p (float): The probability of a success on every given trial.
        """
        if n > 400:
            raise ValueError('The n value for binomial distribution must not exceed 400. ' +
                             'Consider approximating this distribution using the central limit theorem ' +
                             'and continuous.NormalDistribution')
        if n < 1:
            raise ValueError('The n value must be an int greater than or equal to 1.')
        if p < 0 or p > 1:
            raise ValueError('The p value must be a float on the range [0, 1]')
        super().__init__()
        self._n = n
        self._p = p

        self._low = 0
        self._high = n

        self._mean = binomial_mean(n, p)
        self._variance = binomial_variance(n, p)

    def n(self) -> int:
        return self._n

    def p(self) -> float:
        return self._p

    def probability(self, x: int) -> float:
        if x < self._low or x > self._high:
            return 0
        return choose(self._n, x) * pow(self._p, x) * pow(1 - self._p, self._n - x)


class GeometricDistribution(DiscreteDistribution):
    def __init__(self, p: float):
        """
        Initializes an object for handling a geometric distribution.

        Args:
            p (float): The probability of observing a success on any given trial.
        """
        if not isinstance(p, float):
            raise TypeError('The p value must be of type float on the range [0, 1].')
        if p < 0 or p > 1:
            raise ValueError('The p value must be on the range [0, 1].')
        super().__init__()
        self._p = p

        self._low = 1
        self._high = math.inf

        self._mean = geometric_mean(p)
        self._variance = geometric_variance(p)

    def p(self) -> float:
        return self._p

    def probability(self, x: int):
        if x < self._low or x > self._high:
            return 0
        return self._p * ((1 - self._p)**(x-1))


class HypergeometricDistribution(DiscreteDistribution):
    def __init__(self, total_population: int, successes_in_population: int, sample_size: int):
        """
        Initializes an object for handling a hypergeometric distribution.

        Args:
            total_population (int): The total population size.
            successes_in_population (int): The number of successes in the population.
            sample_size (int): The sample size taken without replacement from the total population.
        """
        if successes_in_population > total_population:
            raise ValueError('The number of successes in the population cannot exceed the size ' 
                             'of the total population.')
        if sample_size > total_population:
            raise ValueError('The sample size cannot exceed the total population.')
        super().__init__()
        self._total_population = total_population
        self._successes_in_population = successes_in_population
        self._sample_size = sample_size

        self._low = max(sample_size - (total_population - successes_in_population), 0)
        self._high = sample_size

        self._mean = hypergeometric_mean(total_population, successes_in_population, sample_size)
        self._variance = hypergeometric_variance(total_population, successes_in_population, sample_size)

    def total_population(self) -> int:
        return self._total_population

    def successes_in_population(self) -> int:
        return self._successes_in_population

    def sample_size(self) -> int:
        return self._sample_size

    def probability(self, x: int) -> float:
        if x < self._low or x > self._high:
            return 0

        prob = choose(self._successes_in_population, x)
        prob *= choose(self._total_population - self._successes_in_population, self._sample_size - x)
        prob /= choose(self._total_population, self._sample_size)

        return prob


class PoissonDistribution(DiscreteDistribution):
    def __init__(self, mean: float):
        """
        Initializes an object for handling a Poisson distribution.

        Args:
            mean (float): The expected value of the Poisson distribution.
        """
        if mean <= 0:
            raise ValueError('The mean of the Poisson distribution must be greater than 0.')
        if mean > 115:
            raise ValueError('The mean of the Poisson distribution must not be greater than 115. '
                             + 'Consider approximating this distribution using continuous.NormalDistribution.')
        super().__init__()
        self._mean = mean
        self._variance = mean

        self._low = 0
        self._high = math.inf

    def probability(self, x: int) -> float:
        if x < 0:
            return 0
        return (self._mean**x)*(math.exp(-self._mean)) / math.gamma(x+1)

    def display_graph(self, plot_title: str = None, low: int = None, high: int = None, save_fig: bool = False):
        mean = self.mean()
        stddev = self.standard_deviation()
        super().display_graph(plot_title, int(max(mean - 3 * stddev, 0)), int(mean + 3 * stddev), save_fig)


def choose(n: int, k: int) -> int:
    """
    Args:
        n (int): The total pool of things to choose from
        k (int): The number of things to choose out of n

    Returns:
        The number of combinations of items you can choose out of n
    """
    if k > n - k:
        k = n - k
    if k == 0:
        return 1
    if k == 1:
        return n
    if n < k or k < 0:
        return 0

    dp = [0] * (k + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            dp[j] = dp[j] + dp[j - 1]

    return dp[k]


def uniform_mean(low: int, high: int) -> float:
    """
    Allows for calculation of uniform distribution expected value without initializing a UniformDistribution object.

    Args:
        low (int): The lower bound of where the uniform distribution is defined (inclusive).
        high (int): The upper bound of where the uniform distribution is defined (inclusive).

    Returns:
        The expected value of a uniform distribution.
    """
    return (low + high) / 2


def uniform_variance(low: int, high: int) -> float:
    """
    Allows for calculation of uniform distribution variance without initializing a UniformDistribution object.

    Args:
        low (int): The lower bound of where the uniform distribution is defined (inclusive).
        high (int): The upper bound of where the uniform distribution is defined (inclusive).

    Returns:
        The variance of a uniform distribution.
    """
    n = (high - low) + 1
    return (n**2 - 1) / 12


def uniform_standard_deviation(low: int, high: int) -> float:
    """
    Allows for calculation of uniform distribution standard deviation without initializing a UniformDistribution object.

    Args:
        low (int): The lower bound of where the uniform distribution is defined (inclusive).
        high (int): The upper bound of where the uniform distribution is defined (inclusive).

    Returns:
        The value of 1 standard deviation of a uniform distribution.
    """
    return math.sqrt(uniform_variance(low, high))


def binomial_mean(n: int, p: float) -> float:
    """
    Allows for calculation of binomial distribution expected value without initializing a BinomialDistribution object.

    Args:
        n (int): The number of trials for the binomial distribution.
        p (float): The probability of success on any given trial.

    Returns:
        The expected value of a binomial distribution.
    """
    if not (0 <= p <= 1):
        raise ValueError('The p vale must be on the range [0, 1].')
    return n * p


def binomial_variance(n: int, p: float) -> float:
    """
    Allows for calculation of binomial distribution variance without initializing a BinomialDistribution object.

    Args:
        n (int): The number of trials for the binomial distribution.
        p (float): The probability of success on any given trial.

    Returns:
        The variance of a binomial distribution.
    """
    if not (0 <= p <= 1):
        raise ValueError('The p vale must be on the range [0, 1].')
    return n * p * (1 - p)


def binomial_standard_deviation(n: int, p: float) -> float:
    """
    Allows for calculation of binomial distribution standard deviation without initializing a BinomialDistribution
    object.

    Args:
        n (int): The number of trials for the binomial distribution.
        p (float): The probability of success on any given trial.

    Returns:
        The value of 1 standard deviation of a binomial distribution.
    """
    if not (0 <= p <= 1):
        raise ValueError('The p vale must be on the range [0, 1].')
    return math.sqrt(binomial_variance(n, p))


def geometric_mean(p: float) -> float:
    """
    Allows for calculation of geometric distribution expected value without initializing a GeometricDistribution
    object.

    Args:
        p (float): The probability of success on any given trial.

    Returns:
        The expected value of a geometric distribution.
    """
    if not (0 <= p <= 1):
        raise ValueError('The p vale must be on the range [0, 1].')
    return 1 / p


def geometric_variance(p: float) -> float:
    """
    Allows for calculation of geometric distribution variance without initializing a GeometricDistribution
    object.

    Args:
        p (float): The probability of success on any given trial.

    Returns:
        The variance of a geometric distribution.
    """
    if not (0 <= p <= 1):
        raise ValueError('The p vale must be on the range [0, 1].')
    return (1 - p) / (p ** 2)


def geometric_standard_deviation(p: float) -> float:
    """
    Allows for calculation of geometric distribution standard deviation without initializing a GeometricDistribution
    object.

    Args:
        p (float): The probability of success on any given trial.

    Returns:
        The value of 1 standard deviation of a geometric distribution.
    """
    return math.sqrt(geometric_variance(p))


def hypergeometric_mean(total_population: int, successes_in_population: int, sample_size: int):
    """
    Allows for calculation of the expected value of a hypergeometric distribution without initializing a
    HypergeometricDistribution object.

    Args:
        total_population (int): The total population size.
        successes_in_population (int): The number of successes in the population.
        sample_size (int): The sample size taken without replacement from the total population.

    Returns:
        The expected value of the Hypergeometric distribution.
    """
    if successes_in_population > total_population:
        raise ValueError('The number of successes in the population cannot exceed the size '
                         'of the total population.')
    if sample_size > total_population:
        raise ValueError('The sample size cannot exceed the total population.')
    mean = (successes_in_population * sample_size) / total_population
    return mean


def hypergeometric_variance(total_population: int, successes_in_population: int, sample_size: int):
    """
    Allows for calculation of the variance of a hypergeometric distribution without initializing a
    HypergeometricDistribution object.

    Args:
        total_population (int): The total population size.
        successes_in_population (int): The number of successes in the population.
        sample_size (int): The sample size taken without replacement from the total population.

    Returns:
        The variance of the Hypergeometric distribution.
    """
    if successes_in_population > total_population:
        raise ValueError('The number of successes in the population cannot exceed the size '
                         'of the total population.')
    if sample_size > total_population:
        raise ValueError('The sample size cannot exceed the total population.')
    variance = successes_in_population * (total_population - successes_in_population)
    variance *= sample_size * (total_population - sample_size)
    variance /= (total_population ** 2) * (total_population - 1)
    return variance


def hypergeometric_standard_deviation(total_population: int, successes_in_population: int, sample_size: int):
    """
    Allows for calculation of the standard deviation of a hypergeometric distribution without initializing a
    HypergeometricDistribution object.

    Args:
        total_population (int): The total population size.
        successes_in_population (int): The number of successes in the population.
        sample_size (int): The sample size taken without replacement from the total population.

    Returns:
        The value of 1 standard deviation of the Hypergeometric distribution.
    """
    if successes_in_population > total_population:
        raise ValueError('The number of successes in the population cannot exceed the size '
                         'of the total population.')
    if sample_size > total_population:
        raise ValueError('The sample size cannot exceed the total population.')
    return math.sqrt(hypergeometric_variance(total_population, successes_in_population, sample_size))


def poisson_standard_deviation(beta: float) -> float:
    """
    Allows for calculation of Poisson distribution standard deviation without initializing a PoissonDistribution
    object.

    Args:
        beta (float): The expected value of the Poisson distribution.

    Returns:
        The value of 1 standard deviation of the Poisson distribution.
    """
    if beta <= 0:
        raise ValueError('The beta value of the Poisson distribution must be greater than 0.')
    return math.sqrt(beta)


def convolve(dist1, dist2):
    """
    Computes the convolution of two discrete random variables.

    Args:
        dist1 (DiscreteDistribution): The first discrete distribution.
        dist2 (DiscreteDistribution): The second discrete distribution.

    Returns:
        CustomDistribution: The distribution that results from the convolution of two discrete random variables.
    """
    h1 = dist1.high()
    h2 = dist2.high()

    if math.isinf(h1):
        h1 = dist1.mean() + 3 * dist1.standard_deviation()

    if math.isinf(h2):
        h2 = dist2.mean() + 3 * dist2.standard_deviation()

    low1, high1 = int(dist1.low()), int(h1)
    low2, high2 = int(dist2.low()), int(h2)

    low = low1 + low2
    high = high1 + high2

    probabilities = [0.0 for _ in range(high - low + 1)]
    for i in range(low1, high1 + 1):
        for j in range(low2, high2 + 1):
            x_conv = i + j - low
            probabilities[x_conv] += dist1.probability(i) * dist2.probability(j)
    x_values = list(range(int(low), int(high) + 1))

    return CustomDistribution(x_values, probabilities)