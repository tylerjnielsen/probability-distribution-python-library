"""
Distributions Supported:
    UniformDistribution
    NormalDistribution
    StudentTDistribution
    GammaDistribution
    ExponentialDistribution
    ChiSquaredDistribution
    CustomDistribution - Define your own probability density function
"""

import math
import random
from typing import Callable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class ContinuousDistribution(ABC):
    """
    Abstract base class for handling continuous probability distributions. Only intended for use in inheritance, not
    as a standalone class. Implements default functionality for several methods.
    """
    def __init__(self, title: str = None):
        """
        Initializes an object to handle continuous distributions.

        Args:
            title (string): The name of the distribution. Default name is the distribution type.
                            Example of default - NormalDistribution
        """
        self._mean = None
        self._variance = None
        if title is not None:
            self._title = title
        else:
            self._title = self.__class__.__name__

    @abstractmethod
    def density_function(self, x: float):
        """
        The probability density function of the continuous random variable.

        Args:
            x (float): The value at which to evaluate the density function.

        Returns:
            The density function evaluated at x.
        """
        pass

    def title(self):
        """
        Returns:
            The title of the distribution.
        """
        return self._title

    def mean(self) -> float:
        """
        Returns:
             The expected value of the distribution.
        """
        return self._mean

    def standard_deviation(self) -> float:
        """
        Returns:
            The value of 1 standard deviation of the distribution.
        """
        return math.sqrt(self.variance())

    def variance(self):
        """
        Returns:
            The variance of the distribution.
        """
        return self._variance

    def cdf(self, x: float) -> float:
        """
        Uses methods of integral approximation to calculate the cumulative density function (CDF) at a given value.

        Args:
            x (float): The value at which to evaluate the CDF.

        Returns:
            An approximation of the probability of observing a value less than or equal to the x value inputted.
        """
        return cdf(self.density_function, x)

    def probability_of_range(self, low: float, high: float):
        """
        Args:
            low (float): The upper bound of the range.
            high (float): The lower bound of the range.

        Returns:
            The probability of observing a value within the range given.
        """
        return self.cdf(high) - self.cdf(low)

    def probability_greater_than(self, low: float) -> float:
        """
        Args:
            low (float): The lower bound of the range.

        Returns:
            The probability of observing a value greater than the lower bound given.
        """
        return 1 - self.cdf(low)

    def probability_less_than(self, high: float) -> float:
        """
        Args:
            high (float): The upper bound of the range.

        Returns:
            The probability of observing a value less than the upper bound given.
        """
        return self.cdf(high)

    def approximate_discrete_range(self, low: int, high: int) -> float:
        """
        Can be used to approximate discrete random variables that share the same general shape, mean, and
        standard deviation as the continuous distribution. Particularly useful with normal distributions to approximate
        discrete random variables, such as a binomial distribution with hundreds of trials.

        Args:
            low (float): The lower bound of the range (inclusive).
            high (float): The upper bound of the range (inclusive).

        Returns:
            The probability of observing a value less than the upper bound given.
        """
        result = 0
        for x in range(low, high + 1, 1):
            result += self.density_function(x)
        return result

    def display_graph(self, plot_title: str = None, low: float = None, high: float = None):
        """
        Displays a graph of the probability density function using Matplotlib.

        Args:
            plot_title (string): The title of the plot.
            low (float): The lower bound of where to display the graph.
            high (float): The upper bound of where to display the graph.
        """
        if plot_title is None:
            plot_title = self.title()

        num_points = 1000
        x = []
        y = []

        if low is None:
            low = self._mean - 3 * self.standard_deviation()
        if high is None:
            high = self._mean + 3 * self.standard_deviation()

        step = (high - low) / (num_points - 1)

        current_x = low
        for _ in range(num_points):
            x.append(current_x)
            y.append(self.density_function(current_x))
            current_x += step

        plt.plot(x, y)
        plt.title(plot_title)
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.grid(True)
        plt.show()


class CustomDistribution(ContinuousDistribution):
    def __init__(self, function: Callable[[float], float], lower_bound: float, upper_bound: float, title: str = None,
                 simpsons_rule: bool = True):
        """
        Initializes an object to handle custom distributions. Only use this if absolutely necessary, because this is
        much less optimized than the built-in distributions because the CDF of most of the built-in distributions has
        been pre-computed.

        The area under the density function along its range MUST evaluate to 1, or an error will be raised.
        The density function must never be negative, or an error will be raised.

        Args:
            function (function): The probability density function for the custom distribution.
            lower_bound (float): The lower bound of where the density function is defined/nonzero.
            upper_bound (float): The upper bound of where the density function is defined/nonzero.
            title (string): The name of the distribution. Default is CustomDistribution.
            simpsons_rule (bool): A bool to determine whether to use Simpson's rule or the Trapezoidal rule of
                                  integral approximation for this random variable. Default is Simpson's rule.
        """
        super().__init__(title)
        if lower_bound > upper_bound:
            raise ValueError('The lower bound must be less than the upper bound.')

        if any(function(x) < 0 for x in frange(lower_bound, upper_bound, 0.001)):
            raise ValueError('This is not a density function. The function must never go below 0 on its bounds.')

        function_area = trapezoidal_rule_integration(function, lower_bound, upper_bound)
        if function_area - 0.02 > 1 or function_area + 0.02 < 1:
            raise ValueError('This is not a density function. The area under the curve within the bounds is not 1.')

        self._function = function
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._simpsons = simpsons_rule

        # Calculate the mean
        mean_integrand = lambda x: x * self._function(x)
        self._mean = simpsons_rule_integration(mean_integrand, lower_bound, upper_bound)

        # Calculate the variance
        variance_integrand = lambda x: x**2 * self._function(x)
        self._variance = simpsons_rule_integration(variance_integrand, lower_bound, upper_bound)
        self._variance -= self._mean**2

    @property
    def lower_bound(self):
        """
        Returns:
             The lower bound of where the distribution is defined/nonzero.
        """
        return self._lower_bound

    @property
    def upper_bound(self):
        """
        Returns:
            The upper bound of where the distribution is defined/nonzero.
        """
        return self._upper_bound

    def density_function(self, x: float) -> float:
        if not (self._lower_bound <= x <= self._upper_bound):
            return 0
        return self._function(x)

    def cdf(self, x: float) -> float:
        if not self._simpsons:
            return trapezoidal_rule_integration(self.density_function, self._lower_bound, x)
        else:
            return simpsons_rule_integration(self.density_function, self._lower_bound, x)

    def display_graph(self, plot_title: str = "Probability Density Function", low: float = None, high: float = None):
        if low is None:
            low = self._lower_bound
        if high is None:
            high = self._upper_bound

        return super().display_graph(plot_title, low + 0.0001, high - 0.0001)


class UniformDistribution(ContinuousDistribution):
    def __init__(self, low: float, high: float, title: str = None):
        """
        Initializes an object for handling a continuous uniform distribution. That is, a distribution where the
        probability of any event is equal within the boundary it is defined along.

        Args:
            low (float): The lower bound of where the distribution is defined/nonzero.
            high (float): The upper bound of where the distribution is defined/nonzero.
            title (string): The name of the distribution. Default is UniformDistribution.
        """
        super().__init__(title)
        if high < low:
            raise ValueError('The lower bound must be less than the upper bound.')

        self._low = low
        self._high = high

        self._mean = (low + high) / 2
        self._variance = (high - low)**2 / 12

    def density_function(self, x: float) -> float:
        if not (self._low <= x <= self._high):
            return 0
        return 1 / (self._high - self._low)

    def cdf(self, x: float) -> float:
        """
        Returns the exact probability, not an approximation.

        Args:
            x (float): The value at which to evaluate the CDF (Cumulative Density Function).

        Returns:
            The probability of observing a value less than the given x value.
        """
        return (x - self._low) / (self._high - self._low)

    def random_number(self):
        """
        Returns:
            A random number with uniform probability for any value within the range of the uniform distribution.
        """
        return random.uniform(self._low, self._high)

    def display_graph(self, plot_title: str = None, low: float = None, high: float = None):
        if low is None:
            low = self._low
        if high is None:
            high = self._high
        return super().display_graph(plot_title, low + 0.0001, high - 0.0001)


class NormalDistribution(ContinuousDistribution):
    def __init__(self, mean: float, sigma: float, title: str = None):
        """
        Initializes an object to handle a normal distribution.

        Args:
            mean: The expected value of the normal distribution.
            sigma: The value of 1 standard deviation of the normal distribution.
        """
        super().__init__(title)
        if sigma <= 0:
            raise ValueError('Standard deviation must be a positive real number.')
        self._mean = mean
        self._standard_deviation = sigma
        self._variance = sigma ** 2

    def density_function(self, x: float) -> float:
        mu = self._mean
        sigma = self._standard_deviation
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def cdf(self, x: float) -> float:
        """
        Returns the exact probability, not an approximation.

        Args:
            x (float): The value at which to evaluate the CDF (Cumulative Density Function).

        Returns:
            The probability of observing a value less than the given x value.
        """
        return (1 / 2) * (1 + math.erf((x - self._mean) / (self._standard_deviation * math.sqrt(2))))

    def random_number(self) -> float:
        """
        Returns:
             A random number with the probability of any given number being determined by the normal distribution.
        """
        while True:
            u1, u2 = random.random(), random.random()
            z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            x = self._mean + z1 * self._standard_deviation
            if -math.inf <= x <= math.inf:
                return x


class StudentTDistribution(ContinuousDistribution):
    def __init__(self, df: float, title: str = None):
        """
        Initializes an object to handle the student-t distribution.

        Args:
            df (float): Number of degrees of freedom of the student-t distribution.
            title (string): The title of the distribution.
        """
        super().__init__(title)
        self._degrees_of_freedom = df
        self._mean = 0
        if df > 2:
            self._variance = df / (df - 2)
        else:
            self._variance = 1  # Technically wrong, but allows for default plotting functionality for df <= 2

    def density_function(self, x: float):
        df = self._degrees_of_freedom
        numerator = math.gamma((df + 1) / 2)
        denominator = math.sqrt(df * math.pi) * math.gamma(df / 2)
        coefficient = numerator / denominator
        exponent = -0.5 * (df + 1) * math.log(1 + (x ** 2) / df)
        return coefficient * math.exp(exponent)

    def cdf(self, x: float):
        # Uses -500 as lower bound because student-t distribution is negligible for x < -500.
        # Because of the increased range, the number of intervals is increased from the default of 1000 to 5000
        # to help improve accuracy
        return simpsons_rule_integration(self.density_function, -500, x, n=5000)


class GammaDistribution(ContinuousDistribution):
    def __init__(self, alpha: float, beta: float, title: str = None):
        """
        Initializes an object to handle a gamma distribution.

        Args:
            alpha (float): Shape parameter of the gamma distribution. Must be greater than 0.
            beta (float): Rate parameter of the gamma distribution. Must be greater than 0.
            title (string): Title for the distribution plot. Defaults to GammaDistribution.
        """
        super().__init__(title)
        if alpha <= 0 or beta <= 0:
            raise ValueError('alpha and beta values must be greater than 0.')
        self._alpha = alpha
        self._beta = beta

        self._mean = alpha / beta
        self._variance = alpha / beta**2

    def density_function(self, x: float) -> float:
        if x < 0:
            return 0
        alpha = self._alpha
        beta = self._beta
        return (x**(alpha - 1) * math.exp(-x / beta)) / (beta**alpha * math.gamma(alpha))

    def display_graph(self, plot_title: str = None, low: float = 0, high: float = None):
        super().display_graph(plot_title, low, high)


class ExponentialDistribution(GammaDistribution):
    def __init__(self, mean: float, title: str = None):
        """
        Initializes an object to handle an exponential distribution

        Args:
            mean (float): The expected value of the exponential distribution.
            title (string): The title of the distribution.
        """
        super().__init__(1, 1 / mean, title)

    def cdf(self, x: float):
        """
        Returns the exact probability, not an approximation.

        Args:
            x (float): The value at which to evaluate the CDF (Cumulative Density Function).

        Returns:
            The probability of observing a value less than the given x value.
        """
        if x < 0:
            return 0
        return 1 - math.exp(-x / self._mean)


class ChiSquaredDistribution(GammaDistribution):
    def __init__(self, degrees_of_freedom: float, title: str = None):
        """
        Initializes an object to handle a chi-squared distribution

        Args:
            degrees_of_freedom (float): The degrees of freedom for the chi-squared distribution.
            title (string): The title of the distribution.
        """
        super().__init__(degrees_of_freedom / 2, 2, title)


def trapezoidal_rule_integration(function: Callable[[float], float], a, b, n: int = 1000):
    """
    As a general rule, Trapezoidal Rule approximations are better when your function is linear or constant.
    If your function is curvy, such as sin(x), Simpson's rule is preferable.

    Args:
        function (function): The function to integrate, that is, the integrand.
        a: The lower limit of integration.
        b: The upper limit of integration.
        n (int): The number of intervals to use in the trapezoidal rule approximation. Default is 1000.

    Returns: An approximation of the definite integral of the function from a to b.
    """
    if a > b:
        raise ValueError('The lower bound must be less than or equal to the upper bound.')

    delta_x = (b - a) / n
    result = 0.5 * (function(a) + function(b))
    for i in range(1, n):
        result += function(a + i * delta_x)
    result *= delta_x
    return result


def simpsons_rule_integration(function: Callable[[float], float], a, b, n: int = 1000):
    """
    As a general rule, Simpson's Rule approximations are better when your function is curvy.
    If you have a linear or constant function, the trapezoidal rule is preferable.

    Args:
        function (function): The function to integrate, that is, the integrand.
        a: The lower limit of integration.
        b: The upper limit of integration.
        n (int): The number of intervals to use in the simpson's rule approximation. Default is 1000.

    Returns: An approximation of the definite integral of the function from a to b.
    """
    if a > b:
        raise ValueError('The lower bound must be less than or equal to the upper bound.')
    if n % 2 != 0:
        raise ValueError('Simpson\'s rule requires an even number of sub-intervals')

    delta_x = (b - a) / n
    result = (delta_x / 3) * function(a)
    four = True
    for i in range(1, n):
        if four:
            result += (delta_x / 3) * 4 * function(a + i * delta_x)
        else:
            result += (delta_x / 3) * 2 * function(a + i * delta_x)
        four = not four
    result += (delta_x / 3) * function(b)
    return result


def derivative_at_x(function: Callable[[float], float], x: float) -> float:
    """
    Args:
        function (function): The function to find the derivative at a point of.
        x (float): The point to find the derivative at.

    Returns:
        An approximation of the derivative of a function at x.
    """
    return (function(x + 0.001) - function(x)) / 0.001


def cdf(function: Callable[[float], float], x: float, n: int = 1000):
    """
    Uses methods of integral approximation to approximate the CDF of a density function at an upper limit.

    Your density function must not be defined for negative numbers.

    Args:
        function (function): The function whose CDF is to be calculated.
        x (float): The upper limit of integration for calculating the CDF.
        n (int, optional): The number of intervals to use in the numerical integration. Default is 1000.

    Returns:
        float: The approximate value of the CDF at the given upper limit.
    """
    return simpsons_rule_integration(function, x, n)


def frange(x, y, jump):
    """
    A range function designed to handle floating point numbers.

    Args:
        x: The lower bound of the range.
        y: The upper bound of the range.
        jump: The increment between consecutive values in the sequence.

    Yields:
        The next consecutive value in the sequence of the range.
    """
    while x < y:
        yield x
        x += jump


def plot_several_distributions(*distributions: 'ContinuousDistribution'):
    """
    Allows for the plotting of several distributions on the same graph using matplotlib.

    Args:
        *distributions (ContinuousDistribution): One or more of the distributions to be plotted.
    """
    for dist in distributions:
        x = []
        y = []
        min_x = dist.mean() - 3 * dist.standard_deviation()
        max_x = dist.mean() + 3 * dist.standard_deviation()
        num_points = 1000
        step = (max_x - min_x) / (num_points - 1)

        current_x = min_x
        for _ in range(num_points):
            x.append(current_x)
            y.append(dist.density_function(current_x))
            current_x += step

        plt.plot(x, y, label=dist.title())

    plt.title('Probability Density Functions')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()
