from math import floor
import numpy
from sympy import symbols, Eq, solve, log, floor


def partial(element, function):
    """
    partial : sympy.core.symbol.Symbol * sympy.core.add.Add -> sympy.core.add.Add
    partial(element, function) Performs partial derivative of a function of several variables is its derivative with respect to one of those variables, with the others held constant. Return partial_diff.
    """
    partial_diff = function.diff(element)

    return partial_diff


def gradient(partials):
    """
    gradient : List[sympy.core.add.Add] -> numpy.matrix
    gradient(partials) Transforms a list of sympy objects into a numpy matrix. Return grad.
    """
    grad = numpy.matrix([[partials[0]], [partials[1]]])

    return grad


def gradient_to_zero(symbols_list, partials):
    """
    gradient_to_zero : List[sympy.core.symbol.Symbol] * List[sympy.core.add.Add] -> Dict[sympy.core.numbers.Float]
    gradient_to_zero(symbols_list, partials) Solve the null equation for each variable, and determine the pair of coordinates of the singular point. Return singular.
    """
    partial_x = Eq(partials[0], 0)
    partial_y = Eq(partials[1], 0)

    singular = solve((partial_x, partial_y),
                     (symbols_list[0], symbols_list[1]))

    return singular


def hessian(partials_second, cross_derivatives):
    """
    hessian : List[sympy.core.add.Add] * sympy.core.add.Add -> numpy.matrix
    hessian(partials_second, cross_derivatives) Transforms a list of sympy objects into a numpy hessian matrix. Return hessianmat.
    """
    hessianmat = numpy.matrix([[partials_second[0], cross_derivatives], [
                              cross_derivatives, partials_second[1]]])

    return hessianmat


def main(current, compensated, x, y):
    """
    Fonction principale.
    """
    a0, a1, a2, a3, a4, a5, a6, a7 = symbols('a0 a1 a2 a3 a4 a5 a6 a7')
    symbols_list = [a0, a1, a2, a3, a4, a5, a6, a7]
    a = floor((a0+a2*x+a3*y)/(a6*x+a7*y+1))
    function = compensated[floor((a0+a2*x+a3*y)/(a6*x+a7*y+1))
                           ][floor((a1+a4*x+a5*y)/(a6*x+a7*y+1))] - current[x][y]
    partials, partials_second = [], []

    for element in symbols_list:
        partial_diff = partial(element, function)
        partials.append(partial_diff)

    grad = gradient(partials)
    singular = gradient_to_zero(symbols_list, partials)

    cross_derivatives = partial(symbols_list[0], partials[1])

    for i in range(0, len(symbols_list)):
        partial_diff = partial(symbols_list[i], partials[i])
        partials_second.append(partial_diff)

    hessianmat = hessian(partials_second, cross_derivatives)
    # det = determat(partials_second, cross_derivatives, singular, symbols_list)

    print("Hessian matrix that organizes all the second partial derivatives of the function {0} is :\n {1}".format(
        function, hessianmat))
    return hessianmat

#! does not work
"""
	This does not work because we are not able to use the values of the matrix as a function of the parameters. Ora at least not in the way the framework would like us to do. 
"""