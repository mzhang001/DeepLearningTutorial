__author__ = 'mengzhang'

import theano

a = theano.tensor.vector()
b = theano.tensor.vector()
out = a ** 2 + b ** 2 + 2 * a * b
f = theano.function([a, b], out)
print f([0, 1, 2],[2, 4 ,6])