__author__ = 'mengzhang'

from theano import tensor as T, function

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = function([x], s)
print(logistic([[0, 1], [-1, -2]]))

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = function([x], s2)
print(logistic2([[0, 1], [-1, -2]]))

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = function([a, b], [diff, abs_diff, diff_squared])
print(f([[1,1],[1,1]],[[2,2],[3,3]]))

from theano import Param

x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, Param(y, default=1)], z)
print(f(33))

x, y, w = T.dscalars('x', 'y', 'w')
f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
f(33)

from theano import shared
state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

