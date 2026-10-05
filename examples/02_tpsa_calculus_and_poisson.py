"""TPSA differentiation, integration, parameters, and Poisson brackets."""

from madng_tpsa import Descriptor

# Canonical coordinates are taken in pairs: (q, p).
descriptor = Descriptor(
    variables=['q', 'p'],
    order=5,
    params=['k'],
    param_order=2,
)

q, p = descriptor.vars()
k = descriptor.param('k')

hamiltonian = (p**2 + q**2) / 2 + k * q**3 / 3

print('Hamiltonian:')
print('H =', hamiltonian.format('code'))
print()

print('Parameter derivative:')
print('dH/dk =', hamiltonian.derivative('k').format('code'))
print()

print('First-order parameter coefficient at the expansion point:')
print('param_grad(H) =', hamiltonian.param_grad())

dq = hamiltonian.derivative('q')
dp = hamiltonian.derivative('p')

print('Partial derivatives:')
print('dH/dq =', dq.format('code'))
print('dH/dp =', dp.format('code'))
print()

print('Formal integration:')
print('integral(dH/dq, dq) =', dq.integrate('q').format('code'))
print()

# Poisson bracket in canonical coordinates:
# {f, g} = df/dq dg/dp - df/dp dg/dq
print('Poisson brackets:')
print('[q, p] =', q.poisson_bracket(p).format('code'))
print('[p, q] =', p.poisson_bracket(q).format('code'))
print('[H, H] =', hamiltonian.poisson_bracket(hamiltonian).format('code'))
print()

observable = q**2 + p**2
print('Observable:')
print('A =', observable.format('code'))
print('[A, H] =', observable.poisson_bracket(hamiltonian).format('code'))
