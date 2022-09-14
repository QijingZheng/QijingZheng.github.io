#!/usr/bin/env python

import sympy as sp
from sympy.matrices import Matrix

E1, E2, L = sp.symbols("E1 E2 L")
Hamil = Matrix([
    [E1, L],
    [L, E2],
])

sp.pprint(Hamil.eigenvects())

# U, T = Hamil.diagonalize(normalize=True)
# sp.pprint(U)
