"""
Shared Gauss-Legendre nodes, computed once per order.

Both the Jeans projection (``sph_model``) and the generalised-Gaussian
likelihood (``vkurtosis``) evaluate a fixed-order Gauss-Legendre rule
inside a hot loop, and both used to rebuild the nodes on every call.

``scipy.special.roots_legendre`` rather than
``numpy.polynomial.legendre.leggauss``: the two agree to machine
precision at every order (checked from 5 to 1000 below), but leggauss
solves a companion-matrix eigenproblem through LAPACK, so it is ~8x
more expensive and routes a trivial 40x40 problem through the threaded
BLAS. roots_legendre uses Newton iteration instead.

The cache is what actually matters for speed: rebuilding the nodes was
75% of the cost of a projection.
"""

import numpy as np
from scipy.special import roots_legendre

_NODES = {}


def leg_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre nodes and weights on [-1, 1], cached per order.

    The arrays are shared between callers, so treat them as read-only.

    Args:
        n_nodes: Number of quadrature nodes.

    Returns:
        Tuple (nodes, weights), each of shape (n_nodes,).

    Example:
        >>> x, w = leg_nodes(40)
        >>> float(np.sum(w))
        2.0
    """
    if n_nodes not in _NODES:
        _NODES[n_nodes] = roots_legendre(n_nodes)
    return _NODES[n_nodes]


def _self_check():
    """Assert the cached nodes match numpy's to machine precision."""
    from numpy.polynomial.legendre import leggauss

    worst = 0.0
    for n in (5, 10, 20, 40, 54, 80, 160, 300, 1000):
        x_ref, w_ref = leggauss(n)
        x, w = leg_nodes(n)
        worst = max(worst, np.abs(x - x_ref).max(), np.abs(w - w_ref).max())
        assert np.isclose(w.sum(), 2.0, rtol=0, atol=1e-12), n
        # Reason: exactness on a degree 2n-1 polynomial is the property
        # the projection actually relies on; x^2 is the cheapest probe.
        assert np.isclose((w * x ** 2).sum(), 2.0 / 3.0, rtol=1e-13), n
    assert worst < 1e-12, worst
    assert leg_nodes(40) is leg_nodes(40), "cache returns a fresh object"
    print(f"quadrature self-check OK (worst vs leggauss: {worst:.2e})")


if __name__ == "__main__":
    _self_check()
