# JAX TOV structure

Numerical building blocks from a BSc final-year project (Mathematics and Physics, University of Bath, 2025) on integrating the Tolman-Oppenheimer-Volkoff equations for slowly rotating neutron stars in JAX: a differentiable cubic-spline equation of state, a fourth-order Runge-Kutta step, `lax.scan` integration that stops at the stellar surface, and a quadratic extrapolation of the radius to the surface pressure.

## Files

| file | what it is |
|---|---|
| `spline.py` | natural cubic spline interpolant for the equation of state, built with `lax.fori_loop` so it can be jitted and differentiated |
| `rk4.py` | one fourth-order Runge-Kutta step |
| `tov_scan.py` | outward integration with `lax.scan`; the state carries mass, density, the frame-dragging function and its derivative, and the metric potential, and freezes once the pressure drops below the surface value |
| `tov_scan_tracked.py` | the same integration keeping the last three valid steps |
| `surface_extrapolation.py` | quadratic fit through those three steps to place the surface at the table's minimum pressure, then one final RK4 step to it |

These are the pieces as they were used in the project, not a packaged solver. The TOV right-hand side `f`, the `pressure(rho)` closure from the spline, the equation-of-state table and the starting state are set up in the project notebook and are not included here.

## Requirements

```
pip install jax jaxlib
```

## References

- Hartle, J. B. (1967), Slowly rotating relativistic stars
- Yagi, K. and Yunes, N. (2013), I-Love-Q relations
- CompOSE equation-of-state database
- JAX: https://github.com/google/jax

## Licence

MIT. See `LICENSE`.
