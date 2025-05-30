def rk4_step(x, r, dr, f):
k1 = dr * f(x, r)
k2 = dr * f(x + 0.5 * k1, r + 0.5 * dr)
k3 = dr * f(x + 0.5 * k2, r + 0.5 * dr)
k4 = dr * f(x + k3, r + dr)
return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
