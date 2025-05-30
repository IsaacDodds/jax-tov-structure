P_final = pressure(x_valid[1])
P2_final = pressure(x_second_last[1])
P_trial = pressure(x_last[1])
P_vals = jnp.array([P2_final, P_final, P_trial]) # shape (3,)
r_vals = jnp.array([r_second_last, r_valid, r_last]) # shape (3,)
P0 = P_surface
p1, p2, p3 = P_vals
r1, r2, r3 = r_vals
# Compute numerator and denominator for quadratic fit
numerator = (
p1**2 * r2 - p1**2 * r3
- p2**2 * r1 + p2**2 * r3
+ p3**2 * r1 - p3**2 * r2
)
denominator = (
p1**2 * p2 - p1**2 * p3
- p1 * p2**2 + p1 * p3**2
+ p2**2 * p3 - p2 * p3**2
)
# Solve for quadratic coefficients
b = numerator / denominator
a = (-b * (p1 - p2) + r1 - r2) / (p1**2 - p2**2)
d = -a * p1**2 - b * p1 + r1
# Compute extrapolated radius at surface pressure
r1 = a * P0**2 + b * P0 + d
# Perform final RK4 step to update state at r1
dr1 = r1 - r3
X = rk4_step(x_last, r3, dr1, f)
# Extract final mass
r = r1
m = X[0]
