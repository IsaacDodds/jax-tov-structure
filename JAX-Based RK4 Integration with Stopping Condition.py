P_surface = jnp.min(P_table) * G / c**4
def scan_body(carry, _):
r, x, done, last_valid_r, last_valid_x = carry
m, rho, w, dw_dr, v = x
P = pressure(rho)
still_valid = jnp.logical_and(~done, jnp.isfinite(P) & (P > P_surface))
new_done = jnp.logical_or(done, ~still_valid)
last_valid_r = jnp.where(still_valid, r, last_valid_r)
last_valid_x = jnp.where(still_valid, x, last_valid_x)
x_new = rk4_step(x, r, dr, f)
x_out = jnp.where(done, x, x_new) # freeze if done
r_new = r + dr
return (r_new, x_out, new_done, last_valid_r, last_valid_x), (r_new, x_out)
init_carry = (r0, x0, False, r0, x0)
max_steps = 50000
(r_final, x_final, _, r_valid, x_valid), history = \
lax.scan(scan_body, init_carry, xs=None, length=max_steps)
