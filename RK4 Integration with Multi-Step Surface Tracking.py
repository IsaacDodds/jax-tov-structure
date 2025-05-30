def scan_body(carry, _):
r, x, done, third_r, third_x, second_r, second_x, last_r, last_x = carry
# Check pressure BEFORE stepping
P = pressure(x[1])
still_valid = jnp.logical_and(~done, jnp.isfinite(P) & (P > P_surface))
new_done = jnp.logical_or(done, ~still_valid)
# Shift: second → third, last → second, current → last
updated_third_r = jnp.where(still_valid, second_r, third_r)
updated_third_x = jnp.where(still_valid, second_x, third_x)
updated_second_r = jnp.where(still_valid, last_r, second_r)
updated_second_x = jnp.where(still_valid, last_x, second_x)
updated_last_r = jnp.where(still_valid, r, last_r)
updated_last_x = jnp.where(still_valid, x, last_x)
# Advance step
x_new = rk4_step(x, r, dr, f)
x_out = jnp.where(done, x, x_new)
r_new = r + dr
return (
r_new, x_out, new_done,
updated_third_r, updated_third_x,
updated_second_r, updated_second_x,
updated_last_r, updated_last_x
), None
init_carry = (r0, x0, False, r0, x0, r0, x0, r0, x0)
max_steps = 200000
(r_final, x_final, _,
r_second_last, x_second_last,
r_valid, x_valid,
r_last, x_last), _ = lax.scan(
scan_body, init_carry, xs=None, length=max_steps
)
