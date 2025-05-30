def natural_cubic_spline(x, y):
n = x.shape[0] - 1
h = jnp.diff(x)
# === Compute alpha vector ===
y_im1 = y[:-2]
y_i = y[1:-1]
y_ip1 = y[2:]
h_im1 = h[:-1]
h_i = h[1:]
diffs = 3 * ((y_ip1 - y_i) / h_i - (y_i - y_im1) / h_im1)
alpha = jnp.zeros(n)
alpha = alpha.at[1:n].set(diffs)
# === Tridiagonal forward pass ===
l = jnp.ones(n + 1)
mu = jnp.zeros(n + 1)
z = jnp.zeros(n + 1)
def forward_body(i, state):
l, mu, z = state
li = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
mui = h[i] / li
zi = (alpha[i] - h[i-1] * z[i-1]) / li
l = l.at[i].set(li)
mu = mu.at[i].set(mui)
z = z.at[i].set(zi)
return l, mu, z
l, mu, z = lax.fori_loop(1, n, forward_body, (l, mu, z))
# === Back substitution to solve M ===
M = jnp.zeros(n + 1)
def backward_body(i, M):
j = n - 1 - i
val = z[j] - mu[j] * M[j + 1]
return M.at[j].set(val)
M = lax.fori_loop(0, n, backward_body, M)
# === Spline coefficients ===
a = jnp.array(y[:-1])
b = jnp.array((y[1:] - y[:-1]) / h - h * (2*M[:-1] + M[1:]) / 3)
c = jnp.array(M[:-1])
d = jnp.array((M[1:] - M[:-1]) / (3 * h))
# === Evaluate the spline ===
def spline_function(x_eval):
x_eval = jnp.atleast_1d(x_eval)
def single_spline_eval(xi):
idx = jnp.searchsorted(x, xi, side=’right’) - 1
idx = jnp.clip(idx, 0, n - 1)
dx = xi - x[idx]
spline_val = a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
linear_left = b[0] * xi + (a[0] - b[0] * x[0])
return jnp.where(xi < x[0], linear_left, spline_val)
result = jax.vmap(single_spline_eval)(x_eval)
return result if result.shape[0] > 1 else result[0]
return spline_function
