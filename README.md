# JAX TOV Structure

This repository contains a JAX-based numerical solver for the Tolman–Oppenheimer–Volkoff (TOV) equations, which describe the internal structure of static, spherically symmetric neutron stars in general relativity. The solver supports spline-based equations of state, Runge–Kutta 4th-order integration, and automatic surface extrapolation using JAX and `lax.scan`.

---

## 📘 Features

* RK4 integrator for TOV and related ODE systems
* Natural cubic spline EOS interpolation (JAX differentiable)
* `lax.scan`-based integration with logical stopping condition
* Extrapolation of radius at surface pressure using quadratic fit
* JAX autodiff compatible for future gradient-based analysis

---

## 📁 Repository Structure

```
jax-tov-structure/
├── README.md
├── requirements.txt
├── src/
│   ├── spline.py             # Natural cubic spline function
│   ├── rk4.py                # RK4 integrator step
│   ├── tov_scan.py           # Integration with scan + extrapolation
│   ├── eos_loader.py         # (Optional) EOS loading and preprocessing
│   └── main.py               # Main example runner
├── figures/                  # Output figures (e.g., M-R plots)
├── data/                     # EOS tables (pressure-density)
```

---

## 🧠 Core Functions

### `natural_cubic_spline(x, y)`

Returns a JAX-compatible callable cubic spline interpolant for the EOS.

### `rk4_step(x, r, dr, f)`

Performs a single 4th-order Runge–Kutta integration step.

### `extrapolated_surface_step(P_surface, pressure, r0, x0, dr, f)`

Integrates outward in radius using `lax.scan` until pressure drops below `P_surface`. Then, it applies a quadratic fit to extrapolate the true surface radius and final state.

---

## ▶️ Running the Code

Install dependencies:

```bash
pip install -r requirements.txt
```

Then, run:

```bash
python src/main.py
```

---

## 📈 Output

* Final radius and mass at surface pressure
* Intermediate integration history (optional)
* Figures for $M(R)$, sensitivity, or EOS-dependent results

---

## 📝 License

MIT License

---

## 👤 Author

Isaac Dodds – 2025 BSc Physics Final Year Project

---

## 🔬 References

* Hartle, J. B. (1967). Slowly Rotating Relativistic Stars.
* Yagi & Yunes (2013). I-Love-Q Relations.
* CompOSE EOS database
* JAX: [https://github.com/google/jax](https://github.com/google/jax)
