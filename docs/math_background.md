# Mathematical Background

## 1. First Variation of Area and the Minimal Surface Condition

Let $\mathbf{x}: \Omega \to \mathbb{R}^3$ be a smooth immersion with fixed boundary $\partial\Omega$.
The total area is
$$A[\mathbf{x}] = \int_\Omega \left|\frac{\partial \mathbf{x}}{\partial u} \times \frac{\partial \mathbf{x}}{\partial v}\right| \, du\, dv.$$

A **normal perturbation** $\mathbf{x}_\varepsilon = \mathbf{x} + \varepsilon h \hat{n}$ (with $h|_{\partial\Omega} = 0$) gives the first variation
$$\frac{d}{d\varepsilon}\bigg|_{\varepsilon=0} A[\mathbf{x}_\varepsilon] = -2\int_\Omega h \, H \, dA,$$
where $H = \tfrac{1}{2}(\kappa_1 + \kappa_2)$ is the **mean curvature** (half the sum of principal curvatures).

**Minimal surface** $\Leftrightarrow$ $H \equiv 0$ $\Leftrightarrow$ the surface is a critical point of area with fixed boundary.

---

## 2. The Cotangent Laplacian as Area Gradient

For a triangle mesh $(V, F)$, let $w_{ij} = \tfrac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ be the cotangent weight of edge $(i,j)$, where $\alpha_{ij}, \beta_{ij}$ are the angles opposite that edge in the two incident triangles.

The **gradient of total area** with respect to vertex $i$ is
$$\nabla_{\mathbf{x}_i} A = \frac{1}{2}\sum_{j \in N(i)} (\cot\alpha_{ij} + \cot\beta_{ij})(\mathbf{x}_i - \mathbf{x}_j) = (\mathbf{L}\mathbf{x})_i,$$
where $\mathbf{L}$ is the **cotangent Laplacian matrix**:
$$L_{ij} = \begin{cases} -w_{ij} & j \in N(i) \\ \sum_{k \in N(i)} w_{ik} & j = i \\ 0 & \text{otherwise.}\end{cases}$$

The **discrete mean-curvature vector** at vertex $i$ is
$$\mathbf{H}_i \hat{n}_i = \frac{(\mathbf{L}\mathbf{x})_i}{2 A_i},$$
where $A_i$ is the **Voronoi area** (with obtuse-triangle safeguard; Meyer et al. 2003).

Minimal surface $\Leftrightarrow$ $\mathbf{L}\mathbf{x} = 0$ (coordinate functions are discrete-harmonic).

---

## 3. Explicit and Implicit Mean-Curvature Flow

**Explicit (forward Euler) step:**
$$\mathbf{x}_i^{n+1} = \mathbf{x}_i^n - \tau (\mathbf{L}\mathbf{x}^n)_i / (2 A_i), \quad i \notin B.$$
Stability requires $\tau < h^2 / 4$ where $h$ is the minimum edge length.

**Implicit (semi-implicit) step** (Desbrun et al. 1999):
$$(\mathbf{M} + \tau \mathbf{L})\,\mathbf{x}^{n+1} = \mathbf{M}\,\mathbf{x}^n, \quad \text{boundary rows pinned,}$$
where $\mathbf{M} = \text{diag}(A_i)$ is the lumped mass matrix.  
The system $(\mathbf{M} + \tau\mathbf{L})$ is symmetric positive definite for all $\tau > 0$ (since $\mathbf{L}$ is PSD), so CG converges in every step. This is the default method.

At convergence $\mathbf{x}^* = \lim \mathbf{x}^n$, the system gives $\tau\,\mathbf{L}\,\mathbf{x}^* = 0 \Rightarrow \mathbf{L}\mathbf{x}^* = 0$ — the discrete harmonic condition. ✓

---

## 4. Discrete Gaussian Curvature (Angle Defect)

The discrete Gaussian curvature at an interior vertex $i$ is the **angle defect**:
$$K_i = \frac{2\pi - \sum_{t \ni i} \theta_{i,t}}{A_i},$$
where $\theta_{i,t}$ is the interior angle of triangle $t$ at vertex $i$.  
**Gauss–Bonnet** in the discrete setting:
$$\sum_{i \in \text{int}} K_i A_i + \text{boundary terms} = 2\pi\chi,$$
where $\chi = V - E + F$ is the Euler characteristic.

---

## 5. The Catenoid: Exact Computations

The **catenoid** with neck parameter $a > 0$ is
$$\mathbf{X}(\theta, z) = \bigl(a\cosh(z/a)\cos\theta,\; a\cosh(z/a)\sin\theta,\; z\bigr).$$

**First fundamental form** (isothermal coordinates):
$$E = G = \cosh^2(z/a), \quad F = 0.$$

**Second fundamental form:**
$$e = -1, \quad f = 0, \quad g = 1.$$

**Mean curvature:**
$$H = \frac{1}{2}\!\left(\frac{e}{E} + \frac{g}{G}\right) = \frac{1}{2}\!\left(\frac{-1}{\cosh^2} + \frac{1}{\cosh^2}\right) = 0. \checkmark$$

**Gaussian curvature:** $K = -1/\cosh^4(z/a)$.

**Boundary condition:** two coaxial rings of radius $R$ at $z = \pm h$ require
$$R = a \cosh(h/a).$$
This has two solutions for $R > h\cosh(1) \approx 1.543\,h$: a stable larger-$a$ root and an unstable smaller-$a$ root. The `catenoid_for_rings` function returns the stable root.

---

## 6. Weierstrass–Enneper Representation

Every minimal surface locally admits a **Weierstrass–Enneper** (W-E) parametrization:
$$\mathbf{X}(\zeta) = \operatorname{Re} \int_{\zeta_0}^{\zeta} \bigl(\tfrac{1}{2}f(1-g^2),\; \tfrac{i}{2}f(1+g^2),\; fg\bigr)\,d\zeta,$$
where $f, g$ are holomorphic with $fg^2$ holomorphic and nonzero.

**Standard surfaces** ($a=1$, with appropriate integration constants):

| Surface | $f(\zeta)$ | $g(\zeta)$ | Notes |
|---------|-----------|-----------|-------|
| Catenoid | $-e^{-\zeta}$ | $e^{\zeta}$ | $\Phi = (\sinh\zeta, -i\cosh\zeta, -1)$ |
| Enneper | $1$ | $\zeta$ | Self-intersecting for $|\zeta| > 1$ |
| Helicoid | $ie^{-\zeta}$ | $e^{\zeta}$ | Ruled; associate partner of catenoid |

**Associate family:** $f \mapsto e^{it} f$ (phase rotation), $g$ fixed. The map
$$t \in [0, \pi/2]: \quad \text{catenoid} \longrightarrow \text{helicoid}$$
is an **isometric deformation**: the Riemannian metric $E = G = \cosh^2 u$, $F = 0$ is preserved for all $t$. In coordinates $(u, v)$ with $\zeta = u + iv$:
$$\mathbf{X}_t(u,v) = a\bigl(\cosh u\cos v\cos t - \sinh u\sin v\sin t,\;\cosh u\sin v\cos t + \sinh u\cos v\sin t,\;{-u\cos t + v\sin t}\bigr).$$

---

## 7. Stability: The 1.3255 Threshold

A catenoid spanning two equal coaxial rings (radius $R$, separation $2h$) exists only while the dimensionless ratio satisfies
$$\frac{2h}{R} \leq \frac{2\,\operatorname{arccosh}(x_c)}{x_c} \approx 1.3255,$$
where $x_c \approx 1.5088$ is the unique positive root of $\tanh(x) = 1/x$.

Beyond this threshold, **no connected minimal surface spanning the two rings exists**. The area-minimising configuration degenerates to the **Goldschmidt solution**: two flat disks (one per ring) plus a degenerate line segment of zero area.

The **stability Jacobi operator** $\mathcal{J} = -\Delta_S - |A|^2$ (where $|A|^2 = \kappa_1^2 + \kappa_2^2 = -2K$ for minimal surfaces) determines the Morse index of the surface. A negative smallest eigenvalue of $\mathcal{J}$ indicates an unstable critical point of area.

---

## References

- Meyer, M., Desbrun, M., Schröder, P., Barr, A. (2003). *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds.* Visualization and Mathematics III.
- Desbrun, M., Meyer, M., Schröder, P., Barr, A. (1999). *Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow.* SIGGRAPH.
- Pinkall, U., Polthier, K. (1993). *Computing Discrete Minimal Surfaces and Their Conjugates.* Experimental Mathematics.
- Osserman, R. (1969). *A Survey of Minimal Surfaces.* Dover.
- Isenberg, C. (1978). *The Science of Soap Films and Soap Bubbles.* Dover.
