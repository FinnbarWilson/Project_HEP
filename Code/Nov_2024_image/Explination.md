### **1. Gyroradius Calculation**

```
R = (pt / (0.3 * B_FIELD)) * 1000.0
```

$$R [\text{mm}] = \frac{p_T}{0.3 \cdot B} \times 1000$$

- **Physics:** Derived from the balance of centripetal and Lorentz forces ($qvB = \frac{mv^2}{R}$).

- **Constant:** The factor **0.3** is an approximation of the speed of light ($c$) to convert units.

- Derivation:

	$$p_{\text{SI}} = q_{\text{SI}} B_{\text{SI}} R_{\text{SI}}$$

	$$p_{\text{GeV/c}} = \frac{p_{\text{SI}} \cdot c}{e \cdot 10^9} = \frac{e B R \cdot c}{e \cdot 10^9} = \frac{c}{10^9} B R \approx \mathbf{0.3} B R$$

- **Scaling:** Multiplied by **1000** to convert meters to millimeters (standard detector unit).

- `B_FIELD = 3.8` I belive?

### **2. Helix Pitch (Pseudorapidity)**

```
slope_z = np.sinh(eta)
```

$$\frac{dz}{ds} = \sinh(\eta)$$

- **Physics:** Defines the "steepness" of the helix along the beamline.

- Derivation:

	Using the definition of pseudorapidity $\eta = -\ln[\tan(\theta/2)]$, we use the identity $\cot(\theta) = \sinh(\eta)$.

	$$p_z = p_T \cot(\theta) = p_T \sinh(\eta)$$

	Since velocity is proportional to momentum:

	$$\frac{v_z}{v_T} = \frac{dz}{ds} = \sinh(\eta)$$

### **3. Deflection Angle**

```
alpha = s * curvature * curvature_sign
```

$$\alpha(s) = \pm \frac{s}{R}$$

- **Geometry:** The angle turned is simply the arc length divided by the radius ($\theta = s/r$).
- **Charge:** `curvature_sign` ($q/|q|$) determines the chirality (clockwise vs. counter-clockwise) of the helix.

### **4. Transverse Position (X, Y)**

```
dx = R * (np.sin(phi + alpha) - np.sin(phi))
dy = R * (np.cos(phi) - np.cos(phi + alpha))
```

$$x(s) = x_0 + R[\sin(\phi + \alpha) - \sin(\phi)]$$

$$y(s) = y_0 - R[\cos(\phi + \alpha) - \cos(\phi)]$$

- **Geometry:** Parametric equations for a circle offset so the track starts at $(0,0)$ with initial tangent angle $\phi$.

- Derivation:

	Integrate the velocity vector components $\vec{v} = (v_x, v_y) \propto (\cos(\phi+\alpha), \sin(\phi+\alpha))$.

	$$x(s) = \int_0^s \cos(\phi + s'/R) \, ds' = R_0^s$$

	$$x(s) = R(\sin(\phi + \alpha) - \sin(\phi))$$

	(Note: The dy term in the code effectively integrates $\sin$, resulting in $-\cos$).

### **5. Longitudinal Position (Z)**

```
z = vz + (s * slope_z)
```

$$z(s) = z_0 + s \cdot \sinh(\eta)$$

- **Physics:** Motion in the $z$-direction is uniform (drift) because the Lorentz force component $F_z = q(\vec{v} \times \vec{B})_z$ is zero when $\vec{B}$ is aligned with $z$.

### **6. Boundary Checks**

```
if current_r >= MAX_R: break
if abs(z) >= MAX_Z: break
```

$$r > 1200 \text{ mm} \quad \lor \quad |z| > 3000 \text{ mm}$$

- **Hardware:** Represents the physical acceptance of the **CMS Tracker**. Particles exiting this volume enter the calorimeter and are no longer tracked.4