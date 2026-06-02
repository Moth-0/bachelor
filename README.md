# Deuterium Ground State Energy via Pion-Exchange

Simulating relativistic effects of exchanged pions in light nuclei. This project finds the ground state energy of a deuteron (proton-neutron bound pair) using quantum mechanics with pion exchange forces.

## Overview

The nuclear force arises from pion exchange between nucleons. This project:
- Models a **2-body bare state** (proton-neutron) and **3-body dressed states** (PN + virtual pion)
- Uses **Correlated Shifted Gaussians (CSG)** as spatial basis functions
- Optimizes the basis via the **Stochastic Variational Method (SVM)** with competitive growth
- Supports both **non-relativistic** and **relativistic** kinetic energies
- Applies **box regularization** (harmonic oscillator potential) for numerical convergence
- Includes **Python scripts** for parameter sweeps and contour plots

**Target Result**: Ground state energy ≈ -2.224 MeV (experimental deuteron binding energy)

## Physical Model

### System Channels
The deuteron couples between 10 distinct physics channels:

| Channel Group | States | Composition | Notes |
|---------------|--------|-------------|--------|
| PN (Bare)     | 1      | Proton-neutron pair | Ground state, even parity |
| π⁰ (Neutral)  | 3      | PN + neutral pion | 3 spin-flip configurations |
| π⁺ (Positive) | 3      | PN + positive pion | 3 spin-flip configurations |
| π⁻ (Negative) | 3      | PN + negative pion | 3 spin-flip configurations |

Spin-flip types (per pion channel):
- **NO_FLIP (0f)**: No nucleon spin flipped
- **FLIP_PARTICLE_1**: Proton spin flipped
- **FLIP_PARTICLE_2**: Neutron spin flipped

### Key Physics Constants
```
m_p   = 938.27 MeV     (proton mass)
m_n   = 939.56 MeV     (neutron mass)
m_π⁰  = 134.97 MeV     (neutral pion)
m_πc  = 139.57 MeV     (charged pion)
ℏc    = 197.327 MeV·fm (conversion constant)
```

### Hamiltonian Structure

The Hamiltonian has block structure:
```
H[i,j] = T[i,j] + V_rest[i,j] + W[i,j]
        + V_box[i,j]  (if box regularization active)
```

Where:
- **T[i,j]** = Kinetic energy matrix element (classical or relativistic)
- **V_rest** = Pion rest mass (for dressed states only)
- **W[i,j]** = Pion exchange coupling (off-diagonal between channels)
- **V_box** = Artificial harmonic oscillator confining potential (optional)

**Matrix properties**:
- Hermitian (ensures real eigenvalues)
- Positive definite overlap matrix N[i,j]
- Sparse structure by selection rules

### Kinetic Energy Options

1. **Classical (non-relativistic)**:
   ```
   T_classical = ℏ²/2μ × <ψ | p² | ψ'>
   ```
   Analytical for Gaussians; fast computation

2. **Relativistic**:
   ```
   T_rel = <ψ | (√(p² + m²) - m) | ψ'>
   ```
   Computed via Simpsons in momentum space

### Pion Coupling (W-operator)

**Physical Process**: Nucleon pair emits/absorbs virtual pion

**Mathematical Form**:
```
W[i,j] = S × g(b) × f(w_π·r) × Tensor_operator
```

Where:
- **S** = Coupling strength (tuning parameter, typically 25-45 MeV)
- **b_form** = Form factor range controlling interaction localization (fm)
- **b_range** = Search space width for basis function widths (fm)
- **Tensor operator** = Spin structure (3 types per pion channel)

### Basis Functions (Correlated Shifted Gaussians)

Each basis state is a product of correlated Gaussians in relative coordinates:
```
ψ(r) = [exp(-r·A·r + s·r) + P·exp(-r·A·r - s·r)] / N
```

**Components**:
- **A** = (N-1)×(N-1) correlation matrix (positive definite)
- **s** = (N-1)×3 shift matrix (translates center in space)
- **P** = ±1 parity sign (symmetric or antisymmetric)
- **N** = Normalization factor

**Physical interpretation**:
- Exponential exp(-r·A·r) controls range of wavefunction
- Shift s·r couples Gaussian to kinetic energy
- Two terms (+ and -) handle exchange symmetry/antisymmetry

## Project Structure

```
bachelor/
├── README.md                    # This file
├── Makefile                     # Build targets and sweep automation
├── deu.cc                       # Main SVM driver
├── deuterium.h                  # Channel definitions & Hamiltonian
├── SVM.h                        # SVM algorithm (competitive growth, optimization)
├── qm/                          # Quantum mechanics library
│   ├── matrix.h                 # Vector/matrix algebra (real & complex)
│   ├── gaussian.h               # Basis functions & overlaps
│   ├── jacobi.h                 # N-body coordinate transformations
│   ├── operators.h              # Kinetic energy & W-operator
│   ├── solver.h                 # GEVP solver & Nelder-Mead optimization
│   ├── serialization.h          # I/O for basis states
│   └── csv_writer.h             # CSV output for results
├── scripts/                     # Python automation scripts
│   ├── sweep_S.py              # Sweep coupling strength S
│   ├── sweep_b_form.py          # Sweep pion form factor range
│   ├── sweep_b_range.py         # Sweep basis search space
│   ├── sweep_basis_size.py      # Basis convergence study
│   ├── contour_plot_b_form.py  # Generate b_form contour
│   ├── contour_plot_b_range.py  # Generate b_range contour
│   ├── smart_contour_search.py  # Adaptive mesh for radius target
│   ├── plot_results.py          # Publication-quality plotting
│   └── plot_wavefunction.py     # Visualize ground state wavefunction
└── results/                     # Output directory
    ├── all_configs_*/           # Complete runs with all configurations
    ├── energy_sweep_*/          # Parameter sweep results
    ├── contour_*/               # 2D contour data
    └── smart_contour/           # Adaptive mesh search
```

## Algorithm Workflow

### Core SVM Algorithm:

**Competitive SVM Growth** (inside widening HO box)
```
For each harmonic oscillator box strength (K):
  Twice for each of 9 pion channels:
    - Generate ~1000 random candidate states
    - Evaluate each via GEVP solve
    - Lock the best (lowest energy) into basis
  - After all channels: sweep-optimize entire expanded basis
  
Then: Move specialized states to master pool
```

**Final Free-Space Refinement** (ho_k = 0)
```
- Single shallow sweep at box strength 0
- Optmize all basis states
- Allows core/pocket/tail states to equilibrate
- Compute final observables (energy, radius, kinetic energy)
```

### Box Regularization Strategy

The **harmonic oscillator box** is a tuning tool that:
1. Prevents basis from spreading too wide initially
2. Forces states to stay physically localized
3. Gradually weakened to zero (free space)

**Effect**: Energy monotonically decreases as box strength → 0

### Basis State Selection

**Competitive growth criterion**: Keep state i+1 if:
```
E(basis + state) < E(basis) - 1e-6  [in MeV]
```

This ensures rigorous variational improvement.

## Command-Line Interface

### Main Executable: `./deu`

```bash
./deu [options]

Options:
  -b_range FLOAT          Search space for Gaussian widths (fm)
                         Default: 2.24
  
  -b_form FLOAT           Pion form factor range (fm)
                         Default: 1.4
  
  -S FLOAT                Pion coupling strength (MeV)
                         Default: 31.29
  
  --output-csv PATH      Write full results to CSV (metadata + convergence)
  
  -box-strengths LIST    Comma-separated HO box strengths
                         Example: "0.0" or "0.5,0.1,0.0"
  
  --pn-rel              Use relativistic PN channel (default: false)
  
  --pi-rel              Use relativistic pion channel (default: false)
  
  -h, --help            Show help message
```

### Example Usage

```bash
# Find ground state with default parameters
./deu

# Scan for optimal S value (find binding energy ≈ -2.224 MeV)
./deu -S 35.0 --output-csv results/scan_s35.csv

# Use box regularization with free-space final step
./deu -S 31.29 -b_form 1.4 -b_range 2.24 \
  -box-strengths "1.0,0.5,0.2,0.0"

# Fully relativistic calculation
./deu --pn-rel --pi-rel -S 32.0
```

## Automated Parameter Sweeps

### Using Python Scripts

All sweeps use **parallel execution** via `ProcessPoolExecutor`. Results are aggregated into CSV files and plotted automatically.

**Sweep S (coupling strength)**:
```bash
python3 scripts/sweep_S.py \
  --b_range 2.24 --b_form 1.4 \
  --S_min 25.0 --S_max 45.0 --S_steps 20 \
  --jobs 10
```
→ Output: `results/energy_sweep_S/aggregated.csv` + plots

**Sweep b_form (pion interaction range)**:
```bash
python3 scripts/sweep_b_form.py \
  --b_range 2.24 --S 31.29 \
  --b_form_min 0.8 --b_form_max 2.0 --b_form_steps 12 \
  --jobs 8
```
→ Output: `results/energy_sweep_b_form/aggregated.csv`

**2D Contour Search** (b_range vs S):
```bash
python3 scripts/contour_plot_b_range.py \
  --b_range_min 2.0 --b_range_max 3.0 --b_range_steps 10 \
  --S_init_anchor 30.0 --S_window 10.0 --S_steps 8 \
  --b_form 1.4 --jobs 12
```
→ Output: `results/contour_b_range/grid_data.csv` + contour plot

**Adaptive Radius Contour** (find 2D surface where radius = 2.128 fm):
```bash
python3 scripts/smart_contour_search.py \
  --S_init 31.29 --b_form_init 1.4 --b_range_init 2.24 \
  --radius_target 2.128 --max_iterations 50 --jobs 10
```
→ Output: Adaptive mesh points converging to charge radius target

**Basis Size Convergence**:
```bash
python3 scripts/sweep_basis_size.py \
  --b_range 2.24 --b_form 1.4 --S 31.29 \
  --basis_size_steps 8 --jobs 8
```
→ Shows energy vs. number of basis states

### Using Makefile

Convenience targets for common workflows:

```bash
# Build
make all

# Run all 4 kinematic configs with default parameters (saves timestamped results)
make all-configs  B_RANGE=2.24 B_FORM=1.4 S=31.29

# Sweep S with 10 parallel jobs
make sweep_S  B_RANGE=2.24 B_FORM=1.4

# Generate 2D contour plot for b_range parameter
make contour_b_range  B_FORM=1.4 S=31.29

# Clean build artifacts
make clean
```

All Makefile targets automatically create `results/` subdirectories and run the Python scripts with appropriate job counts.

## Compilation

```bash
# With GNU C++23 and OpenMP support
g++ -std=c++23 -fopenmp -O3 -Wall deu.cc -o deu

# Using Clang
clang++ -std=c++23 -fopenmp -O3 deu.cc -o deu

# Using Intel compiler
icpc -std=c++23 -fopenmp -O3 deu.cc -o deu
```

**Requirements**:
- C++23 compiler (for structured bindings, auto types)
- OpenMP (for parallel optimization)
- Standard library only (no external dependencies)

## Example Output

```
========================================
  DEUTERON SYSTEM (FAST COMPETITIVE SVM)
========================================

Parameters:
  b_range = 2.24 fm
  b_form  = 1.4 fm
  S       = 31.29 MeV
  PN Treatment: Cla, Pion Treatment: Cla
  Box Strengths: 5, 2, 1, 0.5, 0.2, 0.1, 0
========================================

--- 1. Planting Geometric PN Grid & Pion Seeds ---

--- 2. Competitive Growth inside widening HO Box ---

=== Generating Basis for ho_k = 5 ===
=== Generating Basis for ho_k = 2 ===
...
=== FINAL GEVP EVALUATION IN FREE SPACE (ho_k = 0.0) ===

>>>>>>>> RUNNING CONFIGURATION: PN_{Cla} Pi_{Cla} <<<<<<<<
...
--> FINAL PN_{Cla} Pi_{Cla} | E: -2.18234 MeV, R: 2.13567 fm

Saved basis state to basis_final.txt

======================================================================================================================================
                                  FINAL RESULTS SUMMARY                                   
======================================================================================================================================
PN_{Cla} Pi_{Cla}   | E: -2.18234 MeV | R: 2.13567 fm | <T>: 40.234 MeV | PN: 85.2 % | PN+pi: 14.8 % | Time: 3.456 s
--------------------------------------------------------------------------------------------------------------------------------------
Experimental Target | E: -2.22400 MeV | R: 2.12800 fm
======================================================================================================================================

Saved all configurations to all_configurations.txt
```

## Key Physics Interpretation

| Quantity | Physical Meaning | Typical Value |
|----------|------------------|---|
| **E** | Ground state energy | -2.22 MeV (bound) |
| **R** (charge radius) | ⟨r²_charge⟩^(1/2) | 2.13 fm |
| **<T>** (kinetic energy) | Expectation value of kinetic operator | ~40 MeV |
| **Prob PN** | Probability state is bare PN | 80-90% |
| **Prob PN+π** | Probability state contains pion | 10-20% |

## References & Implementation Details

### Numerical Methods Used

1. **Analytical Gaussian Overlaps**: No numerical integration
   - Formula: ⟨g₁|g₂⟩ = (π/det(A₁+A₂))^(3/2) × exp(...)
   
2. **Jacobi Rotation**: Diagonalization of H' matrix
   - Converges when all off-diagonals < 10^(-6)
   - Typically 50-100 sweeps needed
   
3. **Nelder-Mead Simplex**: Derivative-free optimization
   - Non-smooth objective (eigenvalue doesn't have smooth gradients)
   - Ideal for basis parameter tuning
   - Typically 200-500 evaluations per optimization
   
4. **Stochastic Sampling**: Random state generation via quasi-random sequences
   - Van der Corput quasi-random for deterministic pseudo-randomness
   - Ensures good exploration of parameter space

### Performance Characteristics

- **Typical runtime**: 3-5 seconds per point (64-state basis)
- **Parallelization**: OpenMP across candidate screening (10-fold speedup on 10 cores)
- **Basis size**: Grows to ~60-100 states for full convergence
- **Memory usage**: <100 MB for typical run
- `main()`: Run classic + relativistic and report results

### deuterium.h
**Physical system definitions**

- `Channel` enum: PN + 9 pion channels
- `BasisState` struct: Wavefunction + channel + Jacobian + masses
- `build_matrices()`: Compute H and N matrices from basis states

### qm/gaussian.h
**Spatial basis functions**

Functions:
- `gaussian_overlap(...)`: Compute <Gaussian_1 | Gaussian_2> analytically
- `spatial_overlap(...)`: Add parity factor <ψ₁|ψ₂> = overlap + parity × shifted_overlap
- `apply_basis_expansion(...)`: Higher-order function integrating over all parity combinations

Structs:
- `Gaussian`: Elementary Gaussian (A, s parameters)
- `SpatialWavefunction`: Full basis with parity ± symmetry

### qm/jacobi.h
**N-body coordinate transformations**

Jacobi transform:
- Relative coordinates: r_i = r_i - (r_1 + ... + r_{i-1}) / i
- Center of mass: R_cm = (m₁r₁ + ... + m_N r_N) / M_total
- Removes 3 CM degrees of freedom → N-1 internal dimensions

Struct `Jacobian`:
- Precomputes transformation matrices
- Computes reduced masses μᵢ for each coordinate pair

### qm/operators.h
**Quantum mechanics operators**

Functions:
- `classic_kinetic_energy(...)`: T = p²/2μ via Gaussian matrix elements
- `relativistic_kinetic_energy(...)`: T = √(p²+m²) - m via 32-point Gauss-Legendre quadrature
- `total_kinetic_energy(...)`: Sums over all Jacobi coordinates
- `total_w_coupling(...)`: Pion emission/absorption matrix elements

Key concept:
- Kinetic energy = expectation of ∇² operator
- Requires integral of Gaussian derivatives
- Relativistic version uses numerical integration (k-space)

### qm/solver.h
**Eigenvalue solvers & optimization**

Functions:
- `solve_ground_state_energy(...)`: GEVP solver (Cholesky + Jacobi diagonalization)
- `jacobi_lowest_eigenvalue(...)`: Jacobi rotation method for Hermitian matrices
- `nelder_mead(...)`: Simplex optimization for basis parameters

Numerical methods:
- **Cholesky**: Decomposes overlap matrix N = L L†
- **Jacobi rotation**: Similarity transforms to diagonalize
- **Nelder-Mead**: Robust derivative-free optimization

### qm/matrix.h
**Linear algebra library**

Templated for `long double` (real) and `std::complex<long double>` (complex)

Capabilities:
- Vector/matrix arithmetic: +, -, ×, /, determinant, inverse
- Specialized: `cholesky()`, `inverse_lower()`, block operations
- Output: Pretty-printed formatting

## Physics Troubleshooting

### Issue: Energy too high (close to 0 or positive)
**Causes**:
- `S` parameter too small (weak pion coupling)
- Basis not large enough
- `b_form` outside physical range

**Fix**:
- Increase `S` (start with 150, try 180)
- Run more SVM cycles
- Check form factor range: b_form ∈ [0.8, 2.0]

### Issue: Cholesky decomposition fails (999999 MeV)
**Causes**:
- Basis becoming linearly dependent (duplicate states)
- Numerical instability
- Gaussian parameters out of bounds

**Fix**:
- SVM automatically rejects this candidate (expected behavior)
- If persistent: tighten `ZERO_LIMIT` in `qm/matrix.h`

### Issue: Optimization stalls (no improvement after sweeps)
**Causes**:
- Stuck in local minimum
- Basis parameters not being varied enough
- Nelder-Mead tolerance too loose

**Fix**:
- Increase `num_candidates_per_step` (100 → 150)
- Increase `max_sweeps` (20 → 40)
- Loosen `b_range` (1.4 → 2.0)

## References
