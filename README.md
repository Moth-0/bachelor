# Deuterium Ground State Energy via Pion-Exchange

Simulating relativistic effects of exchanged pions in light nuclei. This project finds the ground state energy of a deuteron (proton-neutron bound pair) using quantum mechanics with pion exchange forces.

## Overview

The nuclear force arises from pion exchange between nucleons. This project:
- Models a **2-body bare state** (proton-neutron) and **3-body dressed states** (PN + virtual pion)
- Uses **Correlated Shifted Gaussians (CSG)** as spatial basis functions
- Optimizes the basis via the **Stochastic Variational Method (SVM)**
- Computes both **non-relativistic** and **relativistic** kinetic energies

**Target Result**: Ground state energy ≈ -2.224 MeV (experimental deuteron binding energy)

## Physical Model

### System Channels
| Channel | Composition | Notes |
|---------|-------------|--------|
| PN      | Bare nucleon pair | Ground state seed |
| π⁰      | PN + neutral pion | 3 spin-flip configurations |
| π⁺      | PN + positive pion | 3 spin-flip configurations |
| π⁻      | PN + negative pion | 3 spin-flip configurations |

### Key Physics Constants
```
m_p  = 938.272 MeV     (proton mass)
m_n  = 939.565 MeV     (neutron mass)
m_π⁰ = 134.97  MeV     (neutral pion)
m_πc = 139.57  MeV     (charged pion)
ℏc   = 197.327 MeV·fm  (conversion constant)
```

### Physics Description

**Hamiltonian Structure**:
- Diagonal blocks: Kinetic energy T(r) + pion rest mass (for dressed states)
- Off-diagonal blocks: W-operator coupling (pion exchange)

**Kinetic Energy**:
- Classical: T = p²/2μ
- Relativistic: T = √(p² + m²) - m

**Pion Coupling** (W-operator):
- Form factor with range b (fm) controls interaction strength
- Strength S (tuning parameter) adjusted to match experimental binding

**Basis Functions** (Correlated Shifted Gaussians):
```
ψ(r) = [exp(-r·A·r + s·r) + P·exp(-r·A·r - s·r)]

where:
  A = correlation matrix (positive definite)
  s = shift vector
  P = parity sign (±1)
  r = relative coordinates (Jacobi transformed)
```

## Project Structure

```
bachelor/
├── README.md              # This file
├── deu.cc                 # Main SVM workflow driver
├── deuterium.h            # Channel definitions & basis wrappers
└── qm/                    # Quantum mechanics library
    ├── matrix.h           # Matrix/vector algebra (real & complex)
    ├── gaussian.h         # Gaussian basis & overlap integrals
    ├── jacobi.h           # Coordinate transformation (N-body → relative)
    ├── operators.h        # Kinetic energy & pion coupling
    └── solver.h           # GEVP solver & optimization
```

## Algorithm Workflow

### Phase 1: Skeleton Basis (14 states)
1. **5 deterministic PN states** with geometric widths: {0.02, 0.08, 0.3, 1.2, 4.0} fm
   - Ensures wide-range coverage of spatial scales
2. **9 random pion-dressed seeds** (one per channel: π⁰, π⁺, π⁻ × 3 flips each)
   - Initial guess for coupled channels
3. Optimize all 14 parameters with Nelder-Mead sweeping

### Phase 2: Competitive SVM Growth (configurable cycles)
**Per cycle**, for each of 10 pion channels:
- Test 100 random candidate states (parallel with OpenMP)
- Evaluate each against full current basis via GEVP
- **Lock the best** (lowest energy) into the basis
- After all channels: sweep-optimize the entire expanded basis

**Why this design?**
- Avoids expensive optimization of all parameters simultaneously
- Incrementally builds basis without artificially truncating states
- Parallel candidate screening exploits multi-core hardware
- SVM guarantees energy monotonically decreases (or stays same)

## Key Computational Methods

### Correlated Shifted Gaussians (CSG)
Basis functions naturally handle short-range correlations in nuclear wavefunctions. Parameters:
- **A** controls correlation range (short-to-medium range)
- **s** shifts spatial center (couples to kinetic energy)
- **Parity** ensures proper symmetry under PN ↔ PN exchange

### Stochastic Variational Method (SVM)
- **Advantage**: No need to pre-know optimal parameters
- **Strategy**: Randomly sample basis space, keep improving states
- **Result**: Basis converges to near-optimal ground state

### Generalized Eigenvalue Problem (GEVP)
Solve: **H c = E N c** (Rayleigh-Ritz principle)

**Solution method**:
1. Cholesky: N = L L†
2. Transform: H' = L⁻¹ H (L†)⁻¹
3. Jacobi diagonalization → lowest eigenvalue = ground state energy

### Pion Exchange (W-operator)
**Physical meaning**: Proton emits pion, neutron absorbs it (and vice versa)

**Form factor**: f(r) = exp(-r²/b²) with range b controlling interaction strength

**Coupling strength**: S parameter (tuned from physics literature or fit to data)

## Configuration Parameters

Located in `deu.cc::run_deuteron_svm()`:

```cpp
ld b_range = 1.4;              // Search space for Gaussian widths (fm)
ld b_form = 1.4;               // Pion form factor range (fm)
ld S = 140.0;                  // *** TUNING PARAMETER ***
int num_cycles = 2;            // Number of SVM growth cycles
int num_candidates_per_step = 100;  // Test 100 states per channel
```

### Parameter Meanings

| Parameter | Typical Range | Effect on Ground State |
|-----------|---------------|------------------------|
| `S` | 50-200 | Higher S → more binding (more negative E) |
| `b_form` | 1.0-2.0 fm | Controls pion interaction range |
| `num_cycles` | 1-5 | More cycles → better convergence |
| `num_candidates_per_step` | 50-200 | More tests → better state selection |

**Critical tuning**: Adjust `S` to match target E ≈ -2.224 MeV

## Compilation & Execution

```bash
# Compile with OpenMP support (parallel Nelder-Mead)
g++ -std=c++23 -fopenmp -O3 deu.cc -o deu

# Run
./deu

# Output shows two calculations:
# 1. Classic kinetic energy
# 2. Relativistic kinetic energy
# Reports difference → relativistic correction to binding
```

### Example Output
```
========================================
  DEUTERON SYSTEM (FAST COMPETITIVE SVM)
========================================

--- 1. Planting Geometric PN Grid & Pion Seeds ---
Skeleton Size: 14 states.
Skeleton Energy: -1.234567 MeV

--- 2. Competitive SVM Growth ---
Added State 15 (Cycle 1, Ch 0) -> E = -1.289345 MeV
Added State 16 (Cycle 1, Ch 1) -> E = -1.342891 MeV
...
 - Sweeping Cycle 1 basis -
...

========================================
  FINAL RESULTS
========================================
Classic Energy:      -1.456789 MeV
Relativistic Energy  -1.523456 MeV
Difference            -0.066667 Mev
========================================
```

**Interpretation**:
- Classic should → **-2.224 MeV**
- Difference shows relativistic correction 

## File Descriptions

### deu.cc
**Main driver**: Orchestrates SVM workflow

Key functions:
- `evaluate_basis_energy(...)`: Build H, N matrices and solve GEVP
- `sweep_optimize_basis(...)`: Nelder-Mead loop optimizing all basis parameters
- `run_deuteron_svm(...)`: Phase 1 & Phase 2 SVM algorithm
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
