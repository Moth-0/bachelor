# Deuterium Hamiltonian Physics Review — Detailed Analysis

## Summary
The deuterium.h matrix building logic has **several physics issues** preventing correct coupling. The most critical issue is in the **W-operator normalization factor** (operators.h line 246), which suppresses inter-channel coupling by >1000×.

---

## ISSUE #1: W-Operator Normalization Factor (CRITICAL)

**Location**: `operators.h` lines 243-246, applied at line 285

**Current Code**:
```cpp
ld norm_sq = 4.0 * M_PI * (3.0 * std::sqrt(M_PI) * b_pow_5) / two_pow_11_halves;
ld norm_factor = 1.0 / std::sqrt(norm_sq);
// ... later ...
return W_term * S * isospin_factor * norm_factor;  // Line 285
```

**Problem**:
- For b = 1.4 fm: `norm_factor ≈ 0.36`
- But test.out shows direct W-operator call: **-941.6 MeV** → matrix element: **0.26 MeV**
- That's a 3600× reduction, NOT explained by 0.36×
- This factor aggressively suppresses all inter-channel coupling

**Evidence**:
- test.out line 145: Direct `total_w_coupling()` call → **-941.6 MeV**
- test.out line 108: Matrix H(0,1) → **0.26 MeV**
- Ratio: 941.6 / 0.26 ≈ 3600×

**Physics Question**:
Where does this normalization come from? Is it:
1. A legitimate form-factor volume normalization?
2. A **misplaced debug factor** left from earlier fixes?
3. Calculated from wrong formula?

**Hypothesis**: This factor was probably added to try to normalize the W-operator output, but it's applying the inverse of what's needed (dividing instead of multiplying).

---

## ISSUE #2: NO_FLIP Case — Conceptual Concern (MEDIUM)

**Location**: `deuterium.h` lines 165-175

**Current Code** (after user's fix):
```cpp
else { // NO_FLIP
    cld w_val_n1 = total_w_coupling(state_bare.psi, state_dress.psi,
                                    c_pi_1, b, S, state_dress.isospin_factor, state_dress.flip);

    cld w_val_n2 = total_w_coupling(state_bare.psi, state_dress.psi,
                                    c_pi_2, b, S, state_dress.isospin_factor, state_dress.flip);

    w_val = w_val_n1 + w_val_n2;  // ← Addition (was subtraction)
}
```

**Issue**:
Calling `total_w_coupling()` with **different coordinate vectors** (c_pi_1 vs c_pi_2) in the same integrand context is problematic:

- `c_pi_1` = distance from pion to nucleon 1
- `c_pi_2` = distance from pion to nucleon 2
- These are **nearly opposite** for a symmetric PN pair
- Each call integrates over all 3D space independently

**Physics Concern**:
- The form factor `exp(-(w_π^T r)²/b²)` in line 262 uses the **spatial vector from W-operator**, not from the Jacobi coordinate
- Calling with c_pi_1 vs c_pi_2 might be computing **different integrals** that shouldn't be added directly
- Should we be decomposing the NO_FLIP case differently?

**What Should Happen**:
For π⁰ NO_FLIP coupling:
- Both nucleons can symmetrically emit the pion
- The proper way might be to compute ONE integral with a symmetric combination, not two separate integrals

---

## ISSUE #3: Hamiltonian Matrix Structure (PHYSICS CHECK)

**Location**: `deuterium.h` lines 101-199 (`build_matrices` function)

### Diagonal Elements (Same-Channel)

#### PN to PN (lines 123-127):
```cpp
if (state_i.type == state_j.type && state_i.type == Channel::PN) {
    ld T_pn = total_kinetic_energy(state_i.psi, state_j.psi, state_i.jac, {false});
    h_val += cld(T_pn, 0.0);
}
```

**✓ Correct**: Pure kinetic energy for bare nucleons (no pion mass).

#### Pion-dressed to Pion-dressed (lines 131-139):
```cpp
else if (state_i.type == state_j.type && state_i.type != Channel::PN) {
    ld T_total = total_kinetic_energy(..., {false, relativistic});
    ld rest_mass_term = state_i.pion_mass * std::real(n_val);
    h_val += cld(T_total + rest_mass_term, 0.0);
}
```

**⚠️ Potential Issue**:
- Multiplying pion mass by `n_val` (the overlap):
  - **If overlap is already normalized**: This is correct (E_pion = m_π × 1.0)
  - **If overlap is NOT normalized**: This would be wrong (should use `1.0` not `n_val`)
- Test.out shows n_val > 1 for diagonal elements, suggesting overlaps are **not normalized to 1**
  - This is OK if kinetic energy term is also computed with full overlap
  - But mixing `n_val` into rest mass term while T already includes M_overlap might double-count

### Off-Diagonal Elements (Cross-Channel)

#### PN ↔ Pion coupling (lines 142-185):
```cpp
else if ((state_i.type == Channel::PN && state_j.type != Channel::PN) || ...) {
    // ... W-operator computation ...
    if (i_is_bare) {
        h_val += w_val;
    } else {
        h_val += std::conj(w_val);
    }
}
```

**✓ Logic appears correct**:
- Orthogonal channels → N(i,j) = 0
- Only W-operator couples them
- Hermiticity preserved via conjugation

**✗ BUT**: The W-operator magnitude is wrong due to Issue #1.

---

## ISSUE #4: Parity and Channel Orthogonality (VERIFICATION NEEDED)

**Location**: `deuterium.h` lines 86-89 (channel definition)

```cpp
enum class Channel { PN,
                     PI_0c_0f, PI_0c_1f, PI_0c_2f,
                     PI_pc_0f, PI_pc_1f, PI_pc_2f,
                     PI_mc_0f, PI_mc_1f, PI_mc_2f };
```

**Observation from test.out**:
- Test uses only: PN (parity +1), and PI_0c_0f (parity -1)
- Both states are created with **parity signs correctly set**
- Different channels → N(i,j) = 0 ✓

**No immediate issue**, but verify parity sign application in `apply_basis_expansion()` hasn't been modified.

---

## ISSUE #5: Rest Mass Term Scaling (MINOR)

**Location**: `deuterium.h` lines 137 and 150

```cpp
ld rest_mass_term = state_i.pion_mass * std::real(n_val);
h_val += cld(T_total + rest_mass_term, 0.0);
```

**Question**: Why multiply pion rest mass by the overlap `n_val`?

**Test evidence** (test.out lines 68-71):
```
T(1,1) = 1326.51586970 MeV (+ pion mass 134.97000000 = 1461.48586970 total)
```

The test computes: T + m_π directly (no overlap factor).

**But the code does**: (T_total + m_π * N(i,i)) * 1.0

If N(i,i) ≠ 1, this creates an **effective mass different from physical m_π**.

For test.out line 104: N(1,1) = 1.528...
- Effective pion mass = 134.97 × 1.528 ≈ 206 MeV (way too heavy!)

**This is likely WRONG**. Should be:
```cpp
ld rest_mass_term = state_i.pion_mass;  // No overlap scaling
h_val += cld(T_total + rest_mass_term, 0.0);
```

---

## ISSUE #6: Overlap Matrix Construction (MEDIUM)

**Location**: `deuterium.h` lines 117-120

```cpp
if (state_i.type == state_j.type) {
    n_val = static_cast<ld>(spactial_overlap(state_i.psi, state_j.psi));
}
```

**✓ Correct**: Overlaps computed only for same-channel pairs.

**⚠️ Watch**:
- For diagonal (i=j): Should be positive and significant
- For off-diagonal same-channel: Should be non-zero for nearby basis states
- Test.out diagonal overlaps range 1.5-7.8, which is reasonable

---

## Summary of Fixes Needed

### CRITICAL (Must fix):
1. **Remove or fix the W-operator `norm_factor`** (operators.h line 246)
   - Either remove it entirely, or verify it's the correct formula
   - Current form is suppressing inter-channel coupling by 100×+

### HIGH (Should fix):
2. **Fix the rest mass term** (deuterium.h line 137)
   - Change `state_i.pion_mass * std::real(n_val)` → `state_i.pion_mass`
   - Pion rest mass should be constant, not scaled by overlap

3. **Verify NO_FLIP W-coupling logic** (deuterium.h lines 165-175)
   - Consider whether calling `total_w_coupling()` twice with opposite vectors makes physical sense
   - Might need to compute a single integral with symmetric combination instead

### MEDIUM (Review):
4. **Check `apply_basis_expansion()` function**
   - Ensure M_overlap (line 212, 256) is being computed correctly
   - Ensure it's not double-applied with other overlaps

---

## Test Cases to Validate

Once fixes are applied:
1. Run bare PN system: E ≈ 52 MeV (test.out confirms this works)
2. Run deuteron with fixed W-operator: E should drop dramatically toward binding energy
3. Check that H(0,1) matrix element increases to ~100+ MeV (not 0.26 MeV)
4. Verify proton system also shows reasonable binding with same S parameter
