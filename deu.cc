//
// deu.cc  —  Deuteron ground state via explicit sigma-meson coupling
//
// Reproduces the calculation from:
//   D.V. Fedorov, "A nuclear model with explicit mesons", Few-Body Systems (2020)
//
// System:
//   Two coupled channels:
//     ch[0]  bare np state           (dim=1,  ψ_np(x₀))
//     ch[1]  dressed σnp state       (dim=2,  ψ_σnp(x₀, x₁))
//
// Hamiltonian (paper eq.8):
//   H = | T_np                  W              |
//       | W†    T_np + T_σnp + m_σ · N_dressed |
//
// W-operator (paper eq.11):
//   W(x₀, x₁) = S_σ exp(−(x₀² + x₁²) / b_σ²)
//
// This is a SpinType::SCALAR coupling — both Jacobi coordinates are penalised
// symmetrically.  The W matrix element is a pure (modified) overlap:
//   ⟨g'_dressed | W | g_bare⟩ = S_σ · ⟨g'_dressed | g_ket_eff⟩
//   where A_tilde = promote(A_bare) + (1/b²)·I
//
// This file delegates entirely to the generic run_svm() from solver.h.
// The physics glue needed here:
//   1.  JacobiSystem  {m_n, m_p, m_σ}
//   2.  two Channel descriptors
//   3.  SvmParams
//
// Known result (paper §4):
//   m_σ = 500 MeV,  m_n = m_p = 939 MeV
//   b_σ = 3 fm,  S_σ = 20.35 MeV
//   → E₀ ≈ −2.2 MeV,  R_c ≈ 2.1 fm
//
// Build:
//   g++ -std=c++17 -O2 -o deu deu.cc
//

#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"
#include "qm/solver.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qm;

// ─────────────────────────────────────────────────────────────────────────────
// build_sigma_channels  —  construct the two-channel descriptor list
//
// ch[0]  bare np channel
//   — treated specially by HamiltonianBuilder: uses basis_bare, 1D KE only.
//   — W-coupling parameters are never read for channel 0 (the W loop starts
//     at a = 1), so we fill them with safe zero/dummy values.
//
// ch[1]  dressed σnp channel
//   — pion_mass = m_sigma  →  meson rest mass added to H diagonal block
//   — iso_coeff = 1        →  scalar-isoscalar (no isospin algebra factor)
//   — spin_type = SCALAR   →  W = S · exp(−Σ_α xα²/b²), no (σ·r) factor
//   — w_piN is irrelevant for SCALAR (unused in w_matrix_element); we pass
//     a zero vector to make this explicit in the code.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Channel> build_sigma_channels(const JacobiSystem& sys,
                                           ld                  m_sigma)
{
    std::vector<Channel> channels(2);

    // ── Channel 0: bare np ───────────────────────────────────────────────────
    channels[0].index      = 0;
    channels[0].is_bare    = true;
    channels[0].pion_mass  = ld{0};
    channels[0].iso_coeff  = ld{0};          // unused for bare channel
    channels[0].w_piN      = rvec(sys.N-1);  // zero vector — unused
    channels[0].spin_type  = SpinType::NO_FLIP;  // unused for bare channel
    channels[0].dim        = 1;

    // ── Channel 1: dressed σnp ───────────────────────────────────────────────
    channels[1].index      = 1;
    channels[1].is_bare    = false;
    channels[1].pion_mass  = m_sigma;
    channels[1].iso_coeff  = ld{1};          // scalar-isoscalar coupling
    channels[1].w_piN      = rvec(sys.N-1);  // zero vector — not used by SCALAR
    channels[1].spin_type  = SpinType::SCALAR;
    channels[1].dim        = sys.N - 1;

    return channels;
}


// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Deuteron binding — sigma-meson model (Fedorov 2020) ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    // ── Physical parameters (paper §4) ───────────────────────────────────────
    const ld m_nucleon = ld{939.0L};   // MeV  (paper uses equal masses)
    const ld m_sigma   = ld{500.0L};   // MeV  sigma meson rest mass
    const ld b_sigma   = ld{3.0L};     // fm   coupling form-factor range
    const ld S_sigma   = ld{20.35L};   // MeV  scalar coupling strength

    // ── Jacobi system: particle 0 = neutron, 1 = proton, 2 = σ ──────────────
    //   x₀ = r_p − r_n          (np relative coordinate,  μ₀ = m/2)
    //   x₁ = r_σ − r_{np CM}    (σ relative to np CM,     μ₁ = 2m·m_σ/(2m+m_σ))
    JacobiSystem sys({m_nucleon, m_nucleon, m_sigma}, {"n", "p", "sigma"});

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "System:  n + p + σ\n";
    std::cout << "  m_n = m_p = " << m_nucleon << " MeV\n";
    std::cout << "  m_σ        = " << m_sigma   << " MeV\n";
    std::cout << "  μ_np  (μ₀) = " << sys.mu[0] << " MeV  (np reduced mass)\n";
    std::cout << "  μ_σnp (μ₁) = " << sys.mu[1] << " MeV  (σ−NN reduced mass)\n\n";

    std::cout << "Coupling parameters (Fedorov 2020):\n";
    std::cout << "  b_σ = " << b_sigma << " fm\n";
    std::cout << "  S_σ = " << S_sigma << " MeV\n";
    std::cout << "  Target:  E₀ ≈ −2.2 MeV,  R_c ≈ 2.1 fm\n\n";

    // ── Channels ─────────────────────────────────────────────────────────────
    auto channels = build_sigma_channels(sys, m_sigma);

    // ── SVM parameters ────────────────────────────────────────────────────────
    // run_svm grows one bare + one dressed Gaussian per step (paired), so
    // K_max = 40 yields a 2×40 = 80-function basis.
    //
    // Paper uses b₀ = b_σ = 3 fm and unshifted Gaussians (s_max = 0).
    //
    // b_ff and S_coupling are forwarded directly into w_matrix_element()
    // by HamiltonianBuilder — they must match b_σ and S_σ.
    //
    // refine_every = 10  →  every 10 accepted steps, loop over the existing
    // basis and try to replace individual pairs with better candidates.
    // Set to 0 to disable refinement and reproduce the original two-phase
    // behaviour (faster but converges less tightly).
    SvmParams params;
    params.K_max          = 25;
    params.N_trial        = 50;
    params.refine_every   = 10;     // 0 = disable
    params.N_refine_trial = 20;
    params.b0             = b_sigma; // same length scale as coupling range
    params.s_max          = ld{0};   // paper: unshifted Gaussians
    params.b_ff           = b_sigma; // form-factor range  → w_matrix_element
    params.S_coupling     = S_sigma; // coupling strength  → w_matrix_element
    params.relativistic   = false;   // switch to true for relativistic KE
    params.verbose        = true;

    // ── Run ──────────────────────────────────────────────────────────────────
    std::mt19937 rng(42);
    auto t0 = std::chrono::steady_clock::now();

    SvmState result = run_svm(sys, channels, params, rng);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  Results                                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Final basis:  K = " << result.K()
              << "  (" << 2 * result.K() << " total functions: "
              << result.K() << " bare + " << result.K() << " dressed)\n";
    std::cout << "  E₀ (computed) = " << result.E0      << " MeV\n";
    std::cout << "  E₀ (target)   = -2.2000 MeV\n";
    std::cout << "  Deviation     = " << result.E0 - ld{-2.2L} << " MeV\n";
    std::cout << "  Elapsed time  = " << elapsed << " s\n\n";

    return 0;
}