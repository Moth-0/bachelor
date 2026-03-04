#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <string>

#include "qm/matrix.h"
#include "qm/eigen.h"
#include "qm/particle.h"
#include "qm/jacobian.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"

// ============================================================
//  deu.cc  —  Deuteron via sigma meson exchange
//
//  Block Hamiltonian (2x2):
//
//    H = | H_pn        W      |
//        | W^T   H_pnσ + m*N  |
//
//  SVM procedure from Varga & Suzuki, Sec. 4.2:
//    Phase 1 — Competitive selection: build basis one slot
//              at a time, n_cand candidates per slot.
//    Phase 2 — Refining: cycle through all slots n_refine
//              times and try to improve each one.
//
//  Parameters (Fedorov, "A nuclear model with explicit mesons"):
//    S = -20.35 MeV,  b = 3.0 fm,  m_sigma = 500 MeV
//  Target: E_0 = -2.225 MeV
// ============================================================

using namespace qm;

// -----------------------------------------------------------
//  Build the full block H and N matrices
// -----------------------------------------------------------
static void build_full(
    const std::vector<gaus>& basis_bare,
    const std::vector<gaus>& basis_sig,
    const jacobian& jac_bare,
    const jacobian& jac_sig,
    const hamiltonian& H,
    const matrix& Omega,
    long double S_sigma,
    long double m_sigma,
    matrix& H_tot,
    matrix& N_tot)
{
    size_t n1 = basis_bare.size();
    size_t n2 = basis_sig.size();
    size_t N  = n1 + n2;

    H_tot = matrix(N, N);
    N_tot = matrix(N, N);

    constexpr std::array<long double,3> beta_zero = {0.0L, 0.0L, 0.0L};

    // Block (0,0): bare pn
    for (size_t i = 0; i < n1; ++i)
        for (size_t j = 0; j < n1; ++j) {
            N_tot(i,j) = overlap(basis_bare[i], basis_bare[j]);
            H_tot(i,j) = H.K_cla(basis_bare[i], basis_bare[j],
                                   jac_bare.c(0), jac_bare.mu(0));
        }

    // Block (1,1): clothed pnσ + rest mass
    for (size_t i = 0; i < n2; ++i)
        for (size_t j = 0; j < n2; ++j) {
            long double nij  = overlap(basis_sig[i], basis_sig[j]);
            long double k_np = H.K_cla(basis_sig[i], basis_sig[j],
                                        jac_sig.c(0), jac_sig.mu(0));
            long double k_s  = H.K_cla(basis_sig[i], basis_sig[j],
                                        jac_sig.c(1), jac_sig.mu(1));
            N_tot(n1+i, n1+j) = nij;
            H_tot(n1+i, n1+j) = k_np + k_s + m_sigma * nij;
        }

    // Off-diagonal: W coupling (scalar, alpha=S, beta=0)
    for (size_t i = 0; i < n1; ++i)
        for (size_t j = 0; j < n2; ++j) {
            long double w = H.W(basis_bare[i], basis_sig[j],
                                 Omega,
                                 jac_sig.meson_index(),
                                 S_sigma,
                                 beta_zero);
            H_tot(i,    n1+j) = w;
            H_tot(n1+j, i   ) = w;
            N_tot(i,    n1+j) = 0.0L;
            N_tot(n1+j, i   ) = 0.0L;
        }
}

// -----------------------------------------------------------
//  Solve H c = E N c, return lowest eigenvalue or NaN
// -----------------------------------------------------------
static long double solve(const matrix& H_tot, const matrix& N_tot)
{
    matrix L = cholesky(N_tot);
    if (L.size1() == 0) return std::numeric_limits<long double>::quiet_NaN();

    matrix L_inv = L.inverse_lower();
    if (L_inv.size1() == 0) return std::numeric_limits<long double>::quiet_NaN();

    vector evals = jacobi_eigenvalues(L_inv * H_tot * L_inv.transpose());
    if (evals.size() == 0) return std::numeric_limits<long double>::quiet_NaN();

    long double E0 = evals[0];
    for (size_t k = 1; k < evals.size(); ++k)
        if (evals[k] < E0) E0 = evals[k];
    return E0;
}

static gaus make_swave(size_t dim, long double A_min, long double A_max)
{
    gaus g(dim, A_min, A_max);
    g.zero_shifts();
    return g;
}

// ============================================================
int main()
{
    std::cerr << "[deu] ============================================\n";
    std::cerr << "[deu] Deuteron + sigma meson SVM\n";
    std::cerr << "[deu] ============================================\n";

    // ----------------------------------------------------------
    // 1. Physics
    // ----------------------------------------------------------
    Particle proton  = make_proton();
    Particle neutron = make_neutron();
    Particle sigma   = make_sigma();

    jacobian jac_bare({proton, neutron});
    jacobian jac_sig ({proton, neutron, sigma});

    std::cerr << "[deu] Jacobians:\n";
    std::cerr << "  bare:  dim=" << jac_bare.dim()
              << "  mu(0)=" << jac_bare.mu(0) << " MeV\n";
    std::cerr << "  sigma: dim=" << jac_sig.dim()
              << "  mu(0)=" << jac_sig.mu(0)
              << "  mu(1)=" << jac_sig.mu(1)
              << "  meson_idx=" << jac_sig.meson_index() << "\n";

    hamiltonian H;
    std::cerr << "  hbar_c=" << H.hbar_c << " MeV*fm\n";

    const long double S_sigma = -20.35L;
    const long double b_sigma =   3.0L;
    const long double m_sigma = sigma.mass;

    // Omega: form factor exp(-r_np^2/b^2) acts ONLY on x_0
    const long double w = 1.0L / (b_sigma * b_sigma);
    matrix Omega(jac_sig.dim(), jac_sig.dim());
    Omega(0,0) = w;    // np coordinate
    Omega(1,1) = 0.0L; // sigma coordinate is free
    Omega(0,1) = 0.0L;
    Omega(1,0) = 0.0L;

    std::cerr << "[deu] Coupling: S=" << S_sigma
              << " MeV  b=" << b_sigma
              << " fm  w=1/b^2=" << w << " fm^-2\n";
    std::cerr << "[deu] m_sigma=" << m_sigma << " MeV\n\n";

    // ----------------------------------------------------------
    // 2. SVM parameters
    // ----------------------------------------------------------
    const size_t n_bare   = 20;
    const size_t n_sig    = 40;
    const int    n_cand   = 30;
    const int    n_refine = 5;
    const long double A_min = 0.00001L;
    const long double A_max = 0.1L;

    std::cerr << "[deu] SVM: n_bare=" << n_bare
              << "  n_sig=" << n_sig
              << "  n_cand=" << n_cand
              << "  n_refine=" << n_refine
              << "  A=[" << A_min << "," << A_max << "]\n\n";

    std::vector<gaus> basis_bare;
    std::vector<gaus> basis_sig;
    basis_bare.reserve(n_bare);
    basis_sig .reserve(n_sig);

    matrix H_cur, N_cur;

    // ----------------------------------------------------------
    // 3. Phase 1 — Competitive selection
    //
    //    Interleave sectors so W coupling is active early.
    //    For each slot: test n_cand candidates, keep the best.
    // ----------------------------------------------------------
    std::cerr << "=== Phase 1: Competitive selection ===\n";
    std::cerr << std::fixed << std::setprecision(4);

    size_t total_slots = n_bare + n_sig;

    for (size_t slot = 0; slot < total_slots; ++slot) {

        // Decide sector: interleave until one is full
        bool add_bare;
        if      (basis_bare.size() >= n_bare) add_bare = false;
        else if (basis_sig .size() >= n_sig ) add_bare = true;
        else                                  add_bare = (slot % 2 == 0);

        size_t dim   = add_bare ? jac_bare.dim() : jac_sig.dim();
        const char* sec = add_bare ? "bare " : "sigma";

        long double best_E = std::numeric_limits<long double>::infinity();
        gaus        best_g = make_swave(dim, A_min, A_max);
        int         n_valid = 0;

        for (int c = 0; c < n_cand; ++c) {
            gaus trial = make_swave(dim, A_min, A_max);

            if (add_bare) basis_bare.push_back(trial);
            else          basis_sig .push_back(trial);

            build_full(basis_bare, basis_sig,
                       jac_bare, jac_sig,
                       H, Omega, S_sigma, m_sigma,
                       H_cur, N_cur);

            long double E = solve(H_cur, N_cur);

            if (!std::isnan(E)) {
                ++n_valid;
                if (E < best_E) { best_E = E; best_g = trial; }
            } else {
                std::cerr << "  [!] slot=" << slot
                          << " cand=" << c << " → NaN (ill-conditioned)\n";
            }

            if (add_bare) basis_bare.pop_back();
            else          basis_sig .pop_back();
        }

        if (n_valid == 0) {
            std::cerr << "  [!!] slot=" << slot << " (" << sec
                      << "): ALL " << n_cand
                      << " candidates were ill-conditioned!\n"
                      << "       Using random fallback. "
                      << "Check A_min/A_max or overlap formula.\n";
        }

        if (add_bare) basis_bare.push_back(best_g);
        else          basis_sig .push_back(best_g);

        // Recompute energy with accepted function
        build_full(basis_bare, basis_sig,
                   jac_bare, jac_sig,
                   H, Omega, S_sigma, m_sigma,
                   H_cur, N_cur);
        long double E_cur = solve(H_cur, N_cur);

        std::cerr << "  slot=" << std::setw(3) << slot
                  << "  [" << sec << "]"
                  << "  n_bare=" << std::setw(2) << basis_bare.size()
                  << "  n_sig=" << std::setw(2) << basis_sig.size()
                  << "  valid=" << n_valid << "/" << n_cand
                  << "  E=" << std::setw(10) << E_cur << " MeV\n";
    }

    long double E_sel = solve(H_cur, N_cur);
    std::cerr << "\n[deu] After selection: E=" << E_sel << " MeV\n\n";

    // ----------------------------------------------------------
    // 4. Phase 2 — Refining cycles
    // ----------------------------------------------------------
    std::cerr << "=== Phase 2: Refining ===\n";
    long double E_best = E_sel;

    for (int cycle = 1; cycle <= n_refine; ++cycle) {
        long double E_start = E_best;

        for (size_t k = 0; k < basis_bare.size(); ++k) {
            gaus orig = basis_bare[k];
            gaus best_g = orig;
            long double best_E = E_best;

            for (int c = 0; c < n_cand; ++c) {
                basis_bare[k] = make_swave(jac_bare.dim(), A_min, A_max);
                build_full(basis_bare, basis_sig,
                           jac_bare, jac_sig,
                           H, Omega, S_sigma, m_sigma,
                           H_cur, N_cur);
                long double E = solve(H_cur, N_cur);
                if (!std::isnan(E) && E < best_E) {
                    best_E = E; best_g = basis_bare[k];
                }
            }
            basis_bare[k] = best_g;
            E_best = best_E;
        }

        for (size_t k = 0; k < basis_sig.size(); ++k) {
            gaus orig = basis_sig[k];
            gaus best_g = orig;
            long double best_E = E_best;

            for (int c = 0; c < n_cand; ++c) {
                basis_sig[k] = make_swave(jac_sig.dim(), A_min, A_max);
                build_full(basis_bare, basis_sig,
                           jac_bare, jac_sig,
                           H, Omega, S_sigma, m_sigma,
                           H_cur, N_cur);
                long double E = solve(H_cur, N_cur);
                if (!std::isnan(E) && E < best_E) {
                    best_E = E; best_g = basis_sig[k];
                }
            }
            basis_sig[k] = best_g;
            E_best = best_E;
        }

        std::cerr << "  cycle=" << cycle
                  << "  E=" << std::setw(10) << E_best
                  << " MeV   delta=" << (E_start - E_best) << " MeV\n";
    }

    // ----------------------------------------------------------
    // 5. Result
    // ----------------------------------------------------------
    build_full(basis_bare, basis_sig,
               jac_bare, jac_sig,
               H, Omega, S_sigma, m_sigma,
               H_cur, N_cur);
    E_best = solve(H_cur, N_cur);

    std::cerr << "\n[deu] Done.\n";

    std::cout << "\n==========================================\n";
    std::cout << "  Target  :  -2.225000 MeV\n";
    std::cout << "  Result  : " << std::setw(10) << E_best << " MeV\n";
    std::cout << "  S_sigma :  " << S_sigma << " MeV\n";
    std::cout << "  b_sigma :  " << b_sigma << " fm\n";
    std::cout << "  n_bare  :  " << basis_bare.size() << "\n";
    std::cout << "  n_sig   :  " << basis_sig.size() << "\n";
    std::cout << "==========================================\n";

    return 0;
}