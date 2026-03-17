//
// fastscan.cc  —  Fast (S, b) parameter scan for the pion-deuterium system
//
// Strategy:
//   1. Run a full SVM optimisation at one reference point (S_ref, b_ref)
//      to obtain an optimised Gaussian basis.
//   2. For every (S, b) grid point, rebuild H and N using that FIXED basis
//      with the new coupling parameters, then solve the GEP directly.
//      No SVM trial loop is run for any other point.
//
// This is orders of magnitude faster than running a full SVM per grid point
// (as heatmap.cc does), at the cost of basis quality away from the reference.
// For mapping the landscape and locating the binding region it is excellent;
// for precise energies at specific points use heatmap.cc or main.cc.
//
// The grid evaluation in step 2 is embarrassingly parallel — each (S,b)
// point only needs a matrix build + eigensolve with the shared read-only basis.
//
// Usage:
//   ./fastscan [options]
//
// Options:
//   Reference point (where SVM is run):
//     --S_ref v        Coupling strength for basis optimisation  (default: 20.0)
//     --b_ref v        Form-factor range  for basis optimisation (default: 1.4)
//
//   SVM (only affects the single reference run):
//     --K_max N        Basis size                               (default: 25)
//     --N_trial N      Trial candidates per SVM step            (default: 50)
//     --refine_every N Refinement period (0=off)                (default: 10)
//     --N_refine N     Candidates per refinement step           (default: 20)
//     --b0 v           Gaussian length scale [fm]               (default: 1.4)
//     --s_max v        Shift vector bound [fm^-1]               (default: 0.0)
//     --seed N         RNG seed                                 (default: 42)
//     --relativistic   Use relativistic KE                      (default: classical)
//
//   Grid (evaluated with fixed basis):
//     --S_min v        (default: 5.0)
//     --S_max v        (default: 60.0)
//     --N_S n          (default: 20)
//     --b_min v        (default: 0.5)
//     --b_max v        (default: 4.0)
//     --N_b n          (default: 20)
//
//   Output:
//     --output file    (default: fastscan.dat)
//     --E_clip v       Clip unbound values  (default: 10.0)
//
// Build:
//   g++ -std=c++17 -O2 -fopenmp -o fastscan fastscan.cc
//

#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"
#include "qm/solver.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <omp.h>


using namespace qm;

// ─────────────────────────────────────────────────────────────────────────────
// build_pion_channels  (identical to main.cc / heatmap.cc)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Channel> build_pion_channels(const JacobiSystem& sys)
{
    constexpr ld m_pi0   = ld{134.977L};
    constexpr ld m_pi_pm = ld{139.570L};
    constexpr ld c1 =  ld{1};
    const     ld c2 =  std::sqrt(ld{2});
    const     ld c3 =  std::sqrt(ld{2});
    constexpr ld c4 = -ld{1};

    rvec w_pp = sys.w_meson_proton();
    rvec w_pn = sys.w_meson_neutron();

    auto make = [&](int idx, ld mp, ld iso, const rvec& w, SpinType st) {
        Channel ch;
        ch.index = idx; ch.is_bare = false; ch.pion_mass = mp;
        ch.iso_coeff = iso; ch.w_piN = w; ch.spin_type = st;
        ch.dim = sys.N - 1;
        return ch;
    };

    std::vector<Channel> ch(9);
    ch[0].index = 0; ch[0].is_bare = true; ch[0].pion_mass = ld{0};
    ch[0].iso_coeff = ld{0}; ch[0].w_piN = rvec(sys.N-1);
    ch[0].spin_type = SpinType::NO_FLIP; ch[0].dim = 1;

    ch[1] = make(1, m_pi0,   +c1, w_pp, SpinType::NO_FLIP);
    ch[2] = make(2, m_pi_pm, +c2, w_pp, SpinType::NO_FLIP);
    ch[3] = make(3, m_pi_pm, +c3, w_pn, SpinType::NO_FLIP);
    ch[4] = make(4, m_pi0,   +c4, w_pn, SpinType::NO_FLIP);

    ch[5] = make(5, m_pi0,   +c1, w_pp, SpinType::SPIN_FLIP);
    ch[6] = make(6, m_pi_pm, +c2, w_pp, SpinType::SPIN_FLIP);
    ch[7] = make(7, m_pi_pm, +c3, w_pn, SpinType::SPIN_FLIP);
    ch[8] = make(8, m_pi0,   +c4, w_pn, SpinType::SPIN_FLIP);
    return ch;
}

int main() {
    // 1. Physical Setup
    JacobiSystem sys({938.27, 939.57, 139.57}, {"p", "n", "pi"});
    auto channels = build_pion_channels(sys); // Assumes this is available in your headers or main.cc

    // 2. Reference Point & SVM Parameters
    ld S_ref = 18.0, b_ref = 1.4;
    SvmParams params;
    params.K_max = 20; 
    params.N_trial = 50;
    params.refine_every = 5;
    params.S_coupling = S_ref;
    params.b_ff = b_ref;
    params.relativistic = false;

    std::cout << "Optimizing basis at S=" << S_ref << ", b=" << b_ref << "..." << std::endl;
    std::random_device rd;
    std::mt19937 rng(rd());
    SvmState optimized_state = run_svm(sys, channels, params, rng);

    // 3. Define the Grid for Heatmap
    int n_points = 20;
    ld S_min = 10, S_max = 40.0;
    ld b_min = 0.8, b_max = 2.0;

    std::ofstream out("fastscan.dat");
    out << "# S [MeV]   b [fm]   E0 [MeV]" << std::endl;

    std::cout << "Starting fixed-basis scan..." << std::endl;

    for (int i = 0; i < n_points; ++i) {
        ld S = S_min + i * (S_max - S_min) / (n_points - 1);
        for (int j = 0; j < n_points; ++j) {
            ld b = b_min + j * (b_max - b_min) / (n_points - 1);

            // Directly build H and N using the FIXED basis from optimized_state
            HamiltonianBuilder hb(sys, channels, 
                                  optimized_state.basis_bare, 
                                  optimized_state.basis_dressed, 
                                  b, S, params.relativistic);
            
            cmat H = hb.build_H();
            cmat N = hb.build_N();
            ld E0 = solve_gevp(H, N);

            // Clip unbound/divergent results for cleaner plotting
            if (!std::isfinite(E0) || E0 > 5.0) E0 = 5.0;

            out << S << " " << b << " " << E0 << "\n";
        }
        out << "\n"; // Gnuplot block separator
        std::cout << "\r Progress: " << (i + 1) * 100 / n_points << "%" << std::flush;
    }

    std::cout << "Scan complete. Data saved to heatmap_data.dat" << std::endl;
    return 0;
}