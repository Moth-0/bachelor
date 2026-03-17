//
// fastscan.cc  —  Lightning-Fast (S, b) parameter scan
//
// Compile: g++ -std=c++17 -O2 -fopenmp -o fastscan fastscan.cc
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
#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace qm;

// ─────────────────────────────────────────────────────────────────────────────
// build_pion_channels 
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
    auto channels = build_pion_channels(sys); 

    // 2. Reference Point & SVM Parameters
    ld S_ref = 15.0, b_ref = 1.4;
    SvmParams params;
    params.K_max = 40; 
    params.N_trial = 50;
    params.refine_every = 10;
    params.S_coupling = S_ref;
    params.b_ff = b_ref;
    params.s_max = 0.1;
    params.relativistic = false;

    std::cout << "Optimizing basis at S=" << S_ref << ", b=" << b_ref << "..." << std::endl;
    std::random_device rd;
    std::mt19937 rng(rd()); // Using fixed seed here for reproducibility, or rd()
    SvmState ref = run_svm(sys, channels, params, rng);

    // 3. Define the Grid for Heatmap
    int n_points = 20;
    ld S_min = 10.0, S_max = 20.0;
    ld b_min = 1.2, b_max = 1.6;
    ld E_clip = 5.0; // Clip positive unbound energies

    std::vector<ld> S_vals(n_points), b_vals(n_points);
    for (int i = 0; i < n_points; ++i) {
        S_vals[i] = S_min + i * (S_max - S_min) / (n_points - 1);
        b_vals[i] = b_min + i * (b_max - b_min) / (n_points - 1);
    }

    std::cout << "\nPre-computing constant matrices..." << std::endl;

    // ── Pre-compute the Constant Matrices (Outside the loops!) ────────────────
    // Build N and H_diag (setting S=0 so we only get Kinetic Energy and masses)
    HamiltonianBuilder hb_base(sys, channels, ref.basis_bare, ref.basis_dressed, 1.0, 0.0, params.relativistic);
    cmat N = hb_base.build_N();
    cmat H_diag = hb_base.build_H();
    size_t dim = H_diag.size1();

    // Pre-compute the Cholesky inversion of N 
    cmat L = N.cholesky();
    if (L.size1() == 0) {
        std::cerr << "Error: Reference basis is linearly dependent.\n";
        return 1;
    }
    cmat Linv = L.inverse_lower();
    cmat Linv_dag = Linv.adjoint();

    std::cout << "Starting lightning-fast grid scan..." << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    std::atomic<int> done{0};

    // Store results to write sequentially later
    std::vector<std::vector<ld>> results(n_points, std::vector<ld>(n_points, 0.0));

    // ── Outer Loop: Form-factor 'b' ───────────────────────────────────────────
    #pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < n_points; ib++) {
        ld b = b_vals[ib];

        // Build the Hamiltonian for this 'b', with S=1.0 to get the raw W matrix
        HamiltonianBuilder hb_b(sys, channels, ref.basis_bare, ref.basis_dressed, b, 1.0, params.relativistic);
        cmat H_b_S1 = hb_b.build_H();

        // Extract JUST the interaction block W(b) by subtracting the kinetic energy
        cmat W_raw = zeros<cld>(dim, dim);
        for(size_t r = 0; r < dim; ++r) {
            for(size_t c = 0; c < dim; ++c) {
                W_raw(r, c) = H_b_S1(r, c) - H_diag(r, c);
            }
        }

        // ── Inner Loop: Coupling 'S' ──────────────────────────────────────────
        for (int iS = 0; iS < n_points; iS++) {
            ld S = S_vals[iS];

            // 1. Assemble the full H instantly: H = H_diag + S * W_raw
            cmat H_final = zeros<cld>(dim, dim);
            for(size_t r = 0; r < dim; ++r) {
                for(size_t c = 0; c < dim; ++c) {
                    H_final(r, c) = H_diag(r, c) + cld{S, 0.0} * W_raw(r, c);
                }
            }

            // 2. Transform to standard Eigenvalue Problem H' = L⁻¹ H L⁻†
            cmat Hp = Linv * H_final * Linv_dag;
            
            // 3. Diagonalize H'
            jacobi_diag(Hp);

            // 4. Extract the ground state energy
            ld E0 = std::numeric_limits<ld>::max();
            for (size_t d = 0; d < dim; d++) {
                ld e = std::real(Hp(d, d));
                if (e < E0) E0 = e;
            }

            // Clip for plotting
            if (!std::isfinite(E0) || E0 > E_clip) E0 = E_clip;
            results[iS][ib] = E0; // Store based on S (outer) and b (inner)
        }

        // Progress
        int n_done = ++done;
        #pragma omp critical
        {
            std::cout << "\r Progress: " << (n_done * 100) / n_points << "%" << std::flush;
        }
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::cout << "\nGrid scan finished in " << std::fixed << std::setprecision(2) << elapsed << " s!\n";

    // ── Write output for Gnuplot ──────────────────────────────────────────────
    std::ofstream out("fastscan.dat");
    out << "# S [MeV]   b [fm]   E0 [MeV]\n";
    out << std::fixed << std::setprecision(6);

    for (int iS = 0; iS < n_points; ++iS) {
        for (int ib = 0; ib < n_points; ++ib) {
            out << S_vals[iS] << "  " << b_vals[ib] << "  " << results[iS][ib] << "\n";
        }
        out << "\n"; // Gnuplot block separator for pm3d
    }

    std::cout << "Data saved to fastscan.dat\n";
    return 0;
}