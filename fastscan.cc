//
// fastscan_1d.cc  —  Independent SVM evaluations across S
//
// Compile: g++ -std=c++17 -O2 -fopenmp -o fastscan_1d fastscan_1d.cc
// Run:     ./fastscan_1d
//

#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"
#include "qm/solver.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

    // 2. Base SVM Parameters
    ld b_target = 1.4;
    SvmParams params;
    params.K_max = 10; // Lower K slightly so 120 runs don't take all day
    params.N_trial = 30;
    params.refine_every = params.K_max; // Refine once at the end
    params.b_ff = b_target;
    params.s_max = 0.5;
    params.relativistic = false;
    params.verbose = true; // Turn off inner loop printing so it doesn't flood terminal

    std::cout << "========================================================\n";
    std::cout << "  Full SVM 1D S-Scan (Fixed b = 1.4 fm)\n";
    std::cout << "========================================================\n";
    std::cout << "Running independent optimization from scratch for every S...\n\n";
    
    // 3. Setup CSV
    std::string filename = "data_1d.csv";
    std::ofstream out(filename);
    out << "S_MeV,b_fm,E0_MeV\n";
    out << std::fixed << std::setprecision(6);

    auto t1 = std::chrono::steady_clock::now();

    // 4. The Independent Scan Loop
    for (ld S = 12.0; S <= 20.01; S += 0.1) {
        
        params.S_coupling = S;

        // Reset the RNG seed for each run so they are compared fairly
        std::random_device rd;
        std::mt19937 rng(rd());

        // Run the FULL SVM loop from K=1 to K_max
        SvmState state = run_svm(sys, channels, params, rng);
        
        ld E0 = state.E0;

        // Write to CSV
        out << S << "," << b_target << "," << E0 << "\n";
        
        // Terminal Progress
        std::cout << "\r  [Scanning] S = " << std::fixed << std::setprecision(1) << S 
                  << " MeV  |  Optimized E0 = " << std::setprecision(4) << E0 
                  << " MeV    " << std::flush;
    }

    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::cout << "\n\nFull independent scan finished in " << std::fixed << std::setprecision(2) << elapsed << " s!\n";
    std::cout << "Data saved to " << filename << "\n";
    
    return 0;
}