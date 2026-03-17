//
// main.cc — Streamlined Deuteron SVM (Energy & Relativistic Deviation)
//
// Compile: g++ -std=c++17 -O2 -fopenmp -o main main.cc
// Usage:   ./main --config run.cfg --S 20.0 --b 1.4
//

#include "qm/matrix.h"
#include "qm/jacobi.h"
#include "qm/gaussian.h"
#include "qm/hamiltonian.h"
#include "qm/solver.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>

using namespace qm;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Simplified Configuration & Parser
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    ld S = 20.0;          // Coupling strength [MeV]
    ld b = 1.4;           // Form-factor range [fm]
    int K_max = 25;       // Basis size
    int N_trial = 50;     // Stochastic trials per step
    ld b0 = 1.4;          // Gaussian range [fm]
    ld s_max = 0.1;       // Shift bound [fm^-1]
    uint64_t seed = 0;   // RNG Seed
    bool verbose = true;       // Print SVM-loop
};

void apply_arg(Config& cfg, const std::string& key, const std::string& val) {
    if      (key == "S")       cfg.S = std::stold(val);
    else if (key == "b")       cfg.b = std::stold(val);
    else if (key == "K_max")   cfg.K_max = std::stoi(val);
    else if (key == "N_trial") cfg.N_trial = std::stoi(val);
    else if (key == "b0")      cfg.b0 = std::stold(val);
    else if (key == "s_max")   cfg.s_max = std::stold(val);
    else if (key == "seed")    cfg.seed = std::stoull(val);
    else if (key == "verbose") cfg.verbose = (val == "true" || val == "1");
}

Config parse_config(int argc, char* argv[]) {
    Config cfg;
    
    // 1. Attempt to read from run.cfg if it exists
    std::ifstream file("run.cfg");
    std::string line;
    while (std::getline(file, line)) {
        // Strip comments FIRST
        auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash); 
        }
        
        // THEN look for the equals sign on the clean line
        auto eq = line.find('=');
        if (eq != std::string::npos) {
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            
            // Remove all spaces
            key.erase(remove_if(key.begin(), key.end(), isspace), key.end());
            val.erase(remove_if(val.begin(), val.end(), isspace), val.end());
            
            if (!key.empty() && !val.empty()) apply_arg(cfg, key, val);
        }
    }

    // 2. Command line overrides
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Handle standalone boolean flags immediately
        if (arg == "-verbose") { cfg.verbose = true;  continue; }
        if (arg == "-quiet")   { cfg.verbose = false; continue; }
        
        // Handle key-value pairs (e.g., --S 25.0)
        if (arg.rfind("-", 0) == 0 && i + 1 < argc) { 
            apply_arg(cfg, arg.substr(2), argv[i+1]);
            i++; // Skip the value since we just consumed it
        }
    }
    return cfg;

}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Channel Setup (9-channel system)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Channel> build_pion_channels(const JacobiSystem& sys) {
    // Standard setup matching your physical model
    rvec w_pp = sys.w_meson_proton();
    rvec w_pn = sys.w_meson_neutron();

    auto make = [&](int idx, ld m_pi, ld iso, const rvec& w, SpinType st) {
        Channel ch;
        ch.index = idx; ch.is_bare = false; ch.pion_mass = m_pi;
        ch.iso_coeff = iso; ch.w_piN = w; ch.spin_type = st; ch.dim = sys.N - 1;
        return ch;
    };

    std::vector<Channel> ch(9);
    ch[0].index = 0; ch[0].is_bare = true; ch[0].pion_mass = 0;
    ch[0].iso_coeff = 0; ch[0].w_piN = rvec(sys.N - 1);
    ch[0].spin_type = SpinType::NO_FLIP; ch[0].dim = 1;

    ch[1] = make(1, 134.977, +1.0,           w_pp, SpinType::NO_FLIP);
    ch[2] = make(2, 139.570, +std::sqrt(2.), w_pp, SpinType::NO_FLIP);
    ch[3] = make(3, 139.570, +std::sqrt(2.), w_pn, SpinType::NO_FLIP);
    ch[4] = make(4, 134.977, -1.0,           w_pn, SpinType::NO_FLIP);
    ch[5] = make(5, 134.977, +1.0,           w_pp, SpinType::SPIN_FLIP);
    ch[6] = make(6, 139.570, +std::sqrt(2.), w_pp, SpinType::SPIN_FLIP);
    ch[7] = make(7, 139.570, +std::sqrt(2.), w_pn, SpinType::SPIN_FLIP);
    ch[8] = make(8, 134.977, -1.0,           w_pn, SpinType::SPIN_FLIP);
    return ch;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Execution & Output
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg = parse_config(argc, argv);
    
    std::cout << "========================================================\n";
    std::cout << "  Deuteron 9-Channel SVM (Classical vs Relativistic)\n";
    std::cout << "========================================================\n";
    std::cout << "Params: S = " << cfg.S << " MeV, b = " << cfg.b << " fm, K = " << cfg.K_max << "\n\n";

    JacobiSystem sys({938.272, 939.565, 139.570}, {"p", "n", "pi"});
    auto channels = build_pion_channels(sys);

    SvmParams params;
    params.K_max = cfg.K_max;
    params.N_trial = cfg.N_trial;
    params.b0 = cfg.b0;
    params.s_max = cfg.s_max;
    params.b_ff = cfg.b;
    params.S_coupling = cfg.S;
    params.refine_every = 5; // Fixed refinement rate
    params.verbose = true;

    // ── Pre-warm Gaussian Integrator 
    {
        Gaussian g(1.0, 0.0); GaussianPair gp(g, g); rvec c(1); c[0] = 1.0;
        ke_relativistic(gp, KineticParams(gp, c), sys.mu[0]);
    }

    // ── Evaluate the seed
    std::random_device rd;
    uint64_t seed = (cfg.seed == 0) ? rd() : cfg.seed;

    // ── 1. Classical Run
    std::cout << "Running Classical SVM..." << std::flush;
    std::mt19937 rng_cla(seed); // Ensure exact same seed
    params.relativistic = false;
    
    auto t0 = std::chrono::steady_clock::now();
    SvmState state_cla = run_svm(sys, channels, params, rng_cla);
    double t_cla = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << " Done (" << std::setprecision(1) << std::fixed << t_cla << " s)\n\n";

    // ── 2. Relativistic Run
    std::cout << "Running Relativistic SVM..." << std::flush;
    std::mt19937 rng_rel(seed); // Ensure exact same seed again
    params.relativistic = true;
    
    t0 = std::chrono::steady_clock::now();
    SvmState state_rel = run_svm(sys, channels, params, rng_rel);
    double t_rel = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << " Done (" << std::setprecision(1) << std::fixed << t_rel << " s)\n\n";

    // ── 3. Deviation Calculations
    ld e_cla = state_cla.E0;
    ld e_rel = state_rel.E0;
    ld delta_E = std::abs(e_rel - e_cla);
    ld rel_diff = std::abs(delta_E / e_cla) * 100.0;

    // ── 4. Final Data Table
    std::cout << "========================================================\n";
    std::cout << std::left << std::setw(20) << "Metric" 
              << std::setw(15) << "Classical" 
              << std::setw(15) << "Relativistic" << "\n";
    std::cout << "--------------------------------------------------------\n";
    
    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::left << std::setw(20) << "Energy E0 [MeV]" 
              << std::setw(15) << e_cla 
              << std::setw(15) << e_rel << "\n";
              
    std::cout << std::left << std::setw(20) << "Charge Rad [fm]" 
              << std::setw(15) << "TODO" // Placeholder for your r.m.s matrix implementation
              << std::setw(15) << "TODO" << "\n";
              
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Absolute Relativistic Shift: " << std::showpos << delta_E << " MeV\n" << std::noshowpos;
    std::cout << "Relative Shift Percentage:   " << std::setprecision(2) << rel_diff << " %\n";
    std::cout << "========================================================\n";

    return 0;
}