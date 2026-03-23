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
// 1. Configuration & Parser
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    ld S = 16;          // Final/Target physical coupling [MeV] (if not annealing)
    ld b = 1.4;           // Form-factor range [fm]
    int K_max = 5;       // Basis size
    int N_trial = 50;     // Stochastic trials per step
    int refine_every = 5; // Periodic refinement (0 during annealing build)
    int N_refine_trial = 20;
    ld b0 = 1.4;          // Gaussian range [fm]
    ld s_max = 0.5;       // Shift bound [fm^-1]
    uint64_t seed = 0;    // RNG Seed
    bool verbose = true;  // Print SVM-loop
};

void apply_arg(Config& cfg, const std::string& key, const std::string& val) {
    if      (key == "S")              cfg.S = std::stold(val);
    else if (key == "b")              cfg.b = std::stold(val);
    else if (key == "K_max")          cfg.K_max = std::stoi(val);
    else if (key == "N_trial")        cfg.N_trial = std::stoi(val);
    else if (key == "refine_every")   cfg.refine_every = std::stoi(val);
    else if (key == "N_refine_trial") cfg.N_refine_trial = std::stoi(val);
    else if (key == "b0")             cfg.b0 = std::stold(val);
    else if (key == "s_max")          cfg.s_max = std::stold(val);
    else if (key == "seed")           cfg.seed = std::stoull(val);
    else if (key == "verbose")        cfg.verbose = (val == "true" || val == "1");
}

Config parse_config(int argc, char* argv[]) {
    Config cfg;
    
    // 1. Attempt to read from run.cfg if it exists
    std::ifstream file("run.cfg");
    std::string line;
    while (std::getline(file, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash); 
        
        auto eq = line.find('=');
        if (eq != std::string::npos) {
            std::string key = line.substr(0, eq);
            std::string val = line.substr(eq + 1);
            key.erase(remove_if(key.begin(), key.end(), isspace), key.end());
            val.erase(remove_if(val.begin(), val.end(), isspace), val.end());
            if (!key.empty() && !val.empty()) apply_arg(cfg, key, val);
        }
    }

    // 2. Command line overrides
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-verbose") { cfg.verbose = true;  continue; }
        if (arg == "-quiet")   { cfg.verbose = false; continue; }
        if (arg.rfind("--", 0) == 0 && i + 1 < argc) { 
            apply_arg(cfg, arg.substr(2), argv[i+1]);
            i++; 
        }
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Channel Setup (14-channel system)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Channel> build_pion_channels(const JacobiSystem& sys) {
    constexpr ld m_pi0   = ld{134.977L};
    constexpr ld m_pi_pm = ld{139.570L};
    constexpr ld c1 =  ld{1};
    const     ld c2 =  std::sqrt(ld{2});
    const     ld c3 =  std::sqrt(ld{2});
    constexpr ld c4 = -ld{1};

    rvec w_pp = sys.w_meson_proton();
    rvec w_pn = sys.w_meson_neutron();

    auto make = [&](int idx, ld mp, ld iso, const rvec& w, SpinType st) {
        Channel ch; ch.index = idx; ch.is_bare = false; ch.pion_mass = mp;
        ch.iso_coeff = iso; ch.w_piN = w; ch.spin_type = st; ch.dim = sys.N - 1;
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

// ─────────────────────────────────────────────────────────────────────────────
// 3. Execution & Output
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg = parse_config(argc, argv);
    
    std::cout << "========================================================\n";
    std::cout << "  Deuteron 9-Channel SVM (Classical vs Relativistic)\n";
    std::cout << "========================================================\n";
    std::cout << "[Parameters]\n";
    std::cout << "  K_max        = " << cfg.K_max << "\n";
    std::cout << "  b (Form Fac) = " << cfg.b << " fm\n";
    std::cout << "  b0 (Gauss)   = " << cfg.b0 << " fm\n";
    std::cout << "  s_max        = " << cfg.s_max << " fm^-1\n";
    std::cout << "  Annealing    = " << (cfg.do_annealing ? "ON" : "OFF") << "\n";
    if (cfg.do_annealing) {
        std::cout << "    S_build    = " << cfg.S_build << " MeV\n";
        std::cout << "    E_target   = " << cfg.E_target << " MeV\n";
        std::cout << "    Relax Swps = " << cfg.relax_sweeps << "\n";
    } else {
        std::cout << "  S (Coupling) = " << cfg.S << " MeV\n";
    }
    std::cout << "========================================================\n\n";

    JacobiSystem sys({938.272, 939.565, 139.570}, {"p", "n", "pi"});
    auto channels = build_pion_channels(sys);

    SvmParams params;
    params.K_max = cfg.K_max;
    params.N_trial = cfg.N_trial;
    params.b0 = cfg.b0;
    params.s_max = cfg.s_max;
    params.b_ff = cfg.b;
    params.S_coupling = cfg.S;
    params.refine_every = cfg.refine_every;
    params.N_refine_trial = cfg.N_refine_trial;
    params.verbose = cfg.verbose;
    
    params.do_annealing = cfg.do_annealing;
    params.S_build = cfg.S_build;
    params.E_target = cfg.E_target;
    params.relax_sweeps = cfg.relax_sweeps;

    // Force phase 1 to be Classical
    params.relativistic = false;

#ifdef _OPENMP
    omp_set_max_active_levels(1);
#endif

    std::random_device rd;
    uint64_t seed = (cfg.seed == 0) ? rd() : cfg.seed;
    std::mt19937 rng(seed); 

    // ── 1. Classical Run (Annealing workflow)
    std::cout << "Running Classical SVM (with Annealing)...\n";
    auto t0 = std::chrono::steady_clock::now();
    SvmState state_cla = run_svm(sys, channels, params, rng);
    double t_cla = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "Classical Run Done (" << std::setprecision(1) << std::fixed << t_cla << " s)\n\n";

    ld e_cla = state_cla.E0;
    ld tuned_S = state_cla.S_final;

    // ── 2. Relativistic Evaluation
    std::cout << "Evaluating Relativistic Shift on locked wavefunction...\n";
    t0 = std::chrono::steady_clock::now();
    
    // Build Relativistic Hamiltonian using the exact same basis and tuned S
    HamiltonianBuilder hb_rel(sys, channels, state_cla.basis_bare, state_cla.basis_dressed, 
                              params.b_ff, tuned_S, true); // true = relativistic
    cmat H_rel = hb_rel.build_H();
    
    // Evaluate against the same overlap matrix N
    ld e_rel = solve_gevp(H_rel, state_cla.N);
    double t_rel = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "Relativistic Eval Done (" << std::setprecision(1) << std::fixed << t_rel << " s)\n\n";

    // ── 3. Deviation Calculations
    ld delta_E = e_rel - e_cla;
    ld rel_diff = (std::abs(e_cla) > 1e-12) ? std::abs(delta_E / e_cla) * 100.0 : 0.0;

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