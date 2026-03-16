//
// main.cc  —  Deuteron ground state via explicit pion-nucleon coupling
//
// Thesis: "Pion-Nucleon Coupling in the Deuteron" (Mikkel Moth Billing)
//
// System:
//   Three particles:  proton (0),  neutron (1),  pion (2)
//   Jacobi coordinates:
//     x₀ = r_p − r_n          (nucleon–nucleon relative coordinate,  μ₀ = m_p m_n / (m_p+m_n))
//     x₁ = r_π − r_{pn CM}    (pion relative to NN centre of mass,   μ₁ = reduced)
//
// Nine-channel Hamiltonian  (thesis §3–4):
//   ch[0]  bare pn              (dim=1)
//   ch[1]  pnπ⁰  NO_FLIP        proton  emits π⁰,  C_iso=+1,   σ_z
//   ch[2]  nnπ⁺  NO_FLIP        proton  emits π⁺,  C_iso=+√2,  σ_z
//   ch[3]  ppπ⁻  NO_FLIP        neutron emits π⁻,  C_iso=+√2,  σ_z
//   ch[4]  npπ⁰  NO_FLIP        neutron emits π⁰,  C_iso=−1,   σ_z
//   ch[5]  pnπ⁰  SPIN_FLIP      proton  emits π⁰,  C_iso=+1,   σ_+
//   ch[6]  nnπ⁺  SPIN_FLIP      proton  emits π⁺,  C_iso=+√2,  σ_+
//   ch[7]  ppπ⁻  SPIN_FLIP      neutron emits π⁻,  C_iso=+√2,  σ_+
//   ch[8]  npπ⁰  SPIN_FLIP      neutron emits π⁰,  C_iso=−1,   σ_+
//
// This program runs the SVM twice with identical parameters —
// once with classical KE and once with relativistic KE — and
// prints a side-by-side comparison.  Convergence histories for
// both runs are written to a gnuplot-compatible .dat file.
//
// ── Parameter sources (lowest to highest priority) ───────────────────────────
//   1. Built-in defaults (see Config struct below)
//   2. Config file  (--config run.cfg)
//      Lines:  key = value   (comments with #,  blank lines ignored)
//   3. Command-line flags (override everything)
//
// ── Usage ─────────────────────────────────────────────────────────────────────
//   ./main [--config file.cfg] [options]
//
// ── Options ───────────────────────────────────────────────────────────────────
//   Physics:
//     --S value          Coupling strength S   [MeV]    (default: 20.0)
//     --b value          Form-factor range  b  [fm]     (default: 1.4)
//
//   SVM:
//     --K_max N          Basis size                     (default: 25)
//     --N_trial N        Trial candidates per SVM step  (default: 50)
//     --refine_every N   Refinement period  (0=off)     (default: 10)
//     --N_refine N       Candidates per refinement      (default: 20)
//     --b0 value         Gaussian length scale  [fm]    (default: 1.4)
//     --s_max value      Shift vector bound [fm⁻¹]      (default: 0.0)
//     --seed N           RNG seed                       (default: 42)
//
//   Output:
//     --output file      Convergence data file          (default: convergence.dat)
//     --no_classical     Skip the classical KE run
//     --no_relativistic  Skip the relativistic KE run
//
// ── Config file example (run.cfg) ─────────────────────────────────────────────
//   # Deuteron run parameters
//   S            = 20.0
//   b            = 1.4
//   K_max        = 30
//   N_trial      = 60
//   refine_every = 10
//   b0           = 1.4
//   seed         = 42
//   output       = convergence.dat
//
// ── Build ─────────────────────────────────────────────────────────────────────
//   g++ -std=c++17 -O2 -fopenmp -o deuteron main.cc
//
// ── Expected result ───────────────────────────────────────────────────────────
//   E₀ ≈ −2.2 MeV  (deuteron binding energy)
//   The relativistic correction shifts E₀ by a few hundred keV.
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
#include <sstream>
#include <string>
#include <vector>

using namespace qm;

// ─────────────────────────────────────────────────────────────────────────────
// Config  —  all run parameters with defaults
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    // Physics
    ld   S            = ld{20.0L};   // MeV  coupling strength
    ld   b            = ld{1.4L};    // fm   form-factor range

    // SVM
    int  K_max        = 25;
    int  N_trial      = 50;
    int  refine_every = 5;
    int  N_refine     = 20;
    ld   b0           = ld{1.4L};    // fm   Gaussian length scale
    ld   s_max        = ld{0.1L};    // fm⁻¹ shift vector bound
    uint64_t seed     = 42;

    // Output
    bool run_classical       = true;
    bool run_relativistic    = true;
};


// ─────────────────────────────────────────────────────────────────────────────
// apply_key_value  —  set one Config field from a key/value pair
// ─────────────────────────────────────────────────────────────────────────────
bool apply_key_value(Config& cfg, const std::string& key, const std::string& val)
{
    auto f  = [&]{ return static_cast<ld>(std::stold(val)); };
    auto i  = [&]{ return std::stoi(val); };
    auto u  = [&]{ return static_cast<uint64_t>(std::stoull(val)); };

    if      (key == "S"            || key == "S_coupling") { cfg.S            = f(); }
    else if (key == "b"            || key == "b_ff")       { cfg.b            = f(); }
    else if (key == "K_max")                               { cfg.K_max        = i(); }
    else if (key == "N_trial")                             { cfg.N_trial      = i(); }
    else if (key == "refine_every")                        { cfg.refine_every = i(); }
    else if (key == "N_refine"     || key == "N_refine_trial") { cfg.N_refine = i(); }
    else if (key == "b0")                                  { cfg.b0           = f(); }
    else if (key == "s_max")                               { cfg.s_max        = f(); }
    else if (key == "seed")                                { cfg.seed         = u(); }
    else return false;   // unknown key
    return true;
}


// ─────────────────────────────────────────────────────────────────────────────
// load_config_file  —  read key = value pairs from a text file
//
// Format:
//   # comment lines and blank lines are ignored
//   key = value
//   key=value          (spaces around = are optional)
// ─────────────────────────────────────────────────────────────────────────────
void load_config_file(Config& cfg, const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open config file '" << path << "'\n";
        std::exit(1);
    }

    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        lineno++;
        // Strip comments
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);

        // Trim whitespace
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r\n");
            size_t b = s.find_last_not_of(" \t\r\n");
            return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
        };
        line = trim(line);
        if (line.empty()) continue;

        // Split on '='
        auto eq = line.find('=');
        if (eq == std::string::npos) {
            std::cerr << "Warning: config line " << lineno
                      << " has no '=' — skipping: " << line << "\n";
            continue;
        }
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        if (!apply_key_value(cfg, key, val))
            std::cerr << "Warning: unknown config key '" << key
                      << "' on line " << lineno << "\n";
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// parse_args  —  command-line argument parser
//
// Config file is loaded first (if --config is present), then individual
// command-line flags override whatever the file set.
// ─────────────────────────────────────────────────────────────────────────────
Config parse_args(int argc, char* argv[])
{
    Config cfg;
 
    // Auto-load run.cfg from the current directory if it exists.
    // This means you never need to pass --config explicitly for the default file.
    // Command-line flags and an explicit --config always override it.
    {
        std::ifstream probe("run.cfg");
        if (probe.is_open()) {
            probe.close();
            load_config_file(cfg, "run.cfg");
        }
    }
 
    // First pass: find explicit --config and load it (overrides run.cfg)
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--config")
            load_config_file(cfg, argv[i + 1]);
    }
 
    // Second pass: all command-line flags (override config file)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
 
        // Skip --config (already handled)
        if (a == "--config") { i++; continue; }
 
        // Boolean flags
        if (a == "--no_classical")    { cfg.run_classical    = false; continue; }
        if (a == "--no_relativistic") { cfg.run_relativistic = false; continue; }
 
        // Key–value flags
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing value after '" << a << "'\n";
                std::exit(1);
            }
            return argv[++i];
        };
 
        // Try stripping the leading '--' and treating as a key=value pair
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            std::string key = a.substr(2);
            std::string val = next();
            if (!apply_key_value(cfg, key, val)) {
                std::cerr << "Error: unknown option '" << a << "'\n";
                std::exit(1);
            }
        } else {
            std::cerr << "Error: unrecognised argument '" << a << "'\n";
            std::exit(1);
        }
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_pion_channels  —  9-channel pion-deuterium descriptor
//
// Channel layout (thesis §3–4):
//   ch[0]  bare pn          (dim=1)
//   ch[1]  pnπ⁰  NO_FLIP    proton  emits π⁰,  C_iso=+1
//   ch[2]  nnπ⁺  NO_FLIP    proton  emits π⁺,  C_iso=+√2
//   ch[3]  ppπ⁻  NO_FLIP    neutron emits π⁻,  C_iso=+√2
//   ch[4]  npπ⁰  NO_FLIP    neutron emits π⁰,  C_iso=−1
//   ch[5]  pnπ⁰  SPIN_FLIP  proton  emits π⁰,  C_iso=+1
//   ch[6]  nnπ⁺  SPIN_FLIP  proton  emits π⁺,  C_iso=+√2
//   ch[7]  ppπ⁻  SPIN_FLIP  neutron emits π⁻,  C_iso=+√2
//   ch[8]  npπ⁰  SPIN_FLIP  neutron emits π⁰,  C_iso=−1
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Channel> build_pion_channels(const JacobiSystem& sys)
{
    constexpr ld m_pi0   = ld{134.977L};
    constexpr ld m_pi_pm = ld{139.570L};

    constexpr ld c1 = ld{1};
    const     ld c2 = std::sqrt(ld{2});
    const     ld c3 = std::sqrt(ld{2});
    constexpr ld c4 = ld{-1};

    rvec w_pp = sys.w_meson_proton();
    rvec w_pn = sys.w_meson_neutron();

    auto make_dressed = [&](int idx, ld m_pion, ld iso, const rvec& w, SpinType st) {
        Channel ch;
        ch.index     = idx;
        ch.is_bare   = false;
        ch.pion_mass = m_pion;
        ch.iso_coeff = iso;
        ch.w_piN     = w;
        ch.spin_type = st;
        ch.dim       = sys.N - 1;
        return ch;
    };

    std::vector<Channel> channels(9);
    channels[0].index     = 0;
    channels[0].is_bare   = true;
    channels[0].pion_mass = ld{0};
    channels[0].iso_coeff = ld{0};
    channels[0].w_piN     = rvec(sys.N - 1);
    channels[0].spin_type = SpinType::NO_FLIP;
    channels[0].dim       = 1;

    channels[1] = make_dressed(1, m_pi0,   +c1, w_pp, SpinType::NO_FLIP);
    channels[2] = make_dressed(2, m_pi_pm, +c2, w_pp, SpinType::NO_FLIP);
    channels[3] = make_dressed(3, m_pi_pm, +c3, w_pn, SpinType::NO_FLIP);
    channels[4] = make_dressed(4, m_pi0,   +c4, w_pn, SpinType::NO_FLIP);
    channels[5] = make_dressed(5, m_pi0,   +c1, w_pp, SpinType::SPIN_FLIP);
    channels[6] = make_dressed(6, m_pi_pm, +c2, w_pp, SpinType::SPIN_FLIP);
    channels[7] = make_dressed(7, m_pi_pm, +c3, w_pn, SpinType::SPIN_FLIP);
    channels[8] = make_dressed(8, m_pi0,   +c4, w_pn, SpinType::SPIN_FLIP);

    return channels;
}


// ─────────────────────────────────────────────────────────────────────────────
// print_separator / print_header helpers
// ─────────────────────────────────────────────────────────────────────────────
void sep(char c = '-', int width = 62) {
    std::cout << std::string(width, c) << "\n";
}

void box(const std::string& title) {
    int w = 60;
    std::cout << "+" << std::string(w, '=') << "+\n";
    int pad = (w - (int)title.size()) / 2;
    std::cout << "|" << std::string(pad, ' ') << title
              << std::string(w - pad - (int)title.size(), ' ') << "|\n";
    std::cout << "+" << std::string(w, '=') << "+\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// run_and_record  —  run SVM and return the full SvmState (with energy history)
// ─────────────────────────────────────────────────────────────────────────────
SvmState run_and_record(const JacobiSystem&         sys,
                         const std::vector<Channel>& channels,
                         const Config&               cfg,
                         bool                        relativistic)
{
    SvmParams params;
    params.K_max          = cfg.K_max;
    params.N_trial        = cfg.N_trial;
    params.refine_every   = cfg.refine_every;
    params.N_refine_trial = cfg.N_refine;
    params.b0             = cfg.b0;
    params.s_max          = cfg.s_max;
    params.b_ff           = cfg.b;
    params.S_coupling     = cfg.S;
    params.relativistic   = relativistic;
    params.verbose        = true;

    std::mt19937 rng(static_cast<uint32_t>(cfg.seed));
    return run_svm(sys, channels, params, rng);
}


// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    Config cfg = parse_args(argc, argv);

    std::cout << "\n";
    box("Deuteron  —  pion-nucleon coupling  (9-channel SVM)");
    std::cout << "\n";

    // ── Physical setup ────────────────────────────────────────────────────────
    const ld m_proton  = ld{938.272046L};
    const ld m_neutron = ld{939.565379L};
    const ld m_pion    = ld{139.570L};     // representative for reduced masses

    JacobiSystem sys({m_proton, m_neutron, m_pion}, {"p", "n", "pi"});
    auto channels = build_pion_channels(sys);

    // ── Print system info ─────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(4);
    sep();
    std::cout << "Physical system:\n";
    std::cout << "  m_p  = " << m_proton  << " MeV\n";
    std::cout << "  m_n  = " << m_neutron << " MeV\n";
    std::cout << "  m_π  = " << m_pion    << " MeV  (representative)\n";
    std::cout << "  μ₀ (p-n)    = " << sys.mu[0] << " MeV\n";
    std::cout << "  μ₁ (π-NN)   = " << sys.mu[1] << " MeV\n";
    sep();
    std::cout << "Coupling parameters:\n";
    std::cout << "  S  = " << cfg.S << " MeV\n";
    std::cout << "  b  = " << cfg.b << " fm\n";
    sep();
    std::cout << "SVM parameters:\n";
    std::cout << "  K_max        = " << cfg.K_max        << "\n";
    std::cout << "  N_trial      = " << cfg.N_trial      << "\n";
    std::cout << "  refine_every = " << cfg.refine_every << "\n";
    std::cout << "  N_refine     = " << cfg.N_refine     << "\n";
    std::cout << "  b0           = " << cfg.b0    << " fm\n";
    std::cout << "  s_max        = " << cfg.s_max << " fm⁻¹\n";
    std::cout << "  seed         = " << cfg.seed  << "\n";
    sep();
    std::cout << "\n";

    // Pre-warm the Gauss-Legendre cache before any threaded work
    {
        Gaussian g_warm(ld{1.0L}, ld{0.0L});
        GaussianPair gp_warm(g_warm, g_warm);
        rvec c1(1); c1[0] = ld{1};
        KineticParams kp_warm(gp_warm, c1);
        (void) ke_relativistic(gp_warm, kp_warm, sys.mu[0]);
    }

    // ── Classical KE run ─────────────────────────────────────────────────────
    SvmState result_cla;
    double   time_cla = 0.0;

    if (cfg.run_classical) {
        std::cout << "\n";
        box("Run 1 — Classical kinetic energy");
        std::cout << "\n";

        auto t0 = std::chrono::steady_clock::now();
        result_cla = run_and_record(sys, channels, cfg, /*relativistic=*/false);
        time_cla   = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();

        std::cout << "\nClassical run finished in " << std::fixed
                  << std::setprecision(1) << time_cla << " s\n";
    }

    // ── Relativistic KE run ──────────────────────────────────────────────────
    SvmState result_rel;
    double   time_rel = 0.0;

    if (cfg.run_relativistic) {
        std::cout << "\n";
        box("Run 2 — Relativistic kinetic energy");
        std::cout << "\n";

        auto t0 = std::chrono::steady_clock::now();
        result_rel = run_and_record(sys, channels, cfg, /*relativistic=*/true);
        time_rel   = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count();

        std::cout << "\nRelativistic run finished in " << std::fixed
                  << std::setprecision(1) << time_rel << " s\n";
    }

    // ── Comparison summary ────────────────────────────────────────────────────
    std::cout << "\n";
    box("Results & Comparison");
    std::cout << "\n";

    const ld E_target = ld{-2.2L};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Parameter:        S = " << cfg.S << " MeV,  b = " << cfg.b << " fm\n\n";

    // Table header
    std::cout << "  " << std::left
              << std::setw(22) << "Quantity"
              << std::setw(18) << "Classical"
              << std::setw(18) << "Relativistic"
              << "\n";
    sep('-', 62);

    ld E_cla = cfg.run_classical    ? result_cla.E0 : std::numeric_limits<ld>::quiet_NaN();
    ld E_rel = cfg.run_relativistic ? result_rel.E0 : std::numeric_limits<ld>::quiet_NaN();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << std::left
              << std::setw(22) << "E0  [MeV]"
              << std::setw(18) << E_cla
              << std::setw(18) << E_rel
              << "\n";

    std::cout << "  " << std::left
              << std::setw(22) << "E0 - E_target [MeV]"
              << std::setw(18) << E_cla - E_target
              << std::setw(18) << E_rel - E_target
              << "\n";

    std::cout << "  " << std::left
              << std::setw(22) << "Basis size K"
              << std::setw(18) << (cfg.run_classical    ? (int)result_cla.K() : 0)
              << std::setw(18) << (cfg.run_relativistic ? (int)result_rel.K() : 0)
              << "\n";

    std::cout << "  " << std::left
              << std::setw(22) << "Wall time  [s]"
              << std::setw(18) << time_cla
              << std::setw(18) << time_rel
              << "\n";

    sep('=', 62);

    if (cfg.run_classical && cfg.run_relativistic) {
        ld delta = E_rel - E_cla;
        std::cout << "\n  Relativistic correction:  dE = E_rel - E_cla = "
                  << std::showpos << std::fixed << std::setprecision(4)
                  << delta << " MeV\n" << std::noshowpos;

        ld frac = (E_cla != ld{0}) ? std::abs(delta / E_cla) * ld{100} : ld{0};
        std::cout << "  Fractional shift:         |dE/E_cla| = "
                  << std::fixed << std::setprecision(2) << frac << " %\n";
    }

    std::cout << "\n  Target (deuteron):  E0 = " << E_target << " MeV\n";
    std::cout << "\n";

    return 0;
}