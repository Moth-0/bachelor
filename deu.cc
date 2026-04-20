/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                  deu.cc - DEUTERON SVM GROUND STATE FINDER                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Main driver orchestrating the complete Stochastic Variational Method (SVM)   ║
║   to find the ground state energy of a deuteron with pion exchange coupling.   ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <omp.h>
#include <string>

#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "deuterium.h" 
#include "SVM.h"

using namespace qm;

// Performs iterative stochastic refinement using shrinking noise (Simulated Annealing)
inline void refinement(std::vector<BasisState>& basis, int max_passes, ld tolerance, 
                       ld b_form, ld S, const std::vector<bool>& relativistic, rvec& convergence_energies) 
{
    ld previous_pass_E = evaluate_basis_energy(basis, b_form, S, relativistic);
    
    // Start with 20% noise, shrink it down to 1% by the last pass
    ld initial_noise = 0.20; 
    ld final_noise = 0.01;

    for (int pass = 0; pass < max_passes; ++pass) {
        int num_improved_this_pass = 0;
        
        // Calculate dynamic noise scale for this pass
        ld noise_scale = initial_noise * std::pow(final_noise / initial_noise, static_cast<ld>(pass) / (max_passes - 1));

        for (size_t k = 0; k < basis.size(); ++k) {
            
            // Allow optimizing ALL states, including PN bare states!
            bool improved = refine_basis_state(basis, k, noise_scale, b_form, S, relativistic);
            
            if (improved) num_improved_this_pass++;
            
            std::cout << "\rPass " << pass+1 << "/" << max_passes 
                      << " [Noise: " << std::fixed << std::setprecision(2) << noise_scale * 100.0 << "%] "
                      << "| Refined state " << k+1 << "/" << basis.size()
                      << " (" << num_improved_this_pass << " improved)" << std::flush;
        }

        ld current_pass_E = evaluate_basis_energy(basis, b_form, S, relativistic);
        ld delta_E = previous_pass_E - current_pass_E;

        std::cout << "\r -> End of Pass " << pass+1 << ": E = " 
                  << std::fixed << std::setprecision(6) << current_pass_E 
                  << " MeV (ΔE = " << delta_E << " MeV)          " << std::flush;

        //convergence_energies.push_back(current_pass_E);

        if (delta_E < tolerance) {
            std::cout << "\n -> Refinement Converged: Total improvement below " << tolerance << " MeV.\n";
            break; 
        }
        previous_pass_E = current_pass_E;
    }
}

// Run two-phase SVM: skeleton (Phase 1) + competitive growth (Phase 2)
SvmResult run_deuteron_svm(const std::vector<bool>& relativistic, ld b_range, ld b_form, ld S) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;
    ld m_pi0 = 134.97, m_pic = 139.57;
    ld iso_c = std::sqrt(2.0L);

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

    std::vector<BasisState> basis;
    rvec convergence_energies;

    std::vector<BasisState> channel_templates = {
        {SpatialWavefunction(1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0},
        
        {SpatialWavefunction(-1), Channel::PI_0c_0f, NO_FLIP, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_1f, FLIP_PARTICLE_1, 1.0, jac_dressed_0, m_pi0},
        {SpatialWavefunction(-1), Channel::PI_0c_2f, FLIP_PARTICLE_2, 1.0, jac_dressed_0, m_pi0},
        
        {SpatialWavefunction(-1), Channel::PI_pc_0f, NO_FLIP, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_1f, FLIP_PARTICLE_1, iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_pc_2f, FLIP_PARTICLE_2, iso_c, jac_dressed_c, m_pic},
        
        {SpatialWavefunction(-1), Channel::PI_mc_0f, NO_FLIP, -iso_c, jac_dressed_c, m_pic}, 
        {SpatialWavefunction(-1), Channel::PI_mc_1f, FLIP_PARTICLE_1, -iso_c, jac_dressed_c, m_pic},
        {SpatialWavefunction(-1), Channel::PI_mc_2f, FLIP_PARTICLE_2, -iso_c, jac_dressed_c, m_pic}
    };

    // ------- PHASE 1: SKELETON BASIS WITH GEOMETRIC GRID --------
    std::cout << "--- 1. Planting Geometric PN Grid & Pion Seeds ---\n";
    
    std::vector<ld> deterministic_widths = {0.2, 3.0, 10.0};
    for (ld width : deterministic_widths) {
        rmat A_fixed = eye<ld>(1) * 1.0L /(width * width);
        rmat s_fixed = zeros<ld>(1, 3);
        basis.push_back({SpatialWavefunction(A_fixed, s_fixed, +1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    }

    // ------- PHASE 2: COMPETITIVE SVM GROWTH --------
    int num_cycles = 3;

    std::cout << "--- 2. Competitive SVM Growth ---\n";
    for (int cycle = 1; cycle < num_cycles+1; ++cycle) {
        std::cout << " - Cycle " << cycle << " - \n";

        // 1. Competitive Search
        competitive_search(basis, channel_templates, 100000, b_range, b_form, S, relativistic);
        ld E_now = evaluate_basis_energy(basis, b_form, S, relativistic);
        convergence_energies.push_back(E_now);

        std::cout << "\nStarting Sweep optimize\n";
        sweep_optimize_basis(basis, b_form, S, relativistic, convergence_energies);
        
        std::cout << "\n-------------------------------------------------------\n";
    }

    

    SvmResult result = evaluate_observables(basis, b_form, S, relativistic);
    result.convergence_history = convergence_energies;

    print_basis_details(basis, result.coefficients);
    std::cout << "\n";  
    
    return result;
}

int main(int argc, char* argv[]) {
    // Default values
    ld b_range = 200;
    ld b_form = 1.4;
    ld S = 30.0;
    bool do_s_sweep = false;

    std::string file_name = "convergence.data";

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-b_range") && i + 1 < argc) {
            b_range = std::stold(argv[++i]);
        } else if ((arg == "-b_form") && i + 1 < argc) {
            b_form = std::stold(argv[++i]);
        } else if ((arg == "-S") && i + 1 < argc) {
            S = std::stold(argv[++i]);
        } else if ((arg == "-sweep") || (arg == "--s-sweep")) {
            do_s_sweep = true;
        } else if ((arg == "-f" || arg == "--file") && i+1 < argc) {
            file_name = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./deu [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -b_range <value>    Search space for Gaussian width (default: 1.4 fm)\n";
            std::cout << "  -b_form <value>     Pion interaction range (default: 1.4 fm)\n";
            std::cout << "  -S <value>          Pion coupling strength (default: 100.0 MeV)\n";
            std::cout << "  -sweep, --s-sweep   Run sweep over S values (50 to 500 MeV)\n";
            std::cout << "  -f, --file <string> Choose convergence data file location\n";
            std::cout << "  -h, --help          Show this help message\n";
            return 0;
        }
    }

    // Enable nested parallelism
    omp_set_nested(0);
    //omp_set_max_active_levels(2);

    std::cout << "========================================\n";
    std::cout << "  DEUTERON SYSTEM (FAST COMPETITIVE SVM)\n";
    std::cout << "========================================\n";
    std::cout << "Parameters:\n";
    std::cout << "  b_range = " << b_range << " fm\n";
    std::cout << "  b_form  = " << b_form << " fm\n";
    if (do_s_sweep) {
        std::cout << "  MODE: S-parameter sweep (50 to 500 MeV)\n";
    } else {
        std::cout << "  S       = " << S << " MeV\n";
    }
    std::cout << "========================================\n\n";

    std::ofstream outfile(file_name);
    std::vector<SvmResult> all_results;

    if (do_s_sweep) {
        // S-sweep mode
        std::vector<ld> S_values = {30, 50, 75, 100, 125};
        std::vector<bool> flags = {false, false}; // PN_Cla, Pi_Cla

        outfile << "S_value\tEnergy\tRadius\tPN_percent\tPion_percent\n";

        std::cout << "\n========== S-PARAMETER SWEEP ==========\n";
        std::cout << "Testing S values: ";
        for (auto s : S_values) std::cout << s << " ";
        std::cout << "\n\n";

        for (ld S_test : S_values) {
            std::cout << "\n>>> Running S = " << S_test << " MeV\n";

            SvmResult res = run_deuteron_svm(flags, b_range, b_form, S_test);
            all_results.push_back(res);

            ld pion_pct = (100.0 - res.prob_bare * 100.0);
            ld pn_pct = res.prob_bare * 100.0;

            std::cout << "    E = " << std::fixed << std::setprecision(6) << res.energy
                      << " MeV | R = " << res.charge_radius
                      << " fm | PN = " << pn_pct
                      << "% | Pions = " << pion_pct << "%\n";

            outfile << S_test << "\t"
                    << std::fixed << std::setprecision(6) << res.energy << "\t"
                    << res.charge_radius << "\t"
                    << pn_pct << "\t"
                    << pion_pct << "\n";
        }

        std::cout << "\n========== SWEEP COMPLETE ==========\n";
        std::cout << "Results saved to: " << file_name << "\n";

    } else {
        // Single S value mode
        std::vector<std::pair<std::string, std::vector<bool>>> configurations = {
            {"PN_{Cla} Pi_{Cla}", {false,  false}},
        };

        // Run the configurations loop
        for (const auto& config : configurations) {
            std::string label = config.first;
            std::vector<bool> flags = config.second;

            std::cout << "\n>>>>>>>> RUNNING CONFIGURATION: " << label << " <<<<<<<<\n";

            SvmResult res = run_deuteron_svm(flags, b_range, b_form, S);
            all_results.push_back(res);

            std::cout << "--> FINAL " << label << " | E: " << res.energy << " MeV, R: " << res.charge_radius << " fm\n";

            outfile << "\"Iteration\"\t\"" << label << "\"\n";
            for (size_t iter = 0; iter < res.convergence_history.size(); ++iter) {
                outfile << iter << "\t" << std::fixed << std::setprecision(8) << res.convergence_history[iter] << "\n";
            }
            outfile << "\n\n";
        }

        // Print final summary comparison table
        std::cout << "\n========================================================================================\n";
        std::cout << "                                  FINAL RESULTS SUMMARY                                 \n";
        std::cout << "========================================================================================\n";
        std::cout << std::fixed << std::setprecision(5);

        for (size_t i = 0; i < configurations.size(); ++i) {
            std::cout << std::setw(24) << std::left << configurations[i].first
                      << " | E: "       << std::right << all_results[i].energy  << " MeV"
                      << " | R: "       << all_results[i].charge_radius         << " fm"
                      << " | <T>: "     << all_results[i].avg_kinetic_energy    << " MeV"
                      << " | PN: "      << (all_results[i].prob_bare * 100.0)   << " %"
                      << " | PN+pi: "   << (all_results[i].prob_dressed * 100.0)<< " %\n";
        }

        std::cout << "----------------------------------------------------------------------------------------\n";
        std::cout << "Experimental Target      | E: -2.22400 MeV | R: 2.12800 fm\n";
        std::cout << "========================================================================================\n";
    }

    outfile.close();
    return 0;
}