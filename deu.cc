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
#include "qm/serialization.h"
#include "SVM.h"

using namespace qm;


// Run two-phase SVM: skeleton (Phase 1) + competitive growth (Phase 2)
SvmResult run_deuteron_svm(const std::vector<bool>& relativistic, ld b_range, ld b_form, ld S) {
    // Physical Constants
    ld m_p = 938.27, m_n = 939.56;
    ld m_pi0 = 134.97, m_pic = 139.57;
    ld iso_c = std::sqrt(2.0L);

    Jacobian jac_bare({m_p, m_n});
    Jacobian jac_dressed_0({m_p, m_n, m_pi0});
    Jacobian jac_dressed_c({m_p, m_n, m_pic});

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
    
    std::vector<BasisState> bare_basis;
    std::vector<ld> deterministic_widths = {1.0, 10.0, 100.0};
    for (ld width : deterministic_widths) {
        rmat A_fixed = eye<ld>(1) * 1.0L /(width * width);
        rmat s_fixed = zeros<ld>(1, 3);
        bare_basis.push_back({SpatialWavefunction(A_fixed, s_fixed, +1), Channel::PN, NO_FLIP, 1.0, jac_bare, 0.0});
    }

    // ------- PHASE 2 & 3: COMPETITIVE GROWTH & BOX REGULARIZATION --------
    std::cout << "--- 2. Competitive Growth inside widening HO Box ---\n";
    
    std::vector<ld> box_strengths = {10.0, 1.0, 0.1, 0.01};
    std::vector<BasisState> grand_basis;
    grand_basis.insert(grand_basis.end(), bare_basis.begin(), bare_basis.end());

    for (ld ho_k : box_strengths) {
        std::cout << "\n=== Generating Basis for ho_k = " << ho_k << " ===\n";
        
        std::vector<BasisState> local_basis;
        local_basis.insert(local_basis.end(), bare_basis.begin(), bare_basis.end());
        
        // 1. Add 1 state per channel (10 states total)
        competitive_search(local_basis, channel_templates, 5000, b_range, b_form, S, relativistic, ho_k);
        
        // 2. Fast sweep on just these 10 states!
        sweep_optimize_basis(local_basis, b_form, S, relativistic, convergence_energies, 10, 1e-4, ho_k);
        
        // 3. Move these highly specialized states into the master pool
        grand_basis.insert(grand_basis.end(), local_basis.begin() + bare_basis.size(), local_basis.end());
    }

    std::cout << "\n=== FINAL GEVP EVALUATION IN FREE SPACE (ho_k = 0.0) ===\n";
    
    // (Optional) Do one final, shallow sweep of the grand basis at k=0 
    // to let the core, pocket, and tail states slightly adjust to each other.
    sweep_optimize_basis(grand_basis, b_form, S, relativistic, convergence_energies, 3, 1e-4, 0.0);

    SvmResult result = evaluate_observables(grand_basis, b_form, S, relativistic);

    result.convergence_history = convergence_energies;

    print_basis_details(grand_basis, result.coefficients);
    std::cout << "\n";

    // Save final basis state for analysis
    save_basis_state(grand_basis, result.coefficients, result.energy, result.charge_radius,
                     result.avg_kinetic_energy, "basis_final.txt");
    std::cout << "Saved basis state to basis_final.txt\n";

    return result;
}

int main(int argc, char* argv[]) {
    // Default values
    ld b_range = 100;
    ld b_form = 1.4;
    ld S = 30.0;

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
        } else if ((arg == "-f" || arg == "--file") && i+1 < argc) {
            file_name = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ./deu [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -b_range <value>    Search space for Gaussian width (default: 1.4 fm)\n";
            std::cout << "  -b_form <value>     Pion interaction range (default: 1.4 fm)\n";
            std::cout << "  -S <value>          Pion coupling strength (default: 100.0 MeV)\n";
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
    std::cout << "  S       = " << S << " MeV\n";
    std::cout << "========================================\n\n";

    std::ofstream outfile(file_name);
    std::vector<SvmResult> all_results;

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

        ld binding_energy = res.energy_excited - res.energy;  // E_1 - E_0
        std::cout << "--> FINAL " << label << " | E_0: " << res.energy << " MeV, E_1: " << res.energy_excited
                  << " MeV, Binding: " << binding_energy << " MeV, R: " << res.charge_radius << " fm\n";

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
        ld binding_energy = all_results[i].energy_excited - all_results[i].energy;
        std::cout << std::setw(24) << std::left << configurations[i].first
                  << " | E_0: "     << std::right << all_results[i].energy  << " MeV"
                  << " | E_1: "     << all_results[i].energy_excited        << " MeV"
                  << " | Binding: " << binding_energy                        << " MeV"
                  << " \n                        "
                  << " |    R: "       << all_results[i].charge_radius         << " fm"
                  << " | <T>: "     << all_results[i].avg_kinetic_energy    << " MeV"
                  << " |  PN: "      << (all_results[i].prob_bare * 100.0)   << " %"
                  << " | PN+pi: "   << (all_results[i].prob_dressed * 100.0)<< " %\n";
    }

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "Experimental Target      | E: -2.22400 MeV | R: 2.12800 fm\n";
    std::cout << "========================================================================================\n";

    outfile.close();
    return 0;
}