/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                   qm/serialization.h - BASIS STATE PERSISTENCE                 ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║ PURPOSE:                                                                       ║
║   Save and load optimized basis states for post-analysis. Allows inspection    ║
║   of final basis configuration, overlap matrices, coefficients, and debugging  ║
║   of convergence behavior.                                                     ║
║                                                                                ║
║ KEY FUNCTIONS:                                                                 ║
║   • save_basis_state(): Write basis to plaintext file (uses pack_wavefunction) ║
║   • load_basis_state(): Reconstruct basis from file (uses unpack_wavefunction) ║
║                                                                                ║
║ STRATEGY:                                                                       ║
║   Reuse pack_wavefunction() and unpack_wavefunction() which handle Cholesky    ║
║   decomposition and shift packing/unpacking. Store packed parameters + metadata║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

#pragma once

#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "matrix.h"
#include "gaussian.h"

// Forward declarations (BasisState is global, from deuterium.h)
struct BasisState;
enum class Channel;

namespace qm {

// Forward declaration of SpinChannel (defined in operators.h)
enum SpinChannel;

// Helper to convert Channel enum to string
inline std::string channel_to_string(int channel_id) {
    static const std::vector<std::string> names = {
        "PN",
        "PI_0c_0f", "PI_0c_1f", "PI_0c_2f",
        "PI_pc_0f", "PI_pc_1f", "PI_pc_2f",
        "PI_mc_0f", "PI_mc_1f", "PI_mc_2f"
    };
    if (channel_id >= 0 && channel_id < (int)names.size()) {
        return names[channel_id];
    }
    return "UNKNOWN";
}

// Helper to convert SpinChannel enum to string
inline std::string spinch_to_string(int flip_id) {
    static const std::vector<std::string> names = {"NO_FLIP", "FLIP_PARTICLE_1", "FLIP_PARTICLE_2"};
    if (flip_id >= 0 && flip_id < (int)names.size()) {
        return names[flip_id];
    }
    return "UNKNOWN";
}

// String-to-enum converters
inline int string_to_channel(const std::string& name) {
    static const std::vector<std::string> names = {
        "PN",
        "PI_0c_0f", "PI_0c_1f", "PI_0c_2f",
        "PI_pc_0f", "PI_pc_1f", "PI_pc_2f",
        "PI_mc_0f", "PI_mc_1f", "PI_mc_2f"
    };
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == name) return i;
    }
    return 0; // Default to PN
}

inline int string_to_spinch(const std::string& name) {
    if (name == "NO_FLIP") return 0;
    if (name == "FLIP_PARTICLE_1") return 1;
    if (name == "FLIP_PARTICLE_2") return 2;
    return 0; // Default
}

/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                           SAVE BASIS STATE TO FILE                             ║
╚════════════════════════════════════════════════════════════════════════════════╝

Save optimized basis to plaintext format using pack_wavefunction():
  - Observables (energy, radius, kinetic_energy, coefficients)
  - For each basis state:
    • Channel, flip, isospin_factor, pion_mass, parity, jacobian masses
    • Packed parameters (via pack_wavefunction)
    • Dimension info for reconstruction
*/
inline void save_basis_state(const std::vector<BasisState>& basis,
                             const rvec& coefficients,
                             ld energy, ld radius, ld kinetic_energy,
                             const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file << std::setprecision(12);

    // Header
    file << "# OPTIMIZED BASIS STATE DUMP\n";
    file << "# " << basis.size() << " basis states\n\n";

    // Save observables
    file << "ENERGY " << energy << "\n";
    file << "RADIUS " << radius << "\n";
    file << "KINETIC_ENERGY " << kinetic_energy << "\n";
    file << "COEFFICIENTS " << coefficients.size() << "\n";
    for (size_t i = 0; i < coefficients.size(); ++i) {
        file << "  " << coefficients[i] << "\n";
    }
    file << "\n";

    // Save each basis state
    for (size_t idx = 0; idx < basis.size(); ++idx) {
        const BasisState& b = basis[idx];

        file << "STATE " << idx << "\n";
        file << "  CHANNEL " << channel_to_string((int)b.type) << "\n";
        file << "  FLIP " << spinch_to_string((int)b.flip) << "\n";
        file << "  ISOSPIN_FACTOR " << b.isospin_factor << "\n";
        file << "  PION_MASS " << b.pion_mass << "\n";
        file << "  PARITY " << b.psi.parity_sign << "\n";

        // Jacobian masses
        file << "  JACOBIAN_MASSES " << b.jac.masses.size() << "\n";
        for (size_t j = 0; j < b.jac.masses.size(); ++j) {
            file << "    " << b.jac.masses[j] << "\n";
        }

        // Dimension
        file << "  DIMENSION " << b.psi.A.size1() << "\n";

        // Packed parameters (using pack_wavefunction)
        rvec packed = pack_wavefunction(b.psi, true);
        file << "  PACKED_PARAMS " << packed.size() << "\n";
        for (size_t i = 0; i < packed.size(); ++i) {
            file << "    " << packed[i] << "\n";
        }
        file << "\n";
    }

    file.close();
}

/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                          LOAD BASIS STATE FROM FILE                            ║
╚════════════════════════════════════════════════════════════════════════════════╝

Reconstruct basis state and observables from saved file using unpack_wavefunction().
Returns tuple: (basis_states, coefficients, energy, radius, kinetic_energy)
*/
inline std::tuple<std::vector<BasisState>, rvec, ld, ld, ld>
load_basis_state(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<BasisState> basis;
    rvec coefficients;
    ld energy = 0, radius = 0, kinetic_energy = 0;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "ENERGY") {
            iss >> energy;
        } else if (key == "RADIUS") {
            iss >> radius;
        } else if (key == "KINETIC_ENERGY") {
            iss >> kinetic_energy;
        } else if (key == "COEFFICIENTS") {
            size_t n;
            iss >> n;
            coefficients.resize(n);
            for (size_t i = 0; i < n; ++i) {
                std::getline(file, line);
                coefficients[i] = std::stold(line);
            }
        } else if (key == "STATE") {
            // Parse new basis state
            size_t state_idx;
            iss >> state_idx;

            // Temporary holders
            int channel_id = 0;
            int flip_id = 0;
            ld isospin_factor = 1.0;
            ld pion_mass = 0.0;
            int parity_sign = 1;
            std::vector<ld> jac_masses_vec = {1.0};
            size_t dimension = 1;
            rvec packed_params;

            // Read state properties until blank line
            while (std::getline(file, line)) {
                if (line.empty()) break;

                std::istringstream line_iss(line);
                std::string prop;
                line_iss >> prop;

                if (prop == "CHANNEL") {
                    std::string ch_name;
                    line_iss >> ch_name;
                    channel_id = string_to_channel(ch_name);
                } else if (prop == "FLIP") {
                    std::string flip_name;
                    line_iss >> flip_name;
                    flip_id = string_to_spinch(flip_name);
                } else if (prop == "ISOSPIN_FACTOR") {
                    line_iss >> isospin_factor;
                } else if (prop == "PION_MASS") {
                    line_iss >> pion_mass;
                } else if (prop == "PARITY") {
                    line_iss >> parity_sign;
                } else if (prop == "JACOBIAN_MASSES") {
                    size_t n;
                    line_iss >> n;
                    jac_masses_vec.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        std::getline(file, line);
                        jac_masses_vec[i] = std::stold(line);
                    }
                } else if (prop == "DIMENSION") {
                    line_iss >> dimension;
                } else if (prop == "PACKED_PARAMS") {
                    size_t n;
                    line_iss >> n;
                    packed_params.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        std::getline(file, line);
                        packed_params[i] = std::stold(line);
                    }
                }
            }

            // Reconstruct BasisState
            SpatialWavefunction psi(parity_sign);
            psi.A = zeros<ld>(dimension, dimension);
            psi.s = zeros<ld>(dimension, 3);

            // Unpack parameters using unpack_wavefunction
            unpack_wavefunction(psi, packed_params, true);

            // Convert std::vector to qm::rvec for Jacobian
            rvec jac_masses(jac_masses_vec.size());
            for (size_t i = 0; i < jac_masses_vec.size(); ++i) {
                jac_masses[i] = jac_masses_vec[i];
            }

            BasisState b{psi, (Channel)channel_id, (SpinChannel)flip_id,
                        isospin_factor, Jacobian(jac_masses), pion_mass};
            basis.push_back(b);
        }
    }

    file.close();
    return {basis, coefficients, energy, radius, kinetic_energy};
}

/*
╔════════════════════════════════════════════════════════════════════════════════╗
║                   SAVE/LOAD ALL 4 CONFIGURATIONS                              ║
╚════════════════════════════════════════════════════════════════════════════════╝
*/

struct ConfigurationResult {
    std::string label;
    std::vector<BasisState> basis;
    rvec coefficients;
    ld energy;
    ld charge_radius;
    ld avg_kinetic_energy;
    ld prob_bare;
    ld prob_dressed;
};

inline void save_all_configurations(const std::vector<ConfigurationResult>& configs,
                                    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file << std::setprecision(12);
    file << "# ALL 4 CONFIGURATIONS\n";
    file << "# Total configurations: " << configs.size() << "\n\n";

    for (const auto& config : configs) {
        file << "CONFIGURATION " << config.label << "\n";
        file << "  ENERGY " << config.energy << "\n";
        file << "  RADIUS " << config.charge_radius << "\n";
        file << "  KINETIC_ENERGY " << config.avg_kinetic_energy << "\n";
        file << "  PROB_BARE " << config.prob_bare << "\n";
        file << "  PROB_DRESSED " << config.prob_dressed << "\n";
        file << "  NUM_BASIS_STATES " << config.basis.size() << "\n";
        file << "  COEFFICIENTS " << config.coefficients.size() << "\n";
        for (size_t i = 0; i < config.coefficients.size(); ++i) {
            file << "    " << config.coefficients[i] << "\n";
        }
        file << "\n";

        // Save each basis state in this configuration
        for (size_t idx = 0; idx < config.basis.size(); ++idx) {
            const BasisState& b = config.basis[idx];

            file << "  STATE " << idx << "\n";
            file << "    CHANNEL " << channel_to_string((int)b.type) << "\n";
            file << "    FLIP " << spinch_to_string((int)b.flip) << "\n";
            file << "    ISOSPIN_FACTOR " << b.isospin_factor << "\n";
            file << "    PION_MASS " << b.pion_mass << "\n";
            file << "    PARITY " << b.psi.parity_sign << "\n";

            // Jacobian masses
            file << "    JACOBIAN_MASSES " << b.jac.masses.size() << "\n";
            for (size_t j = 0; j < b.jac.masses.size(); ++j) {
                file << "      " << b.jac.masses[j] << "\n";
            }

            // Dimension
            file << "    DIMENSION " << b.psi.A.size1() << "\n";

            // Packed parameters
            rvec packed = pack_wavefunction(b.psi, true);
            file << "    PACKED_PARAMS " << packed.size() << "\n";
            for (size_t i = 0; i < packed.size(); ++i) {
                file << "      " << packed[i] << "\n";
            }
            file << "\n";
        }

        file << "\n";
    }

    file.close();
}

inline std::vector<ConfigurationResult> load_all_configurations(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<ConfigurationResult> configs;
    std::string line;
    ConfigurationResult current_config;
    bool in_configuration = false;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string key;
        iss >> key;

        if (key == "CONFIGURATION") {
            if (in_configuration && !current_config.label.empty()) {
                configs.push_back(current_config);
            }
            iss >> current_config.label;
            current_config.basis.clear();
            current_config.coefficients.resize(0);
            current_config.energy = 0;
            current_config.charge_radius = 0;
            current_config.avg_kinetic_energy = 0;
            current_config.prob_bare = 0;
            current_config.prob_dressed = 0;
            in_configuration = true;
        } else if (key == "ENERGY" && in_configuration) {
            iss >> current_config.energy;
        } else if (key == "RADIUS" && in_configuration) {
            iss >> current_config.charge_radius;
        } else if (key == "KINETIC_ENERGY" && in_configuration) {
            iss >> current_config.avg_kinetic_energy;
        } else if (key == "PROB_BARE" && in_configuration) {
            iss >> current_config.prob_bare;
        } else if (key == "PROB_DRESSED" && in_configuration) {
            iss >> current_config.prob_dressed;
        } else if (key == "COEFFICIENTS" && in_configuration) {
            size_t n;
            iss >> n;
            current_config.coefficients.resize(n);
            for (size_t i = 0; i < n; ++i) {
                std::getline(file, line);
                current_config.coefficients[i] = std::stold(line);
            }
        } else if (key == "STATE" && in_configuration) {
            size_t state_idx;
            iss >> state_idx;

            // Temporary holders
            int channel_id = 0;
            int flip_id = 0;
            ld isospin_factor = 1.0;
            ld pion_mass = 0.0;
            int parity_sign = 1;
            std::vector<ld> jac_masses_vec = {1.0};
            size_t dimension = 1;
            rvec packed_params;

            // Read state properties until blank line
            while (std::getline(file, line)) {
                if (line.empty()) break;

                std::istringstream line_iss(line);
                std::string prop;
                line_iss >> prop;

                if (prop == "CHANNEL") {
                    std::string ch_name;
                    line_iss >> ch_name;
                    channel_id = string_to_channel(ch_name);
                } else if (prop == "FLIP") {
                    std::string flip_name;
                    line_iss >> flip_name;
                    flip_id = string_to_spinch(flip_name);
                } else if (prop == "ISOSPIN_FACTOR") {
                    line_iss >> isospin_factor;
                } else if (prop == "PION_MASS") {
                    line_iss >> pion_mass;
                } else if (prop == "PARITY") {
                    line_iss >> parity_sign;
                } else if (prop == "JACOBIAN_MASSES") {
                    size_t n;
                    line_iss >> n;
                    jac_masses_vec.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        std::getline(file, line);
                        jac_masses_vec[i] = std::stold(line);
                    }
                } else if (prop == "DIMENSION") {
                    line_iss >> dimension;
                } else if (prop == "PACKED_PARAMS") {
                    size_t n;
                    line_iss >> n;
                    packed_params.resize(n);
                    for (size_t i = 0; i < n; ++i) {
                        std::getline(file, line);
                        packed_params[i] = std::stold(line);
                    }
                }
            }

            // Reconstruct BasisState
            SpatialWavefunction psi(parity_sign);
            psi.A = zeros<ld>(dimension, dimension);
            psi.s = zeros<ld>(dimension, 3);

            // Unpack parameters
            unpack_wavefunction(psi, packed_params, true);

            // Convert std::vector to qm::rvec for Jacobian
            rvec jac_masses(jac_masses_vec.size());
            for (size_t i = 0; i < jac_masses_vec.size(); ++i) {
                jac_masses[i] = jac_masses_vec[i];
            }

            BasisState b{psi, (Channel)channel_id, (SpinChannel)flip_id,
                        isospin_factor, Jacobian(jac_masses), pion_mass};
            current_config.basis.push_back(b);
        }
    }

    if (in_configuration && !current_config.label.empty()) {
        configs.push_back(current_config);
    }

    file.close();
    return configs;
}

} // namespace qm