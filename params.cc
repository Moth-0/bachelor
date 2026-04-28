/*
 * s_finder.cc - Finds the exact S for a specific b_range and b_form
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "qm/matrix.h"
#include "qm/gaussian.h"
#include "qm/operators.h"
#include "qm/solver.h"
#include "deuterium.h"
#include "qm/serialization.h"
#include "SVM.h"

using namespace qm;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./s_finder <b_range> <b_form>\n";
        return 1;
    }

    ld b_r = std::stold(argv[1]);
    ld b_f = std::stold(argv[2]);
    const ld TARGET_E = -2.2245;

    std::vector<BasisState> basis;
    auto [b, c, e, r, ke] = load_basis_state("basis_final.txt");
    basis = b;

    std::vector<bool> rel = {false, false};

    // Binary Search for S
    ld low_S = 1.0, high_S = 2000.0;
    SvmResult res;

    std::cout << "Finding S for b_r=" << b_r << ", b_f=" << b_f << "...\n";

    for (int i = 0; i < 25; ++i) {
        ld mid_S = (low_S + high_S) / 2.0;
        res = evaluate_observables(basis, b_f, b_r, mid_S, rel);

        if (res.energy > TARGET_E) low_S = mid_S;
        else high_S = mid_S;
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "RESULTS FOR b_f = " << b_f << "\n";
    std::cout << "Required S: " << std::fixed << std::setprecision(6) << (low_S + high_S) / 2.0 << " MeV\n";
    std::cout << "Final E:    " << res.energy << " MeV\n";
    std::cout << "Radius:     " << res.charge_radius << " fm\n";
    std::cout << "----------------------------------------\n";

    return 0;
}