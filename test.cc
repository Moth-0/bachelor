#include"qm/matrix.h"
#include"qm/jacobi.h"

using namespace qm;

int main () {
    // --- Jacobian ---
    Jacobian sys({938.0, 939.0, 139.0});
    std::cout << sys.J << std::endl;
    rvec r_phys = {1.0, -0.5, 3.0}; 
    ld true_distance = r_phys[0] - r_phys[2]; // 1.0 - 3.0 = -2.0

    rvec r = sys.J*r_phys;

    rvec w_p = sys.transform_w(0);
    rvec w_pi = sys.transform_w(2);
    rvec w_1 = w_p - w_pi;

    ld jacobi_distance = dot_no_conj(w_1, r);

    std::cout << "True physical distance (r_p - r_pi): " << true_distance << " fm\n";
    std::cout << "Calculated distance in Jacobi space: " << jacobi_distance << " fm\n";
    
    return 0;
}