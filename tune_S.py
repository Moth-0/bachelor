import subprocess
import re

# Targets
target_energy = -2.224
target_radius = 2.128
b_range = 3.0

# Outer Binary Search Bounds (for b_form)
b_low = 0.1   # Extremely narrow well -> small radius
b_high = 1.0  # Wide well -> large radius
b_iters = 10

def find_S_for_b_form(b_form):
    """Inner binary search: Finds the S that gives E = -2.224 for a given b_form"""
    # S needs to be much higher when b_form is small (narrow wells need to be deep)
    S_low = 10.0   
    S_high = 200.0 
    S_iters = 15
    tolerance = 0.005 # Energy tolerance
    
    best_S, best_E, best_R = None, None, None
    
    print(f"  [Inner Loop] Tuning S for b_form = {b_form:.4f}...")
    
    for j in range(S_iters):
        S_mid = (S_low + S_high) / 2.0
        
        cmd = ["./deu", "-b_range", str(b_range), "-b_form", f"{b_form:.4f}", "-S", f"{S_mid:.4f}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        match = re.search(r"Energy \(MeV\):\s*([-0-9.]+)\s*-\s*Radius \(fm\):\s*([-0-9.]+)", result.stdout)
        
        if match:
            E = float(match.group(1))
            R = float(match.group(2))
            best_S, best_E, best_R = S_mid, E, R
            
            # If we hit the energy target, stop tuning S
            if abs(E - target_energy) <= tolerance:
                break
                
            if E < target_energy:
                # Too bound (e.g., -5.0), need less attraction
                S_high = S_mid
            else:
                # Not bound enough (e.g., -1.0), need more attraction
                S_low = S_mid
        else:
            print("  [Error] Could not parse output. Skipping.")
            break
            
    return best_S, best_E, best_R

# ---------------------------------------------------------
# Main Outer Loop
# ---------------------------------------------------------
print("=====================================================")
print(f" HUNTING FOR TARGETS: E = {target_energy} MeV | R = {target_radius} fm")
print("=====================================================")

for i in range(b_iters):
    b_mid = (b_low + b_high) / 2.0
    print(f"\nOuter Iteration {i+1}: Testing b_form = {b_mid:.4f}")
    
    # Run the inner search to lock the energy at -2.224 MeV
    S_found, E_found, R_found = find_S_for_b_form(b_mid)
    
    if R_found is None:
        print("Simulation failed. Adjust bounds.")
        break
        
    print(f"-> Locked Energy! For b_form={b_mid:.4f}, we need S={S_found:.4f}")
    print(f"-> Resulting Physics: Energy = {E_found:.4f} MeV | Radius = {R_found:.4f} fm")
    
    # Check if we hit the radius target
    if abs(R_found - target_radius) <= 0.1:
        print("\n*** SUCCESS! ALL PHYSICAL OBSERVABLES MATCHED! ***")
        break
        
    # Outer binary search logic for the Radius
    if R_found > target_radius:
        print("   Radius is too large. Making the potential well narrower...")
        b_high = b_mid
    else:
        print("   Radius is too small. Making the potential well wider...")
        b_low = b_mid

print("\n=====================================================")
print(f"FINAL CLASSIC PARAMETERS:")
print(f" b_form = {b_mid:.4f} fm")
print(f" S      = {S_found:.4f} MeV")
print("=====================================================")