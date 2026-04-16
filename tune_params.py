import subprocess
import re
import csv
import math

# --- Search Configuration ---
TARGET_E = -2.224        # Experimental Deuteron Binding Energy (MeV)
TOLERANCE = 0.001        # Stop when we are within 0.001 MeV of the target
MAX_ITERATIONS = 20      # Failsafe to prevent infinite loops

# The list of form factor widths (b) to test
b_list = [0.8, 1.0, 1.2, 1.4]

# Starting bounds for S (MeV). S=60 gave -6.5 MeV, so 0 to 100 is a very safe bracket.
S_MIN_INIT = 0.0
S_MAX_INIT = 100.0

def run_svm(b_form, S):
    """Runs the C++ deu executable and extracts E and R."""
    cmd = ["./deu", "-b_range", "200", "-b_form", str(b_form), "-S", str(S)]
    
    # Run the command and capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Use Regex to find the exact Energy and Radius from the Summary Table
    # Looks for: "| E: -X.XXXX MeV | R: Y.YYYY fm"
    match = re.search(r"\|\s*E:\s*([-\d\.]+)\s*MeV\s*\|\s*R:\s*([-\d\.]+)\s*fm", result.stdout)
    
    if match:
        E = float(match.group(1))
        R = float(match.group(2))
        return E, R
    else:
        print(f"\n[Error] Could not parse output for b={b_form}, S={S}. Ensure C++ code compiled.")
        return None, None

def main():
    print("=========================================================")
    print("   AUTOMATED BINARY SEARCH FOR DEUTERON BINDING ENERGY   ")
    print("=========================================================\n")
    
    results_data = []

    for b in b_list:
        print(f"--- Starting optimization for b = {b} fm ---")
        
        S_low = S_MIN_INIT
        S_high = S_MAX_INIT
        best_S, best_E, best_R = None, None, None
        
        for iteration in range(MAX_ITERATIONS):
            S_mid = (S_low + S_high) / 2.0
            
            print(f" Iteration {iteration+1:2d} | Testing S = {S_mid:.5f} MeV...", end="", flush=True)
            
            E, R = run_svm(b, S_mid)
            
            if E is None:
                break # Parsing failed, skip this b
                
            error = E - TARGET_E
            print(f" Result: E = {E:.5f} MeV (Error: {error:+.5f}) | R = {R:.5f} fm")
            
            # Save the current best just in case we hit the max iterations
            best_S, best_E, best_R = S_mid, E, R
            
            # Check if we hit the target within tolerance
            if abs(error) < TOLERANCE:
                print(f" -> [CONVERGED] Found S = {S_mid:.5f} for b = {b}\n")
                break
                
            # Physics Binary Search Logic:
            # If E < -2.224 (e.g. -6.5): System is OVERBOUND. We need a weaker attraction (lower S).
            if E < TARGET_E:
                S_high = S_mid
            # If E > -2.224 (e.g. -1.0): System is UNDERBOUND. We need a stronger attraction (higher S).
            else:
                S_low = S_mid
                
        # Store the converged data
        results_data.append({
            "b_form": b,
            "S_converged": best_S,
            "E_0": best_E,
            "R_c": best_R
        })

    # Save everything to a CSV
    csv_filename = "deuteron_results.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["b_form", "S_converged", "E_0", "R_c"])
        writer.writeheader()
        writer.writerows(results_data)
        
    print("=========================================================")
    print(f"Search complete! Data saved to '{csv_filename}'.")
    print("=========================================================")

if __name__ == "__main__":
    main()