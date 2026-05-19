#!/usr/bin/env python3
"""
sweep_S.py - Orchestrates S (coupling strength) parameter sweeps with integrated plotting

Runs multiple deu instances with varying S, collects results, plots automatically.

Usage:
  python3 scripts/sweep_S.py \
    --b_range 2.5 --b_form 1.5 \
    --S_min 20 --S_max 150 --S_steps 15 \
    [--jobs 4]
"""

import argparse
import subprocess
import os
import sys
import csv
import json
import glob
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import hashlib

class ParameterSweep:
    """Manages S parameter sweep with integrated plotting"""
    
    def __init__(self, fixed_params, sweep_params, num_jobs=1):
        self.scan_type = "S"
        self.fixed_params = fixed_params
        self.sweep_params = sweep_params
        self.num_jobs = num_jobs
        self.results_dir = f"results/energy_sweep_S"
        self.run_dir = self.results_dir
        
    def generate_parameter_combinations(self):
        """Generate all S parameter combinations"""
        combinations = []
        
        S_values = np.linspace(
            self.sweep_params["S_min"],
            self.sweep_params["S_max"],
            self.sweep_params["S_steps"]
        )
        for S in S_values:
            params = {
                "b_range": self.fixed_params["b_range"],
                "b_form": self.fixed_params["b_form"],
                "S": S
            }
            combinations.append(params)
        
        return combinations
    
    def run_deu(self, params):
        """Execute a single deu run with given parameters"""
        param_str = "_".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                             for k, v in sorted(params.items())])
        
        run_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        run_id = f"run_{run_hash}"
        csv_path = os.path.join(self.run_dir, f"{run_id}.csv")
        
        cmd = ["./deu"]
        cmd.extend(["-b_range", str(params["b_range"])])
        cmd.extend(["-b_form", str(params["b_form"])])
        cmd.extend(["-S", str(params["S"])])
        cmd.extend(["-box-strengths", "10.0,5.0,2.0,1.0"])
        cmd.extend(["--output-csv", csv_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
            
            if result.returncode != 0:
                return {"status": "FAILED", "params": params, "run_id": run_id, "csv_path": csv_path, "error": result.stderr}
            
            return {"status": "SUCCESS", "params": params, "run_id": run_id, "csv_path": csv_path}
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT", "params": params, "run_id": run_id, "csv_path": csv_path}
        except Exception as e:
            return {"status": "ERROR", "params": params, "run_id": run_id, "csv_path": csv_path, "error": str(e)}
    
    def extract_final_row(self, csv_path):
        """Extract the FINAL row from a run CSV"""
        try:
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            data_lines = [l for l in lines if not l.startswith("#")]
            
            if len(data_lines) < 2:
                return None
            
            reader = csv.DictReader(data_lines)
            rows = list(reader)
            
            if rows:
                final_row = rows[-1]
                if float(final_row.get("iteration", 0)) == -1:
                    return final_row
                return rows[-1]
            
            return None
        except Exception as e:
            print(f"Error extracting final row from {csv_path}: {e}", file=sys.stderr)
            return None
    
    def aggregate_results(self, run_results):
        """Aggregate all run results into a single CSV"""
        aggregated_path = os.path.join(self.results_dir, "aggregated.csv")
        aggregated_rows = []
        
        for result in run_results:
            if result["status"] != "SUCCESS":
                print(f"Skipping {result['run_id']}: {result['status']}", file=sys.stderr)
                continue
            
            csv_path = result["csv_path"]
            if not os.path.exists(csv_path):
                print(f"Warning: CSV not found for {result['run_id']}: {csv_path}", file=sys.stderr)
                continue
            
            final_row = self.extract_final_row(csv_path)
            if final_row:
                row = dict(result["params"])
                row.update(final_row)
                aggregated_rows.append(row)
        
        # Sort by S for readability
        aggregated_rows.sort(key=lambda r: float(r.get("S", 0)))
        
        if aggregated_rows:
            headers = list(aggregated_rows[0].keys())
            with open(aggregated_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(aggregated_rows)
            
            print(f"\nAggregated results written to: {aggregated_path}")
            print(f"  Rows: {len(aggregated_rows)}")
            
            run_files = glob.glob(os.path.join(self.results_dir, "run_*.csv"))
            for run_file in run_files:
                try:
                    os.remove(run_file)
                except:
                    pass
            if run_files:
                print(f"  Cleaned up {len(run_files)} individual run files")
            
            return aggregated_path
        else:
            print(f"ERROR: No results to aggregate!", file=sys.stderr)
            return None
    
    def print_summary(self, run_results):
        """Print summary of sweep results"""
        print("\n" + "="*80)
        print(f"SWEEP SUMMARY: S parameter sweep")
        print("="*80)
        
        success = sum(1 for r in run_results if r["status"] == "SUCCESS")
        failed = sum(1 for r in run_results if r["status"] != "SUCCESS")
        
        print(f"Total runs: {len(run_results)}")
        print(f"Successful: {success}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed runs:")
            for r in run_results:
                if r["status"] != "SUCCESS":
                    print(f"  {r['run_id']}: {r['status']}")
                    if "error" in r:
                        print(f"    Error: {r['error'][:100]}")
        
        print(f"\nResults directory: {self.results_dir}")
        print("="*80 + "\n")
    
    def plot_energy_vs_S(self, csv_file):
        """Plot energy vs S (coupling strength)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("ERROR: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
            return False
        
        data = {"S": [], "energy": [], "radius": []}
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data["S"].append(float(row["S"]))
                    data["energy"].append(float(row["energy_mev"]))
                    try:
                        data["radius"].append(float(row["radius_fm"]))
                    except (KeyError, ValueError):
                        pass
        except Exception as e:
            print(f"ERROR reading CSV: {e}", file=sys.stderr)
            return False
        
        if not data["S"]:
            print("No data found in CSV", file=sys.stderr)
            return False
        
        sorted_data = sorted(zip(data["S"], data["energy"], data["radius"]))
        S_vals, energies, radii = zip(*sorted_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(S_vals, energies, 'o-', linewidth=2, markersize=8, label='Computed')
        ax1.axhline(y=-2.224, color='r', linestyle='--', linewidth=2, label='Target (-2.224 MeV)')
        ax1.set_xlabel('S - Pion Coupling Strength (MeV)', fontsize=12)
        ax1.set_ylabel('Energy (MeV)', fontsize=12)
        ax1.set_title('Deuteron Ground State Energy vs Coupling Strength', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        if radii and any(r > 0 for r in radii):
            ax2.plot(S_vals, radii, 's-', linewidth=2, markersize=8, color='green', label='Computed')
            ax2.axhline(y=2.128, color='r', linestyle='--', linewidth=2, label='Target (2.128 fm)')
            ax2.set_xlabel('S - Pion Coupling Strength (MeV)', fontsize=12)
            ax2.set_ylabel('Charge Radius (fm)', fontsize=12)
            ax2.set_title('Charge Radius vs Coupling Strength', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
        
        plt.tight_layout()
        output_file = csv_file.replace('aggregated.csv', 'plot.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        return True
    
    def run(self):
        """Execute the complete sweep"""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Starting S parameter sweep")
        print(f"  Fixed b_range: {self.fixed_params['b_range']}")
        print(f"  Fixed b_form: {self.fixed_params['b_form']}")
        print(f"  S range: [{self.sweep_params['S_min']}, {self.sweep_params['S_max']}]")
        print(f"  Results directory: {self.results_dir}")
        print(f"  Parallel jobs: {self.num_jobs}")
        
        combinations = self.generate_parameter_combinations()
        print(f"  Parameter combinations: {len(combinations)}")
        
        print(f"\nExecuting runs...")
        if self.num_jobs == 1:
            run_results = [self.run_deu(params) for params in combinations]
        else:
            with Pool(self.num_jobs) as pool:
                run_results = pool.map(self.run_deu, combinations)
        
        print(f"\nAggregating results...")
        agg_path = self.aggregate_results(run_results)
        self.print_summary(run_results)
        
        # Plot automatically
        if agg_path:
            print(f"Generating plots...")
            self.plot_energy_vs_S(agg_path)
        
        # Save metadata
        metadata = {
            "scan_type": "S",
            "timestamp": datetime.now().isoformat(),
            "fixed_params": self.fixed_params,
            "sweep_params": self.sweep_params,
            "num_jobs": self.num_jobs,
            "total_runs": len(run_results),
            "successful_runs": sum(1 for r in run_results if r["status"] == "SUCCESS"),
            "aggregated_csv": agg_path
        }
        
        metadata_path = os.path.join(self.results_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return agg_path

def main():
    parser = argparse.ArgumentParser(
        description="S parameter sweep orchestrator for deuteron SVM solver"
    )
    
    parser.add_argument("--b_range", type=float, required=True,
                       help="Fixed Gaussian width search space (fm)")
    parser.add_argument("--b_form", type=float, required=True,
                       help="Fixed pion form factor range (fm)")
    
    parser.add_argument("--S_min", type=float, required=True,
                       help="Minimum S value")
    parser.add_argument("--S_max", type=float, required=True,
                       help="Maximum S value")
    parser.add_argument("--S_steps", type=int, default=10,
                       help="Number of S steps (default: 10)")
    
    parser.add_argument("--jobs", type=int, default=1,
                       help="Number of parallel jobs (default: 1)")
    
    args = parser.parse_args()
    
    fixed_params = {
        "b_range": args.b_range,
        "b_form": args.b_form
    }
    
    sweep_params = {
        "S_min": args.S_min,
        "S_max": args.S_max,
        "S_steps": args.S_steps
    }
    
    sweep = ParameterSweep(fixed_params, sweep_params, args.jobs)
    agg_path = sweep.run()
    
    return 0 if agg_path else 1

if __name__ == "__main__":
    sys.exit(main())
