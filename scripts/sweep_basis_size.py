#!/usr/bin/env python3
"""
sweep_basis_size.py - Orchestrates basis size convergence sweeps with integrated plotting

Runs deu with increasing box strengths, collects results, plots convergence automatically.

Usage:
  python3 scripts/sweep_basis_size.py \
    --b_range 2.5 --b_form 1.5 --S 39.17 \
    --basis_size_steps 8 \
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

class BasisSizeSweep:
    """Manages basis size convergence sweep with integrated plotting"""
    
    def __init__(self, fixed_params, num_steps, num_jobs=1):
        self.scan_type = "basis_size"
        self.fixed_params = fixed_params
        self.num_steps = num_steps
        self.num_jobs = num_jobs
        self.results_dir = f"results/energy_sweep_basis_size"
        self.run_dir = self.results_dir
        
    def generate_parameter_combinations(self):
        """Generate parameter combinations with increasing basis sizes"""
        base_strengths = [0.0, 1.0, 0.1, 0.2, 0.5, 2.0, 0.01, 5.0]
        combinations = []
        
        for step in range(2, self.num_steps + 1):
            step_box_strengths = sorted(base_strengths[:step], reverse=True)
            params = dict(self.fixed_params)
            params.update({
                "box_strengths": step_box_strengths,
                "step": step
            })
            combinations.append(params)
        
        return combinations
    
    def run_deu(self, params):
        """Execute a single deu run with given parameters"""
        step = params.get("step", 0)
        param_str = f"step{step}_b_range={params['b_range']:.3f}_b_form={params['b_form']:.3f}_S={params['S']:.3f}"
        
        if "box_strengths" in params:
            box_str = ",".join([f"{x:.1f}" for x in params["box_strengths"]])
            param_str += f"_box={box_str}"
        
        run_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        run_id = f"run_{run_hash}"
        csv_path = os.path.join(self.run_dir, f"{run_id}.csv")
        
        cmd = ["./deu"]
        cmd.extend(["-b_range", str(params["b_range"])])
        cmd.extend(["-b_form", str(params["b_form"])])
        cmd.extend(["-S", str(params["S"])])
        cmd.extend(["--output-csv", csv_path])
        
        if "box_strengths" in params:
            box_strengths_str = ",".join([str(x) for x in params["box_strengths"]])
            cmd.extend(["-box-strengths", box_strengths_str])
        
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
                row = {"step": result["params"].get("step", 0)}
                row.update(result["params"])
                row.update(final_row)
                aggregated_rows.append(row)
        
        # Sort by step for readability
        aggregated_rows.sort(key=lambda r: int(r.get("step", 0)))
        
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
        print(f"SWEEP SUMMARY: Basis size convergence")
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
    
    def plot_basis_size_convergence(self, csv_file):
        """Plot energy, radius, and execution time convergence vs basis size"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("ERROR: matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
            return False
        
        data = {"step": [], "num_boxes": [], "energy": [], "basis_size": [], "radius": [], "execution_time": []}
        
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        step = int(row.get("step", 0))
                        box_strengths = row.get("box_strengths", "")
                        num_boxes = len([x for x in box_strengths.split(',') if x.strip()])
                        
                        data["step"].append(step)
                        data["num_boxes"].append(num_boxes)
                        data["energy"].append(float(row["energy_mev"]))
                        data["basis_size"].append(float(row.get("basis_size", 0)))
                        try:
                            data["radius"].append(float(row["radius_fm"]))
                        except (KeyError, ValueError):
                            data["radius"].append(None)
                        try:
                            data["execution_time"].append(float(row.get("execution_time_s", 0)))
                        except (KeyError, ValueError):
                            data["execution_time"].append(0)
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            print(f"ERROR reading CSV: {e}", file=sys.stderr)
            return False
        
        if not data["step"]:
            print("No basis_size convergence data found in CSV", file=sys.stderr)
            return False
        
        sorted_data = sorted(zip(data["step"], data["num_boxes"], data["energy"], data["basis_size"], data["radius"], data["execution_time"]))
        steps, num_boxes, energies, basis_sizes, radii, exec_times = zip(*sorted_data)
        
        fig, axes = plt.subplots(3, 1, figsize=(11, 10))
        
        ax1 = axes[0]
        ax1.plot(basis_sizes, energies, 'o-', linewidth=2.5, markersize=10, label='Computed', color='steelblue')
        ax1.axhline(y=-2.224, color='r', linestyle='--', linewidth=2, label='Target (-2.224 MeV)')
        ax1.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Energy (MeV)', fontsize=12, fontweight='bold')
        ax1.set_title('Basis Size Convergence: Ground State Energy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.legend(fontsize=11, loc='best')
        ax1.set_xticks(basis_sizes)
        
        if radii and any(r is not None and r > 0 for r in radii):
            ax2 = axes[1]
            radii_clean = [r if r is not None else 0 for r in radii]
            ax2.plot(basis_sizes, radii_clean, 's-', linewidth=2.5, markersize=10, color='green', label='Computed')
            ax2.axhline(y=2.128, color='r', linestyle='--', linewidth=2, label='Target (2.128 fm)')
            ax2.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Charge Radius (fm)', fontsize=12, fontweight='bold')
            ax2.set_title('Basis Size Convergence: Charge Radius', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle=':')
            ax2.legend(fontsize=11, loc='best')
            ax2.set_xticks(basis_sizes)
        
        if exec_times and any(t > 0 for t in exec_times):
            ax3 = axes[2]
            ax3.plot(basis_sizes, exec_times, '^-', linewidth=2.5, markersize=10, color='darkorange', label='Execution Time')
            ax3.set_xlabel('Basis Size', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
            ax3.set_title('Basis Size Convergence: Execution Time', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle=':')
            ax3.legend(fontsize=11, loc='best')
            ax3.set_xticks(basis_sizes)
        
        plt.tight_layout()
        output_file = csv_file.replace('aggregated.csv', 'basis_convergence.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        return True
    
    def run(self):
        """Execute the complete sweep"""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Starting basis size convergence sweep")
        print(f"  Fixed b_range: {self.fixed_params['b_range']}")
        print(f"  Fixed b_form: {self.fixed_params['b_form']}")
        print(f"  Fixed S: {self.fixed_params['S']}")
        print(f"  Basis size steps: {self.num_steps}")
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
            self.plot_basis_size_convergence(agg_path)
        
        # Save metadata
        metadata = {
            "scan_type": "basis_size",
            "timestamp": datetime.now().isoformat(),
            "fixed_params": self.fixed_params,
            "num_steps": self.num_steps,
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
        description="Basis size convergence sweep orchestrator for deuteron SVM solver"
    )
    
    parser.add_argument("--b_range", type=float, required=True,
                       help="Fixed Gaussian width search space (fm)")
    parser.add_argument("--b_form", type=float, required=True,
                       help="Fixed pion form factor range (fm)")
    parser.add_argument("--S", type=float, required=True,
                       help="Fixed pion coupling strength (MeV)")
    
    parser.add_argument("--basis_size_steps", type=int, default=8,
                       help="Number of basis size steps (default: 8)")
    parser.add_argument("--jobs", type=int, default=1,
                       help="Number of parallel jobs (default: 1)")
    
    args = parser.parse_args()
    
    fixed_params = {
        "b_range": args.b_range,
        "b_form": args.b_form,
        "S": args.S
    }
    
    sweep = BasisSizeSweep(fixed_params, args.basis_size_steps, args.jobs)
    agg_path = sweep.run()
    
    return 0 if agg_path else 1

if __name__ == "__main__":
    sys.exit(main())
