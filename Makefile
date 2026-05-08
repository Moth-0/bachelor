MAKEFLAGS += -r
# Variables
CXXFLAGS = -Wall -Werror -O3 -std=c++23 -fopenmp
CXX = c++
LDFLAGS += -fopenmp
LDLIBS = -lstdc++ -lm
RM = rm -f

all : deu

HEADERS = qm/matrix.h qm/gaussian.h qm/operators.h qm/jacobi.h qm/solver.h qm/serialization.h qm/csv_writer.h deuterium.h SVM.h

% : %.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o : %.cc $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

search: deu params
	@echo "======================================================================"
	@echo "Step 1: Computing binding energy with b_range=$(b_range) b_form=$(b_form) S=$(S)"
	@echo "======================================================================"
	./deu -b_range $(b_range) -b_form $(b_form) -S $(S) 
	@echo ""
	@echo "======================================================================"
	@echo "Step 2: Binary search for S with  b_range=$(b_range) b_form=$(b_form)"
	@echo "======================================================================"
	./params $(b_range) $(b_form)

convergence.png: convergence.data Makefile
	echo '\
	set terminal pngcairo size 800,600 enhanced font "Arial,11" ;\
	set output "convergence.png" ;\
	set title "Convergence of SVM under Different Kinematics" ;\
	set xlabel "SVM Iterations" ;\
	set ylabel "Ground State Energy (MeV)" ;\
	set grid ;\
	set key top right box ;\
	plot for [i=0:*] "convergence.data" index i using 1:2 with lines lw 2 title columnheader(2), \
	     -2.224 with lines dt 2 lw 2 lc rgb "purple" title "Target (-2.224 MeV)" \
	' | tee plot.gpi | gnuplot

clean :
	$(RM) *.o *.log *.dat *.gpi

# ============================================================================
# PHASE 3: SWEEP & PLOTTING TARGETS (NEW AUTOMATED PIPELINE)
# ============================================================================

# === SWEEP TARGETS ===

# Sweep b_form parameter (varies 1.0 to 2.5, 10 steps)
sweep-b_form: deu
	@echo "=========================================="
	@echo "Sweeping b_form (1.0 - 2.5, 10 steps)"
	@echo "=========================================="
	@mkdir -p results/energy_sweep_b_form
	python3 scripts/sweep.py --scan b_form \
	  --b_range 2.5 --S 38.4 \
	  --b_form_min 1.0 --b_form_max 2.5 --b_form_steps 10 \
	  --jobs 4

# Sweep S parameter (coupling strength: 20 - 100, 15 steps)
sweep-S: deu
	@echo "=========================================="
	@echo "Sweeping S (20 - 100 MeV, 15 steps)"
	@echo "=========================================="
	@mkdir -p results/energy_sweep_S
	python3 scripts/sweep.py --scan S \
	  --b_range 2.5 --b_form 1.5 \
	  --S_min 20 --S_max 100 --S_steps 15 \
	  --jobs 4

# Sweep basis size limit (10 - 60, 10 steps)
sweep-basis-size: deu
	@echo "=========================================="
	@echo "Sweeping max_basis_size (10 - 60, 10 steps)"
	@echo "=========================================="
	@mkdir -p results/basis_size_convergence
	python3 scripts/sweep.py --scan basis_size \
	  --b_range 2.5 --b_form 1.5 --S 38.4 \
	  --max_basis_size_min 10 --max_basis_size_max 60 --max_basis_size_steps 10 \
	  --jobs 4

# === PLOTTING TARGETS ===

results/energy_sweep_b_form/aggregated.csv: sweep-b_form
	@echo "Aggregated results available at: $@"

results/energy_sweep_S/aggregated.csv: sweep-S
	@echo "Aggregated results available at: $@"

results/basis_size_convergence/aggregated.csv: sweep-basis-size
	@echo "Aggregated results available at: $@"

plot-energy-vs-b_form: results/energy_sweep_b_form/aggregated.csv
	@echo "Plotting energy vs b_form..."
	python3 scripts/plot_results.py energy_sweep_b_form

plot-energy-vs-S: results/energy_sweep_S/aggregated.csv
	@echo "Plotting energy vs S..."
	python3 scripts/plot_results.py energy_sweep_S

# === ORCHESTRATION TARGETS ===

all-sweeps: sweep-b_form sweep-S sweep-basis-size
	@echo "=========================================="
	@echo "All parameter sweeps complete!"
	@echo "Results in: results/energy_sweep_*/"
	@echo "=========================================="

all-plots: plot-energy-vs-b_form plot-energy-vs-S
	@echo "=========================================="
	@echo "All plots generated!"
	@echo "View: results/energy_sweep_*/plot.png"
	@echo "=========================================="

full-experiment: all-sweeps all-plots
	@echo "=========================================="
	@echo "FULL EXPERIMENT COMPLETE"
	@echo "=========================================="
	@ls -lh results/*/plot.png 2>/dev/null || echo "(Plots may still be generating...)"
	@echo "CSV results: results/energy_sweep_*/aggregated.csv"
	@echo "=========================================="

.PHONY: sweep-b_form sweep-S sweep-basis-size plot-energy-vs-b_form plot-energy-vs-S all-sweeps all-plots full-experiment