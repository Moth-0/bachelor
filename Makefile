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

# ALL CONFIGURATIONS: Run all 4 kinematic configs with timestamped results
# Default parameters (override with: make all-configs B_RANGE=2.0 B_FORM=1.5 S=40.0)
B_RANGE ?= 2.5
B_FORM ?= 1.2
S ?= 38.4

all-configs: deu
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	RESULTS_DIR="results/all_configs_$$TIMESTAMP"; \
	mkdir -p $$RESULTS_DIR; \
	echo "========================================"; \
	echo "Running all 4 configurations..."; \
	echo "Parameters: b_range=$(B_RANGE) b_form=$(B_FORM) S=$(S)"; \
	echo "Timestamp: $$TIMESTAMP"; \
	echo "Results: $$RESULTS_DIR"; \
	echo "========================================"; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --output-csv $$RESULTS_DIR/PN_Cla_Pi_Cla.csv; \
	echo "✓ Config 1/4: PN_Cla Pi_Cla"; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pi-rel --output-csv $$RESULTS_DIR/PN_Cla_Pi_Rel.csv; \
	echo "✓ Config 2/4: PN_Cla Pi_Rel"; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pn-rel --output-csv $$RESULTS_DIR/PN_Rel_Pi_Cla.csv; \
	echo "✓ Config 3/4: PN_Rel Pi_Cla"; \
	./deu -b_range $(B_RANGE) -b_form $(B_FORM) -S $(S) --pn-rel --pi-rel --output-csv $$RESULTS_DIR/PN_Rel_Pi_Rel.csv; \
	echo "✓ Config 4/4: PN_Rel Pi_Rel"; \
	echo "========================================"; \
	echo "All configurations complete!"; \
	echo "Results saved to: $$RESULTS_DIR"; \
	echo "========================================"


# Parameter Sweeps 
sweep_S : deu scripts/sweep.py script/plot_result.py Makefile
	python3 scripts/sweep.py --scan S --S_min 35.0 --S_max 37.0 --S_steps 8
	python3 scripts/plot_result.py energy_sweep_S

sweep_b : deu scripts/sweep.py script/plot_result.py Makefile
	python3 scripts/sweep.py --scan b_form --b_form_min 1.0 --b_form_max 1.4 --b_form_steps 4
	python3 scripts/plot_result.py energy_sweep_S
	python3 scripts/sweep.py --scan b_range --b_range_min 2.0 --b_range_max 3.0 --b_range_steps 5
	python3 scripts/plot_result.py energy_sweep_S

sweep_size : deu scripts/sweep.py script/plot_result.py Makefile
	python3 scripts/sweep.py --scan basis_size 10
	python3 scripts/plot_result.py energy_sweep_basis_size

clean :
	$(RM) *.o *.log *.dat *.gpi *.out *.err

.PHONY: all clean all-configs sweep_S sweep_b sweep_size