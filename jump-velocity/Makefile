### ====================================================================
### Makefile for the jump-velocity plots
### Current target list:
###	all			generate all plots and data files		
###	clean			same as exact-clean
###	distclean		same as exact-clean and simulate-clean
###	exact			generate all plots of FP solns
###	exact-animations	generate all animations of the FP solns
###	exact-clean		remove plots and animations of solutions
###				to the Fokker-Planck equations
###	exact-data		save data from FP solns to .dat files
###	exact-plots		plot exit time distributions for all
###				parameter files
###	install			install files on system (dummy target)
###	simulate		run all simulate scripts
###	simulate-bootstrap	run bootstrap trials
###	simulate-clean		remove all simulated data, MLE fits, and
###				bootstrap data
###	simulate-data		generate simulated data
###	simulate-estimate	find MLE of parameter values
###	uninstall		uninstall installed files (dummy target)
###	variables		print definitions of all variables in
###				this Makefile
###
###	[23-Aug-2019]
### ====================================================================

### Variables are defined in config.mk
include config.mk

### ====================================================================

.PHONY : all
all : exact simulate

.PHONY : clean
clean : exact-clean

.PHONY : distclean
distclean : exact-clean simulate-clean

.PHONY : exact
exact : exact-animations exact-plots

.PHONY : exact-animations
exact-animations: $(MP4_FILES)

.PHONY : exact-clean
exact-clean :
	rm *~
	rm $(PNG_FILES)
	rm $(MP4_FILES)
	rm $(DST_FILES)

.PHONY : exact-data
exact-data : $(NPZ_FILES)

.PHONY : exact-plots
exact-plots : $(PNG_FILES) $(DST_FILES)

.PHONY : install
install :
	@echo There is nothing to install for this package

.PHONY : simulate
simulate : simulate-bootstrap simulate-estimate

.PHONY : simulate-bootstrap
simulate-bootstrap : $(BOO_FILES)

.PHONY : simulate-clean
simulate-clean :
	rm $(SIM_FILES)
	rm $(EST_FILES)
	rm $(BOO_FILES)

.PHONY : simulate-data
simulate-data : $(SIM_FILES)

.PHONY : simulate-estimate
simulate-estimate : $(EST_FILES)

.PHONY : uninstall
uninstall :
	@echo There is nothing to uninstall for this package
.PHONY : variables
variables :
	@echo TXT_FILES: $(TXT_FILES)
	@echo PNG_FILES: $(PNG_FILES)
	@echo MP4_FILES: $(MP4_FILES)
	@echo NPZ_FILES: $(NPZ_FILES)
	@echo DST_FILES: $(DST_FILES)
	@echo EXP_FILES: $(EXP_FILES)
	@echo SIM_FILES: $(SIM_FILES)
	@echo EST_FILES: $(EST_FILES)
	@echo BOO_FILES: $(BOO_FILES)
	@echo TRIALS: $(TRIALS)
	@echo PROC: $(PROC)

### ====================================================================
### Define rules for specific targets

### Exact numerical simulations

## Animate solution to the FP equations
plots/%.mp4 : npz-files/%.npz $(ANI_SRC)
	$(ANI_EXE) $*

## Plot PDF and CDF of exit times
plots/%.png : npz-files/%.npz $(PNG_SRC)
	$(PNG_EXE) $*

###### IS THIS CORRECT?
dat-files/distributions/%-dst.dat : npz-files/%.npz $(PNG_SRC)
	$(PNG_EXE) $*

### Solve FP equations and save to .npz file
npz-files/%.npz : par-files/%.txt $(NPZ_SRC)
	$(NPZ_EXE) $*

### Statistical simulations

## Simulate test data from a given set of parameter values
dat-files/simulations/%-sim.dat : par-files/experiments/%.txt $(SIM_SRC)
	$(SIM_EXE) $*

## Estimate parameters from the simulated data
dat-files/ml-estimates/%-est.dat : dat-files/simulations/%-sim.dat $(EST_SRC)
	$(EST_EXE) $*

## Run bootstrap trials on the simulated data
dat-files/bootstrap/%-boot.dat : dat-files/simulations/%-sim.dat $(BOO_SRC)
	$(BOO_EXE) $* $(TRIALS) $(PROC)

### Fit model parameters to experimental data

## Estimate maximum likelihood parameters based on both models
dat-files/ml-estimates/%-est.dat : dat-files/simulations/%-sim.dat $(EST_SRC)
	$(EST_EXE) $*

## Run bootstrap trials on the data
dat-files/bootstrap/%-boot.dat : dat-files/simulations/%-sim.dat $(BOO_SRC)
	$(BOO_EXE) $* $(TRIALS) $(PROC)
