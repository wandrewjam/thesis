### ====================================================================
### Makefile for the jump-velocity plots
### Current target list:
###	all
###	clean
###	distclean
###	exact
###	exact-animations
###	exact-clean
###	exact-data
###	exact-plots
###	install
###	simulate
###	simulate-bootstrap
###	simulate-clean
###	simulate-data
###	simulate-estimate
###	variables

include config.mk

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

###########################################
# Defines rules for specific targets

# Exact numerical simulations
plots/%.mp4 : npz-files/%.npz $(ANI_SRC)
	$(ANI_EXE) $*

plots/%.png : npz-files/%.npz $(PNG_SRC)
	$(PNG_EXE) $*

dat-files/distributions/%-dst.dat : npz-files/%.npz $(PNG_SRC)
	$(PNG_EXE) $*

npz-files/%.npz : par-files/%.txt $(NPZ_SRC)
	$(NPZ_EXE) $*

# Statistical simulations
dat-files/simulations/%-sim.dat : par-files/experiments/%.txt $(SIM_SRC)
	$(SIM_EXE) $*

dat-files/ml-estimates/%-est.dat : dat-files/simulations/%-sim.dat $(EST_SRC)
	$(EST_EXE) $*

dat-files/bootstrap/%-boot.dat : dat-files/simulations/%-sim.dat $(BOO_SRC)
	$(BOO_EXE) $* $(TRIALS) $(PROC)
