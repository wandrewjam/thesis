### ====================================================================
### Makefile for the jump-velocity plots
### Current target list:
###	all
###	clean
###	avg-vel_test

TXT_FILES = $(wildcard par-files/*.txt)
PNG_FILES = $(patsubst par-files/%.txt, plots/%.png, $(TXT_FILES))
MP4_FILES = $(patsubst par-files/%.txt, plots/%.mp4, $(TXT_FILES))
NPZ_FILES = $(patsubst par-files/%.txt, npz-files/%.npz, $(TXT_FILES))
PYTHON = python
ANI_SRC = src/animate.py
PNG_SRC = src/avg-vel.py
NPZ_SRC = src/jv.py
ANI_EXE = $(PYTHON) $(ANI_SRC)
PNG_EXE = $(PYTHON) $(PNG_SRC)
NPZ_EXE = $(PYTHON) $(NPZ_SRC)

.PHONY : all
all : animations plots

.PHONY : animations
animations: $(MP4_FILES)

.PHONY : clean
clean :
	rm *~
	rm $(PNG_FILES)
	rm $(MP4_FILES)

.PHONY : distclean
distclean :
	rm *~
	rm $(PNG_FILES)
	rm $(MP4_FILES)
	rm $(NPZ_FILES)

.PHONY : data
data : $(NPZ_FILES)

.PHONY : install
install :
	@echo There is nothing to install for this package

.PHONY : plots
plots : $(PNG_FILES)

.PHONY : variables
variables :
	@echo TXT_FILES: $(TXT_FILES)
	@echo PNG_FILES: $(PNG_FILES)
	@echo MP4_FILES: $(MP4_FILES)
	@echo NPZ_FILES: $(NPZ_FILES)

###########################################

plots/%.mp4 : npz-files/%.npz $(ANI_SRC)
	$(ANI_EXE) $*

plots/%.png : npz-files/%.npz $(PNG_SRC)
	$(PNG_EXE) $*

npz-files/%.npz : par-files/%.txt $(NPZ_SRC)
	$(NPZ_EXE) $*
