### ====================================================================
### Makefile for the jump-velocity plots
### Current target list:
###	all
###	clean
###	avg-vel_test

avg-vel_fourstate-largea.png : fourstate-largea.txt jv.py
	python jv.py fourstate-largea.txt

avg-vel_fourstate-meda.png : fourstate-meda.txt jv.py
	python jv.py fourstate-meda.txt

avg-vel_fourstate-smalla.png : fourstate-smalla.txt jv.py
	python jv.py fourstate-smalla.txt

avg-vel_test.png : test.txt jv.py
	python jv.py test.txt

avg-vel_twostate-largeeps-largea.png : twostate-largeeps-largea.txt jv.py
	python jv.py twostate-largeeps-largea.txt

avg-vel_twostate-largeeps-meda.png : twostate-largeeps-meda.txt jv.py
	python jv.py twostate-largeeps-meda.txt

avg-vel_twostate-largeeps-smalla.png : twostate-largeeps-smalla.txt jv.py
	python jv.py twostate-largeeps-smalla.txt

avg-vel_twostate-smalleps-largea.png : twostate-smalleps-largea.txt jv.py
	python jv.py twostate-smalleps-largea.txt

avg-vel_twostate-smalleps-meda.png : twostate-smalleps-meda.txt jv.py
	python jv.py twostate-smalleps-meda.txt

avg-vel_twostate-smalleps-smalla.png : twostate-smalleps-smalla.txt jv.py
	python jv.py twostate-smalleps-smalla.txt

clean :
	rm *~

install :
	@echo There is nothing to install for this package

plots : avg-vel_fourstate-largea.png avg-vel_fourstate-meda.png avg-vel_fourstate-smalla.png avg-vel_test.png avg-vel_twostate-largeeps-largea.png avg-vel_twostate-largeeps-meda.png avg-vel_twostate-largeeps-smalla.png avg-vel_twostate-smalleps-largea.png avg-vel_twostate-smalleps-meda.png avg-vel_twostate-smalleps-smalla.png
