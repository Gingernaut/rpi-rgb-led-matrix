# The *.cpp files are included in the distribution; this is only needed when
# working on the pyx files.
#
# Please check in modified *.cpp files with distribution to not require cython
# to be installed on the users' machine.
# for python3: make PYTHON=$(which python3) CYTHON=$(which cython3)
CYTHON ?= cython3

all : core.cpp graphics.cpp

# Cython reads the .pxd declarations too, so regenerate when they change
core.cpp : core.pxd cppinc.pxd
graphics.cpp : graphics.pxd cppinc.pxd core.pxd

%.cpp : %.pyx
	$(CYTHON) --cplus -o $@ $<

clean:
	rm -rf core.cpp graphics.cpp
