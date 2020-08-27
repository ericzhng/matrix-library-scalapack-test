

# define compiler and compiler flag variables
CC		= mpicc
FC		= mpif90
CXX		= mpic++
CFLAGS = -O3 -Wall
FCFLAGS = -O3 -Wall
CPPFLAGS = -O3 -Wall -std=c++11
LDFLAGS = -lm -lrt -lgfortran

LDINC = 
REQUIRED_LDFLAGS = $(SCALAPACKLIB) $(LAPACKLIB) $(BLASLIB)
# if shared library, then add option: -Wl,-rpath,$(SCALAPACK_DIR)/lib 

EXTRA_INC = -I./common
EXTRA_LIB = ./common/pdscaexinfo.o

# define source folder (src or .) and its content
SRCDIR	= src

# find all other file names or specify them
SUF  ?= .f
SRCS := $(shell find $(SRCDIR) -maxdepth 1  -name '*$(SUF)' -or -name '*.cpp')
ifeq ($(SRCDIR),.)
SRCS1 = $(notdir $(SRCS))
else
SRCS1 = $(SRCS)
endif
SRCOBJS = $(SRCS1:%$(SUF)=%.o)
TARGETS := $(patsubst %.o,%,$(SRCOBJS))

default: $(TARGETS)
	@echo Make Complete

# pattern rules to compile objectives
$(TARGETS): %: %.o $(EXTRA_LIB)
	$(FC) $(CFLAGS) $(LDINC) $(EXTRA_INC) -o $@ $< $(EXTRA_LIB) $(REQUIRED_LDFLAGS) $(LDFLAGS)
	@rm $<
	mpirun --mca btl_vader_single_copy_mechanism none -np 4 ./$@ > $(notdir $@.txt)
	@echo

EXTRA_LIB_C = $(EXTRA_LIB:%.o=%$(SUF))
$(EXTRA_LIB):$(EXTRA_LIB_C)
	$(FC) $(FCFLAGS) -c $< -o $@ $(LDINC) $(EXTRA_INC)
.f.o:
	@echo '====================================='
	@echo 'Compile and execute: ' $<
	$(FC) $(FCFLAGS) -c $< -o $@ $(LDINC) $(EXTRA_INC)
.c.o:
	@echo '====================================='
	@echo 'Compile and execute: ' $<
	$(CC) $(CFLAGS) -c $< -o $@ $(LDINC) $(EXTRA_INC)
.cpp.o:
	@echo '====================================='
	@echo 'Compile and execute: ' $<
	$(CXX) $(CPPFLAGS) -c $< -o $@ $(LDINC) $(EXTRA_INC)

clean:
	rm -rf *.o $(TARGETS)
