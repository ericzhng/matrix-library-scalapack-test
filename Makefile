
all: cc cxx fortran

cc:
	make -C test-scalapack-c

cxx:
	make -C test-scalapack-cpp

fortran:
	make -C test-scalapack-fortran

clean :
	make -C test-scalapack-c clean
	make -C test-scalapack-cpp clean
	make -C test-scalapack-fortran clean
