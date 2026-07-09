# makefile EDABRAID 


GCC = g++

CC_FLAGS = -c -Wall $(OPTIMIZE)
LN_FLAGS = -lm

# Standard include directories
#CCINC=-I/home/rsantana/Work/Centro/DecimationEDAs/libdai/include/

# Destination directory of libDAI library
#LIB=lib

# LINKER
# Standard libraries to include
#LIBS=-ldai -lgmpxx -lgmp

defines = -DFAKE_QUATERNIONS


.SUFFIXES: .cpp


.cpp.o:
	$(GCC) $(defines) $(CC_FLAGS) $<
 
SRC =    AbstractTree.h  AbstractTree.cpp Popul.h Popul.cpp  auxfunc.h auxfunc.cpp ipmpar.cpp cdflib.h dcdflib.cpp  random.h random.cpp quaternion.h quaternion.cpp edabraid.cpp 

OBJS =   AbstractTree.o  Popul.o auxfunc.o ipmpar.o dcdflib.o random.o quaternion.o edabraid.o

all: edabraid

edabraid: $(OBJS)
	$(GCC) $(defines) $(LN_FLAGS) $(OBJS) -o edabraid
edit:
	emacs $(SRC) &

tags:
	etags *.h *.cpp

clean:
	rm -f $(OBJS) edabraid








