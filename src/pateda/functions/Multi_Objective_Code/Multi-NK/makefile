# makefile snps



GCC = g++

CC_FLAGS =  -c -Wno-deprecated -fno-rtti $(DEBUGGING) -Wall $(OPTIMIZE) $(PROFILING) 
LN_FLAGS = $(DEBUGGING) $(OPTIMIZE) $(PROFILING) 

.SUFFIXES: .cpp


.cpp.o:
	$(GCC) $(CC_FLAGS) $<
 
SRC =  Set.h Set.cpp Chi.h Chi.lib Chi.cpp auxfunc.h auxfunc.cpp ipmpar.cpp cdflib.h dcdflib.cpp NKModel.h NKModel.cpp Popul.h Popul.cpp mainMultiNK.cpp



OBJS =  Set.o Chi.o auxfunc.o ipmpar.o dcdflib.o NKModel.o Popul.o mainMultiNK.o


all: multink

multink  : $(OBJS)
	$(GCC) $(LN_FLAGS) $(OBJS) -o multink

edit:
	emacs $(SRC) &

tags:
	etags *.h *.cpp

clean:
	rm -f $(OBJS) multink










