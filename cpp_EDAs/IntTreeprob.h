#ifndef __INTTREEPROB_H
#define __INTTREEPROB_H

#include <stdlib.h>
#include <stdio.h>
#include "Popul.h"
#include "Treeprob.h"




class IntProbTree: public ProbTree
 {
	 public:
     
	  
	  unsigned *MaxValues; // Maxima cantidad de valores para cada variable
	  int **GenInd;
	  int **ParentValues;
	  unsigned Maximum_among_Minimum;

	  
	  IntProbTree(int,int*,int,int,unsigned*,Popul*);
	 ~IntProbTree();

	 
	 void CalMutInf();
	 double Prob(unsigned*);
	 void GenIndividual (Popul*,int);
	 void GenPop(int, Popul*);
	 int UnivFreq(int,unsigned);
	 double UnivProb(int, unsigned);
	 double CondProb(int, unsigned,int, unsigned);
	 
	 
	 void MakeProbStructures(); 
	 unsigned SonValue(int,unsigned,int);
	};

#endif 
