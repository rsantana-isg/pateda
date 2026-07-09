#ifndef __TREEPROB_H
#define __TREEPROB_H

#include <stdlib.h>
#include <stdio.h>
#include "Popul.h"



	class ProbTree {
	 public:
 
	      
	 int  length;             //  Cantidad de variables
     double *AllProb;           //Se almacenaran frecuencias y luego probabilidades de cada gen
	 int genepoollimit; // Cantidad de muestras o vectores de la poblacion
     double *MutualInf;
     double **AllSecProb;	  
	 int rootnode;
	 int actualpoolsize;
	 int* Tree;
	 int* Queue;
	 int *actualindex;
	 Popul* Pop;

	 FILE *f1;

	  
  	  void CalProbFvect(Popul*,double*);
	  void AddCase(int* );
	  void ResetProb();
	  void IndepTestInit();
	  void PrintMut();

	  ProbTree(int,int*,int,int,Popul*);
	 ~ProbTree();

	 void CalProb(Popul* );
	 void CalMutInf();
	 int FindRootNode();
	 int RandomRootNode();
	 void MakeTree(int);
     void MakeRandomTree(int);
	 void GenPop(int, Popul*);
	 void SetGenePoolSize(int);
	 double Prob(unsigned*);
	 void GenIndividual (Popul*,int);
	 int NextInOrder(int);
	 void ArrangeNodes();
	};

#endif

 
