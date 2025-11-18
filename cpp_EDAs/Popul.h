#ifndef __POPUL_H
#define __POPUL_H 
 
#include <stdlib.h>  
#include <stdio.h> 
 
extern int* params; 
 
 	class Popul { 
	 public: 
	  int vars; 
      unsigned **P; 
	  unsigned *dim;  
	  unsigned int *index; 
	  double *Evaluations; 
	  double meaneval; 
            int psize; 
	  int elit; 
	  int Tour; 
      int genepoollimit; 
 
	  Popul(int,int,int); 
	  Popul(int,int,int,unsigned*); 
	  void RandInit();
          void  RandInitIndiv(int) ;
          void Print();  
          void Print(int);
          void Print(int,int);
	  void ProbInit(); 
	  void TournSel(Popul*,int); 
	  void TruncSel(Popul*,int); 
	  void SetElit(int, Popul*); 
      void InitIndex(); 
	  void AssignChrom(int ,Popul* ,int ); 
	  void Evaluate(int,int); 
	  void EvaluateAll(int); 
	  void SetGenePoolSize(int); 
	  double Fitness(int);  
	  unsigned* Ind(int); 
	  void SetVal(int,double); 
          double GetVal(int); 
	  void Repair(int,int); 
	  void Repair(int,int,double*); 
	   ~Popul(); 
          int CompactPop(Popul*,double*);
          void CopyPop(Popul*);
          int CompactPopNew(Popul*,double*); 
          void BotzmannDist(double, double*); 
          void ProporDist(double*); 
          void UniformProb(int, double* );
          void SUSSel(int,Popul* ,double* );
          double FindBestVal();
          int FindBestIndPos();
          void Merge(int, int, Popul*);
          int FindBestClosestChrom(int,int,unsigned int*,double); 
          void OrderPop(); 
          void Merge2Pops(Popul*,int,Popul*,int);
 
	 }; 
#endif  
