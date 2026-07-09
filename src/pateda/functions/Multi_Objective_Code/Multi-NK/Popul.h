#ifndef __POPUL_H
#define __POPUL_H 
 
#include <stdlib.h>  
#include <stdio.h> 
#include <iostream>    
#include <fstream>    
#include "Set.h"    

using namespace std;
 
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
          Popul();
	  Popul(int,int,int); 
	  Popul(int,int,int,unsigned*); 
	  void RandInit();
          void  RandInitIndiv(int) ;
          virtual void Print();  
          virtual void Print(int);
          virtual void Print(int,int);
	  void ProbInit(); 
	  virtual void TournSel(Popul*,int); 
	  virtual void TruncSel(Popul*,int); 
	  virtual void SetElit(int, Popul*); 
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
          virtual int CompactPop(Popul*,double*);
          virtual void CopyPop(Popul*);
          virtual int CompactPopNew(Popul*,double*); 
          virtual void BotzmannDist(double, double*); 
          virtual void ProporDist(double*); 
          virtual void UniformProb(int, double* );
          virtual void SUSSel(int,Popul* ,double* );
          virtual double FindBestVal();
          virtual int FindBestIndPos();
          virtual void Merge(int, int, Popul*);
          virtual int FindBestClosestChrom(int,int,unsigned int*,double); 
          virtual void OrderPop(); 
          virtual void Merge2Pops(Popul*,int,Popul*,int);
          void Mutation(int,double); 
 
	 }; 


	class MultiPopul :public Popul {  
	 public: 
	  int NObj;
	  double **MultiEvaluations; 
          CSet *listaindividuosdominados;
	  int *listacontindividuosquedominana;
	  int *listarank;

          MultiPopul(int,int,int,int); 
	  MultiPopul(int,int,int,unsigned*,int); 
	  virtual void Print();  
          virtual void Print(int);
          virtual void Print(int,int);
          virtual double FindBestVal(int); // The best val for objective number  given by parameter
 	  virtual int  FindBestIndPos(int); 
          virtual void TournSel(MultiPopul*,int); 
	  virtual void TruncSel(MultiPopul*,int);
	  void FindDominanceRelationships();
          void FindParetoSet();
          void FillPopFromParetoSet(MultiPopul*,int);
          void ParetoRankingSel(MultiPopul*,int);   

          void FindParetoSetRestricted(int*);
          void ParetoRankingSelRestricted(MultiPopul*,int*);   
          virtual void BotzmannDist(double, double*); 
          virtual void ProporDist(double*);        
          virtual void SUSSel(int,MultiPopul* ,double* );
          virtual void CopyPop(MultiPopul*);  
	  virtual int CompactPop(MultiPopul*,double*);
          virtual int CompactPopNew(MultiPopul*,double*); 
          virtual void SetElit(int, MultiPopul*); 
          virtual void OrderPop(); 
          virtual void Merge2Pops(MultiPopul*,int,MultiPopul*,int);   
	  void SetVals(int,double*); 
          void GetVals(int,double*); 
          virtual void Merge(int, int, MultiPopul*);
          virtual int FindBestClosestChrom(int,int,unsigned int*,double); 
          ~MultiPopul();   
             
	 }; 

#endif  
