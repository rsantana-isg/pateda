#ifndef __FDA_H   
#define __FDA_H   
   
#include <stdlib.h>   
#include <stdio.h>   
#include "Popul.h"   
#include "TriangSubgraph.h"   
#include "AbstractTree.h"   
   
 class DynFDA:public AbstractProbModel{   
	 public:   
    
     double *AllProb;   // Univariate Probabilities of the individuals 
     maximalsubgraph* SetOfCliques;   
     unsigned int** Matrix;  
     double* MChiSquare;  
	 int overlappedcliques;   
         int CliqMLength,MaxNCliq;  
	 int*  order;  
         double* PopProb;  
         double threshold;  
         int* OrderCliques; 
         int NeededCliques;  
         double freqxyz[4096],freqxz[2048],freqyz[2048],freqz[1024]; //auxiliary arrays for cal.marginals  
	 int Pot[12]; //We assume max clique = 12  
         int Priors;  
         double* CliqWeights;  
         KikuchiApprox* CondKikuchiApprox;  
         memberlistKikuchiClique** ListKikuchiCliques; 
         int LearningType; 
         int cycles; 
         int* auxvectGen;  
         int* legalconfGen;  
 
    DynFDA(int,int,int,double,int,int,int);   
    DynFDA(int,Popul*,int,int,double,int,int,int);   
    DynFDA(int,Popul*,int,int,int*,double,int,int,int);    
   
   virtual ~DynFDA();   
     
    virtual void CallProb();	     
    virtual void GenIndividualMNFDA (Popul*,int,int*,int*);   
    virtual void GenPop(int, Popul*);  
    void MarkovCallProb(); 
    void FDACallProb(); 
    void FindChiSquareMat(double);  
    void FindChiSquareMatBiv(double);  
    void CorrectChiSquareMat();  
    void FindRandomSimMatrix();   
    void CreateGraphCliques();   
    void DestroyGraphCliques();   
    void FindCliquesWeights();  
    void SimpleOrdering();  
    void SetOrderofCliques();  
    virtual void UpdateModel(); 
   virtual void UpdateModel(int**);
    void SetNPoints(int,double*);  
    void SetNPoints(int,int,double*);  
    void LearnMatrix();  
    void LearnMatrix(int**);  
    void PrintMatrix(double*);  
    void PrintSimMatrix();  
    void TruncChiSquareMatrix(int,int);  
    void OrderCliquesSizes(int);  
    //double FindChiVal(double,int,double);  
    double DepTest(int,int,int*,int*);   
    void  DepSecOrder(int,double);  
    void SetEdgesPerm(clique*);  
    double FindChiSquarelimit();  
    double FindBivlimit();  
    void kmeans(int, double , int*,double*,double*);   
    void SetMatrix(int,double,double);  
    double CliqWeight(clique*);  
    void OrderCliquesWeights(int);  
    //void AncestralOrdering(int); 
    //void AncestralOrderingOnlyCliq(int); 
    void AncestralOrderingFact(int);  
    void AncestralOrderingFactOnlyCliq(int);  
    int FindMargForGS(int*,int);   
    //int KikuchiFindMargForGS(int*,int);  
    void GenIndividualMRF(Popul*,int,int*,int*);  
    void printmarg();  
    int SamplingVarKikuchi(int*,int); 
    void CreateMarg(int); 
    void ComputeMarg(int); 
    void Normalize(int); 
    void FindUnivProb(); 
    virtual void SetPriors();
    virtual void UpdateModel(double*,int,Popul*);
    void InitTree(int,double*,Popul*,int);
    virtual void GenIndividual (Popul*,int,int);
};   
  
  
#endif   
   
  
  
  
  
  
  
  
