#ifndef __FDA_H   
#define __FDA_H   
   
#include <stdlib.h>   
#include <stdio.h>   
#include "Popul.h"   
#include "TriangSubgraph.h"   
#include "AbstractTree.h" 
#include "ProteinClass.h"   
   
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
         int NeededOvelapp;
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
         clique** ListOvelap;
         HPProtein* FoldingProtein;
         int current_gen;

    DynFDA(int,int,int,double,int,int,int);   
    DynFDA(int,Popul*,int,int,double,int,int,int);   
    DynFDA(int,Popul*,int,int,int*,double,int,int,int);    
   
   virtual ~DynFDA();   
     
    virtual void CallProb();	     
    virtual void GenIndividualMNFDA (Popul*,int,int*,int*);  
    virtual void GenPop(int, Popul*); 
    virtual void GenPop(int,int, Popul*);  
    void GenPopMPM(int,int, Popul*);  
    void GenPopProtein(int, Popul*, double);   
    void GenBiasedPop(int,Popul*,unsigned int*,double); 
    void GenStructuredCrossPop(int,Popul*,Popul*);
    void GenCrossPop(int,Popul*,Popul*);
    void MarkovCallProb(); 
    void FDACallProb(); 
    void FGDACallProb();
    void FDACallProbFromMatrix();  
    void FindChiSquareMat(double);  
    void FindChiSquareMatBiv(double);  
    void CorrectChiSquareMat();  
    void CorrectChiSquareMatGTest(unsigned int*,double);
    void FindRandomSimMatrix();   
    void CreateGraphCliques();   
    void CreateGraphCliquesProtein(int,int);
    void CreateGraphCliquesProteinMPM(int, int, unsigned long**);
    virtual void Destroy(); 
    void DestroyFDAOverl();
    void DestroyKikuchi();
    void FindCliquesWeights();  
    void SimpleOrdering();  
    void SetOrderofCliques();  
    virtual void UpdateModel(); 
    virtual void UpdateModel(int**);
    virtual void UpdateModelSAT(int**);
    virtual void UpdateModelProtein(int,int);
    virtual void UpdateModelProteinMPM(int, int, unsigned long**);
    virtual void UpdateModelGTest(unsigned int*,double); 
    virtual void UpdateModelFromMatrix(int**);   
    void SetFactorizationFromMatrix(int**);
 
    void SetNPoints(int,double*);  
    void SetNPoints(int,int,double*);  
    void LearnMatrix();  
    void LearnMatrix(int**); 
    void LearnMatrixSAT(int**); 
    void LearnMatrixGTest(unsigned int*,double); 
    void LearnMatrixFromOther(int**);  
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
    void OrderCliquesForProteins();
    //void AncestralOrdering(int); 
    //void AncestralOrderingOnlyCliq(int); 
    void AncestralOrderingFact(int);  
    void AncestralOrderingFactOnlyCliq(int);  
    int FindMargForGS(int*,int);   
    //int KikuchiFindMargForGS(int*,int);  
    void GenIndividualMRF(Popul*,int,int*,int*);  
    void GenIndividualMOA(Popul*,int,int*,int*,int);  
    void GenIndividualMNFDAProtein(Popul*,int,int*,int*);  
    void NewGenIndividualMNFDAProtein(Popul*,int,int*,int*);   
    void GenBiasedIndividual(Popul*,int,unsigned int*,double);
    void GenStructuredCrossInd(Popul*,int, unsigned int*, unsigned int*); 
    void GenOnePointCXInd(Popul*,int, unsigned int*, unsigned int*); 
    void printmarg();  
    int SamplingVarKikuchi(int*,int); 
    void CreateMarg(int); 
    void ComputeMarg(int); 
    void Normalize(int); 
    void FindUnivProb(); 
    void FindNeighbors(long unsigned**,int,double);  
    void FindStrongNeighbors(unsigned int*, double,long unsigned**,int,double);  
    virtual void SetPriors();
    virtual void UpdateModel(double*,int,Popul*);
    void InitTree(int,double*,Popul*,int);
    virtual void GenIndividual (Popul*,int,int);
    virtual double Prob(unsigned*,int); 
    virtual double Prob(unsigned*); 
    double ProbKikuchi(unsigned*);
    void SetProtein(HPProtein*);
   };   
#endif   
   
  
  
  
  
  
  
  
