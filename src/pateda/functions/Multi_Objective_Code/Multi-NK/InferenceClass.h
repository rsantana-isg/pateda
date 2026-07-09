#ifndef __INF_H 
#define __INF_H 

//#include "mex.h"
//#include "fillMethods.h"
#include <iostream> 
#include <fstream>
#include <memory> 
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include "Loopy.h"
#include "GBP.h"
#include "Gibbs.h"
#include "Wolff.h"
#include "SwendsenWang.h"
#include "Metropolis.h"
#include "MeanField.h"
#include "GBPPreProcessor.h"




  // **************************************************************************
  // reading input arguments
  // **************************************************************************

  // check number of input arguments.
  // arguments should be:
  //
  // adjMat - 1xN cell array, each cell {i} is a row vector with the indices of
  //          i's neighbours
  //
  // lambda - there are 2 forms for lambda:
  //          1. in general MRF algorithms (loopy, gbp, gibbs, mean-field) :
  //             lambda should be a cell array of 1xN, each cell {i} is a cell
  //             array of 1xneighbNum(i). each cell {i}{n} is a VixVj matrix,
  //             where j is the n-th neighbour of i
  //          2. in PottsMRF alhorithms (monte-carlo algorithms which are planned
  //             for Potts model, i.e. metropolis and the cluster algorithms
  //             wolff and swendsen-wang) :
  //             here lambda should be 1xN cell array, each cell {i} is a row
  //             vector with the strength of interaction of i with each of its
  //             neighbours
  // note: Psi{i,j} = exp( [lambda(i,j), 0; 0, lambda(i,j)] )
  //
  // local - cell array of Nx1, each cell {i} is a row vector of length Vi
  //
  // algorithm - integer representing the inference algorithm to use, see the
  //             enumerator algorithmType at the top of this page
  //
  // temperature - double scalar, the temperature of the system
  //
  // model - integeger representing the model, see the enumerator in "definitions.h"
  //
  // trw - use Tree-Reweighted
  //
  // for other parameters required for each algorithm see the header of "inference.m"
  //
  //
  // note: N = number of nodes, V = number of possible values


// *****************************************************************************
// enumerators
// *****************************************************************************

enum algorithmType {AT_LOOPY,AT_GBP,AT_GIBBS,AT_WOLFF,AT_SWENDSEN_WANG,
		    AT_METROPOLIS,AT_MEAN_FIELD};


class CandidateNode{ 
public: 

  double v;
  int node;
  int state;
  int initBN;
  int verified;
  
  CandidateNode(double,int,int,int,int); 
  CandidateNode(); 
  void setvalues(double,int,int,int,int);
  void assign(CandidateNode);
  
  // void pop_front();
};
/*

class VectCandNode{ 
public: 
  CandidateNode* content;
  VectCandNode* previous;
  VectCandNode* next;
  VectCandNode(CandidateNode*, VectCandNode*, VectCandNode*);
  ~VectCandNode();
};

class VectCandidate{ 
public:    
  VectCandNode* first;
 VectCandNode*  current;
 VectCandNode*  last;
  VectCandidate();
  ~VectCandidate();
  void push_back(CandidateNode*);
  void begin();
  void end();
  void erase(VectCandNode*);
  void sort();  
  int  empty();
  void print();
};

*/
class InferenceClass{ 
public: 
  
   int num_nodes;
   int algo_type;
   int model;
   vector<Nodes> *adjMatA;
   //MRF* mrf;
   MRF* main_mrf;
   InferenceAlgorithm* algorithm;
   int converged;
   double** beliefs;
   double** singleBeliefs;
   double**** pairBeliefs;
   int MaxCardNumber; 
   double temperature;
   bool potts_model;
   bool monte_carlo;
   double BestEnergy;
   unsigned* BestConf;

  // variables for Loopy
  Strategy strategy;
  SumOrMax sumOrMax;
  double gbp_alpha;
  
  
  // variables for GBP
  int*** assignInd;
  double* bethe;
  GBPPreProcessor* processor;
  MRF* reg_mrf;
  RegionLevel* regions;
  vector<RegionLevel>* allRegions;
  bool allLevels;
  bool trw;
  
  // variables for Monte-Carlo
  int burningTime, samplingInterval, num_samples;
  int* startX;

  // for Loopy,GBP,Mean-Field
  int maxIter;

  InferenceClass(algorithmType, Model, int, unsigned int**, unsigned int*, double**, double***, double**, double);
  InferenceClass(InferenceClass*);

  ~InferenceClass();
    void fillAdjMat(unsigned int** Mat, vector<Nodes>&);
   void fillLocalMat(unsigned int*, double**, MRF*);
   void fillPsiMat(double***, MRF*); 
   void fillLambdaMat(double**, PottsMRF*); 
    void CopyAdjMat(vector<Nodes>&, vector<Nodes>&);
   void CopyLocalMat(MRF*);
   void CopyPsiMat( MRF*); 
   void CopyLambdaMat(PottsMRF*); 
   void fillInitialAssignment(int*); 
   void fillRegions(int, int*, int**, RegionLevel&); 
  void SetMonteCarlo(int  burnT, int samplingI,int,int*);
  void SetLoopy(int,SumOrMax,Strategy);
  void SetGBP(int, SumOrMax, double, bool, int,  int*, int**);
  void CreateAlgorithm();
  void MakeInferenceAlgorithm();
  void FindBest(unsigned*);
  void ResolveTies(unsigned*,int*,int);
  int MaxConfigurationsLoopy(unsigned**, double* , InferenceClass* ,int,int);
  //void AddNewCandidates(VectCandidate*, double**, int, unsigned*, double);
  void AddNewCandidates(vector<CandidateNode>*, double**, int, unsigned*, double);
  int LoopyPropagation(int, SumOrMax , Strategy);
  void FindMaxUnivariateMarginals(double*, unsigned*);
  double  GetEnergy(unsigned*);  
};

class NodeBestConf{

 public:

  int number_nodes,initBN;
  double best_energy;
  unsigned* best_conf;
  InferenceClass* inf_ente;
  NodeBestConf(int,int,double, unsigned*,InferenceClass*);   
  NodeBestConf(int,int,InferenceClass*);  
  NodeBestConf(int,int);   
  ~NodeBestConf();
  void SetBestConf(unsigned*); 
  void PrintBestConf();  
 
};



//bool operator<(const CandidateNode& x, const CandidateNode& y) { return(x.v < y.v); }
// bool operator==(const CandidateNode& x, const  CandidateNode& y) { return(x.v == y.v); }

#endif    




