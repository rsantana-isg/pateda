#ifndef __ROT_H 
#define __ROT_H 
#define MAXPSIZE 2500
#define MAXNUMBERROT 10

#include <stdio.h> 
#include <stdlib.h>
#include <iostream> 
#include <fstream>
#include "auxfunc.h"  
#include "TriangSubgraph.h"   
#include "InferenceClass.h"   

const int MaxnumberSNPs = 5000; //A maximum number of SNPs is assumed to read the file information


class RotProtein{ 
public: 
  
   int numberresidues;
   int NResiduesAfterDEE;
   int  NActiveContacts;
   int ncells;
   int ncomp;

   double*** Psi;    
   double** Ls;
   int*  RotNumber;
   int** ProteinContacts;
   
   int** FinalRot;
   int*  NFinalRot;
   int*  Fixed;
   double FixedValue;
   int*  RemainResPositions;
   int**  AcceptRes;
   int* DissComponents;
   int* VarPos;
   int* ActiveContacts;
   unsigned int** Matrix;
   int* order;
   maximalsubgraph* MaxSubgraph;
  

   double*** RedPsi;    
   double** RedLs;
   unsigned int** RedMatrix;
   int* Redorder;

   int AuxArray[MAXNUMBERROT];   

    RotProtein(FILE*);
    RotProtein(FILE*,int);
    ~RotProtein();
    void EmptyPsi(); 
    double CalculateEnergy(unsigned*);
   double CalculateEnergySCP(unsigned*);
    double CalculateReducedEnergy(unsigned*);
    double CalculateEnergyWithDEE(unsigned*);
    double CalculateEnergyDECOMP(unsigned*,int,int*);
    void FillFixed(unsigned*);
    void FillFixedComp(unsigned*,int,int*);
    void DEE(double);
    void SecondDEE(double);
    void FinishDEE();
    void ApplyDEE(double);
    void FindDisconnectedComponents();
    double EvaluateFromFixed();
    int SearchNeighborhood(int,unsigned*,int*);
    int Best1Neighborhood(unsigned*,int*);
    int Best2Neighborhood(unsigned*,int*);
    int Best3Neighborhood(unsigned*,int*);
    int SearchRandomNeighborhood(int,unsigned*,int,int*);
    int RandomBest1Neighborhood(unsigned*,int,int*);
    int RandomBest2Neighborhood(unsigned*,int,int*);
    int RandomBest3Neighborhood(unsigned*,int,int*);
    void FindReducedCodification(unsigned*, unsigned*);   
    void FindActiveContacts();
    void PrintAdjacencyMatrix();  
    void FillAdjacencyMatrix();  
    void BestLoopySolution(unsigned int*,double,int);
    void BestGBPSolution(unsigned int*,double,int);
    void BestReducedLoopySolution(unsigned int*,double,int);
    void BestReducedGBPSolution(unsigned int*,double,int);
    void SimplifyProteinFunction();
    void DeleteSimplified();
    unsigned* FindBestSol(InferenceClass*);
    void FindEnlargedCodification(unsigned int*,unsigned int*);
    void BestReducedLoopyMaxConfigurations(unsigned int*, double,int, unsigned**, double*, int, int*);
    void BestLoopySolutionMaxConfigurations(unsigned int*, double,int, unsigned**, double*, int, int*);
};


class FoldPotential{ 
public: 
  
   int numberresidues;
   double t;
   int NumberAmiAcids;

   unsigned int* sequence;
   char allfiledetails[30];

   double*** Psi;    
   double** Ls;
   int** ProteinContacts;
   
   unsigned int** Matrix;
   int* order;
   maximalsubgraph* MaxSubgraph;
  
    
   FoldPotential(FILE*,double);
    ~FoldPotential();
    void EmptyPsi(); 
    void PrintAdjacencyMatrix(); 
    double PotTable(double,int,int);
    double CalculateEnergy(unsigned*);
    void BestLoopySolution(unsigned int*,double,int);
    void BestGBPSolution(unsigned int*,double,int);
    unsigned* FindBestSol(InferenceClass*);
    void FillPotential(FILE*);
    void BestLoopySolutionMaxConfigurations(unsigned int*, double,int, unsigned**, double*, int, int*);
};



class SNPs{ 
public: 
  
   int numberSNPs; // Total number of SNPs
   int numberSelfTagged; // SNPs correlated only with themselves
   int numberSNPsNeedingTag; // SNPs that can be tagged by others
   int npairs; // Number of SNP pairs;
   int ntriples; // Number of SNP triples

   char **SNPsNames;  //Array of SNPs names
   int* IndexOrderedTags;
   int* SelfTagged;            // List of SelfTagged SNPs
   int* SNPsNeedingTag; // List SNPs that can be tagged by others
   int*  ntagging;            // For each SNP, number of SNPs it tags   
   int*  npairtagging;  //  For each SNP, number of SNPs it tags as member of a pair.
   int*  npairtagged;  //  For each SNP, number of SNP pairs that tag  it
   int** tagged;           // tagged[i][j]:  SNP jth of those tagged by SNP i.
   int** tagging_as_pair; // tagging_as_pair[i][j] is the SNP that forms pair
                          // with SNP i to tag SNP in tagged_by_pair[i][j]
   int** tagged_by_pair;   // tagged_by_pair[i][j] is the jth SNP of those tagged by SNP i as member of a pair.
  
  double** tagged_corr;          //  corr_value[i][j] is the correlation between SNPs in i and tagged[i][j]
  double** tagging_as_pair_corr; // corr_pair_value[i][j] is the correlation between SNPs in i, 
                                 // taggeg_as_pair[i][j] and taggeg_by_pair[i][j]    
  //double** tagged_by_pair_corr; // corr_pair_value[i][j] is the correlation between SNPs in i,
                                 // taggeg_as_pair[i][j] and taggeg_by_pair[i][j]     
  
   SNPs(FILE*,FILE*);
   SNPs(){};
  ~SNPs();

  void InitBasicPairStructures(int);
  void InitBasicTripleStructures(int); 
  void DeleteBasicPairStructures();
  void DeleteBasicTripleStructures();
  void ReadFilePairSNPs(FILE*);
  void ReadFileTripleSNPs(FILE*);
  int CalculateNumberTaggedSNPs(unsigned*);
  int CalculateNumberTaggedSNPsWithAdded(unsigned*,int);
  int MultiCalculateNumberTaggedSNPs(unsigned*);
  void SaveTags(FILE*,unsigned*);
  void OrderTags();
  void CreateMatrix(unsigned**);
 };

class SNPSets{ 
public: 
  int  SetIdentifier;
   int numberSNPSets; // Number of different SNP sets
   int numberOptSNPSets;
   int* SelSNPSets;  // For each SNP set,  SelSNPSets[i] = 1 means the set i is included in the multi-obj optimiztion
   int numberSNPs; // Total number of different (UNION) SNPs in all the sets
   int numberOptSNPs; // Number of SNPs used in the optimization (only some of the sets of SNPs) 
   SNPs** TheSNPSets; // There is one set of SNPs for every involved population
   int** TheSNPIndex; // For each SNP in the union of SNPs, specifies its postion in the original SNP population
   int** BackSNPIndex;  // For each SNP, in each population, specifies its position in the UNION
   int*  TheOptSNPIndex; // Index of  SNPs from the UNION that are involved in the optimization process
   int*  OptBack;
   char **SNPsNames;  //Array of SNPs names

   SNPSets(int,int);
  ~SNPSets();
   void Initialize();
   void FillSetSNPSets(int);
   void MultiCalculateNumberTaggedSNPs(unsigned*,double*);
   void OptMultiCalculateNumberTaggedSNPs(unsigned*,double*,int);
   void CreateMatrix(unsigned**);
   void OptCreateMatrix(unsigned**);
   void SaveTags(FILE*,unsigned*);

};

#endif  



