#ifndef __ABSTRACTPROBM_H 
#define __ABSTRACTPROBM_H 
#define max(a,b) (((a)>(b))?(a):(b))  
#include <stdlib.h> 
#include <stdio.h> 
#include <iostream> 
#include <fstream>
#include <math.h>
#include <string.h>
#include "Popul.h" 
//#include "TreePartition.h" 
 
const int ForbidValue = -1; // Value -1 is not allowed for vector's variables 
 
 class AbstractProbModel { 
	 public: 
  
	 int  length;        // Number of Variables of the problem 
	 int genepoollimit;  // Number of individuals in the population 
         int actualpoolsize; // Number of ind. of the pop. considered in the prob. model 
	 int *actualindex;   // Index of individuals consd. for the prob. model 
	 Popul* Pop;         // Population of individuals 
         double TreeProb;  //Total probability of the population given the model
         double TreeL;    //Sum of biv likehood of the tree
         double Likehood;   //Likehood of the population; 
         int NPoints;          //Number of points with repetition
         double Prior;
          AbstractProbModel(int,int*,int,Popul*); 
    AbstractProbModel(int,int); 
 AbstractProbModel(int); 
   virtual ~AbstractProbModel(); 
	 int toclean;	   
        virtual void SetPop(Popul*); 
  	virtual void CalProbFvect(double*) {};  
        virtual void CalProbFvect(Popul*,double*,int){}; 
	virtual void CalProb(){}; 
	virtual void UpdateModel(){}; 
        virtual void UpdateModelForest(double*,int,Popul*){};  
        virtual void GenPop(int, Popul*); 
	virtual void InitPop(Popul*); 
	void SetGenePoolSize(int); 
	virtual double Prob(unsigned*){return 0;};
 	virtual double Prob(unsigned*,int){return 0;};
	virtual void GenIndividual (Popul*,int){}; 
        virtual void GenIndividual (Popul*,int,int){}; 
        virtual void GenPartialIndividual (Popul*,int,int,int*,int){}; 
	virtual void UpdateModel(double*){}; 
        virtual void UpdateModel(double*,int){};  
        virtual void UpdateModel(double*,int,Popul*){}; 
        virtual void PrintModel(){}; 
       	void PopMutation(Popul*,int,int, double); 
    void IndividualMutation(Popul*,int,int,int); 
    double Calculate_Best_Prior(int, int , double ); 
    virtual int Included_Edge(int,int){return 0;}; 
    virtual int Add_Edge(int,int){return 0;}; 
    virtual void Reorder_Edges(int,int){}; 
    virtual int Oldest_Ancestor(int){return 0;}; 
    virtual int Edge_Cases(int,int){return 0;};  
    virtual int Has_descendants(int){return 0;}; 
    virtual void PrintProbMod(){}; 
    void SetGenPoolLimit(int); 
    void CalculateILikehood(Popul*,double*); 
    virtual void RandParam(){};   
    virtual void AdjustBivProb(int,int,double*){}; 
    virtual void SetPrior(double,double,int){}; 
    void  PutPriors(int,int,double);
    virtual void Destroy(){};
  }; 
 
 


 class UnivariateModel:public AbstractProbModel { 
	 public:     
	  
     double *AllProb;   // Univariate Probabilities of the individuals 
    
  
	   
	  UnivariateModel(int,int*,int,Popul*); 
          UnivariateModel(int,int*,int); 
          UnivariateModel(int); 
         virtual	 ~UnivariateModel(); 
         virtual void SetPrior(double,double,int); 
	 virtual double Prob(unsigned*); 
	 virtual void CalProb(); 
         virtual void CalProb(Popul*,int); 
   	 virtual void CalProbFvect(double*); 
         virtual void CalProbFvect(Popul*,double*,int); 
         void ResetProb(); 
	 virtual void GenIndividual (Popul*,int); 
         virtual void GenPartialIndividual (Popul*,int,int,int*){}; 
	 virtual void UpdateModel(); 
	 //virtual void PopMutation(Popul*,int,int, double); 
	 //void IndividualMutation(Popul*,int,int,int);  
         virtual void RandParam(){};   
        }; 


 class SContraintUnivariateModel:public UnivariateModel{ 
	 public:     
	  
      double *NormAllProb;   // Univariate Probabilities of the individuals 
      int constraint;
  
	   
	  SContraintUnivariateModel(int,int); 
         virtual ~SContraintUnivariateModel(); 
         virtual void CalProbFvect(Popul*,double*,int); 
       	 virtual void GenIndividual (Popul*,int); 
         virtual void SetPrior(double,double,int); 

        }; 





	class AbstractTreeModel: public AbstractProbModel{ 
	 public: 
 	   
     int rootnode; 
	 double *MutualInf; 
     int* Tree; 
	 int* Queue; 
     FILE *f1; 
	 int DoReArrangeTrees; 
      	 double Complexity; // Maximum number of nodes in the tree det by chi-square
	 double threshchival; // Chi-square threshold
	 void PrintMut(); 
 
	  AbstractTreeModel(int,int*,int,Popul*,double); 
          AbstractTreeModel(int,double,int);
           AbstractTreeModel(int,double);
	 virtual ~AbstractTreeModel(); 
         virtual void CalMutInf(){};  
         virtual void CalMutInf(double**){}; 
   	 virtual void UpdateModel(double*,int){}; 
         virtual void UpdateModelForest(double*,int,Popul*){};  
         virtual void UpdateModel(double*){};  
         virtual void SetPrior(double,double,int){}; 
	 virtual void UpdateModel(){}; 
	 int RandomRootNode(); 
	 virtual void MakeTree(int); 
         //virtual void MakeTree(int,int); 
         void MakeRandomTree(int); 
	 void SetGenePoolSize(int); 
	 int NextInOrder(int); 
	 void ArrangeNodes(); 
	 void MutateTree(); 
	 void ReArrangeTree(); 
         virtual void PrintModel(); 
	 virtual void SetPop(Popul* pop){};       
    virtual int Included_Edge(int,int); 
    virtual int Add_Edge(int,int); 
    virtual int Correct_Edge(int,int); 
    virtual void Reorder_Edges(int,int); 
    virtual int Oldest_Ancestor(int); 
    virtual void RandParam(){};   
    virtual int Edge_Cases(int,int); 
    virtual int Has_descendants(int); 
    virtual void AdjustBivProb(int,int,double*){};  
    virtual void Propagation(int){}; 
    virtual void PutBivInTree(double**){}; 
    int  Other_root(int);
    void CleanTree();
  }; 
	 
 class BinaryTreeModel:public AbstractTreeModel{ 
	 public: 
	     
	 double **AllSecProb;	   
         double *AllProb;   // Univariate Probabilities of the individuals
         unsigned   *BestConf;  
         double    HighestProb;  
         int       Marca; 
        
       
        int* RootCharge; 

	  BinaryTreeModel(int,int*,int,Popul*,double); 
          BinaryTreeModel(int,double,int);
          BinaryTreeModel(BinaryTreeModel*,int);
	 virtual ~BinaryTreeModel(); 
	 virtual double Prob(unsigned*,int); 
	 virtual void CalProb(); 
	 int FindRootNode();  
         virtual void SetPrior(double,double,int); 
	 virtual void CalMutInf(); 
         virtual void CalMutInf(double**); 
	 virtual void GenIndividual (Popul*,int,int);
         virtual void GenPartialIndividual (Popul*,int,int,int*,int); 
	 virtual void CalProbFvect(Popul*,double*,int,int); 
         virtual void CalProbFvect(Popul*,double*,int); 
	 void ResetProb(); 
	 virtual void UpdateModel(double*,int,Popul*);
         virtual void UpdateModelForest(double*,int,Popul*); 
 	 virtual void UpdateModel(); 
         void ImportProb(double**,double*);  
         virtual void PrintProbMod(); 
         void PrintAllProbs(); 
	 virtual void RandParam();  
         double SumProb (Popul*,int); 
         virtual void AdjustBivProb(int,int,double*);
         void InitTree(int, int,double*,Popul*,int);
         void CalProbUnif(); 
         void FindSumMarg(int ,int ,int , double*);  
         void FindMaxMarg(int ,int ,int , double*); 
         void FindMaxMargR(int ,int ,int , double*); 
         void IncorporateEvidence(int,int, int , double*); 
         void NewDistributeEvidence(int,double*,double*);
         void NewCollectEvidence(int,double*,double*); 
         void DistributeEvidence(int);
         void CollectEvidence(int);  
         virtual void Propagation(int);
         virtual void NewPropagation(int);
         void NormalizeBiv(int, int);
         void Findlamdaprop(double*,double*,double*);
         void Findlamdaprop(double*,double,double,double*);
         virtual void PutBivInTree(double**); 
         void FindRootCharges();
         void FinalUniv();
         double CalculateGainLikehood(int,int,int);
         void ConstructTree();
         double UnivGainLikehood(int); 
         double BivGainLikehood(int,int);
         void SetNPoints(int);
         void FindLikehood();
         void MakeTreeLog();
         void TreeStructure(double**);
         double Calculate_Best_Prior(int,int, int, double);      
	 void SetNoParents();
         void NormalizeProbabilities(int); 
         
	 // Las siguientes funciones se relacionan con el calculo
         // de las configuraciones mas probables.
 void FindBestConf();  
 inline unsigned* GetBestConf(){return BestConf;};  
 inline double GetMaxProb(){ return HighestProb;};  
 inline int  GetMarca(){ return Marca;};  
 inline void  SetMarca(int _Marca){ Marca = _Marca;};
 void  UpdateCliqFromConf(unsigned*);
 //void CollectKConf(int, int,BinaryTreeModel*,Popul*); 
 void UpdateSecProb(BinaryTreeModel*);
 void UpdateSecProb(BinaryTreeModel*,BinaryTreeModel*);
 void InitCliques(double*,double*);
 void ImportProbFromTree(BinaryTreeModel*); 
 void ImportMutInf(double*);
 void ImportMutInfFromTree(BinaryTreeModel*);
 void PutInMutInfFromTree( BinaryTreeModel*); 
 }; 
 
  
 

//Class that represents trees for problems with integer codification

class IntTreeModel:public AbstractTreeModel{ 
	 public: 
	 
         int *Card;    //Cardinality of each variable
         int TotUnivEntries;
         int TotBivEntries;
         int* IndexUnivEntries;
         int* IndexBivEntries;
	 double *AllSecProb;	   
         double *AllProb;   // Univariate Probabilities of the individuals
         
      
       
           IntTreeModel(int,int*,int,Popul*,double,unsigned int*); 
          IntTreeModel(int,double,int,unsigned int*);
        
	 virtual ~IntTreeModel(); 
	 virtual double Prob(unsigned*,int); 
	 virtual void CalProb(); 
	 virtual void CalMutInf(); 
         virtual void CalMutInfDeception();
         virtual void CalMutInf(unsigned int** ); 
         virtual void GenIndividual (Popul*,int);
         virtual void GenPartialIndividual (Popul*,int,int,int*,int); 
	 virtual void CalProbFvect(Popul*,double*,int,int); 
         virtual void CalProbFvect(Popul*,double*,int); 
         virtual void CalProbFvect(Popul*,double*,int,unsigned int**); 
	 virtual void UpdateModel(double*,int,Popul*);
      	 virtual void UpdateModel(); 
         void InitTree(int, int,double*,Popul*,int);
         void SetNPoints(int);
         void TreeStructure(double**);
         double Calculate_Best_Prior(int,int, int, double);
         void CalMutInf(double* SecProbPros, double* UnivProbPros);  
         double SumProb (Popul* , int);
	 void ImportProbFromTree(IntTreeModel*); 
         void ImportMutInf(double*);
         void ImportMutInfFromTree(IntTreeModel*);
         void PutInMutInfFromTree( IntTreeModel*); 
	 void ImportProb(double*,double*);  
         virtual void SetPrior(double,double,int); 
         int  FindRootNode();
 }; 


 
#endif 
