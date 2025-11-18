#ifndef __TRIANGSUBGRAPH_H    
#define __TRIANGSUBGRAPH_H    
#include "cdflib.h"   
#include <iostream>    
#include <fstream>    
class clique    
{    
public:    
	int MaxLength; // Maximal length of the clique    
	int *vars; // Variables in the clique    
	//unsigned *cards;    
	int *exp_card;    
	int NumberVars; // Number of variables in the clique (at least 1)   	 
	double *marg; // Marginal of the clique;    
	int parent;    
        int r_NumberCases;    
    
	clique(int,int);    
        ~clique();    
	int isclosed() {return NumberVars == MaxLength;};    
	void Add(int); // Adds a node to clique;    
	   
	int VarIsInClique(int); // Determines if a var is in the clique    
	   
	void print();   
        void printmarg();      
	void Instantiate(int*);        
	void Instantiate(int*,int,int,int,int*,int*);        
        void PutCase(int,int*);
	void PutCase(int,unsigned int*);    
	int Sum_and_PutCase(int, int*);     
        void Compute(unsigned*,double);          
        int rvars_index(unsigned*);        
	double Prob(unsigned*);        
	void CreateMarg(unsigned*);    
        void Fill_exp_card(unsigned*);    
        int NumberInstantiated(int*);   
        void PartialInstantiate(int* ,int*);   
        int  CheckCase(int,int*);    
        void Normalize(int);  
        void Normalize(int,double);  
        void NormalizeBoltzmann(double);
        void GetValProb(int*,int,double*);   
        int  GetValVar(int,int*,int);   
        void CondProb(int*,int*); 
        int PosVarInClique(int);
        int rvars_index_fv(int*,int*);
        void PutCase_fv(int*,int,int*); 
        void CompMargFromFather(clique*);
        void GiveProb(int*, double*);   
};    
   
class memberlistclique    
{                     
 public:    
	   
      int cliq;    
	memberlistclique* nextcliq;    
	memberlistclique(int,memberlistclique*);    
	~memberlistclique();
};    
   
   
class KikuchiClique    
{    
 public:    
      clique* father; // clique in the list of cliques the Kikuchi clique will be calculated from   
      clique* current;   
      int sign; // whether in the Kikuchi approx this factor is multiplied (sign=1) or divided -1   
      int count; // times the factor is in the list   
	int MaxLength; // Maximal length of the clique    
	int *vars; // Variables in the clique    
	int NumberVars; // Number of variables in the clique (at least 1)    
	       
      KikuchiClique(int,int);   
      KikuchiClique(int,int,clique*,int);    
   
      ~KikuchiClique();  
     
	int isclosed() {return NumberVars == MaxLength;};    
	void Add(int); // Adds a node to clique;    
	int VarIsInClique(int); // Determines if a var is in the clique    
	void print();   
        int rvars_index(unsigned*);        
	double Prob(unsigned*);     
	int IdemCliques(KikuchiClique*);
	void CreateMarg(unsigned*);    
        void CompMargFromFather();
       
};    
   
class memberlistKikuchiClique    
{                     
 public:    
      KikuchiClique* KCliq;      
	memberlistKikuchiClique* nextcliq;    
	memberlistKikuchiClique(KikuchiClique*);   
      ~memberlistKikuchiClique();    
 };  
  
class KikuchiApprox        
 {   
 public:  
  memberlistKikuchiClique** FirstKCliqLevels; //Pointer to the first cliq at each level  
  memberlistKikuchiClique** LastKCliqLevels; //idem before but for the last clique  
  int currentvar; //Which is the variable the cond. Kikuchi Approx. is about  
  int* neighbors; // Which are the neighbors  
  int* count_overl; //How many times is each neighbor in the Kikuchi Approx.  
  int MaxLength; // Maximal length of the cliques in the approx    
  int NumberCliques;  
  int numberneighbors;  
  int MaxVars; //Maximum number of vars   
  int level;  
   
  KikuchiApprox();    
  ~KikuchiApprox();  
  
  KikuchiApprox(int,int,int);   
  KikuchiClique* FindIntersection(clique*,clique*);  
  KikuchiClique* FindIntersection(KikuchiClique*,KikuchiClique*);  
  KikuchiClique* FindSelfIntersection(KikuchiClique*); 
  //int CreateFirstListCliquesInter(clique**,memberlistclique*);  
  int CreateFirstListCliquesInter(clique**,int);   
  int CreateOtherListCliquesInter(int);   
  int CreateAllListCliquesInter(clique**,int);  
  int ContainsClique(KikuchiClique*, KikuchiClique*);
  int ContainsClique(KikuchiClique*, clique*);
  //int Subsume(KikuchiClique*);      
  int IdemCliques();   
  void CheckPresentVars(clique*); 
  void CheckPresentVars(KikuchiClique*);  
  int AlreadyFoundUniqueVars(KikuchiClique*);  
  int AlreadyFoundApproximation();  
  //int FindKikuchiApproximation(clique**,memberlistclique*);  
  int FindKikuchiApproximation(clique**,int);  
  int FindOneLevelKikuchiApproximation(clique**,int);  
  int Insert(int,KikuchiClique*); 
  //void SimplifyClique(KikuchiClique*); 
  void SimplifyAllCliques(int,int); 
  void FindCR(clique**,int,int);
  void  FillListKikuchiCliques(memberlistKikuchiClique**); 
  void CleanLevel(int);
  void CreateMarg(unsigned*); 
 };  
   
   
   
class maximalsubgraph    
{    
 public:      
   int NumberCliques;    // Number of Cliques in the triangulated Subgraphg    
   int NumberNodes;      // Number of nodes of the graph     
   int CliqueMaxLength;  // Max number of variables the clique can have    
   int AddedEdges;    //Edges that have remained from the original graph due to the triangulization    
   int TotalEdges;     // Edges of the original graph    
   int MaxNumberOfCliquesInTheGraph;    
    
   unsigned int ** Adjacency_Matrix;    
   clique** ListCliques;    
   int *Nodes_Order;     // Order used for the triangulation algorithm    
   int *Nodes_Degrees;   // Degrees of the vertices in the original graph    
   int *compsub; //Auxiliar array for finding cliques   
   int c; //Auxiliar var for finding cliques.   
   int *CliquesSizes;  //Sizes of Cliques   
   
   memberlistclique** CliquesPerNodes; // It points each node to all the cliques it is included    
   maximalsubgraph(int,unsigned int**,int,int,int*);     
   maximalsubgraph(int,unsigned int**,int,int);     
   maximalsubgraph();      
   ~maximalsubgraph();    
   void AddToClique(int,int,int);    
   int FindCliqueToAdd(int,int&);    
   void CreateGraphCliques();     
   void CreateGraphCliquesProtein(int,int);   
   void CreateGraphCliquesProteinMPM(int, int, unsigned long**);   
   void UpdateListFromClique(clique*,int,int);    
   void	CardinalityOrdering();    
   void LexicalOrdering();    
   void UpdateLex(int,unsigned**&);    
   int FindNextInLex(int,unsigned**);    
   void MaximumCardinalityLexicalOrdering();    
   int withoutOverlappedCliques();    
   void printCliques();    
   void printNodesDegrees();    
   void printOrdering();   
   void AddClique(int*, int);    
   void version2(int* ,int,int);   
   void FindAllCliques();   
};   
     
    
#endif     
   
   
   
   
   
   
