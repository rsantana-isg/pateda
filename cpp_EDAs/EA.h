#ifndef __EA_H 
#define __EA_H 
#define max(a,b) (((a)>(b))?(a):(b))  
#include <stdlib.h> 
#include <stdio.h> 
#include <iostream> 
#include <fstream>
#include <math.h>
#include <string.h>
#include "Popul.h" 
//#include "TreePartition.h" 
 
 
 class EA { 
  public: 
  

       	div_t ImproveStop;
	double auxtime, alltime,bestalltime;
	time_t ltime_init,ltime_end;
	struct tm *gmt;
	struct tm *gmtnew;
        int *timevector; 
	               // PROBLEM PARAMETERS
	int vars;      // Number of Variables of the problem 	  
	double Max;           	
       	unsigned *Cardinalities;  	
	
	               // ALGORITHM PARAMETERS 
	int psize;    
	int Elit;  	 
	int fgen;
	int Maxgen;  
	int current_gen;
	int printvals;   
        int Prior; 
	double auxprob;
        Popul *pop,*selpop,*elitpop,*compact_pop; 
	double *fvect;       
	

	
	                   //SELECTION PARAMETERS
        int OldWaySel;     // Selection with sel pop (1) or straight on Sel prob (0) 
	int BestElitism; 
        double  Trunc; 	
	int  Tour;  
	int TruncMax; 
                           // STATISTICS PARAMETERS      
	double meangen;            
	double meaneval;  
	double BestEval,AbsBestEval,AuxBest; 
        int GenFound;
        unsigned int  *BestInd, *AbsBestInd;              
	int NPoints;                               //Number of points without repetition 	
 	int TotEvaluations;
        
        void (*ObjFunction)(Popul*,int, int);      // Population-based evaluation function

 

         // PROCEDURES AND FUNCTIONS
	 EA(int,double,unsigned*, void(*)(Popul*,int, int),int,int,int,int,int,int,int,double,int);
         ~EA(){}; 
 
         void init_time();
         void end_time();
         int Selection(); 
         void FindBestVal();
         void InitPopulations();
         void ApplySelection();
         void DeletePopulations(); 
         void MutatePop(Popul*,int, int);
         //virtual RunEA(){};
         void PrintBestSolutions(); 
         void PrintLastGen();
         void InitAlgorithm();      
 

 };  
 

 class GA:public EA { 
	 public:             
         int CX_Type;       //   Crossover type. 0) One-point crossover. 1) Uniform crossover
       	 int Mutation;      // Population based mutation  
         
         GA(int,double,unsigned*, void(*)(Popul*,int, int),int,int,int,int,int,int,int,double,int,int); 
         ~GA(){};
         void GenUniformCXInd(Popul*,int, unsigned int*, unsigned int*);   
         void GenOnePointCXInd(Popul*,int,unsigned int*,unsigned int*);           
         void GenCrossPop(int,Popul*, Popul*,int);
         int SimpleGA();  
         int CompactGA();           
        }; 


class Tree_EDA:public EA { 
	 public:                 

        double Complexity;           // In this case, complexity is the threshold for chi-square 
        IntTreeModel *IntTree;  
       
	Tree_EDA(int,double,unsigned*, void(*)(Popul*,int, int),int,int,int,int,int,int,int,double,int,double);
         int Intusualinit();  
         int CompactIntusualinit();  
         int FixedStructure_Intusualinit(unsigned int**);  
        }; 


class Mixture_Trees_EDA:public EA { 
	 public:             
           MixtureIntTrees *MixtureInt;  

           int TypeMixture;      // Class of MT-FDA (1-Meila, 2-MutInf)   
  	   int Ntrees;  
	   int Nsteps;  
	   int InitTreeStructure;  
	   int VisibleChoiceVar;  
 	   double MaxMixtProb; 
	   double S_alpha;  
           double SelInt;
	   int StopCrit; //Stop criteria to stop the MT learning alg. 
           double Complexity;  
         
       
         Mixture_Trees_EDA(int,double,unsigned*, void(*)(Popul*,int, int),int,int,int,int,int,int,int,double,int,double,double, double, int,int,int,int,int);

         ~Mixture_Trees_EDA(){};  
         int MixturesIntAlgorithm();        
        }; 


class Factorized_EA:public EA{ 
	 public:      
                double Complexity;         
        	int CliqMaxLength;  // Maximum size of the cliques for Markov  or maximum number of neighbors for MOA
   	        int MaxNumCliq;     // Maximum number of cliques for Markov   	        
  	        int LearningType;    // Learning for MNFDA (0-Markov, 1-JuntionTree) 
    	        int Cycles;         // Number of cycles for GS in the MNEDA or size for the clique in Markov EDA.  
      


	 Factorized_EA(int,double,unsigned*, void(*)(Popul*,int, int),int,int,int,int,int,int,int,double,int,double,int,int,int,int);          
         ~Factorized_EA(){};          
         int Markovinit(int,int);         //In this case, complexity is the threshold for chi-square 
         int MOA();  
         int RobustMNEDA(int,int);         
         int FGEDA(int,int);	 
         int RobustMOA(int,int);           
         int FixedStructure_RobustMNEDA(unsigned int**,int,int);  
         int EfficientStructure_RobustMNEDA(unsigned int**,int,int);    
        
        }; 


#endif 
