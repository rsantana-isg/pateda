#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <math.h> 
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 

#include <map>
#include <dai/alldai.h>  // Include main libDAI header file
#include <dai/jtree.h>
#include <dai/bp.h>
#include <dai/decmap.h>
#include <dai/fbp.h>
#include <dai/bbp.h>
#include <dai/hak.h>
#include <dai/trwbp.h>
#include <dai/treeep.h>
#include <dai/lc.h>

#include "auxfunc.h"  
#include "Popul.h"  
#include "EDA.h" 
#include "AbstractTree.h"  
#include "FDA.h"  
#include "MixtureTrees.h" 
#include "AllFunctions.h" 
#include "FactorGraphMethods.h" 
#include "IsingModel.h" 
#include "EA.h" 


#define itoa(a,b,c) sprintf(b, "%d", a) 

using namespace dai;
using namespace std;

FILE *stream;  
FILE *file,*outfile;  	  
 

int cantexp;  
int vars;  
double Max;  
double  Trunc;  
int psize;  
int  Tour;  
int func;  
int ExperimentMode;  
int Ntrees;  
int Elit;  
double meangen;   
int Nsteps;  
int InitTreeStructure;  
int VisibleChoiceVar;  
int Maxgen;  
int printvals;   
unsigned int Card;  
int seed;  
int* params;  
int *timevector; 
char filedetails[30]; 
int BestElitism; 
double MaxMixtProb; 
double S_alpha;  
int StopCrit; //Stop criteria to stop the MT learning alg. 
int Prior; 
double Complex; 
int Coeftype;  
unsigned *Cardinalities;  
int Mutation; 
int CliqMaxLength; 
int MaxNumCliq; 
int OldWaySel; 
int LearningType;
int TypeMixture; 
int Cycles; 
double meaneval;  
double BestEval,AbsBestEval,AuxBest; 
int GenFound;
int LastGen;
int TruncMax; 
int NPoints;  
unsigned int  *BestInd, *AbsBestInd;  
Popul *pop,*selpop,*elitpop,*compact_pop; 
unsigned int **archivesols;
int NArchiveSols;
double *fvect; 
int TypeInfMethod;


int TotEvaluations;
int EvaluationMode;
int currentexp;
int length;
long int explength;
int MaxMPC;
int TypeMPC;

int CXType;
double auxtime;
int local_search_ntrials;

void (*Objective_Function)(Popul*,int, int);  

unsigned** matrixDSYS;
Ising* MyIsing;
//FactorGraph fg;


void ReadStructureMatrix(char* fname, int nvars)
{ 
 FILE *stream;  
 int i,j,k;  
 stream = fopen(fname, "r+" );  	
 for (i=0; i<nvars; i++)  
   for (j=0; j<nvars; j++)  
	{ 
           fscanf(stream,"%d ",&matrixDSYS[i][j]); 
        }
 fclose(stream);   
}


/*******************************************************************************************************/
/*                              ISING EVALUATION FUNCTIONS                                             */
/*******************************************************************************************************/


// Evaluates the fitness function of the Ising configuration without modifying the 
// solution in any way
void PlainEvalIsing(Popul* epop,int eli, int last)
{
 int k;
 double auxval; 

for(k=eli; k < last; k++) 
  {
     auxval = MyIsing->evalfunc(epop->P[k]); 
     //cout<<auxval<<endl;
     epop->SetVal(k,auxval);  
  }
}

// Evaluates the fitness function applying best-flip moves until no improvements
// are possible. Then, if no improvement with respect to the the initial solution
// obtained, a maximum of local_search_ntrials is applied. For each move the variable select
// the movement that decreases the fitness function the least. This movement is also included in a tabu_list. 
// After that movement, the function tries to increase the fitness again for maximum 
// local_search_ntrial trials

void EvalIsingSA(Popul* epop,int eli, int last)
{
 int k;
 double auxval; 
 
for(k=eli; k < last; k++) 
  {
    auxval = MyIsing->SA_evalfunc(epop->P[k],local_search_ntrials);     
     epop->SetVal(k,auxval);  
  }
}


// Evaluates the fitness function applying best-flip moves until no improvements
// are possible. For each bit-flip, the move that improves the fitness the most is selected.
//  A list of promising moves is updated in every step in such a way that only a reduced number
// of spins are considered 
// The algorithm stops when no improvement is achieved (local minima)

void EvalIsingBestSA(Popul* epop,int eli, int last)
{
 int k;
 double auxval; 
 
for(k=eli; k < last; k++) 
  {
    auxval = MyIsing->Best_SA_evalfunc(epop->P[k]);     
     epop->SetVal(k,auxval);  
  }
}

// Idem Best_SA_evalfunc but instead of selecting the bitflip that improves the fitness
// the most, it is randomly selected among those that improve the fitness
//  A list of promising moves is updated in every step in such a way that only a reduced number
// of spins are considered 
// The algorithm stops when no improvement is achieved (local minima)

void EvalIsingRandomSA(Popul* epop,int eli, int last)
{
 int k;
 double auxval; 
 
for(k=eli; k < last; k++) 
  {
    auxval = MyIsing->Random_SA_evalfunc(epop->P[k]);     
     epop->SetVal(k,auxval);  
  }
}

// Combination of functions Best_SA_evalfunc and SA_evalfunc 
// The idea is to improve the solution as much as possible in an efficient way
//  before allowing movements that to decrease its the fitness


void EvalIsingCombined(Popul* epop,int eli, int last)
{
 int k;
 double auxval; 
 
 for(k=eli; k < last; k++) 
  {
     auxval = MyIsing->Best_SA_evalfunc(epop->P[k]); 
     //cout<<k<<" "<<auxval<<" -- ";
     auxval = MyIsing->SA_evalfunc(epop->P[k],local_search_ntrials);    
     //cout<<auxval<<endl;
     epop->SetVal(k,auxval);  
  }
}


// The fitness function to be used by the optimizers is set here
void Select_Ising_EvalFunction()
{
  cout<<" function "<<func<<endl;
   switch(func)  
         {      
	 case 0: Objective_Function = &PlainEvalIsing; break;
	 case 1: Objective_Function = &EvalIsingSA; break;                       
	 case 2: Objective_Function = &EvalIsingBestSA; break; 
	 case 3: Objective_Function = &EvalIsingRandomSA;break; 
	 case 4: Objective_Function = &EvalIsingCombined;break;   
         }
}



void runOptimizer(int algtype,int nrun)  
{  

    EA* MyEA;
    

     switch(algtype)  
                     {             
                     // Markov Network       
		     case 0: MyEA=new Factorized_EA(vars,Max,Cardinalities, Objective_Function,psize,Elit,BestElitism,Maxgen,printvals,Prior,OldWaySel, Trunc, Tour,Complex,CliqMaxLength,MaxNumCliq,LearningType,Cycles);
		       LastGen = ((Factorized_EA*)MyEA)->Markovinit(1,TypeInfMethod);  
                       break; 
		    
                     // Normal Tree
                     case 1: MyEA = new Tree_EDA(vars,Max,Cardinalities, Objective_Function,psize,Elit,BestElitism,Maxgen,printvals,Prior,OldWaySel, Trunc, Tour,Complex);
		     LastGen =((Tree_EDA*)MyEA)->Intusualinit();  
                     break;                         
                    
                     // Tree that uses problem structure
                     case 14: MyEA = new Tree_EDA(vars,Max,Cardinalities, Objective_Function,psize,Elit,BestElitism,Maxgen,printvals,Prior,OldWaySel, Trunc, Tour,Complex);
		     LastGen =((Tree_EDA*)MyEA)->FixedStructure_Intusualinit(matrixDSYS);  
                     break;
 
                     // Learns MN-EDA, MN-FDA, or MOA using MI and applying G-test after (Biased MPC+Elitist MPC)                  
                     case 21: case 23: MyEA=new Factorized_EA(vars,Max,Cardinalities, Objective_Function,psize,Elit,BestElitism,Maxgen,printvals,Prior,OldWaySel, Trunc, Tour,Complex,CliqMaxLength,MaxNumCliq,LearningType,Cycles); 
  	             LastGen = ((Factorized_EA*)MyEA)->EfficientStructure_RobustMNEDA(matrixDSYS,TypeInfMethod,ExperimentMode);
                     break;
                     

                     // Learns MN-EDA, MN-FDA, or MOA using MI and applying G-test after (Biased MPC+Elitist MPC)
                     case 25: MyEA=new GA(vars,Max,Cardinalities, Objective_Function,psize,Elit,BestElitism,Maxgen,printvals,Prior,OldWaySel, Trunc, Tour, CXType);
 		              //LastGen = ((GA*)MyEA)->CompactGA();
                              LastGen = ((GA*)MyEA)->SimpleGA();
                              break; 
                     }   

     BestEval = MyEA->BestEval;        
     auxtime = MyEA->auxtime;        
     GenFound = MyEA->GenFound;
     delete MyEA;   
} 



int main( int argc, char *argv[] )
{
   
  // Example of calling the function
  //  ./sidan 1 25 0 512 100 500 100 1 100000000 10 4 2 2 0  InstanceInteractions_8_98082.0.csv Matrix_512.csv 

  int i,j;
  int T,MaxMixtP,S_alph,Compl;  
  char InteractionsFileName[50]; 
  char Topology_Matrix_Fname[50]; 
  
  
  if( argc != 17 ) {
    std::cout << "Usage: " <<"cantexp  EDA{0:Markov, 1:Tree  2:Mixture, 4:AffEDA} modeprotein{2,3} prot_inst n psize Trunc max-gen" << std::endl;
    std::cout << "       Please read the README file." << std::endl;
    exit(1);
}

 cantexp = atoi(argv[1]);         // Number of experiments
 ExperimentMode = atoi(argv[2]);   // Type of EDA
 CXType = atoi(argv[3]);   // Type of Crossover for GA. 0) One-point CX. 1) Uniform CX
 vars =  atoi(argv[4]);   //Number of variables (redundant because depends on instance)
 psize = atoi(argv[5]);          // Population size
 T = atoi(argv[6]);              // Percentage of truncation integer number (1:999)
 Maxgen =  atoi(argv[7]);        // Max number of generations 
 BestElitism = atoi(argv[8]);         // If there is or not BestElitism, if thereisnot BestElitism, Elitism = 1 by default;
 Max = atoi(argv[9]);
 CliqMaxLength = atoi(argv[10]);
 func  = atoi(argv[11]);
 printvals  = atoi(argv[12]);
 Card =  atoi(argv[13]);
 TypeInfMethod =  atoi(argv[14]);


 Tour = 0;                       // Truncation Selection is used
 //func = 8;                       // Index of the function, only for OchoaFun functions
 Ntrees = 2;                     // Number of Trees  for MT-EDA
 Elit = 1;                       // Elitism
 Nsteps = 50;                    // Learning steps of the Mixture Algorithm  
 InitTreeStructure = 1;    // 0 for a random init tree structures, 1 for a Chu&Liu learned init Tree Structure  
 VisibleChoiceVar = 0;     // 0 The Mixture choice var is hidden, 1 & 2 depends on a variable and the  unitation respectively  
 //printvals = 2;            // The printvals-1 best values in each generation are printed 
 MaxMixtP = 500;           // Maximum learning parameter mixture 
 S_alph = 0;               // Value alpha for smoothing 
 StopCrit = 1;             // Stop Criteria for Learning of trees alg.  
 Prior = 0;                // Type of prior. [Since for the SIDAN problem we are using Mutation, we set Prior to 0. However Prior=1 is recomended]
 Compl=75;                 // Complexities of the trees. 
 Coeftype=2;               // Type of coefficient calculation for Exact Learning. 
 //params[0] = 3 ;           //  Params for function evaluation 
 //params[1] = 3;  
 // params[2] = 10;  
 
 
 //seed =  1243343896; 
 seed = (unsigned) time(NULL);  
 srand(seed); 
 cout<<"seed"<<seed<<endl; 

TypeMixture = 1;          // Class of MT-FDA (1-Meila, 2-MutInf)
Mutation = 0;             // Population based mutation  
//CliqMaxLength = 2; // Maximum size of the cliques for Markov  or maximum number of neighbors for MOA
MaxNumCliq = 5000; // Maximum number of cliques for Markov 
OldWaySel = 0; // Selection with sel pop (1) or straight on Sel prob (0) 
LearningType = 6; // Learning for MNFDA (0-Markov, 1-JuntionTree) 
Cycles = 0 ; // Number of cycles for GS in the MNEDA or size for the clique in Markov EDA. 
Trunc = T/double(1000);  
Complex  = Compl/double(100);  
MaxMixtProb =MaxMixtP/double(100); 
S_alpha = S_alph/double(100); 
 local_search_ntrials = 10;

Cardinalities  = new unsigned[5000];  

// Reading the matrix with the topology 
matrixDSYS = new unsigned*[vars];
for (i=0;i<vars;i++) matrixDSYS[i] = new unsigned[vars];
Topology_Matrix_Fname[0] = 0;
strcat(Topology_Matrix_Fname,argv[16]);      
ReadStructureMatrix(Topology_Matrix_Fname,vars);
 

 // Reading the matrix with the interactions       
 InteractionsFileName[0]=0; 
 strcat(InteractionsFileName,argv[15]);      
 MyIsing = new Ising(InteractionsFileName);                  

 Select_Ising_EvalFunction();
 for(j=0;j<5000;j++) Cardinalities[j] = Card;   
 AbsBestInd = new unsigned int [vars]; 

  cout<<"Alg : "<<ExperimentMode<<", Crossover Type : "<<CXType<<", n : "<<vars<<", psize : "<<psize<<", Trunc : "<<T<<", max-gen : "<<Maxgen<<", BestElit. : "<<BestElitism<<", NNeighbors  : "<<CliqMaxLength<<", MaxFun  : "<<Max<<", func : "<<func<<endl; 
      
      for(i=0;i<cantexp;i++)
       {
        currentexp = i;          
        runOptimizer(ExperimentMode,i);   
        cout<<BestEval<<" "<<GenFound<<" "<<LastGen<<" "<<i<<" "<<auxtime<<endl; 
        BestEval = 0;	  
       }
            
   
 for (i=0;i<vars;i++) delete[] matrixDSYS[i]; 

 delete[] matrixDSYS; 
 delete[] AbsBestInd; 
 delete MyIsing;  
 delete [] Cardinalities; 

 
 return 0;

}      




