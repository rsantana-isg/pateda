#include <math.h>  
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <iostream.h> 
#include <fstream.h> 
#include "auxfunc.h"  
#include "Popul.h"  
//#include "Treeprob.h"  
//#include "IntTreeprob.h" 
#include "EDA.h" 
#include "AbstractTree.h"  
#include "FDA.h"  
#include "MixtureTrees.h" 
#include "ProteinClass.h" 

#define itoa(a,b,c) sprintf(b, "%d", a) 

  
FILE *stream;  
FILE *file,*outfile;  	  
  
 
double meanlikehood[500]; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798};  
double SelInt; 
    
//int statistics[90][90];  
int cantexp;  
int now;  
int vars;  
int auxMax;  
double Max;  
double  Trunc;  
int psize;  
int  Tour;  
int func;  
int ExperimentMode;  
int Ntrees;  
int Elit;  
int succexp;  
double meangen;   
int Nsteps;  
int InitTreeStructure;  
int VisibleChoiceVar;  
int Maxgen;  
int printvals;   
int Card;  
int seed;  
int* params;  
int *timevector; 
char filedetails[30]; 
char MatrixFileName[30]; 
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
double BestEval; 
int TruncMax; 
int NPoints;  
unsigned int *BestInd; 
Popul *pop,*selpop,*elitpop,*compact_pop; 
double *fvect; 
int nsucc; 


HPProtein* FoldingProtein;
int TotEvaluations;
int sizeProtein;
int EvaluationMode;


/*
#define UMDA 0
#define EBNA_B 1
#define EBNA_LOCAL 2
#define PBIL 3
#define BSC 4
#define TREE 5
#define MIMIC 6
#define EBNA_K2 7
#define EBNA_PC 8
//Needed when PBIL or EBNAPC are executed

extern double ALPHA_PBIL;
extern double ALPHA_EBNAPC;

// Scores for the EBNA-local learning types
extern int SCORE;

#define BIC_SCORE 0
#define K2_SCORE 1

// The simulation type (i.e. PLS is the simplest).
extern int SIMULATION;
#define PLS 0
#define PLS_ALL_VALUES_1 1
#define PLS_ALL_VALUES_2 2
#define PLS_CORRECT		 3
#define PENALIZATION	 4

*/

int LEARNEBNA=0;  
int EBNASCORE= K2_SCORE; //BIC_SCORE;
double  EBNA_ALPHA =0.05;
int  EBNA_SIMUL = PLS;


void ReadParameters()  
 
{ 
  int T,MaxMixtP,S_alph,Compl; 
 
stream = fopen( "Param.txt", "r+" );  
         
 		    
	if( stream == NULL )  
		printf( "The file Param.txt was not opened\n" );  
	else  
	{  
         fscanf( stream, "%s", &MatrixFileName);  
         fscanf( stream, "%d", &cantexp); // Number of Experiments  
	 fscanf( stream, "%d", &vars); // Cant of Vars in the vector  
 	 fscanf( stream, "%d", &auxMax); // Max. of the fitness function  
  	 fscanf( stream, "%d", &T); // Percent of the Truncation selection or tournament size 
	 fscanf( stream, "%d", &psize); // Population Size  
	 fscanf( stream, "%d", &Tour);  // Type of selection 0=Trunc, 1=Tour, 2=Prop, 3=Bolt  
	 fscanf( stream, "%d", &func); // Number of the function, Ochoa's  
	 fscanf( stream, "%d", &ExperimentMode); // Type of Experiment (SEE BELOW case instruction)  
 	 fscanf( stream, "%d", &Ntrees); // Number of Trees  
	 fscanf( stream, "%d", &Elit); // Elistism  
	 fscanf( stream, "%d", &Nsteps); // Learning steps of the Mixture Algorithm  
 	 fscanf( stream, "%d", &InitTreeStructure); // 0 for a random init tree structures, 1 for a Chu&Liu learned init Tree Structure  
	 fscanf( stream, "%d", &VisibleChoiceVar); // 0 The Mixture choice var is hidden, 1 & 2 depends on a variable and the  unitation respectively  
	 fscanf( stream, "%d", &Maxgen);  // Max number of generations  
	 fscanf( stream, "%d", &printvals); // Best value in each generation is printed  
         fscanf( stream, "%d", &BestElitism); // If there is or not BestElitism 
         fscanf( stream, "%d", &MaxMixtP); // Maximum learning parameter mixture    
         fscanf( stream, "%d", &S_alph); // Value alpha for smoothing 
	 fscanf( stream, "%d", &StopCrit); //Stop Criteria for Learning of trees alg.  
         fscanf( stream, "%d", &Prior); //Type of prior. 
         fscanf( stream, "%d", &Compl); //Complexities of the trees. 
         fscanf( stream, "%d", &Coeftype); //Type of coefficient calculation for Exact Learning. 
         fscanf( stream, "%d", &params[0]); // Params for function evaluation 
	 fscanf( stream, "%d", &params[1]);  
	 fscanf( stream, "%d", &params[2]);  
	 fscanf( stream, "%d", &Card); // Cardinal for all variables  
	 fscanf( stream, "%d", &seed); // seed 
         fscanf( stream, "%d", &TypeMixture); // Class of MT-FDA (1-Meila, 2-MutInf)
         fscanf( stream, "%d", &Mutation); // Population based mutation  
	 fscanf( stream, "%d", &CliqMaxLength); // Maximum size of the cliques for Markov  
	 fscanf( stream, "%d", &MaxNumCliq); // Maximum number of cliques for Markov 
         fscanf( stream, "%d", &OldWaySel); // Selection with sel pop (1) or straight on Sel prob (0) 
         fscanf( stream, "%d", &LearningType); // Learning for MNFDA (0-Markov, 1-JuntionTree) 
         fscanf( stream, "%d", &Cycles); // Number of cycles for GS in the MNEDA 
	}  
 fclose( stream );  
if(T>0) 
 {  
   div_t res; 
   res = div(T,5);  
   SelInt = Sel_Int[res.quot]; // Approximation to the selection intensity for truncation. 
 } 
 
  
Trunc = T/double(100);  
Complex  = Compl/double(100);  
Max = auxMax/double(1000);   
MaxMixtProb =MaxMixtP/double(100); 
S_alpha = S_alph/double(100); 
 
} 
 

  
int EBNAAlgorithm(int succ)  
{  
    ReadParametersEBNA(vars, Max, Trunc, psize, func, Elit, Maxgen, LEARNEBNA, EBNASCORE, EBNA_ALPHA, EBNA_SIMUL);
    RunEBNA(&BestEval ,&succ);
    return succ;
}


void runOptimizer()  
{  
    int succ=-1; 
   
   succ = EBNAAlgorithm(succ);
   if (succ>-1)  
   { 
       succexp++; 
       meangen += succ; 
       
   }   
   else nsucc++;
   meaneval += BestEval; 
} 
 
 
void PrintStatistics() 
{  
  double auxmeangen, meanfit; 
 
                   meaneval /=  cantexp; 
                  if (succexp>0)  
                   {  
                    auxmeangen = meangen/succexp; 
                    if (BestElitism)  
                         meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;     
                    else meanfit = (auxmeangen+1)* (psize-1) + 1; 
                    cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" T="<<Trunc<<" N="<<psize<<" Sel="<<Tour<<"  m="<<Ntrees<<"  MaxGen="<<Maxgen<<"  ComplexEM="<<MaxMixtProb<<"  Elit="<<Elit<<" Suc.="<<succexp<<"  g="<<(auxmeangen+1)<<"  ave="<<meanfit<<" meaneval "<<meaneval<<endl;                   
                   } 
                  else  
                   {  
		       cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" T="<<Trunc<<" N="<<psize<<" Sel="<<Tour<<"  m="<<Ntrees<<"  MaxGen="<<Maxgen<<"ComplexEM="<<MaxMixtProb<<"  Elit="<<Elit<<" Suc.="<<0<<"  g="<<0<<"  ave="<<0<<" meaneval "<<meaneval<<endl; 
                   }              
 
} 

  
int main(){  

char number[30]; 
 int EBNAALG[5] ={1,2,5,7,8};
 int i,j,u;  
 unsigned ta = (unsigned) time(NULL);  
 srand(ta); 
 //srand(1067389206); 
 cout<<"seed"<<ta<<endl; 
 params = new int[3]; 
 


// int  IntConf[36] ={1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1};
// int  IntConf[25] ={1,1,0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0};
// int  IntConf[48] ={1,1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0};
// int  IntConf[60] ={1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,1};
// int  IntConf[50] = {0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0} ;
//int  IntConf[20] = {0,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,1,0}; 
 int  IntConf[64] = {0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0};
//  int  IntConf[85] = {0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0};
// int  IntConf[100]= {1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0};
     

//cout<<"Llega aqui "<<endl;
ReadParameters(); 

Cardinalities  = new unsigned[5000]; 


 sizeProtein = vars;
 for(i=0;i<5000;i++) 
     {
        Cardinalities[i] = Card;
      }
 
 FoldingProtein = new HPProtein(sizeProtein,IntConf);
 LEARNEBNA = ExperimentMode;

 //  for  (ExperimentMode=0;ExperimentMode<9;ExperimentMode++)
 for(j=0;j<5;j++)
   {
       ExperimentMode = EBNAALG[j];
       LEARNEBNA = ExperimentMode;
  	succexp = 0;  meangen = 0; meaneval = 0;  i =0;  nsucc =0;   
	while (i<cantexp)
        { 	   
	  runOptimizer(); 
         i++;
        }       
	PrintStatistics();
   }
delete FoldingProtein; 
delete[] Cardinalities; 
delete[] params; 
}      
