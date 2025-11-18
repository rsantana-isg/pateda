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
  
FILE *stream;  
FILE *file,*outfile;  	  
  
 
//double allfvalues[500];
double stddev; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798};  
double SelInt; 
   
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
unsigned int Card;  
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
int Cycles; 
int MaxVars;
int MinVars;
 
double meaneval;  
double BestEval; 
int TruncMax; 
int NPoints;  
unsigned int *BestInd; 
Popul *pop,*selpop,*elitpop,*compact_pop,*bigpop; 
double *fvect; 
int  nsucc;


void ReadParameters()  
 
{ 
  int T,MaxMixtP,S_alph,Compl; 
 Tour = 0;
Card = 2;
Prior = 1;
BestElitism = 0;
VisibleChoiceVar = 0;
S_alph = 0;
StopCrit = 1;
printvals = 0;
OldWaySel = 0;
 Mutation = 0;
stream = fopen( "EParam.txt", "r+" );  
        		    
	if( stream == NULL )  
		printf( "The file EParam.txt  with the parameters of the experiments was not opened\n" );  
	else  
	{        
         fscanf( stream, "%d", &cantexp); // Number of Experiments  
	 fscanf( stream, "%d", &T); // Percent of the Truncation selection or tournament size 
	 fscanf( stream, "%d", &func); // Number of the function 
	 fscanf( stream, "%d", &ExperimentMode); // Type of EDA
         fscanf( stream, "%d", &MinVars); // Minimoo numero de variables
         fscanf( stream, "%d", &MaxVars); // Maximo numero de variables
 	 fscanf( stream, "%d", &Elit); // Elistism (Now best elitism is fixed) 
         fscanf( stream, "%d", &Maxgen);  // Max number of generations  
         fscanf( stream, "%d", &printvals);  //Print values in each generation 
         fscanf( stream, "%d", &Ntrees); // Number of Trees    
	 fscanf( stream, "%d", &Nsteps); // Learning steps of the Mixture Algorithm  
 	 fscanf( stream, "%d", &InitTreeStructure); // 0 for a random init tree structures, 1 for a Chu&Liu learned init Tree Structure  
	 fscanf( stream, "%d", &MaxMixtP); // Maximum learning parameter mixture    
         
         fscanf( stream, "%d", &Compl); //Complexities of the trees. 
         fscanf( stream, "%d", &CliqMaxLength); // Maximum size of the cliques for Markov  
	 fscanf( stream, "%d", &MaxNumCliq); // Maximum number of cliques for Markov        
         fscanf( stream, "%d", &LearningType); // Learning for MNFDA (0-Markov, 1-JuntionTree) 
         fscanf( stream, "%d", &Cycles); // Number of cycles for GS in the MNEDA 
	}  
fclose( stream );  

Trunc = T/double(100);  
Complex  = Compl/double(100);  
Max = auxMax/double(100);   
MaxMixtProb =MaxMixtP/double(100); 
S_alpha = S_alph/double(100); 
 
} 
 

 
inline void FindBestVal() 
{     
      
        BestEval = compact_pop->Evaluations[0]; 
        BestInd =  compact_pop->P[0]; 
     
} 
 
inline void InitPopulations() 
{  
     TruncMax = int(psize*Trunc);  
     selpop = new Popul(psize,vars,Elit,Cardinalities);  
     pop = new Popul(psize,vars,Elit,Cardinalities);  
     compact_pop = new Popul(TruncMax,vars,Elit,Cardinalities);  
     bigpop =  new Popul(2*psize-1,vars,Elit,Cardinalities); 
     fvect = new double[psize]; 
     pop->RandInit();  
} 
 
inline void DeletePopulations() 
{ 
  delete compact_pop; 
  delete pop;  
  delete selpop; 
  delete bigpop; 
  delete[] fvect; 
} 
 

int usualinit(double Complexity)  
{  
  int i,fgen;  
  double auxprob,sumprob;     
  BinaryTreeModel *MyTree;  
 
  InitPopulations(); 
  MyTree = new BinaryTreeModel(vaxrs,Complexity,selpop->psize);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;   NPoints =2;
   
  MyTree->Prior = Prior; 
  while (i<Maxgen && BestEval<Max && NPoints>1 )  
  {  
    
     pop->EvaluateAll(func); 

     pop->TruncSel(compact_pop,TruncMax); 
     NPoints = TruncMax;
    
     if (printvals) compact_pop->Print(0,TruncMax); 
     MyTree->Pop = compact_pop;
     MyTree->actualpoolsize = TruncMax;
     MyTree->CalProb();
     MyTree->rootnode = MyTree->FindRootNode();//MyTree->RandomRootNode();         
     MyTree->CalMutInf();  
     MyTree->MakeTree(MyTree->rootnode); 
     FindBestVal(); 
     //if (printvals) MyTree->PrintAllProbs();
     //if (printvals) MyTree->PrintModel();
     //if (printvals) MyTree->PrintMut();
      //MyTree->PutPriors(Prior,selpop->psize,1); 

     sumprob = MyTree->SumProb(compact_pop,NPoints);             
     auxprob = MyTree->Prob(BestInd);  

     /*      
  for(int ll=0;ll<psize;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<pop->P[ll][l];  
     cout<<ll<<" "<<pop->Evaluations[ll]<<endl; 
    } 
     */
     if(printvals) 
      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<sumprob<<" Likehood "<< MyTree->Likehood<<endl;  
      if (BestEval==Max) fgen  = i;	   
      else 
          { 	                
            MyTree->GenPop(Elit,selpop);  
            selpop->EvaluateAll(func); 
            bigpop->Merge2Pops(selpop,pop->psize-1,pop,pop->psize);
            bigpop->TruncSel(pop,psize); 
            i++; 
          }  
  }  
   if(printvals)  
         cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<MyTree->TreeProb<<" Likehood "<< MyTree->Likehood<<endl;  
        
  DeletePopulations(); 
  delete MyTree;
  //cout<<"fgen is "<<fgen<<endl;  
  return fgen;  
}  

 
int UMDAinit()  
{  
  int i,fgen;  
  double auxprob;     
  UnivariateModel *MyUMDA;  
    
  InitPopulations();   
  MyUMDA = new UnivariateModel(vars);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
  NPoints = 2;
  while (i<Maxgen && BestEval<Max && NPoints>1)  
  {     
     pop->EvaluateAll(func);  
     pop->TruncSel(compact_pop,TruncMax); 
     NPoints = TruncMax;

     MyUMDA->CalProb(compact_pop,NPoints);
     FindBestVal(); 
     auxprob = MyUMDA->Prob(BestInd);  
       
 if(printvals) 
 {
   compact_pop->Print();
   //for(int l=0;l<vars;l++) printf("%d ",selpop->P[0][l]);  
   //  printf("\n ");     
 }   
        if(printvals)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
    
  if (BestEval==Max) fgen  = i;	   
      else 
          { 	                
            MyUMDA->GenPop(Elit,selpop);
            //cout<<"Pop "<<endl;
            //pop->Print();  
            selpop->EvaluateAll(func); 
            //cout<<"GenPop "<<endl;
            //selpop->Print();
            bigpop->Merge2Pops(selpop,pop->psize-1,pop,pop->psize);
            //cout<<"BigPop "<<endl;
            //bigpop->Print();
            bigpop->TruncSel(pop,psize); 
            i++; 
          }       
     
  }  
  if(printvals)  
   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;  
        
 
  DeletePopulations(); 
  delete MyUMDA;  
  return fgen;  
}  
 
 
void runOptimizer()  
{  
    int succ=-1; 
   switch(ExperimentMode)  
                     {                     
                       case 0: succ = usualinit(Complex);break; // Tree  
                       case 2: succ = UMDAinit();break; // UMDA 		    
                     } 
   cout<<succ<<" ";   
  
   if (succ>-1)  
   { 
       meangen += succ;
       //allfvalues[succexp] = succ;
       succexp++;                  
   } 
   else nsucc++;  
  
   meaneval += BestEval; 
} 
 
 
void PrintStatistics() 
{  
  double auxmeangen, meanfit; 
  int i;
           meaneval /=  cantexp;
          stddev = 0;

                  if (succexp>0)  
                   {  
                    auxmeangen = meangen/succexp;                    
		    /* for(i=0;i<succexp;i++)   
		      { 
			stddev +=(auxmeangen-allfvalues[i])*(auxmeangen-allfvalues[i]);
                       cout<<(auxmeangen-allfvalues[i])*(auxmeangen-allfvalues[i])<<endl; 
                     }
                    stddev = sqrt(stddev/(succexp-1));
                    */
                    if (BestElitism)  
                         meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;     
                    else meanfit = (auxmeangen+1)* (psize-1) + 1; 
                    cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" f(x)="<<func<<" N="<<psize<<" Sel="<<Tour<<"  m="<<Ntrees<<"  Prior="<<Prior<<"  LearningT="<<LearningType<<   " Suc.="<<succexp<<"  g="<<(auxmeangen+1)<<"  ave="<<meanfit<<" meaneval "<<meaneval<<" stddev "<<stddev<<endl;                   
                   } 
                  else  
                   {  
		       cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" f(x)="<<func<<" N="<<psize<<" Sel="<<Tour<<"  m="<<Ntrees<<"  Prior="<<Prior<<"  LearningT="<<LearningType<<" Suc.="<<0<<"  g="<<0<<"  ave="<<0<<" meaneval "<<meaneval<<" stddev "<<stddev<<endl; 
                   }              
 
} 
 
  
int  main(){  


 int i;  
 unsigned ta = (unsigned) time(NULL); 
 //ta =1059984488; 
 srand(ta); 
 cout<<"seed"<<ta<<endl; 
 params = new int[3]; 
 ReadParameters(); 

Cardinalities  = new unsigned[5000];  
double optim[15]={81,324,1215,4374};
int k,j,u,toadd;


for(i=0;i<5000;i++) Cardinalities[i] = Card; 
 
 if (func==45) toadd = 2;
 else toadd = 1;
 succexp = 0;  meangen = 0; meaneval = 0;  nsucc =0; 
for(i=0;i<cantexp;i++)
 { 	
   cout<<" "<<i<<" "; 
  for(vars=MinVars;vars<=MaxVars;vars+=toadd) 
   {   
     if (func==5) Max = vars;
      if (func==6) Max = ((vars)*(vars+1))/2;
      if (func==45) Max = 0; //(3*pow(2,(vars-1)/2)-2);
    else if (func==44) Max = vars;
      psize = vars;
       
    runOptimizer(); 
   
   }     
  cout<<endl; 
 } 
 vars -= toadd;
 //PrintStatistics();    
delete [] params; 
//fclose(outfile); 
delete [] Cardinalities; 
 return 1;
}      





