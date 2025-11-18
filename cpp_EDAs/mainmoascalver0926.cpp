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

using namespace dai;
using namespace std;

FILE *stream;  
FILE *file,*outfile;  	  
 
double meanlikehood[500]; 
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
int TypeMixture; 
int Cycles; 
int KDec; 
double meaneval;  
double BestEval,AbsBestEval,AuxBest; 
int TruncMax; 
int NPoints;  
unsigned int  *BestInd, *AbsBestInd;  
Popul *pop,*selpop,*elitpop,*compact_pop; 
double *fvect; 
int  succ,nsucc;
int  BB;
int TypeInfMethod;
int Solap;

int Clock;

div_t ImproveStop;
 double auxtime, alltime,bestalltime;
 time_t ltime_init,ltime_end;
 struct tm *gmt;
 struct tm *gmtnew;

int LEARNEBNA=1;  
int EBNASCORE=K2_SCORE;
double  EBNA_ALPHA =0.05;
int  EBNA_SIMUL = PLS;
 

int TotEvaluations;
int EvaluationMode;
int currentexp;
int length;
long int explength;
 int MaxMPC;
 int TypeMPC;
void init_time()
{
 time( &ltime_init );
 gmt = localtime( &ltime_init );
 auxtime = - ( gmt->tm_mday * 86400 + gmt->tm_hour*3600 + gmt->tm_min*60 + gmt->tm_sec);
}


void end_time()
{
  time( &ltime_end );
  gmtnew = localtime( &ltime_end );
  auxtime = auxtime + gmtnew->tm_mday * 86400 + gmtnew->tm_hour*3600+gmtnew->tm_min*60+gmtnew->tm_sec;
}



void evalfunction(Popul* epop,int nelit, int epsize, int atgen)
{
 double obj_f;
 int k,start;
 if (atgen==0) start=0;
 else start=nelit;


for(k=start; k < epsize;  k++)  
 {
   //obj_f = eval(func,(int*)epop->P[k], vars);
   obj_f = evalua(k,epop,func,params);  
   epop->SetVal(k,obj_f);
   //cout<<k<<" "<<obj_f<<endl;
   TotEvaluations++;
 }
}





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

 
int Selection() 
{ 
   int NPoints=0; 
 
   if (Tour==0)  
         {  
           pop->TruncSel(selpop,TruncMax); 
           selpop->UniformProb(TruncMax,fvect);
           //selpop->BotzmannDist(1.0,fvect);          
           NPoints = selpop->CompactPopNew(compact_pop,fvect); 
	   //NPoints = TruncMax;
           //compact_pop->CopyPop(selpop);           
	 } 
     else if(Tour==1) //Tournament selection 
	 {  
	   pop->TournSel(selpop,TruncMax); 
           selpop->UniformProb(psize,fvect); 
           NPoints = selpop->CompactPopNew(compact_pop,fvect); 
	 }  
    else if(Tour==2) //Proportional selection 
	 {  
	   pop->ProporDist(fvect);   
	   if (OldWaySel) 
           { 
            selpop->SUSSel(psize,pop,fvect);  
            selpop->UniformProb(psize,fvect);    
            NPoints = selpop->CompactPopNew(compact_pop,fvect);         
           } 
           else NPoints = pop->CompactPopNew(compact_pop,fvect);                         
          }  
     else if(Tour==3) //Boltzman selection 
	 {  
	   pop->BotzmannDist(1.0,fvect); 
	   if (OldWaySel) 
           { 
            selpop->SUSSel(psize,pop,fvect);  
            selpop->UniformProb(psize,fvect);    
            NPoints = selpop->CompactPopNew(compact_pop,fvect); 
           } 
           else NPoints = pop->CompactPopNew(compact_pop,fvect); 
           
	 }  
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) pop->TruncSel(elitpop,Elit);  
    
   return NPoints; 
} 
 
inline void FindBestVal() 
{     
      if(Elit && Tour != 0)  
          { 
            BestEval =elitpop->Evaluations[0]; 
            BestInd = elitpop->P[0]; 
	  } 
      else if(Tour==0) 
      {  
        BestEval = selpop->Evaluations[0]; 
        BestInd = selpop->P[0]; 
      } 
      else  
          { 
	   int auxind =  pop->FindBestIndPos();  
           BestInd =  pop->P[auxind]; 
           BestEval = pop->Evaluations[auxind]; 
          } 
} 
 
inline void InitPopulations() 
{
  int i; 
 if (Tour==0) 
   { 
     TruncMax = int(psize*Trunc);  
   
     if (BestElitism)  Elit = TruncMax;   //Only for Trunc Selection  
     selpop = new Popul(TruncMax,vars,Elit,Cardinalities);  
   }  
  else selpop = new Popul(psize,vars,Elit,Cardinalities);  
 
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop = new Popul(Elit,vars,Elit,Cardinalities); 
  pop = new Popul(psize,vars,Elit,Cardinalities);
  pop->RandInit();  
  compact_pop = new Popul(psize,vars,Elit,Cardinalities);  
  fvect = new double[psize];
 
 } 
 
inline void DeletePopulations() 
{ 
  delete compact_pop; 
  delete pop;  
  delete selpop;  
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) delete elitpop; 
 delete[] fvect; 
} 
 






int RobustMNEDA(double Complexity,int typemodel)  
{  
  int i,fgen,gap;  
  double auxprob,sumprob,OldBest;     
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  double lam;   //Parameters used by Affinity 
  int maxits, convits, deph;
  double SimThreshold;  
  unsigned long k,j,l;
  unsigned long nclust; 
  unsigned long* allvars;
  int ncalls;
  int auxnumber;
  unsigned int* auxconfiguration;
  auxconfiguration = new unsigned int[vars];

 
  init_time(); 
  InitPopulations();  
  LearningType=3;  //MOA Model 
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);   
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1; NPoints = 100; 

  // cout<<CliqMaxLength<<" "<<MaxNumCliq<<" "<<Complexity<<" "<<selpop->psize<<" "<<EvaluationMode<<" "<<BestEval<<"  "<<NPoints<<"  "<<KDec<<endl; 
 
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 0;
 BB = 0;

 TotEvaluations = 0;
 //while (i<Maxgen && BestEval<Max && NPoints>10 && BB<( (vars/KDec)-1))  
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
     evalfunction(pop,Elit,psize,i);
     //pop->EvaluateAll(func);

     NPoints = Selection(); 
    
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (i+1);
     MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(selpop);   
     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary
    
     double threshold = 1.5;   
     MyMarkovNet->UpdateModelGTest(Cardinalities,Complexity);
     MyMarkovNet->MChiSquare = (double*) 0;   
     nclust = MyMarkovNet->SetOfCliques->NumberCliques;

     FactorGraph fg;
     fg = CreateFactorGraph(MyMarkovNet,Cardinalities,nclust);
   
       //cout << "factors:" << facs << endl;    
	  //MyMarkovNet->Destroy();
	  //MyMarkovNet->UpdateModelProtein(1,1); //There is one clique for each variable, the first is the variable, the rest, its neighbors
    
     FindBestVal();   
     //auxprob = MyMarkovNet->Prob(BestInd);  
           
   if(func==50)
     {
    BB = 0;    
    for(int ii=0;ii<vars;ii+=KDec)
     {
      int auxval = 0;
      for(int jj=ii;jj<ii+KDec;jj++) auxval += selpop->P[0][jj];
      if(auxval==KDec*(Card-1) ) BB++; //Generalization for the case of integer deceptive      
     } 
     }
  
   if(printvals>1) 
    {     
     
     for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
       cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }


     if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" Nclusters "<<nclust<<endl;    
    }
      if (BestEval>=Max || BB>=( (vars/KDec)-1))   fgen  = i;	 
      else 
          {            
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {
                
		selpop->SetElit(Elit,pop);
	        for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }

       
      double chunkprob = 1/MyMarkovNet->SetOfCliques->NumberCliques;
      if(ExperimentMode==6)
	{    
         MyMarkovNet->GenPop(Elit,pop); 
           if(TypeInfMethod>0) auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
         if (auxnumber==1) 
            for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];         
        }
      else if(ExperimentMode==7)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1) MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==8)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1) 
	       {
                MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
	        for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
               }
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==9)
	{       
          MyMarkovNet->GenStructuredCrossPop(Elit,pop,selpop);   
          if(TypeInfMethod>0) 
	    {            
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1) 
               for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];              
             //MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
	     //for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
            }          
        }

       MyMarkovNet->Destroy();   
      i++; 
  } 
   end_time(); 
 
   if(printvals>0)  cout<<"LastGen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" "<<TotEvaluations<<endl;  //cout<<BestEval<<endl; 
 
  if(NPoints>10) NPoints = 10; 
  DeletePopulations(); 
  delete MyMarkovNet;    
  delete IntTree;
  return fgen;
  delete[] auxconfiguration;
} 



// MN-EDA and MN-FDA for binary problems
int Markovinit(double Complexity, int typemodel, int sizecliq)  //In this case, complexity is the threshold for chi-square 
{  
  int i,j, fgen,gap;  
  double auxprob,OldBest;     
  DynFDA* MyMarkovNet;  
  
    
  init_time();
  InitPopulations(); 
 
  //pop->Print();
  LearningType=3;
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
 
  //cout<<CliqMaxLength<<" "<<MaxNumCliq<<"  "<<Complexity<<" "<<Prior<<"  "<<LearningType<<" "<<Cycles<<endl;
  i=0;  fgen = -1;  auxprob =0; BestEval  = Max -1; NPoints=TruncMax; 

  Popul* aux_pop;
  aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
 
 NPoints = psize;
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 0;
 BB = 0;
 TotEvaluations = 0;

 while (i<Maxgen && BestEval<Max && NPoints>10 && BB<( (vars/KDec)-1))  
  {  

   evalfunction(pop,Elit,psize,i);       
   NPoints = Selection(); 
   MyMarkovNet->current_gen = (i+1);
   MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
   MyMarkovNet->SetPop(selpop);    
   FindBestVal();   


    BB = 0;    
    for(int ii=0;ii<vars;ii+=KDec)
     {
      int auxval = 0;
      for(int jj=ii;jj<ii+KDec;jj++) auxval += selpop->P[0][jj];
      if(auxval==KDec*(Card-1) ) BB++; //Generalization for the case of integer deceptive      
     } 


  if(printvals>1) 
    {     
     cout<<"Best : ";
     for(int l=0;l<vars;l++) cout<<BestInd[l]<<" ";
     printf("%10f \n", BestEval);   
     if(printvals>0)     cout<<"Gen : "<<i<<" Best: "<<BestEval<<" Elit "<<Elit<<" TotEval: "<<TotEvaluations<<" DifPoints: "<<NPoints<<" sizecliq "<<sizecliq<<" "<<OldBest<<" "<<BestEval<<" "<<gap<<endl; 
    }

     MyMarkovNet->UpdateModel(); 

     FactorGraph fg;
     fg = CreateFactorGraph(MyMarkovNet,Cardinalities,MyMarkovNet->NeededCliques);
     
     auxprob = MyMarkovNet->Prob(BestInd);      
     if (BestEval>=Max || BB>=( (vars/KDec)-1))   fgen  = i;	 
      else 
          {            
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {                
		selpop->SetElit(Elit,pop);
	        for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }
     
     
      if(NPoints>10)  MyMarkovNet->GenPop(Elit,pop);  
      int auxnumber;
      if(TypeInfMethod>0) auxnumber  = FindMAP(fg,TypeInfMethod,pop->P[psize-1]);   

      i++;         
     MyMarkovNet->Destroy();  
  }  

  if(NPoints>10) NPoints = 10;
   end_time();  

   if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DiffPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<endl;  
  
  if(printvals>0) 
   {        
    cout<<" Best Solution:  ";   
    for(int ll=0;ll<printvals;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 
  //for(int ll=i;ll<Maxgen;ll++)  AllGen[ll] =   AllGen[ll] + BestEval;
  delete aux_pop; 
  //delete big_pop;
  DeletePopulations(); 
  delete MyMarkovNet;
  return fgen;  
}  
 



int Intusualinit(double Complexity)  
{  
  int i,fgen;  
  double auxprob,sumprob;     
  IntTreeModel *IntTree;  
 
  init_time(); 
  InitPopulations(); 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
  NPoints = 100;
   
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
   
     //pop->Print();
     evalfunction(pop,Elit,psize,i);     
     NPoints = Selection(); 
     
   
       
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();  
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();  
     IntTree->MakeTree(IntTree->rootnode); 
      FindBestVal(); 
      //AllGen[i] += BestEval;
 
      //IntTree->PrintModel();
      // IntTree->PrintMut();
     
   
     IntTree->PutPriors(Prior,selpop->psize,1);
     sumprob = IntTree->SumProb(selpop,NPoints);  
     //auxprob = IntTree->Prob(BestInd); 
     //selpop->Print(0); 
     //cout<<"Now is serious "<<endl;       
      
      
if(printvals>1) 
   {           
 
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }
if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TreProb:"<<sumprob<<" "<<Elit<<endl;    
   }

 
       

      if (BestEval>=Max)   fgen  = i;	 
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {
		selpop->SetElit(Elit,pop);
	        for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }

      if(NPoints>10) IntTree->GenPop(Elit,pop);   
    
     i++;
  }  
  
  end_time(); 
  if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<endl;  //cout<<BestEval<<endl; 
  if(NPoints>10) NPoints = 10;
  /*
 if(printvals>0) 
   {           
    for(int ll=0;ll<NPoints;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
  */
  //for(int ll=i;ll<Maxgen;ll++)  AllGen[ll] =   AllGen[ll] + BestEval;   
  DeletePopulations(); 
  delete IntTree;
  return fgen;  

}  

int  MixturesIntAlgorithm(int Type,unsigned *Cardinalities,double Complexity)  
{  
  int i,fgen;  
  double auxprob;  
  MixtureIntTrees *MixtureInt;  
  init_time(); 
  InitPopulations(); 
  MixtureInt = new MixtureIntTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior,Cardinalities);
  i=0; auxprob = 0; BestEval = Max-1; NPoints = 100; fgen = -1;  


 while (i<Maxgen && BestEval<Max && NPoints>10)  //&& oldlikehood != likehood)  
  { 
     
  
   //pop->Print(); 

   evalfunction(pop,Elit,psize,i); 
   NPoints = Selection(); 
   
   MixtureInt->SetNpoints(NPoints,fvect);
   MixtureInt->SetPop(selpop);
   MixtureInt->MixturesInit(Type,InitTreeStructure,fvect,Complexity,0,0,0,0);
   MixtureInt->LearningMixture(Type);  
 
 
   FindBestVal();
   //AllGen[i] += BestEval; 
   auxprob = MixtureInt->Prob(BestInd);  

        
if(printvals>1) 
   {           
 
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }
if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" "<<Elit<<endl;    
   }


    if (BestEval>=Max) fgen  = i;	   
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
            {
               selpop->SetElit(Elit,pop);  
               //for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];    
	    }
       if(NPoints>10)    MixtureInt->SamplingFromMixture(pop);  
                
      	  }   

       MixtureInt->RemoveTrees(); 
       MixtureInt->RemoveProbabilities();
        
      i++;  

 
  }  


 end_time();  

 if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<endl;   // cout<<BestEval<<endl;
 /*
  if(printvals>0) 
   {           
    for(int ll=0;ll<NPoints;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 */
 //for(int ll=i;ll<Maxgen;ll++)  AllGen[ll] =   AllGen[ll] + BestEval;
  delete MixtureInt;  
  DeletePopulations();

  return fgen;  
}  




int MOA(double Complexity,int typemodel)  
{  
  int i,fgen,gap;  
  double auxprob,sumprob,OldBest;     
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  double lam;   //Parameters used by Affinity 
  int maxits, convits, deph;
  unsigned long **listclusters;
  double SimThreshold;  
  unsigned long k,j;
  unsigned long nclust; 
  unsigned long* allvars;
  int ncalls;
  
  init_time(); 
  InitPopulations(); 
 
  MaxNumCliq = vars;
  LearningType=6; //3; // MOA Model 
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
  
  

  listclusters = new  unsigned long* [vars];
  allvars = new  unsigned long [vars];
  for(i=0;i<vars;i++) allvars[i] = i;
  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1; NPoints = 100; nclust = vars; 

  // cout<<CliqMaxLength<<" "<<MaxNumCliq<<" "<<Complexity<<" "<<selpop->psize<<" "<<EvaluationMode<<" "<<BestEval<<"  "<<NPoints<<"  "<<KDec<<endl; 
 
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 0;
 BB = 0;

 TotEvaluations = 0;
 while (i<Maxgen && BestEval<Max && NPoints>10 && BB<( (vars/KDec)-1))  
  {  
     evalfunction(pop,Elit,psize,i);
     //pop->EvaluateAll(func);

     NPoints = Selection(); 

    
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (i+1);
     MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(selpop); 
    

     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary

    

     double threshold = 1.5; 
       MyMarkovNet->FindNeighbors(listclusters,CliqMaxLength,threshold); // The neighborhood is found from the matrix of mutual information  
       MyMarkovNet->MChiSquare = (double*) 0;    
       MyMarkovNet->UpdateModelProteinMPM(2,nclust,listclusters); //There is one clique for each variable, the first is the variable, the rest, its neighbors    
       FindBestVal();   
     //auxprob = MyMarkovNet->Prob(BestInd);  
           

    BB = 0;    
    for(int ii=0;ii<vars;ii+=KDec)
     {
      int auxval = 0;
      for(int jj=ii;jj<ii+KDec;jj++) auxval += selpop->P[0][jj];
      if(auxval==KDec*(Card-1) ) BB++; //Generalization for the case of integer deceptive      
     } 


  
   if(printvals>1) 
    {     
     
     for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
       cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }



     if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" Nclusters "<<nclust<<endl;    
    }
      if (BestEval>=Max || BB>=( (vars/KDec)-1))   fgen  = i;	 
      else 
          { 
           
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {
                
		selpop->SetElit(Elit,pop);
	        for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }

                   
     
      //if(NPoints>10) IntTree->GenPop(Elit,pop);
      if(NPoints>10) MyMarkovNet->GenPop(Elit,pop);     
     
      //MyMarkovNet->Destroy();
   
      i++;

 
  } 
   end_time(); 
 
   if(printvals>0)  cout<<"LastGen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" "<<TotEvaluations<<endl;  //cout<<BestEval<<endl; 
  if(NPoints>10) NPoints = 10;
 
  DeletePopulations(); 
  delete IntTree;
  delete MyMarkovNet; 
  for(i=0;i<nclust;i++) delete[] listclusters[i];
  delete[] listclusters;       
  delete[] allvars; 

  return fgen;
} 




// Factor graph based EDA (uses decimation as sampling method)


int FGEDA(double Complexity,int typemodel)  
{  
  int i,fgen,gap;  
  double auxprob,sumprob,OldBest;     
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  double lam;   //Parameters used by Affinity 
  int maxits, convits, deph;
  unsigned long **listclusters;
  double SimThreshold;  
  unsigned long k,j,l;
  unsigned long nclust; 
  unsigned long* allvars;
  int ncalls;
  int auxnumber;
  unsigned int* auxconfiguration;
  auxconfiguration = new unsigned int[vars];


  init_time(); 
  InitPopulations(); 
 
  MaxNumCliq = vars;
  LearningType=6; //3; // MOA Model 
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
  
  listclusters = new  unsigned long* [vars];
  allvars = new  unsigned long [vars];
  for(i=0;i<vars;i++) allvars[i] = i;
  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1; NPoints = 100; nclust = vars; 

  // cout<<CliqMaxLength<<" "<<MaxNumCliq<<" "<<Complexity<<" "<<selpop->psize<<" "<<EvaluationMode<<" "<<BestEval<<"  "<<NPoints<<"  "<<KDec<<endl; 
 
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 0;
 BB = 0;

 TotEvaluations = 0;
 //while (i<Maxgen && BestEval<Max && NPoints>10 && BB<( (vars/KDec)-1))  
 while (i<Maxgen && BestEval<Max && NPoints>10) 
  {  
     evalfunction(pop,Elit,psize,i);
     //pop->EvaluateAll(func);

     NPoints = Selection(); 
    
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (i+1);
     MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(selpop); 
    

     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary

    

     double threshold = 1.5; 
     MyMarkovNet->FindNeighbors(listclusters,CliqMaxLength,threshold); // The neighborhood is found from the matrix of mutual information  
     MyMarkovNet->MChiSquare = (double*) 0;
     MyMarkovNet->UpdateModelProteinMPM(2,nclust,listclusters); //There is one clique for each variable, the first is the variable, the rest, its neighbors

     FactorGraph fg;
     fg = CreateFactorGraph(MyMarkovNet,Cardinalities,nclust);
   
       //cout << "factors:" << facs << endl;    
	  //MyMarkovNet->Destroy();
	  //MyMarkovNet->UpdateModelProtein(1,1); //There is one clique for each variable, the first is the variable, the rest, its neighbors
    
     FindBestVal();   
     //auxprob = MyMarkovNet->Prob(BestInd);  
           
     /*
    BB = 0;    
    for(int ii=0;ii<vars;ii+=KDec)
     {
      int auxval = 0;
      for(int jj=ii;jj<ii+KDec;jj++) auxval += selpop->P[0][jj];
      if(auxval==KDec*(Card-1) ) BB++; //Generalization for the case of integer deceptive      
     } 
     */
  
   if(printvals>1) 
    {     
     
     for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
       cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }


     if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" Nclusters "<<nclust<<endl;    
    }
      if (BestEval>=Max || BB>=( (vars/KDec)-1))   fgen  = i;	 
      else 
          {            
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {
                
		selpop->SetElit(Elit,pop);
	        for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }

                   
     

      double chunkprob = 1/MyMarkovNet->SetOfCliques->NumberCliques;
      if(ExperimentMode==2)
	{    
         MyMarkovNet->GenPop(Elit,pop); 
         if(TypeInfMethod>0) auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
         if (auxnumber==1) 
            for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
        }
      else if(ExperimentMode==3)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1) MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==4)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1)
	       {
                MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
	        for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
               }
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==5)
	{       
          MyMarkovNet->GenStructuredCrossPop(Elit,pop,selpop);   
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1)
	        for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];        
             //MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
	     //for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
            }          
        }

       //MyMarkovNet->Destroy();
   
      i++;

 
  } 
   end_time(); 
 
   if(printvals>0)  cout<<"LastGen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" "<<TotEvaluations<<endl;  //cout<<BestEval<<endl; 
  if(NPoints>10) NPoints = 10;
 
  DeletePopulations(); 
  delete IntTree;
  delete MyMarkovNet; 
  for(i=0;i<nclust;i++) delete[] listclusters[i];
  delete[] listclusters;       
  delete[] allvars; 
  delete[] auxconfiguration;
  return fgen;
} 


void PrintStatistics() 
{  
  int i;
  double auxmeangen,meanfit,sigma; 
 
  sigma = 0;
                   meaneval /=  cantexp; 
                   alltime  =  alltime/(1.0*cantexp); 
		   for (i=0;i<cantexp;i++) 
                   {
                    sigma += (meanlikehood[i] - meaneval)*(meanlikehood[i] - meaneval);
                    //cout<<sigma<<endl;
                   } 
                   sigma = sigma/(cantexp);
                   
                if (succexp>0)  
                   {  
                    auxmeangen = meangen/succexp;
                    bestalltime = bestalltime/(1.0*succexp); 
                    if (BestElitism)  
                         meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;     
                    else meanfit = (auxmeangen+1)*(psize-1) + 1; 
                    cout<<"TypeExp="<<ExperimentMode<<" Func="<<func<<" KDec="<<KDec<<" Card="<<Card<<" n="<<vars<<" T="<<Trunc<<" N="<<psize<<" Sel="<<Tour<<"  TypeInfMethod="<<TypeInfMethod<<"  k="<<CliqMaxLength<<"  MaxGen="<<Maxgen<<"  Elit="<<Elit<<" Suc.="<<succexp<<"  g="<<(auxmeangen+1)<<"  ave="<<meanfit<<" meaneval "<<meaneval<<" sigma "<<sigma<<" timebest "<<bestalltime<<" fulltime "<<alltime<<" Solap "<<Solap<<endl;     
                    cout<<"SUMMARY "<<ExperimentMode<<" "<<func<<" "<<KDec<<" "<<Card<<" "<<vars<<" "<<Trunc<<" "<<psize<<" "<<Tour<<" "<<TypeInfMethod<<" "<<CliqMaxLength<<" "<<Maxgen<<" "<<Elit<<" "<<succexp<<" "<<(auxmeangen+1)<<" "<<meanfit<<" "<<meaneval<<" "<<0<<" "<<bestalltime<<" "<<alltime<<" "<<Solap<<endl;                  
                   } 
                  else  
                   {  
		      cout<<"TypeExp="<<ExperimentMode<<" Func="<<func<<" KDec="<<KDec<<" Card="<<Card<<" n="<<vars<<" T="<<Trunc<<" N="<<psize<<" Sel="<<Tour<<"  TypeInfMethod="<<TypeInfMethod<<"  k="<<CliqMaxLength<<"  MaxGen="<<Maxgen<<"  Elit="<<Elit<<" Suc.="<<0<<"  g="<<0<<"  ave="<<0<<" meaneval "<<meaneval<<" sigma "<<sigma<<" fulltime "<<alltime<<" Eval "<<(TotEvaluations/(1.0*cantexp))<<" Solap "<<Solap<<endl; 
                   } 


                  
		  //for(int ll=0;ll<Maxgen;ll++)  cout<<AllGen[ll]/(-1.0*cantexp)<<" ";
                  //cout<<endl;
} 



void runOptimizer(int algtype,int nrun)  
{  
    int succ=-1; 
        
      switch(algtype)  
                     {                     
                       //case 0: succ = Markovinit(Complex,1,Cycles);break;  // Markov Network       1
			 // case 1: succ = Intusualinit(Complex);break;
                       //case 2: succ = MixturesIntAlgorithm(1,Cardinalities,Complex);break;// MT on dependencies 
                       //case 4: succ = MOA(Complex,2);break;  // (MOA)
                       case 0: succ = Markovinit(Complex,1,Cycles);break;  // Markov Network       1
                       case 1: succ = Intusualinit(Complex);break;
                       case 2: succ = FGEDA(Complex,2);break;  // (FGEDA)
                       case 3: succ = FGEDA(Complex,2);break;  // (FGEDA)
                       case 4: succ = FGEDA(Complex,2);break;  // (FGEDA)
                       case 5: succ = FGEDA(Complex,2);break;  // (FGEDA)
		       case 6: succ = RobustMNEDA(Complex,2);break; // Learns MN-EDA, MN-FDA, or MOA using MI and applying G-test after (Elitist MPC)
                       case 7: succ = RobustMNEDA(Complex,2);break; // Learns MN-EDA, MN-FDA, or MOA using MI and applying G-test after (Biased MPC)
                       case 8: succ = RobustMNEDA(Complex,2);break; // Learns MN-EDA, MN-FDA, or MOA using MI and applying G-test after (Biased MPC+Elitist MPC)
                       case 9: succ = RobustMNEDA(Complex,2);break; // Learns MN-EDA, MN-FDA, or MOA using MI and applying G-test after (Biased MPC+Elitist MPC)
                     } 

  

    if (succ>-1)  
   { 
       succexp++; 
       meangen += succ;    
       bestalltime +=auxtime;      
        
   } 
   else nsucc++;
   alltime += auxtime;  
   meaneval += BestEval; 
   meanlikehood[nrun] = BestEval;  
} 



int main( int argc, char *argv[] )
{
   
  // ./moaprot 1 5 1 30 500 50 20 1 10 3 50 3 2 1 2 1 // An example of calling MOA

  int i,a;
  int prot_inst,modeprotein;
  int T,MaxMixtP,S_alph,Compl; 

  
  if( argc != 18 ) {
    std::cout << "Usage: " <<"cantexp  EDA{0:Markov, 1:Tree  2:Mixture, 4:AffEDA} modeprotein{2,3} prot_inst n psize Trunc max-gen" << std::endl;
    std::cout << "       Please read the README file." << std::endl;
    exit(1);
}


 params = new int[3];    
 //MatrixFileName = "newviewtrees.txt";  
 cantexp = atoi(argv[1]);         // Number of experiments
 ExperimentMode = atoi(argv[2]); // Type of EDA
 
 length = atoi(argv[3]);             // Number of bits for  each variable   
 vars =  atoi(argv[4]) * length;           //Number of variables (redundant because depends on instance)
 psize = atoi(argv[5]);          // Population size
 T = atoi(argv[6]);              // Percentage of truncation integer number (1:99)
 Maxgen =  atoi(argv[7]);        // Max number of generations 
 BestElitism = atoi(argv[8]);         // If there is or not BestElitism, if thereisnot BestElitism, Elitism = 1 by default;
 Max = atoi(argv[9]);
 CliqMaxLength = atoi(argv[10]);
 func  = atoi(argv[11]);
 params[0]  = atoi(argv[12]);
 params[1]  = atoi(argv[13]);
 params[2]  = atoi(argv[14]);
 printvals  = atoi(argv[15]);
 Card =  atoi(argv[16]);
 TypeInfMethod =  atoi(argv[17]);
 

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
 Prior = 1;                // Type of prior. 
 Compl=75;                 // Complexities of the trees. 
 Coeftype=2;               // Type of coefficient calculation for Exact Learning. 
 //params[0] = 3 ;           //  Params for function evaluation 
 //params[1] = 3;  
 // params[2] = 10;  
 
 
 //seed =  1243343896; 
 seed = (unsigned) time(NULL);  
 srand(seed); 
 cout<<"seed"<<seed<<endl; 

TypeMixture = 1; // Class of MT-FDA (1-Meila, 2-MutInf)
Mutation = 0 ; // Population based mutation  
//CliqMaxLength = 2; // Maximum size of the cliques for Markov  or maximum number of neighbors for MOA
MaxNumCliq = 300; // Maximum number of cliques for Markov 
OldWaySel = 0; // Selection with sel pop (1) or straight on Sel prob (0) 
LearningType = 6; // Learning for MNFDA (0-Markov, 1-JuntionTree) 
Cycles = 0 ; // Number of cycles for GS in the MNEDA or size for the clique in Markov EDA. 
KDec = params[0]; 
Trunc = T/double(100);  
Complex  = Compl/double(100);  
MaxMixtProb =MaxMixtP/double(100); 
S_alpha = S_alph/double(100); 

//ReadParameters(); 

Cardinalities  = new unsigned[5000];  
int k,j,u;
double eval;


int auxpsize = 32;
//int auxpsize = 8192;

  
   if(func==50)
     {
      Max = (Card-1)*vars;
      params[1] = Card; 
     }
   else if(func==54)
     {
       Solap = params[2];
       params[1] = Card; 
       Max = (Card-1)*KDec*(1+(vars-KDec)/(KDec-Solap));
     }
   else if(func==8)
     Max = 1 + (vars-3)/2;

   for(u=0;u<5000;u++) Cardinalities[u] = Card;  
   explength = 1.0;
  
   for(i=0;i<length;i++) explength*=Card;  

   AbsBestInd = new unsigned int [vars];
   psize = auxpsize;
   succ = 0;

 while(psize<250000 && succ<30) // Experiments for critical pop-size
   {
    //cout<<func<<" "<<params[0]<<" "<<params[1]<<endl;
    cout<<"Alg : "<<ExperimentMode<<", number codifying bits : "<<length<<", n : "<<vars<<", psize : "<<psize<<", Trunc : "<<T<<", max-gen : "<<Maxgen<<", BestElit. : "<<BestElitism<<", NNeighbors  : "<<CliqMaxLength<<", MaxFun  : "<<Max<<", func : "<<func<<", params[0] : "<<params[0]<<", params[1] : "<<params[1]<<", params[1] : "<<params[2]<<endl; 

            
        AbsBestEval = -1;
        TotEvaluations = 0;       
	succexp=0; succ = 0; meangen = 0; meaneval = 0;  i =0;  nsucc =0; alltime = 0; bestalltime = 0;     
  while(nsucc==0 && succ<30)  // Only for Experiments of critical pop-size  //while (i<cantexp) //&& nsucc<1
        { 
          currentexp = i;	  
	  runOptimizer(ExperimentMode,i);
          i++;
         //PrintStatistics();
	  if(func==50)
	    {
             if(BB<vars/KDec-1) nsucc++;
	     else  succ++;       
             cout<<BB<<" "<<nsucc<<" "<<succ<<" "<<i<<" "<<auxtime<<endl; 
            }
	  else       
           {
             if(BestEval!=Max) nsucc++;
	     else  succ++;       
             cout<<BestEval<<" "<<nsucc<<" "<<succ<<" "<<i<<" "<<auxtime<<endl; 
            }
        }
        if(nsucc>0) psize = psize*2;
	else PrintStatistics();                
   }
	delete[] AbsBestInd;   
     
  
 delete [] params; 
 delete [] Cardinalities; 
 return 0;

}      




