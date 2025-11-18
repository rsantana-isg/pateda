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
#include "EA.h" 
#define itoa(a,b,c) sprintf(b, "%d", a) 


using namespace dai;
using namespace std;

/**********************************************************************************************************/
/*                         FUNCTIONS OF CLASS EA                                                           */
/**********************************************************************************************************/

  EA::EA(int nvars,double vMax, unsigned* vCard, void (*objective_function) (Popul*,int, int) , int vpsize, int elit, int best_elit, int maxgen, int pval, int prior, int oldwaysel, double trunc_prop, int type_selection)
   {           
	               // PROBLEM PARAMETERS
        vars = nvars;       	  
	Max  = vMax;       	  
       	Cardinalities = vCard;  	
        ObjFunction = objective_function;	

	               // ALGORITHM PARAMETERS 
	psize = vpsize;    
	Elit = elit ;  	 	
	BestElitism = best_elit; 
	Maxgen = maxgen;  
	printvals = pval;   
        Prior = prior; 	
	                //SELECTION PARAMETERS      
        OldWaySel =oldwaysel;

        Trunc = trunc_prop; 	
	Tour = type_selection;  
      	TruncMax = int(Trunc*psize);          
               
     	TotEvaluations = 0; 

   } 


void EA::init_time()
{
 time( &ltime_init );
 gmt = localtime( &ltime_init );
 auxtime = - ( gmt->tm_mday * 86400 + gmt->tm_hour*3600 + gmt->tm_min*60 + gmt->tm_sec);
}


void  EA::end_time()
{
  time( &ltime_end );
  gmtnew = localtime( &ltime_end );
  auxtime = auxtime + gmtnew->tm_mday * 86400 + gmtnew->tm_hour*3600+gmtnew->tm_min*60+gmtnew->tm_sec;
}

int EA::Selection() 
{ 
    NPoints=0; 
    if (Tour==0)  
         {  
           pop->TruncSel(selpop,TruncMax); 
           selpop->UniformProb(TruncMax,fvect);	  
           //selpop->BotzmannDist(1.0,fvect);          
           NPoints = selpop->CompactPopNew(compact_pop,fvect); 
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


 
void EA::FindBestVal() 
{     
           
      if(Elit && Tour != 0)  
          {          
            BestEval =elitpop->Evaluations[0]; 
            BestInd = elitpop->P[0]; 
	  } 
      else if(Tour==0) 
      {  
       if(selpop->Evaluations[0]>BestEval) GenFound = current_gen;                               
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


 
void EA::InitPopulations() 
{  
 if (Tour==0) 
   { 
     TruncMax = int(psize*Trunc);     
     if (BestElitism)  Elit = TruncMax;   //Only for Trunc Selection  
     selpop = new Popul(TruncMax,vars,Elit,Cardinalities);  
   }  
  else selpop = new Popul(psize,vars,Elit,Cardinalities);  
 
  if (Tour>0 || (Tour==0 && Elit>TruncMax))
    {
     elitpop = new Popul(Elit,vars,Elit,Cardinalities);     
    }
  pop = new Popul(psize,vars,Elit,Cardinalities);
  pop->RandInit();  
  compact_pop = new Popul(psize,vars,Elit,Cardinalities);  
  fvect = new double[psize];
 } 
 

void  EA::ApplySelection()
 {
   int ll; 
       if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);         
           else 
              {                
		selpop->SetElit(Elit,pop);
	        for(ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          
 }



void EA::DeletePopulations() 
{ 
  delete compact_pop; 
  delete pop;  
  delete selpop;  
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) delete elitpop; 
 delete[] fvect; 
} 




void EA::PrintBestSolutions() 
{ 
   int l,ll;
   if(printvals>1) 
    {          
     for(ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
       cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }
     if(printvals)   cout<<"Gen : "<<current_gen<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<endl;    
    }
}

void  EA::PrintLastGen() 
{ 
  if(printvals>0)  cout<<"Gen : "<<current_gen<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<endl;   
}

void EA::InitAlgorithm()
{
  current_gen=0; 
  auxprob =0; 
  BestEval  = -1*Max;
  fgen = -1;  
  NPoints = 100;
  AbsBestEval = 0;
  TotEvaluations = 0;

}


void EA::MutatePop(Popul* NewPop,int eli, int last)
{
  int i,k,auxvar;   

for(k=eli; k < last; k ++) 
  {
    for (i=0;i<vars/5; i++)        
	 {
	   auxvar = randomint(vars);
	   NewPop->P[k][auxvar] =  1-NewPop->P[k][auxvar];
         }   
  } 
}

/***********************************************************************************************************/
/*                         FUNCTIONS OF CLASS GAs                                                          */
/***********************************************************************************************************/


GA::GA(int nvars,double vMax, unsigned* vCard, void (*objective_function)(Popul*,int, int), int vpsize, int elit, int best_elit, int maxgen, int pval, int prior, int oldwaysel, double trunc_prop, int type_selection, int cxtype):EA(nvars,vMax,vCard,objective_function,vpsize,elit,best_elit,maxgen,pval,prior,oldwaysel,trunc_prop,type_selection)
 {
   CX_Type = cxtype;
   
 }


void GA::GenOnePointCXInd(Popul* NewPop, int pos, unsigned int* parind1, unsigned int* parind2)   
{    
  // The vector in position pos is generated   
  int i,auxvar;       
  int CXPoint;
  CXPoint = randomint(vars-1)+1;
  for (i=0;i<CXPoint; i++)  NewPop->P[pos][i] =  parind1[i]; 
  for (i=CXPoint;i<vars; i++)  NewPop->P[pos][i] =  parind2[i];
   
}   




void GA::GenUniformCXInd(Popul* NewPop, int pos, unsigned int* parind1, unsigned int* parind2)   
{ 
   // The vector in position pos is generated   
  int i,auxvar;
       for (i=0;i<vars; i++)  
	      if (myrand()>0.5) NewPop->P[pos][i] =  parind2[i];
	  else   NewPop->P[pos][i] =  parind1[i];          
    
}   


void GA::GenCrossPop(int From,Popul* NPop, Popul* SPop, int nparents)   
{    
  int i,parent1,parent2;   

  if (CX_Type==0)     //One point crossover
    {
       for(i=From; i<NPop->psize; i++)   
	 {   
	   parent1 = randomint(nparents);
           parent2 = randomint(nparents);	                   
           GenOnePointCXInd(NPop,i, SPop->P[parent1],SPop->P[parent2]);          
        }
    } 
  else if (CX_Type==1)     // Uniform crossover
    {
       for(i=From; i<NPop->psize; i++)   
	 {   
	   parent1 = randomint(nparents);
           parent2 = randomint(nparents);	          
          GenUniformCXInd(NPop,i, SPop->P[parent1],SPop->P[parent2]); 
         
        }
    }
}   

int GA::SimpleGA()  
{   
  init_time(); 
  InitPopulations();  
  InitAlgorithm();
  
 
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  { 
   
   if(current_gen==0)  ObjFunction(pop,0,psize);       
    else ObjFunction(pop,Elit,psize);     
       
    NPoints = Selection();                   
    
    FindBestVal();               
    PrintBestSolutions();        
      
    if (BestEval>=Max)   fgen  = current_gen;	 
    else  
      {
       ApplySelection();              
       fgen = current_gen;
      }
  
    GenCrossPop(Elit,pop,selpop,selpop->psize);      
    
    MutatePop(pop,Elit,psize);       
    
    current_gen++;   
  } 
  
  if(NPoints>10) NPoints = 10; 
  PrintLastGen();
  
  end_time();  
  
  DeletePopulations();   
  return fgen;

} 



int GA::CompactGA()  
{    
  init_time(); 
  InitPopulations();  
  InitAlgorithm();

  //cout<<current_gen<<" "<<Maxgen<<" "<<BestEval<<" "<<NPoints<<endl;
 
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {      
     if(current_gen==0)  ObjFunction(pop,0,psize);       
     else ObjFunction(pop,Elit,psize);           
     NPoints = Selection();                   
     FindBestVal();               
     PrintBestSolutions();     
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }
     
     GenCrossPop(Elit,pop,compact_pop,NPoints);             
     MutatePop(pop,Elit,psize);       
     current_gen++; 
  } 

  if(NPoints>10) NPoints = 10; 
  PrintLastGen();
  end_time();  

  DeletePopulations();   
  return fgen;

} 


/***********************************************************************************************************/
/*                         FUNCTIONS OF CLASS Tree_EDA                                               */
/***********************************************************************************************************/


Tree_EDA::Tree_EDA(int nvars,double vMax, unsigned* vCard, void (*objective_function)(Popul*,int, int), int vpsize, int elit, int best_elit, int maxgen, int pval, int prior, int oldwaysel, double trunc_prop, int type_selection, double complexity):EA(nvars,vMax,vCard,objective_function,vpsize,elit,best_elit,maxgen,pval,prior,oldwaysel,trunc_prop,type_selection)
{
  Complexity = complexity;
}

int Tree_EDA::Intusualinit()  
{          
  init_time(); 
  InitPopulations(); 
  InitAlgorithm();
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  
 
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {  
   
     //pop->Print();
     if(current_gen==0)  ObjFunction(pop,0,psize);       
     else ObjFunction(pop,Elit,psize);       
     NPoints = Selection(); 
     
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();  
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();  
     IntTree->MakeTree(IntTree->rootnode); 
     FindBestVal(); 
 
      //IntTree->PrintModel();
      // IntTree->PrintMut();     
   
     IntTree->PutPriors(Prior,selpop->psize,1);      
     PrintBestSolutions(); 
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      } 



     if(NPoints>10)
       {
         IntTree->GenPop(Elit,pop);   
         MutatePop(pop,Elit,psize);       
       }
    
     current_gen++;
  }  
  
  if(NPoints>10) NPoints = 10;
  end_time(); 

  DeletePopulations(); 
  delete IntTree;
  return fgen;  
}  



int Tree_EDA::CompactIntusualinit()  
{ 
  IntTreeModel *IntTree;  
 
  init_time(); 
  InitPopulations(); 
  InitAlgorithm();
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  

   
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {  
      //pop->Print();
    if(current_gen==0)  ObjFunction(pop,0,psize);       
    else ObjFunction(pop,Elit,psize);       
          
     NPoints = Selection();              
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();  
     IntTree->CalProbFvect(compact_pop,fvect,NPoints);        
     IntTree->CalMutInf();  
     IntTree->MakeTree(IntTree->rootnode); 
      FindBestVal(); 
      //AllGen[i] += BestEval;
 
      //IntTree->PrintModel();
      // IntTree->PrintMut();
     
   
     IntTree->PutPriors(Prior,compact_pop->psize,1);     
     PrintBestSolutions();
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }
     
     
      if(NPoints>10) 
	{
          IntTree->GenPop(Elit,pop);   
	  //  MutatePop(pop,Elit,psize);       
        }
    
     current_gen++;
  }  
  
  PrintLastGen();
  if(NPoints>10) NPoints = 10;
  end_time(); 
 
  DeletePopulations(); 
  delete IntTree;
  return fgen;  

}  




int Tree_EDA::FixedStructure_Intusualinit(unsigned int** matrixDSYS)  
{
  IntTreeModel *IntTree;  
 
  init_time(); 
  InitPopulations(); 
  InitAlgorithm(); 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  
   
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {  
     if(current_gen==0)  ObjFunction(pop,0,psize);       
     else ObjFunction(pop,Elit,psize);       

     NPoints = Selection();              
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();  
     //IntTree->CalProbFvect(selpop,fvect,NPoints);     
     IntTree->CalProbFvect(selpop,fvect,NPoints,matrixDSYS);        
     IntTree->CalMutInf(matrixDSYS); //THIS IS ONLY FOR STRUCTURAL INFO
     IntTree->MakeTree(IntTree->rootnode); 
     FindBestVal(); 

     //IntTree->PrintModel();
     // IntTree->PrintMut();
       
     if(Prior>0)  IntTree->PutPriors(Prior,selpop->psize,1);
        
     PrintBestSolutions();                 

     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   

     if(NPoints>10)         
       {
         IntTree->GenPop(Elit,pop);       
         MutatePop(pop,Elit,psize);                
       }
     current_gen++;
  }  

  PrintLastGen();
  if(NPoints>10) NPoints = 10;
  end_time(); 
 
  DeletePopulations(); 
  delete IntTree;
  return fgen;  

}  



/***********************************************************************************************************/
/*                         FUNCTIONS OF CLASS MIXTURE_TREES_EDA                                               */
/***********************************************************************************************************/


 Mixture_Trees_EDA::Mixture_Trees_EDA(int nvars,double vMax, unsigned* vCard, void (*objective_function)(Popul*,int, int), int vpsize, int elit, int best_elit, int maxgen, int pval, int prior, int oldwaysel, double trunc_prop, int type_selection, double complexity,  double s_alpha, double selint,int type_mixture, int ntrees, int nsteps,  int init_tree_structure, int visible_choice_var):EA(nvars,vMax,vCard,objective_function,vpsize,elit,best_elit,maxgen,pval,prior,oldwaysel,trunc_prop,type_selection)
{
  Complexity = complexity;
  S_alpha = s_alpha;
  SelInt = selint;  
  TypeMixture = type_mixture;
  Ntrees = ntrees;  
  Nsteps = nsteps;
  InitTreeStructure = init_tree_structure;  
  VisibleChoiceVar = visible_choice_var;  
}


int  Mixture_Trees_EDA::MixturesIntAlgorithm()  
{  
 
  init_time(); 
  InitPopulations(); 
  InitAlgorithm();
 
  MixtureInt = new MixtureIntTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior,Cardinalities);
 
 while (current_gen<Maxgen && BestEval<Max && NPoints>10)  //&& oldlikehood != likehood)  
  { 

   if(current_gen==0)  ObjFunction(pop,0,psize);       
   else ObjFunction(pop,Elit,psize);       
  
   NPoints = Selection(); 
   
   MixtureInt->SetNpoints(NPoints,fvect);
   MixtureInt->SetPop(selpop);
   MixtureInt->MixturesInit(TypeMixture,InitTreeStructure,fvect,Complexity,0,0,0,0);
   MixtureInt->LearningMixture(TypeMixture);  
  
   FindBestVal();
   //AllGen[i] += BestEval; 
   auxprob = MixtureInt->Prob(BestInd);  

   PrintBestSolutions();     
   if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   

   if(NPoints>10)    
     {
        MixtureInt->SamplingFromMixture(pop);   
	//  MutatePop(pop,Elit,psize);       
     }
   MixtureInt->RemoveTrees(); 

   MixtureInt->RemoveProbabilities();     
   current_gen++;   
  }  

  PrintLastGen();
  end_time();  
  delete MixtureInt;  
  DeletePopulations();
  return fgen;  
}  


/***********************************************************************************************************/
/*                         FUNCTIONS OF CLASS FACTORIZED_EDA                                               */
/***********************************************************************************************************/



 Factorized_EA::Factorized_EA(int nvars,double vMax, unsigned* vCard, void (*objective_function)(Popul*,int, int), int vpsize, int elit, int best_elit, int maxgen, int pval, int prior, int oldwaysel, double trunc_prop, int type_selection, double complexity, int cliq_max_length, int max_num_cliq, int learning_type, int cycles):EA(nvars,vMax,vCard,objective_function,vpsize,elit,best_elit,maxgen,pval,prior,oldwaysel,trunc_prop,type_selection)
{
  Complexity = complexity;
  CliqMaxLength = cliq_max_length;
  MaxNumCliq = max_num_cliq;
  LearningType = learning_type;
  Cycles = cycles;
 
}


// MN-EDA and MN-FDA for binary problems
int Factorized_EA::Markovinit(int sizecliq,int TypeInfMethod)  //In this case, complexity is the threshold for chi-square 
{  
  int auxnumber;          
  DynFDA* MyMarkovNet;  
      
  init_time();
  InitPopulations(); 
  InitAlgorithm();
  //pop->Print();
 
  LearningType=3;
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  
  Popul* aux_pop;
  aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
 
 while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {   

   if(current_gen==0)  ObjFunction(pop,0,psize);       
   else ObjFunction(pop,Elit,psize);        
   
   NPoints = Selection(); 
   MyMarkovNet->current_gen = (current_gen+1);
   MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
   MyMarkovNet->SetPop(selpop);    
   FindBestVal();   

   MyMarkovNet->UpdateModel(); 
   FactorGraph fg;
   fg = CreateFactorGraph(MyMarkovNet,Cardinalities,MyMarkovNet->NeededCliques);     
   auxprob = MyMarkovNet->Prob(BestInd);     

   PrintBestSolutions();  
   if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   

     
      if(NPoints>10) 
	{
           MyMarkovNet->GenPop(Elit,pop);  
	   // MutatePop(pop,Elit,psize);       
        }
   
      if(TypeInfMethod>0) auxnumber  = FindMAP(fg,TypeInfMethod,pop->P[psize-1]);   
      current_gen++;         
     MyMarkovNet->Destroy();  
  }  

  if(NPoints>10) NPoints = 10; 
  PrintLastGen();
  end_time();  
 
  delete aux_pop; 
  //delete big_pop;
  DeletePopulations(); 
  delete MyMarkovNet;
  return fgen;  
}  
 

int Factorized_EA::MOA()  
{  
  int i;
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  
 
  unsigned long **listclusters;
  unsigned long nclust; 
  
  init_time(); 
  InitPopulations(); 
  InitAlgorithm();
  
  MaxNumCliq = vars;
  LearningType=6; //3; // MOA Model 
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);    
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information   
  listclusters = new  unsigned long* [vars];
  
  while (current_gen<Maxgen && BestEval<Max && NPoints>10 )  
  {
    if(current_gen==0)  ObjFunction(pop,0,psize);       
    else ObjFunction(pop,Elit,psize);       
   
     NPoints = Selection(); 
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (current_gen+1);
     MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(selpop); 
    

     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary

    

     double threshold = 1.5; 
       MyMarkovNet->FindNeighbors(listclusters,CliqMaxLength,threshold); // The neighborhood is found from the matrix of mutual information  
       MyMarkovNet->MChiSquare = (double*) 0;    
       MyMarkovNet->UpdateModelProteinMPM(2,nclust,listclusters); //There is one clique for each variable, the first is the variable, the rest, its neighbors    
       FindBestVal();   
     //auxprob = MyMarkovNet->Prob(BestInd);  
           

     PrintBestSolutions();  
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   
     
       if(NPoints>10)
	 { 
           MyMarkovNet->GenPop(Elit,pop);          
	   // MutatePop(pop,Elit,psize);       
         }
       //MyMarkovNet->Destroy();   
      current_gen++; 
  } 

  PrintLastGen();
  if(NPoints>10) NPoints = 10;
  end_time(); 
 
  DeletePopulations(); 
  delete IntTree;
  delete MyMarkovNet; 
  for(i=0;i<nclust;i++) delete[] listclusters[i];
  delete[] listclusters;       

  return fgen;
} 


int Factorized_EA::RobustMNEDA(int TypeInfMethod, int ExperimentMode)  
{  
 
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  unsigned long nclust; 
  int auxnumber;
  unsigned int* auxconfiguration;
  auxconfiguration = new unsigned int[vars];

 
  init_time(); 
  InitPopulations();  
  InitAlgorithm();
 
  LearningType=3;   
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);   
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
 
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {  
     if(current_gen==0)  ObjFunction(pop,0,psize);       
     else ObjFunction(pop,Elit,psize);       
       
     NPoints = Selection(); 
     
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(compact_pop,fvect,NPoints,Prior);   
     IntTree->CalMutInf();
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (current_gen+1);
     MyMarkovNet->SetNPoints(compact_pop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(compact_pop);    
     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary
    
        
     MyMarkovNet->UpdateModelGTest(Cardinalities,Complexity);
     MyMarkovNet->MChiSquare = (double*) 0;   
     nclust = MyMarkovNet->SetOfCliques->NumberCliques;

     FactorGraph fg;
     fg = CreateFactorGraph(MyMarkovNet,Cardinalities,nclust);       
     FindBestVal();   
     //auxprob = MyMarkovNet->Prob(BestInd);  
           
    PrintBestSolutions();
    if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   
     
         
      double chunkprob = 1/MyMarkovNet->SetOfCliques->NumberCliques;
      if(ExperimentMode==6)
	{    
         MyMarkovNet->GenPop(Elit,pop); 
           if(TypeInfMethod>0) auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
	   if (auxnumber==1)   for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];         
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
             if (auxnumber==1)  for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];              
            }          
        }

      //  MutatePop(pop,Elit,psize);         
       MyMarkovNet->Destroy();   
      current_gen++; 
  } 
    
  PrintLastGen();
  if(NPoints>10) NPoints = 10; 
  end_time(); 
 
  DeletePopulations(); 
  delete MyMarkovNet;    
  delete IntTree;
  return fgen;
  delete[] auxconfiguration;
} 




// Factor graph based EDA (uses decimation as sampling method) [Uses MOA model]

int Factorized_EA::FGEDA(int TypeInfMethod,int ExperimentMode)  
{  
  int i;
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  unsigned long **listclusters;
  unsigned long nclust; 
  int auxnumber;
  unsigned int* auxconfiguration;
  auxconfiguration = new unsigned int[vars];


  init_time(); 
  InitPopulations(); 
  InitAlgorithm();
 
  MaxNumCliq = vars;
  LearningType=6;  // MOA Model 
  Cycles =  0; // 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);          
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information  
  listclusters = new  unsigned long* [vars];
  
 while (current_gen<Maxgen && BestEval<Max && NPoints>10) 
  {  
     if(current_gen==0)  ObjFunction(pop,0,psize);       
     else ObjFunction(pop,Elit,psize);       
 
     NPoints = Selection(); 
    
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(compact_pop,fvect,NPoints,Prior);           
     IntTree->CalMutInf();
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (current_gen+1);
     MyMarkovNet->SetNPoints(compact_pop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(compact_pop);    
     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary

     double threshold = 1.5;    
     MyMarkovNet->FindNeighbors(listclusters,CliqMaxLength,threshold); // The neighborhood is found from the matrix of mutual information  
     MyMarkovNet->MChiSquare = (double*) 0;
     MyMarkovNet->UpdateModelProteinMPM(2,nclust,listclusters); //There is one clique for each variable, the first is the variable, the rest, its neighbors

     FactorGraph fg;
     fg = CreateFactorGraph(MyMarkovNet,Cardinalities,nclust);   
     FindBestVal();   
     //auxprob = MyMarkovNet->Prob(BestInd);  
           
      PrintBestSolutions();     
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }        
      
      double chunkprob = 1/MyMarkovNet->SetOfCliques->NumberCliques;
      if(ExperimentMode==2)
	{    
         MyMarkovNet->GenPop(Elit,pop); 
         if(TypeInfMethod>0) 
	   {
              auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
              if (auxnumber==1)  for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
           }
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
             if (auxnumber==1)  for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];                  
            }          
        }

       MutatePop(pop,Elit,psize);       
       //MyMarkovNet->Destroy();
   
      current_gen++;

 
  } 

   PrintLastGen();   
   if(NPoints>10) NPoints = 10;
   end_time(); 
  
  DeletePopulations(); 
  delete IntTree;
  delete MyMarkovNet; 
  for(i=0;i<nclust;i++) delete[] listclusters[i];
  delete[] listclusters;       
  delete[] auxconfiguration;
  return fgen;
} 



// Factor graph based EDA (uses decimation as sampling method) [Uses MOA model  improved with statistical test]

int Factorized_EA::RobustMOA(int TypeInfMethod,int ExperimentMode)  
{  
  int i;
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  unsigned long **listclusters;
  unsigned long nclust; 
  int auxnumber;
  unsigned int* auxconfiguration;
  auxconfiguration = new unsigned int[vars];


  init_time(); 
  InitPopulations(); 
  InitAlgorithm();
 
  MaxNumCliq = vars;
  LearningType=3; //3; // MOA Model 
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information 
  listclusters = new  unsigned long* [vars];
 
 while (current_gen<Maxgen && BestEval<Max && NPoints>10) 
  { 
    if(current_gen==0)  ObjFunction(pop,0,psize);       
    else ObjFunction(pop,Elit,psize);       

     NPoints = Selection();     
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(compact_pop,fvect,NPoints,Prior);        
     IntTree->CalMutInf();
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (current_gen+1);
     MyMarkovNet->SetNPoints(compact_pop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(compact_pop); 
     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary

     double threshold = 1.5; 
     MyMarkovNet->FindStrongNeighbors(Cardinalities,Complexity,listclusters,CliqMaxLength,threshold); // The neighborhood is found from the matrix of mutual information  
     MyMarkovNet->MChiSquare = (double*) 0;
     MyMarkovNet->UpdateModelProteinMPM(2,nclust,listclusters); //There is one clique for each variable, the first is the variable, the rest, its neighbors

     FactorGraph fg;
     fg = CreateFactorGraph(MyMarkovNet,Cardinalities,nclust);
     FindBestVal();   
     

     PrintBestSolutions();     
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   
                           
      double chunkprob = 1/MyMarkovNet->SetOfCliques->NumberCliques;
      if(ExperimentMode==10)
	{    
         MyMarkovNet->GenPop(Elit,pop); 
         if(TypeInfMethod>0) auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
         if (auxnumber==1)  for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
        }
      else if(ExperimentMode==11)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1) MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==12)
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
      else if(ExperimentMode==13)
	{       
          MyMarkovNet->GenStructuredCrossPop(Elit,pop,selpop);   
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             if (auxnumber==1) for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];        
             //MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
	     //for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
            }          
        }

       //MyMarkovNet->Destroy();
      //  MutatePop(pop,Elit,psize);          
      current_gen++; 
  } 

  PrintLastGen();
  if(NPoints>10) NPoints = 10;

  end_time(); 
  DeletePopulations(); 
  delete IntTree;
  delete MyMarkovNet; 
  for(i=0;i<nclust;i++) delete[] listclusters[i];
  delete[] listclusters;          
  delete[] auxconfiguration;
  return fgen;
} 


int Factorized_EA::FixedStructure_RobustMNEDA(unsigned int** matrixDSYS,int TypeInfMethod,int ExperimentMode)  
{  
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  unsigned long nclust; 
  int auxnumber;
  unsigned int* auxconfiguration;
  auxconfiguration = new unsigned int[vars];

 
  init_time(); 
  InitPopulations();  
  InitAlgorithm();
   LearningType=3;   
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);   
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
 
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {  

    if(current_gen==0)  ObjFunction(pop,0,psize);       
    else ObjFunction(pop,Elit,psize);       
       
     NPoints = Selection(); 
     
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(compact_pop,fvect,NPoints,Prior);   
     //IntTree->CalMutInf();
     IntTree->CalMutInf(matrixDSYS); //THIS IS ONLY FOR STRUCTURAL INFO
     IntTree->MakeTree(IntTree->rootnode); 
    
     FindBestVal();

     MyMarkovNet->current_gen = (current_gen+1);
     MyMarkovNet->SetNPoints(compact_pop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(compact_pop);    
     MyMarkovNet->MChiSquare =  IntTree->MutualInf; // We will use matrix MChiSquare as auxiliary
    
     double threshold = 1.5;   
     MyMarkovNet->UpdateModelGTest(Cardinalities,Complexity);
     MyMarkovNet->MChiSquare = (double*) 0;   
     nclust = MyMarkovNet->SetOfCliques->NumberCliques;

     FactorGraph fg;
     fg = CreateFactorGraph(MyMarkovNet,Cardinalities,nclust);       
     FindBestVal();   
     //auxprob = MyMarkovNet->Prob(BestInd);  
           
  
     PrintBestSolutions();     
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   
              
      double chunkprob = 1/MyMarkovNet->SetOfCliques->NumberCliques;

      if(ExperimentMode==16)
	{    
           MyMarkovNet->GenPop(Elit,pop);          
           if(TypeInfMethod>0) 
	     {
              auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
	      for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];         
             }
        }
      else if(ExperimentMode==18)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==15)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob);                
  	     for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];               
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==17)
	{       
          MyMarkovNet->GenStructuredCrossPop(Elit,pop,selpop);   
          if(TypeInfMethod>0) 
	    {            
	          auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
                  for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];              
                  //MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
 	          //for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
            }          
        }
       MyMarkovNet->Destroy();   
       //     MutatePop(pop,Elit,psize);       
      current_gen++; 
  } 


  if(NPoints>10) NPoints = 10; 
  PrintLastGen();
  end_time(); 
 
  DeletePopulations(); 
  delete MyMarkovNet;    
  delete IntTree;
  return fgen;
  delete[] auxconfiguration;
} 


int Factorized_EA::EfficientStructure_RobustMNEDA(unsigned int** matrixDSYS,int TypeInfMethod,int ExperimentMode)  
{  
 
  DynFDA* MyMarkovNet;  

  unsigned long nclust; 
  int auxnumber;
  unsigned int* auxconfiguration;
  auxconfiguration = new unsigned int[vars];

 
  init_time(); 
  InitPopulations();  
  InitAlgorithm();
 
  LearningType=3;   
  Cycles = 4; //Parameter IT of MOA (number of iterations of GS is IT*n*ln(n)
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);   
  MyMarkovNet->SetFactorizationFromMatrix(matrixDSYS);    
  while (current_gen<Maxgen && BestEval<Max && NPoints>10)  
  {  
    if(current_gen==0)  ObjFunction(pop,0,psize);       
    else ObjFunction(pop,Elit,psize);       
       
     NPoints = Selection();      
     FindBestVal();

     MyMarkovNet->current_gen = (current_gen+1);
     MyMarkovNet->SetNPoints(compact_pop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(compact_pop);    
        
        
     //MyMarkovNet->SetFactorizationFromMatrix(matrixDSYS);    
     MyMarkovNet->FDACallProbFromMatrix();
     nclust = MyMarkovNet->SetOfCliques->NumberCliques;

     FactorGraph fg;
     //fg = CreateFactorGraph(MyMarkovNet,Cardinalities,nclust);       
     FindBestVal();   
    
           
     PrintBestSolutions();     
     if (BestEval>=Max )   fgen  = current_gen;	 
     else  
      {
       ApplySelection();              
       fgen = current_gen;
      }   
         
       
      double chunkprob = 1/MyMarkovNet->SetOfCliques->NumberCliques;

      if(ExperimentMode==20)
	{    
           MyMarkovNet->GenPop(Elit,pop); 
           if(TypeInfMethod>0) 
	     {
               auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
 	       for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];         
             }
        }
      else if(ExperimentMode==22)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 

            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==19)
	{          
          if(TypeInfMethod>0) 
	    {
             auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob);                
  	     for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];               
            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      else if(ExperimentMode==21)
	{       
          MyMarkovNet->GenStructuredCrossPop(Elit,pop,selpop);   
          //MyMarkovNet->GenCrossPop(Elit,pop,selpop);   
          if(TypeInfMethod>0) 
	    {            
	          auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
                  for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];              
                  //MyMarkovNet->GenBiasedPop(Elit,pop, auxconfiguration, chunkprob); 
 	          //for(int ll=0;ll<vars;ll++) pop->P[psize-1][ll] = auxconfiguration[ll];
            }          
        }
      else if(ExperimentMode==23)
	{          
          if(TypeInfMethod>0) 
	    {
	      //auxnumber  = FindMAP(fg,TypeInfMethod,auxconfiguration);
             MyMarkovNet->GenBiasedPop(Elit,pop, BestInd, chunkprob); 

            }
          else  MyMarkovNet->GenPop(Elit,pop); 
        }
      MutatePop(pop,Elit,psize);        
      //MyMarkovNet->Destroy();   
      //MutatePop(pop,Elit,psize);           
      current_gen++; 
  } 
  MyMarkovNet->Destroy();   
  PrintLastGen();    
  if(NPoints>10) NPoints = 10;   
  end_time(); 
 
  DeletePopulations(); 
  delete MyMarkovNet;     
  return fgen;
  delete[] auxconfiguration;
} 


