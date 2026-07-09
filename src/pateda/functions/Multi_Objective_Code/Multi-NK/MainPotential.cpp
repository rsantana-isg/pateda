#ifndef __MAIN_H 
#define __MAIN_H 

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
#include "RotamerClass.h" 
  
FILE *stream,*streambestsol;  
FILE *file,*outfile;  	  
  

 
double meanlikehood[500]; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798};  
double SelInt; 
double ModVal = 1;

//double AllGen[100];    
//double statistics[100][100];  
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
char filedetails[60]; 
char MatrixFileName[60]; 
char OutFileName[60];
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
int  nsucc;

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
 
FoldPotential* StatPotential;
int TotEvaluations;
int sizeProtein;
int EvaluationMode;
int currentexp;
int currentinstance;
int* indexcomponents;
double temp = 1.0;
int MaxIter = 200; //Iterations for loopy BP


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


void ProteinRandInit(Popul* epop)  
{
  //  int k;
  epop->RandInit();
  
}

void ReadParameters()  
 
{ 
  int T,MaxMixtP,S_alph,Compl; 
 
stream = fopen( "RParam.txt", "r+" );  
        		    
	if( stream == NULL )  
		printf( "The file RParam.txt was not opened\n" );  
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
  //  int i; 
 if (Tour==0) 
   { 
     TruncMax = int(psize*Trunc);  
   
     if (BestElitism)  Elit = TruncMax;   //Only for Trunc Selection  
     selpop = new Popul(TruncMax,vars,Elit,Cardinalities);  
   }  
  else selpop = new Popul(psize,vars,Elit,Cardinalities);  
 
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop = new Popul(Elit,vars,Elit,Cardinalities); 
  pop = new Popul(psize,vars,Elit,Cardinalities);  
  compact_pop = new Popul(psize,vars,Elit,Cardinalities);  
  fvect = new double[psize];
  ProteinRandInit(pop);  
 } 
 
inline void DeletePopulations() 
{ 
  delete compact_pop; 
  delete pop;  
  delete selpop;  
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) delete elitpop; 
 delete[] fvect; 
} 
 


void EvalPotential(Popul* epop,int nelit, int epsize, int atgen)
{
 double CurrentEval=0;
 int start,pos,i;
 

if (atgen==0) start=0;
else start=nelit;
 
for(pos=start; pos < epsize;  pos ++)  
 {
   //cout<<pos<<endl;
   //for(i=0;i<vars;i++) CurrentEval= CurrentEval - 1.0*epop->P[pos][i];
    CurrentEval = -1*StatPotential->CalculateEnergy(epop->P[pos]);
    epop->SetVal(pos,CurrentEval); 
    TotEvaluations ++;    
  }

}




int Markovinit(double Complexity, int typemodel, int sizecliq)  //In this case, complexity is the threshold for chi-square 
{  
  int i,fgen;  
  double auxprob;     
  DynFDA* MyMarkovNet;  
   
  init_time();
  InitPopulations(); 
 
  LearningType=3;
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
 

  i=0;  fgen = -1;  auxprob =0; BestEval  = Max -1; NPoints=TruncMax; 

  Popul* aux_pop;
  aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
  
  NPoints = psize;
 
  while (i<Maxgen && BestEval<Max && NPoints>100)  
  {  
    
     EvalPotential(pop,Elit,psize,i); 
    //pop->Print(); 
     
       NPoints = Selection(); 
       MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
       MyMarkovNet->SetPop(selpop); 
       MyMarkovNet->UpdateModelProtein(typemodel,sizecliq);         

       /*            
  if(printvals>1) 
   {           
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
      for(int l=vars;l<2000;l++) cout<<"0 ";  
      //cout<<" "<<selpop->Evaluations[ll]<<endl; 
       cout<<endl;
    }  
   }
       */    
   //cout<<"Initial marginals "<<endl; 
  
     FindBestVal(); 
     //AllGen[i] += BestEval; 

     auxprob = MyMarkovNet->Prob(BestInd);  
         
 
     //        if(printvals>0)     cout<<"Gen :"<<i<<" Best:"<<BestEval<<" Elit"<<Elit<<" TotEval:"<<TotEvaluations<<" DifPoints:"<<NPoints<<endl; 
 
      if (BestEval>=Max)   fgen  = i;	 
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {
		selpop->SetElit(Elit,pop);
		//Elit = NPoints; 
                   //compact_pop->SetElit(Elit,pop);  
                 for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }

     
      //         if(NPoints<TruncMax*0.9)    MyMarkovNet->GenPopProtein(Elit,pop,FoldingProtein,1.0+TruncMax);       else   

     

             MyMarkovNet->GenPop(Elit,pop);      
             if(params[1]) pop->Mutation(Elit,0.015);
             i++;         
            MyMarkovNet->Destroy();    
  
   }  

  //if(NPoints>10) NPoints = 10;


 end_time();  

 // if(printvals>0) 
 //cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  
  

 if(printvals>0) 
   {           
    for(int ll=0;ll<printvals;ll++)// NPoints 
    { 
      cout<<currentinstance<<" "<<currentexp<<" "<<i<<" "; 
     for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 
/*
  if(printvals>0) 
   {           
    for(int ll=0;ll<printvals;ll++)// NPoints 
    { 
     if(params[0]==0)  for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     else if(params[0]==1) 
       {
         RotamerProtein->FillFixed(selpop->P[ll]);
         for(int l=0;l<RotamerProtein->numberresidues;l++) cout<<RotamerProtein->Fixed[l]<<" ";    
       }
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 */
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
  NPoints = 10000;
  
   while (i<Maxgen && BestEval<Max && NPoints>100)  
  {  
   
     EvalPotential(pop,Elit,psize,i); 
     //pop->Print(); 
     NPoints = Selection();  
     IntTree->SetNPoints(NPoints);
     IntTree->rootnode =   IntTree->RandomRootNode(); // IntTree->FindRootNode();  
     IntTree->CalProbFvect(selpop,fvect,NPoints,0);        
     IntTree->CalMutInf(); 
     //IntTree->CalMutInf(StatPotential->Matrix); //THIS IS ONLY FOR STRUCTURAL INFO
     IntTree->MakeTree(IntTree->rootnode); 
     FindBestVal(); 
      //AllGen[i] += BestEval;
 
     //IntTree->PrintModel();
     /*
      for(int jj=0;jj<vars;jj++) 
        if(IntTree->Tree[jj] != -1) 
          {
            statistics[IntTree->Tree[jj]][jj] ++;
            statistics[jj][IntTree->Tree[jj]] ++;
          }
     */
      // IntTree->PrintMut();
     
     IntTree->PutPriors(Prior,selpop->psize,1);
     sumprob = IntTree->SumProb(selpop,NPoints);  
     //auxprob = IntTree->Prob(BestInd); 
     //selpop->Print(0); 
     //cout<<"Now is serious "<<endl;       
      
  
     /*     
if(printvals>1) 
   {           
 
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }
 
   }
   if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<sumprob<<" "<<Elit<<endl;    
     */ 
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

     IntTree->GenPop(Elit,pop);   
     //IntTree->PopMutation(pop,Elit,1,0.015);
     if(params[1]) pop->Mutation(Elit,0.015);
     i++;
  }  
  //cout<<BestEval<<" ";   selpop->Print(0); 

    end_time(); 
    //  if(printvals>0)  
    //cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  //cout<<BestEval<<endl; 
 
// printvals = 1;
 if(printvals>0) 
   {           
    for(int ll=0;ll<printvals;ll++)// NPoints 
    { 
      cout<<currentinstance<<" "<<currentexp<<" "<<i<<" "; 
     for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 // printvals = 0;
  DeletePopulations(); 
  delete IntTree;
  return fgen;  

}  


int LoopyIntusualinit(double Complexity)  
{  
  int i,fgen;  
  double auxprob,sumprob;     
  LoopyIntTreeModel *IntTree;  
 
  init_time(); 
  InitPopulations(); 

  IntTree = new LoopyIntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  
  //IntTree->InitPotential(IntTree->Matrix); //Here only whe  matrix is NOT learned from tree

  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
  NPoints = 10000;
   
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
 
     EvalPotential(pop,Elit,psize,i); 

     //pop->Print(); 
     NPoints = Selection();
     IntTree->SetNPoints(NPoints);
     IntTree->CalProbFvect(selpop,fvect,NPoints,0);    
     IntTree->CalMutInf(StatPotential->Matrix); //THIS IS ONLY FOR STRUCTURAL INFO
     IntTree->rootnode =   IntTree->RandomRootNode(); // IntTree->FindRootNode();  

     IntTree->CreateFields(); //Here only whe  matrix is learned from tree
     IntTree->MakeTree(IntTree->rootnode); 
     IntTree->MatrixFromTree(); //Here only whe  matrix is learned from tree
     IntTree->InitLs();                  //Here only whe  matrix is learned from tree
     IntTree->InitPsi();                 //Here only whe  matrix is learned from tree  
     //IntTree->FillPsi();
     
  
     //IntTree->PrintModel();
    
     FindBestVal(); 
         


     //IntTree->PutPriors(Prior,selpop->psize,1);
   
    
     
if(printvals>1) 
   {           
 
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }
 
   }
   if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<sumprob<<" "<<Elit<<endl;    
      
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
       
       int genmaxconf = IntTree->GenMaxConfigurations(temp,MaxIter,10,Elit,pop);
       IntTree->GenPop(Elit+genmaxconf,pop);   
       IntTree->DeletePotential(); //Here only whe  matrix is learned from tree
      
       //pop->Print(); 
     
       if(params[1]) pop->Mutation(Elit,0.015);
     i++;
  }  

   end_time(); 
  if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  

 printvals = 1;
 if(printvals>0) 
   {           
    for(int ll=0;ll<printvals;ll++)// NPoints 
    { 
      cout<<currentinstance<<" "<<currentexp<<" "<<i<<" "; 
     for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 printvals = 0;
 
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
      EvalPotential(pop,Elit,psize,i); 

   //pop->Print(); 

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
if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" "<<Elit<<endl;    
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
          
           MixtureInt->SamplingFromMixture(pop);           
      	  }   

       MixtureInt->RemoveTrees(); 
       MixtureInt->RemoveProbabilities();
       // cout<<"Pass 7 "<<endl;    
      i++;  
  }  
 //EvalProtein(selpop,0,Elit,i);   
 

 if(NPoints>10) NPoints = 10;


 end_time();  

 if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;    cout<<BestEval<<endl;
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
                   sigma = sigma/(cantexp-1);
                   
                  if (succexp>0)  
                   {  
                    auxmeangen = meangen/succexp;
                    bestalltime = bestalltime/(1.0*succexp); 
                    if (BestElitism)  
                         meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;     
                    else meanfit = (auxmeangen+1)*(psize-1) + 1; 
                    cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" T="<<Trunc<<" N="<<psize<<" Sel="<<Tour<<"  k="<<Cycles<<"  MaxGen="<<Maxgen<<"  ComplexEM="<<MaxMixtProb<<"  Elit="<<Elit<<" Suc.="<<succexp<<"  g="<<(auxmeangen+1)<<"  ave="<<meanfit<<" meaneval "<<meaneval<<" sigma "<<sigma<<" timebest "<<bestalltime<<" fulltime "<<alltime<<endl;                   
                   } 
                  else  
                   {  
		     cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" T="<<Trunc<<" N="<<psize<<" Sel="<<Tour<<"  k="<<Cycles<<"  MaxGen="<<Maxgen<<"ComplexEM="<<MaxMixtProb<<"  Elit="<<Elit<<" Suc.="<<0<<"  g="<<0<<"  ave="<<0<<" meaneval "<<meaneval<<" sigma "<<sigma<<" fulltime "<<alltime<<" Eval "<<(TotEvaluations/(1.0*cantexp))<<endl; 
                   } 

		  //for(int ll=0;ll<Maxgen;ll++)  cout<<AllGen[ll]/(-1.0*cantexp)<<" ";
                  //cout<<endl;
} 


double  Partialrun(int algtype,int nrun)  
{  
  int succ=-1; 

        
  switch(algtype)  
                     {                     
                       case 0: succ = Markovinit(Complex,1,Cycles);break;  // Markov Network       1
                       case 1: succ = Intusualinit(Complex);break;
                       case 2: succ = MixturesIntAlgorithm(1,Cardinalities,Complex);break;// MT on dependencies 
                      } 

   return BestEval;   
} 


void runOptimizer(int algtype,int nrun)  
{  
    int succ=-1; 

        
  switch(algtype)  
                     {                     
                       case 0: succ = Markovinit(Complex,1,Cycles);break;  // Markov Network       1
                       case 1: succ = Intusualinit(Complex);break;
                       case 2: succ = MixturesIntAlgorithm(1,Cardinalities,Complex);break;// MT on dependencies 
                      } 

  

   if (succ>-1)  
   { 
       succexp++; 
       meangen += succ;    
       bestalltime +=auxtime;      
       //cout<<"Contact order "<<FoldingProtein->ContactOrderVector(sizeProtein,BestInd)<<endl;       
   } 
   else nsucc++;
   alltime += auxtime;  
   meaneval += BestEval; 
   meanlikehood[nrun] = BestEval;  
} 




int  readdata1()
{  
  int ctime;
  double bestval,ff,sum_ff,max_ff,auxdouble;
  int S,i,j,k,l,sum_gg,auxint,ii,gg,rr;
  int nexper;
  double allmax[62];

  FILE *helpstream; 
  stream = fopen( "../Matlab/pdbfiles/newlistprotein.txt", "r+" );  
  //helpstream = fopen( "ResultsLoopyTree10.txt", "r+" );  
  helpstream = fopen( "Expe1000EstructTree.txt", "r+" );  
  //helpstream = fopen( "Expe1000FullTree.txt", "r+" );  
  // helpstream = fopen( "ResultsLoopy10.txt", "r+" );  
  //helpstream = fopen( "AuxExpeTree.txt", "r+" );  
   //cout<<" It arrived in here "<<endl;
 
for(k=0;k<63;k++) 
{ 
    fscanf(stream, "%s %d", &filedetails,&vars);
    sum_gg = 0;
    sum_ff = 0;
    max_ff = 0;
    for(i=0;i<100;i++) 
     { 
       // for(l=0;l<100;l++)  
      {
       fscanf(helpstream, "%d %d %d", &ii,&rr,&gg );
       //cout<<ii<<" "<<rr<<" "<<gg<<endl;
       for(j=0;j<vars;j++)  fscanf(helpstream, "%d", &auxint);
       fscanf(helpstream, "%lf", &ff );
       //  if (l==0) 
         {
          if (ff>max_ff) max_ff = ff;
         }      
	 // cout<<ff<<"  "<<max_ff<<endl;
      }
      
     }
    allmax[k] = max_ff; 
    fscanf(helpstream, "%s %lf", &MatrixFileName,&auxdouble);
    //cout<<MatrixFileName<<endl;
}
fclose(helpstream);
fclose(stream);

  stream = fopen( "../Matlab/pdbfiles/newlistprotein.txt", "r+" );  
  //  helpstream = fopen( "ResultsLoopyTree10.txt", "r+" );  
   helpstream = fopen( "Expe1000EstructTree.txt", "r+" );  
  // helpstream = fopen( "Expe1000FullTree.txt", "r+" );  
  // helpstream = fopen( "ResultsLoopy10.txt", "r+" );  
  // helpstream = fopen( "AuxExpeTree.txt", "r+" );  
for(k=0;k<63;k++) 
{ 
    fscanf(stream, "%s %d", &filedetails,&vars);
    sum_gg = 0;
    sum_ff = 0;
    S = 0;
 
  
    for(i=0;i<100;i++) 
     { 
       //for(l=0;l<100;l++)  
     {
       fscanf(helpstream, "%d %d %d", &ii,&rr,&gg );
       // cout<<ii<<" "<<rr<<" "<<gg<<endl;
    
       for(j=0;j<vars;j++)  fscanf(helpstream, "%d", &auxint);
       fscanf(helpstream, "%lf", &ff );
       //if(l==0)
	 {
	   //   cout<<ff<<"  "<<allmax[k]<<endl;
         if (ff==allmax[k]) 
          {
           S++;
           sum_gg += gg; 
         }
         sum_ff += ff;
	 }   
       }
      
     }    
       fscanf(helpstream, "%s %lf", &MatrixFileName,&auxdouble);
 //cout<<" TTT{"<< filedetails<<"} & "<<allmax[k]<<"& "<<S<<"  & "<<(sum_ff/100.0)<<" & "<<((1.0*sum_gg)/S)<<" RR"<<endl;    
   printf("TTT{ %s } & %4.2lf & %d & %4.2lf & %4.2lf & RR \n",filedetails,allmax[k],S,(sum_ff/100.0),((1.0*sum_gg)/S));         
}
fclose(helpstream);
fclose(stream);







}  




int  readdata()
{  
  int ctime;
  float bestval;
  int alele,i,j,k,ident;
  int nexper;
  // int proteinsizes[325] = {95, 158, 144, 161, 105, 124, 53, 56, 85, 76, 90, 177, 54, 26, 62, 92, 32, 14, 144, 122, 181, 137, 152, 122, 175, 97, 60, 132, 149, 50, 133, 213, 121, 48, 95, 108, 130, 112, 39, 111, 97, 85, 94, 64, 92, 173, 108, 126, 134, 88, 74, 184, 128, 200, 113, 178, 98, 99, 47, 108, 123, 66, 162, 75, 96, 139, 244, 86, 179, 96, 84, 251, 91, 198, 47, 61, 144, 104, 93, 69, 78, 143, 216, 228, 123, 51, 103, 131, 154, 130, 83, 100, 109, 118, 230, 169, 241, 127, 205, 91, 117, 186, 139, 154, 91, 233, 121, 139, 103, 193, 65, 192, 140, 184, 9, 10, 9, 157, 92, 125, 116, 112, 147, 58, 103, 55, 207, 107, 122, 66, 75, 157, 199, 172, 51, 55, 83, 165, 44, 75, 66, 86, 76, 125, 267, 64, 68, 108, 146, 83, 109, 211, 126, 235, 56, 54, 165, 116, 60, 57, 261, 152, 84, 63, 140, 67, 122, 85, 165, 113, 68, 107, 67, 39, 133, 122, 195, 115, 101, 96, 88, 103, 188, 57, 148, 156, 190, 153, 144, 152, 64, 50, 115, 259, 150, 192, 79, 54, 46, 208, 106, 262, 88, 102, 114, 204, 50, 169, 203, 80, 11, 94, 64, 87, 170, 17, 13, 29, 82, 92, 91, 203, 218, 106, 173, 106, 132, 229, 175, 134, 144, 83, 97, 54, 42, 142, 123, 168, 102, 174, 21, 88, 137, 222, 93, 115, 119, 65, 99, 92, 140, 80, 92, 121, 208, 167, 97, 66, 76, 189, 97, 111, 65, 63, 133, 225, 119, 101, 91, 151, 185, 101, 80, 85, 7, 89, 56, 161, 77, 85, 96, 233, 210, 142, 112, 34, 158, 143, 42, 125, 97, 190, 81, 140, 50, 142, 122, 115, 226, 80, 105, 48, 94, 80, 92, 88, 53, 100, 87, 89, 84, 91, 57, 88, 64, 92, 101, 220, 224, 176, 92, 62, 94, 119, 96};

  //int proteinsizes[45] ={704,428,574,315,380,442, 434,618, 374,441,331,340,345,404,338,512,490,397,495,443,360,445,688,337,463,373,332,544,381,424,381,375,358,509,457,510,344,392,332,441,334,641,496,311,594};

   int proteinsizes[93] = {972,1972,582,397,510,636,441,578,668,1328,413,960, 699,1298,1219,807,750,609,604,964, 880,446, 531,614,630,658, 636,635,1409,650, 750,866, 628,520, 614,804, 520,640, 565,1982, 685,532, 124,544, 744,1671,  314,584, 415,923,919,530, 922,545, 1040,855, 806,454,457,772,739,481,323,522, 536,505, 1060,593, 576,724, 960,1364, 505,629, 283,746, 562,1124, 1108,632, 856,506, 656,465, 610,921,699,926,519,277, 845,576,446};

  
     

  //int proteinsizes[45] = {367, 147, 160, 75, 103, 146, 185, 246, 90, 127, 69, 110, 172, 99, 85, 166, 136, 136, 137, 124, 125, 208, 318, 145, 174, 144, 81, 241, 144, 207, 176, 113, 66, 155, 189, 175, 113, 146, 157, 141, 192, 234, 191, 62, 292};

  FILE *helpstream; 
 //helpstream = fopen( "rotexperiments200.txt", "r+" );  
 //helpstream = fopen( "roteresultsmissing324.txt", "r+" );  
 //helpstream = fopen( "output.26", "r+" );  
 //helpstream = fopen( "roteNewDEElarge1.txt", "r+" );  
 // helpstream = fopen( "output.31", "r+" );  

 helpstream = fopen( "output.41", "r+" ); 
nexper = 10;
 
for(k=0;k<93;k++) 
{ 

  /*  //This is for small proteins
  while(k==17 || k==58 || k==114 || k==115 || k==116 || k==198 || k==216 || k==217 || k==274  || k==280)  
   {
      fscanf(helpstream, "%d", &ident);
      fscanf(helpstream, "%s", &MatrixFileName);
      k++;
   }
  */
 fscanf(helpstream, "%d", &ident);
 fscanf(helpstream, "%s", &MatrixFileName);
  for(j=0;j<nexper;j++) 
   {
    fscanf(helpstream, "%s", &MatrixFileName);
    fscanf(helpstream, "%s", &MatrixFileName);
    fscanf(helpstream, "%s", &MatrixFileName);
    fscanf(helpstream, "%s", &MatrixFileName);
    fscanf(helpstream, "%s", &MatrixFileName);
    fscanf(helpstream, "%s", &MatrixFileName);
    fscanf(helpstream, "%s", &MatrixFileName);
    fscanf(helpstream, "%d", &ctime);
    cout<<ident<<" "<<proteinsizes[ident]<<" "<<j<<" ";
    for(i=0;i<2000;i++) 
     {
       if(i<proteinsizes[ident]) 
        {
         fscanf(helpstream, "%d", &alele);
         cout<<alele<<" ";
        }
       else cout<<0<<" ";
      }
    fscanf(helpstream, "%f", &bestval);
    ctime = abs(ctime);
    cout<<-1.0*bestval<<" "<<ctime<<endl;     
   }
   for(i=0;i<19;i++)  fscanf(helpstream, "%s", &MatrixFileName);
}
fclose(helpstream);
}  






//int  main(){  
int main(int argc, char *argv[ ])
{
 int i,j,k;  
 unsigned ta;
  ta = (unsigned) time(NULL);  
  //ta = 1135934259;

 unsigned int bestsol[95] = {25, 1, 2, 0, 1, 7, 0, 7, 3, 0, 3, 30, 13, 7, 4, 70, 1, 7, 4, 33, 70, 0, 8, 1, 2, 1, 22, 13, 26, 2, 0, 0, 7, 70, 40, 1, 30, 18, 7, 18, 2, 0, 7, 7, 70, 1, 2, 8, 7, 7, 0, 0, 30, 7, 1, 7, 12, 7, 18, 3, 2, 1, 22, 25, 10, 4, 14, 1, 14, 14, 2, 67, 7, 8, 18, 7, 30, 8, 4, 1, 12, 1, 67, 1, 18, 22, 7, 3, 8, 40, 3, 7, 0, 4, 4}; 
 unsigned int loadedsol[2000];
 unsigned int initassign[2000];
 double protxyz[9],dist;
 double** xyzcoord;

unsigned int auxsol[21]  = {8, 13, 8, 8, 8, 13, 8, 8, 8, 13, 8, 8, 8, 8, 8, 13, 8, 8, 17, 8, 8};

 //    ta = 1051750320;
 srand(ta); 
 //cout<<"seed"<<ta<<endl; 

 //readdata1();
 // return 1;
 
params = new int[3]; 
 ReadParameters(); 


 // params[0] = 1;
 // params[1] = 2;

 // stream = fopen( "../Matlab/pdbfiles/protein_list.txt", "r+" );  
stream = fopen( "../Matlab/pdbfiles/newlistprotein.txt", "r+" );  


if(params[0]==0)
 {
  for(j=0;j<3900;j++)
   { 
   //j = notopt[jj]-1;
   //fseek(stream,9*(j),0);
    MatrixFileName[0]=0;
    strcat(MatrixFileName,"../Matlab/pdbfiles/");  
    fscanf(stream, "%s", &filedetails);
    OutFileName[0] = 0;
    strcat(OutFileName,MatrixFileName);
    strcat(OutFileName,filedetails);
    strcat(OutFileName,".dist");  
    strcat(filedetails,".xyz");    
    strcat(MatrixFileName,filedetails);
    filedetails[0]=0;
    //cout<<j<<" "<<MatrixFileName<<endl;
    //cout<<j<<" "<<OutFileName<<endl;
   
    file = fopen(MatrixFileName, "r+" );
    fscanf(file, "%s %d", &filedetails,&vars);
    cout<<filedetails<<" "<<vars<<endl;
    xyzcoord = new double*[3];
    xyzcoord[0] = new double[vars];
    xyzcoord[1] = new double[vars];
    xyzcoord[2] = new double[vars];
    for(i=0;i<vars;i++) 
        {  
         fscanf(file, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &protxyz[0],&protxyz[1], &protxyz[2],&protxyz[3], &protxyz[4],&protxyz[5], &protxyz[6],&protxyz[7], &protxyz[8]);
	 //cout<<protxyz[0]<<" "<<protxyz[1]<<" "<<protxyz[2]<<" "<<protxyz[3]<<" "<<protxyz[4]<<" "<<protxyz[5]<<" "<<protxyz[6]<<" "<<protxyz[7]<<" "<<protxyz[8]<<endl;
	 xyzcoord[0][i] = protxyz[0]; 
         xyzcoord[1][i] = protxyz[1];
         xyzcoord[2][i] = protxyz[2];
        }  
      fclose(file);
    
     file = fopen(OutFileName, "w+" );
     cout<<"file "<<file<<endl;
      for(i=0;i<vars-1;i++) 
        for(k=i+1;k<vars;k++) 
	  {
            dist = 0.0;
	    dist += (xyzcoord[0][i] - xyzcoord[0][k]) *  (xyzcoord[0][i] - xyzcoord[0][k]);
            dist += (xyzcoord[1][i] - xyzcoord[1][k]) *  (xyzcoord[1][i] - xyzcoord[1][k]);
            dist += (xyzcoord[2][i] - xyzcoord[2][k]) *  (xyzcoord[2][i] - xyzcoord[2][k]);
            dist = sqrt(dist);
            fprintf(file, "%lf ",dist);
            //cout<<dist;
          }      
      //cout<<endl;
      fclose(file);
      delete[] xyzcoord[0];
      delete[] xyzcoord[1];
      delete[] xyzcoord[2];  
      delete[] xyzcoord; 
  }
 }
 else if(params[0]==1)
  {
   for(j=0;j<3900;j++) //3900 //62
   { 
     int auxint;
    MatrixFileName[0]=0;
    strcat(MatrixFileName,"../Matlab/pdbfiles/");  
    fscanf(stream, "%s", &filedetails);
    //fscanf(stream, "%s %d", &filedetails,&auxint); //ONLY for newlistprotein.txt
    OutFileName[0] = 0;
    strcat(OutFileName,MatrixFileName);
    strcat(OutFileName,filedetails);
    strcat(OutFileName,".dist");  
    strcat(filedetails,".seq");    
    strcat(MatrixFileName,filedetails);
    filedetails[0]=0;
    //cout<<j<<" "<<MatrixFileName<<endl;
    //cout<<j<<" "<<OutFileName<<endl;

 
    file = fopen(MatrixFileName, "r+" );
    StatPotential = new  FoldPotential (file,temp); 
    fclose(file);
    vars = StatPotential->numberresidues;
    
    file = fopen(OutFileName, "r+" );
    StatPotential->FillPotential(file);
    fclose(file);
    
    double bestEnergy = StatPotential->CalculateEnergy(StatPotential->sequence);

   
    //cout<<" BestEnergy:   "<<bestEnergy<<endl;  
    for(i=0;i<vars;i++) cout<<StatPotential->sequence[i]<<" ";
    //cout<<" Other energy is    "<<StatPotential->CalculateEnergy(auxsol)<<endl;  

      Cardinalities  = new unsigned[vars];  
      for(i=0;i<vars;i++) Cardinalities[i] = 20; //Number of allowed aminoacids  
      if (params[1]==0) StatPotential->BestLoopySolution(Cardinalities,temp,2000);
      else if (params[1]==1) StatPotential->BestGBPSolution(Cardinalities,temp,1000);
       else if (params[1]==2)
	{
         int numberconf;
         int finalnumberconf;
         numberconf = 1000;
	 unsigned** bestconf;
         double* energies; 
       	 bestconf = new unsigned*[numberconf];
         energies = new double[numberconf];  
         for(i=0;i<numberconf;i++)  bestconf[i] = new unsigned[vars];
       
         //cout<<"Beginning Loopy "<<endl;   
	 //StatPotential->PrintAdjacencyMatrix();
         
	 //StatPotential->BestGBPSolution(Cardinalities,temp,200);
         
         StatPotential->BestLoopySolutionMaxConfigurations(Cardinalities,temp,MaxIter, bestconf, energies, numberconf, &finalnumberconf);
          
         for(i=0;i<finalnumberconf;i++) 
          {
            for(k=0;k<vars;k++) cout<<bestconf[i][k]<<" ";
            cout<<energies[i]<<endl;
          }

         cout<<"Tot Conv : "<<finalnumberconf<<endl<<endl;   
         
   	 delete[] energies;
	 for(i=0;i<finalnumberconf;i++) delete[] bestconf[i];
         delete[] bestconf;
	      
        } 
    
    delete[] Cardinalities;
    delete StatPotential; 
   }
  }
  else if(params[0]==2)
    {
      int ii,jj;
      /*
    for(ii=0;ii<100;ii++)
      for(jj=0;jj<100;jj++) 
         statistics[ii][jj] = 0;
      */                     
  
    j = atoi(argv[1]);
    //cout<<argc<<" "<<argv[1]<<endl;
    //   for(j=0;j<61;j++) //3900 //62
    { 
    currentinstance = j;
    int auxint;
    MatrixFileName[0]=0;
    strcat(MatrixFileName,"../Matlab/pdbfiles/");  
    //fscanf(stream, "%s", &filedetails);
     for(i=0;i<=j;i++) 
      {
        fscanf(stream, "%s %d", &filedetails,&auxint); //ONLY for newlistprotein.txt
      }
    cout<<filedetails<<"  "<<auxint<<endl;
    OutFileName[0] = 0;
    strcat(OutFileName,MatrixFileName);
    strcat(OutFileName,filedetails);
    strcat(OutFileName,".dist");  
    strcat(filedetails,".seq");    
    strcat(MatrixFileName,filedetails);
    filedetails[0]=0;
    //cout<<j<<" "<<MatrixFileName<<endl;
    //cout<<j<<" "<<OutFileName<<endl;

 
    file = fopen(MatrixFileName, "r+" );
    StatPotential = new  FoldPotential (file,temp); 
    fclose(file);
    vars = StatPotential->numberresidues;
    
    file = fopen(OutFileName, "r+" );
    StatPotential->FillPotential(file);
    fclose(file);
    
    double bestEnergy = StatPotential->CalculateEnergy(StatPotential->sequence);
    //cout<<"BestEnergy:   "<<-1.0*bestEnergy<<endl;  

    Cardinalities  = new unsigned[vars];  
   for(i=0;i<vars;i++) Cardinalities[i] = 20; //Number of allowed aminoacids  
      if (params[1]==0) StatPotential->BestLoopySolution(Cardinalities,temp,2000);
      else if (params[1]==1) StatPotential->BestGBPSolution(Cardinalities,temp,1000);
       else if (params[1]==2)
	{
         int numberconf;
         int finalnumberconf;
         numberconf = 1000;
	 unsigned** bestconf;
         double* energies; 
       	 //bestconf = new unsigned*[numberconf];
         //energies = new double[numberconf];  
         //for(i=0;i<numberconf;i++)  bestconf[i] = new unsigned[vars];
         for(jj=0;jj<cantexp;jj++) 
          {
	    currentexp = jj;
            //Intusualinit(Complex);    //Tree EDA
            Markovinit(Complex,1,Cycles);  // Markov Network     
            //LoopyIntusualinit(Complex);         
          }


	 /*
         for(i=0;i<finalnumberconf;i++) 
          {
            for(k=0;k<vars;k++) cout<<bestconf[i][k]<<" ";
            cout<<energies[i]<<endl;
          }
	 */
         //cout<<"Tot Conv : "<<finalnumberconf<<endl<<endl;   
         
   	 //delete[] energies;
	 //for(i=0;i<finalnumberconf;i++) delete[] bestconf[i];
	}
      delete[] Cardinalities;
      delete StatPotential; 
 
    }
    /*
 for(ii=0;ii<100;ii++)
   {
      for(jj=0;jj<100;jj++)
         {
	   cout<<statistics[ii][jj]<<" ";
         }
      cout<<endl;
   }
    */   
  }
   
 fclose(stream);
 delete[]  params;
 return 1;
}       
    


  
#endif  
