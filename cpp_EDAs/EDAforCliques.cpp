#include <math.h>  
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <iostream.h> 
#include <fstream.h> 
#include "auxfunc.h"  
#include "Popul.h"  
#include "EDA.h" 
#include "AbstractTree.h"  
#include "FDA.h"  
#include "MixtureTrees.h" 
#include "CNF.h" 
#define itoa(a,b,c) sprintf(b, "%d", a) 
CNF *AllClauses;

  
FILE *stream;  
FILE *file,*outfile;  	  
  
int f; 
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
char MatrixFileName[50]; 
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
int maxclique; 
int extraedges;
int minunit; 
int maxunit; 
int reward=500; 
int thresh; 
double meaneval;  
double BestEval; 
int TruncMax; 
int NPoints;  
unsigned int *BestInd; 
Popul *pop,*selpop,*elitpop,*compact_pop; 
double *fvect; 
int nsucc; 
int number_nodes;

int LEARNEBNA=1;  
int EBNASCORE= BIC_SCORE; //K2_SCORE;
double  EBNA_ALPHA =0.05;
int  EBNA_SIMUL = PLS;




void EvalCliques(Popul* epop)
{
int k;
 double auxval;
for(k=0; k < psize; k ++) 
{
    //epop->SetVal(k,AllClauses->SatClauses(epop->P[k]));
    epop->SetVal(k,AllClauses->SatClausesChange(epop->P[k]));
  if (AllClauses->Satisfied>BestEval)
  {
    BestEval=AllClauses->Satisfied;
    //cout<<BestEval<<endl;
  }
}

//for(k=0; k < psize; k ++)  epop->SetVal(k,AllClauses->SatClausesChange(epop->P[k]));
}
 


void ReadParameters()  
 
{ 
  int T,MaxMixtP,S_alph,Compl; 
 
  stream = fopen("SATparams.txt", "r+" ); 
    if( stream == (FILE *)0 ) 
	{ 
		printf( "The file SATParams.txt was not opened\n" ); 
		return; 
	} 
	else 	    	
	{  
         fscanf( stream, "%s", &MatrixFileName);  
         fscanf( stream, "%d", &cantexp); // Number of Experiments  
	 fscanf( stream, "%d", &number_nodes); // Cant of Vars in the vector  
 	 fscanf( stream, "%d", &auxMax); // Max. of the fitness function  
  	 fscanf( stream, "%d", &T); // Percent of the Truncation selection or tournament size 
	 fscanf( stream, "%d", &psize); // Population Size  
	 fscanf( stream, "%d", &Tour);  // Type of selection 0=Trunc, 1=Tour, 2=Prop, 3=Bolt  
	 fscanf( stream, "%d", &func); // Number of the function, Ochoa's  
	 fscanf( stream, "%d", &ExperimentMode); // Type of Experiment (SEE BELOW case instruction)  
 	 fscanf( stream, "%d", &Ntrees); // Number of Trees  
	 fscanf( stream, "%d", &Elit); // Elistism (Now best elitism is fixed)  
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
 vars = number_nodes; 
if(T>0) 
 {  
   div_t res; 
   res = div(T,5);  
   SelInt = Sel_Int[res.quot]; // Approximation to the selection intensity for truncation. 
 } 
 
  
Trunc = T/double(100);  
Complex  = Compl/double(100);  
Max = auxMax/double(100);   
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
 if (Tour==0) 
   { 
     TruncMax = int(psize*Trunc);  
     if (BestElitism)  Elit = TruncMax;   //Only for Trunc Selection  
     selpop = new Popul(TruncMax,vars,Elit);  
   }  
  else selpop = new Popul(psize,vars,Elit);  
 
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop = new Popul(Elit,vars,Elit); 
 
  pop = new Popul(psize,vars,Elit);  
  compact_pop = new Popul(psize,vars,Elit); 
  fvect = new double[psize]; 
  pop->RandInit();  
 
} 
 
inline void DeletePopulations() 
{ 
  delete compact_pop; 
  delete pop;  
  delete selpop;   
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) delete elitpop;  
  delete[] fvect;
}   
 
int Intusualinit(double Complexity)  
{  
  int i,fgen;  
  double auxprob,sumprob;     
  IntTreeModel *IntTree;  
 
  InitPopulations(); 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
   
   
  while (i<Maxgen && auxprob<1 && BestEval<Max)  
  {  
     EvalCliques(pop);  
     NPoints = Selection(); 
/*
   for(int ll=0;ll<10;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<compact_pop->P[ll][l];  
     cout<<" "<<fvect[ll]<<endl; 
     //pop->Print(); 
    }
*/
     IntTree->rootnode = IntTree->RandomRootNode();  
     IntTree->CalProbFvect(compact_pop,fvect,NPoints);           
     IntTree->CalMutInf();  
     IntTree->MakeTree(IntTree->rootnode); 
     //MyTree->CalculateILikehood(compact_pop,fvect);  
     //MyTree->MakeTreeLog(); 
     FindBestVal(); 
 
     sumprob = IntTree->SumProb(compact_pop,NPoints); 
       
     //cout<<"Now is serious "<<endl;       
     auxprob = IntTree->Prob(BestInd,0);  
     
//     if(printvals)     cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<IntTree->TreeProb<<endl;  
       
      if (BestEval==Max) fgen  = i;	   
      else 
          { 
	      if (Tour>0 || (Tour==0 && Elit>TruncMax)) 
                { 
                  elitpop->SetElit(Elit,pop);  
                  // MyTree->CollectKConf(Elit,vars,MyTree,pop);           
                } 
              else  selpop->SetElit(Elit,pop);               
              IntTree->GenPop(Elit,pop);   
	      if (Mutation) IntTree->PopMutation(pop,Elit,1,0.01); 	  
              i++; 
          }  
  }  
   if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<IntTree->TreeProb<<endl; //" Likehood "<< MyTree->Likehood<<endl;  
        
  DeletePopulations(); 
  delete IntTree;  
  return fgen;  
}  
 

int usualinit(double Complexity)  
{  
  int i,fgen;  
  double auxprob,sumprob;     
  BinaryTreeModel *MyTree;  
 
  InitPopulations(); 
  MyTree = new BinaryTreeModel(vars,Complexity,selpop->psize);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
   
   

 
  while (i<Maxgen && auxprob<1 && BestEval<Max)  
  {  
   EvalCliques(pop);  
    
     
     NPoints = Selection(); 
     //pop->Print(); 
     MyTree->rootnode =  MyTree->FindRootNode();//MyTree->RandomRootNode();  
     MyTree->CalProbFvect(compact_pop,fvect,NPoints);           
     MyTree->CalMutInf();  
     MyTree->MakeTree(MyTree->rootnode); 
     //MyTree->CalculateILikehood(compact_pop,fvect);  
     //MyTree->MakeTreeLog(); 
     FindBestVal(); 
 
     sumprob = MyTree->SumProb(compact_pop,NPoints); 
               MyTree->PutPriors(Prior,selpop->psize,1); 
	       auxprob = MyTree->Prob(BestInd,0);  
     
	       // if(printvals)     cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<MyTree->TreeProb<<" Likehood "<< MyTree->Likehood<<endl;  
       
      if (BestEval==Max) fgen  = i;	   
      else 
          { 
	      if (Tour>0 || (Tour==0 && Elit>TruncMax)) 
                { 
                  elitpop->SetElit(Elit,pop);  
                  // MyTree->CollectKConf(Elit,vars,MyTree,pop);           
                } 
              else  selpop->SetElit(Elit,pop);               
              MyTree->GenPop(Elit,pop);   
	      if (Mutation) MyTree->PopMutation(pop,Elit,1,0.01); 	  
              i++; 
          }  //lsantana@fecsa.es 
  }  
   if(printvals)  
         cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<MyTree->TreeProb<<" Likehood "<< MyTree->Likehood<<endl;  
        
  DeletePopulations(); 
 delete MyTree;  
  return fgen;  
}  


int Markovinit(double Complexity) //In this case, complexity is the threshold for chi-square 
{  
  int i,fgen;  
  double auxprob;     
  DynFDA* MyMarkovNet;  
   
  InitPopulations(); 
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  i=0;  fgen = -1;  auxprob =0; BestEval  = 0 ; NPoints=5; 
 
  
 Popul* aux_pop;
   aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
 
  while (i<Maxgen && BestEval<Max && NPoints>3)  
  {  
   
      EvalCliques(pop);
     	 
/*        
   if(i>0) 
       {
       // cout<<"Population pop"<<endl;
       //   pop->Print();            
        pop->Merge(Elit,vars,aux_pop);   
        pop->CopyPop(aux_pop); 
      
       }
      
      else aux_pop->CopyPop(pop);
*/     
      NPoints = Selection(); 
      MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
      MyMarkovNet->SetPop(compact_pop); 
/* 
  pop->TruncSel(selpop,TruncMax);   
  NSelPoints = TruncMax; 
  selpop->BotzmannDist(1,fvect); 
  MyMarkovNet->SetPop(selpop); 
  NPoints = psize; */ 
    
             
   
    
  //cout<<"Initial marginals "<<endl; 
  MyMarkovNet->UpdateModel();
  // MyMarkovNet->UpdateModelSAT(AllClauses->adjmatrix); 
  //AllClauses->UpdateWeights(MyMarkovNet->AllProb);
  //AllClauses->AdaptWeights(reward,selpop->P[0]);
  //FindBestVal(); 
  BestInd = selpop->P[0];
  


/*
 for(int ll=0;ll<1;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<selpop->P[ll][l];  
     //cout<<" "<<fvect[ll]<<endl;
     cout<<" "<<BestEval<<endl;
    }  
*/
  auxprob = MyMarkovNet->Prob(BestInd);  
     
     // cout<<BestEval<<" ";   selpop->Print(0); 
   if(printvals)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
 
      if (BestEval==Max) 
        {
           
            fgen  = i;	 
        } 
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);                    else  selpop->SetElit(Elit,pop);              
           MyMarkovNet->GenPop(Elit,pop);            
            /*  
           selpop->UniformProb(pop->psize,fvect); 
           NPoints = pop->CompactPopNew(compact_pop,fvect); 
           MyMarkovNet->SetNPoints(pop->psize,NPoints,fvect); 
           MyMarkovNet->SetPop(compact_pop); 
           cout<<"New marginals "<<endl; 
           MyMarkovNet->CallProb(); 
	   */ 
	   //if (Mutation) MyMarkovNet->PopMutation(pop,Elit,1,0.01);
 	   if (Mutation) MyMarkovNet->PopMutation(pop,Elit,1,0.01);
           i++; 
          }   
       MyMarkovNet->Destroy();       
  }  

  if(printvals)  
  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;  
        
 
  DeletePopulations(); 
  delete aux_pop; 
  delete MyMarkovNet;  
  return fgen;  
}  
 
 
int UMDAinit()  
{  
  int i,fgen;  
  double auxprob;     
  UnivariateModel *MyUMDA;  
    
  InitPopulations();   
  MyUMDA = new UnivariateModel(vars);  
  i=0; auxprob =0; BestEval  = 0; fgen = -1;  
  
  while (i<Maxgen && auxprob<1 && BestEval<Max)  
  {     
      EvalCliques(pop);
     /* if (i==0)  
        { 
         pop->Print(); 
         cout<<endl; 
	 }*/ 
     NPoints = Selection(); 
 
     MyUMDA->CalProbFvect(compact_pop,fvect,NPoints);    
     MyUMDA->PutPriors(Prior,selpop->psize,1); 
     // FindBestVal(); 
     BestInd = selpop->P[0];
     //AllClauses->UpdateWeights(MyUMDA->AllProb);
     // AllClauses->AdaptWeights(reward,selpop->P[0]);
     auxprob = MyUMDA->Prob(BestInd);  
  
  
     /*  
      for(int l=0;l<vars;l++) printf("%d ",selpop->P[0][l]);  
       printf("\n ");     
     */       
     //  if(printvals)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
      
    
      if (BestEval==Max) fgen  = i;	   
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else  selpop->SetElit(Elit,pop);   
           MyUMDA->GenPop(Elit,pop);   
	   if (Mutation) MyUMDA->PopMutation(pop,Elit,1,0.01); 	  
           i++; 
          }   
  }  
  if(printvals)  
   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;  
        
 
  DeletePopulations(); 
  delete MyUMDA;  
  return fgen;  
}  
 
 
 
int  MixturesAlgorithm(int Type,unsigned *Cardinalities,double Complexity)  
{  
  int i,fgen;  
  double auxprob;  
  MixtureTrees *Mixture;  
   
  InitPopulations(); 
  Mixture = new MixtureTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior);//NSteps+1  
  i=0; auxprob = 0; BestEval = 0; NPoints = 2; fgen = -1;  
  
Popul *aux_pop, *big_pop;
 aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
 big_pop = new Popul(psize,vars,Elit,Cardinalities);

 while (i<Maxgen && BestEval<Max && NPoints>1)  //&& oldlikehood != likehood)  
  { 
   EvalCliques(pop);
/*   
   if(i>0) 
       {        
       big_pop->Merge2Pops(aux_pop,aux_pop->psize,pop,pop->psize);
       pop->CopyPop(big_pop);
       aux_pop->CopyPop(big_pop);     
       }    
   else aux_pop->CopyPop(pop);
*/ 
   NPoints = Selection(); 
   Mixture->SetNpoints(NPoints,fvect); 
   Mixture->SetPop(compact_pop); 
  
 
   /*     
    for(int ll=0;ll<10;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<compact_pop->P[ll][l];  
     cout<<" "<<fvect[ll]<<endl; 
    }  
   */

   Mixture->MixturesInit(Type,InitTreeStructure,fvect,Complexity,0,0,0,0); 
   Mixture->LearningMixture(Type);  
   
   
   //sumprob = Mixture->SumProb(compact_pop,NPoints);  
   //AllClauses->AdaptWeights(reward,selpop->P[0]);
  //FindBestVal(); 
  BestInd = selpop->P[0]; 
 
   auxprob = Mixture->Prob(BestInd);  
   auxprob  =0;
   //Mixture->SamplingFromMixtureMixt(pop);  
   //Mixture->SamplingFromMixtureHMixing(pop);  
  
//     if(printvals)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;   
      
    if (BestEval==Max) fgen  = i;	   
     else if(NPoints>1)
          { 
          if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);           
           else  selpop->SetElit(Elit,pop);   
	  Mixture->SamplingFromMixture(pop); 
	  //Mixture->SamplingFromMixtureMixt(pop); 
	  //  Mixture->SamplingFromMixtureCycles(pop);
           if (Mutation) Mixture->EveryTree[0]->PopMutation(pop,Elit,1,0.01); 
	  }   
       Mixture->RemoveTrees(); 
       Mixture->RemoveProbabilities();
      i++;  
  }  
if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;   
    
  delete Mixture; 
  delete aux_pop;
  delete big_pop;
  DeletePopulations();  
  return fgen;  
}  
 


int  MixturesKikuchiAlgorithm(int Type,unsigned *Cardinalities,double Complexity)  
{  
  int i,fgen;  
  double auxprob;  
  MixtureTrees *Mixture;  
   
  InitPopulations(); 
  Mixture = new MixtureTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior);//NSteps+1  
  i=0; auxprob = 0; BestEval = 0; NPoints = 2; fgen = -1;  
  
Popul *aux_pop, *big_pop;
 aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
 big_pop = new Popul(psize,vars,Elit,Cardinalities);

 while (i<Maxgen && BestEval<Max && NPoints>1)  //&& oldlikehood != likehood)  
  { 
   EvalCliques(pop);
/*   
   if(i>0) 
       {        
       big_pop->Merge2Pops(aux_pop,aux_pop->psize,pop,pop->psize);
       pop->CopyPop(big_pop);
       aux_pop->CopyPop(big_pop);     
       }    
   else aux_pop->CopyPop(pop);
*/ 
   NPoints = Selection(); 
   Mixture->SetNpoints(NPoints,fvect); 
   Mixture->SetPop(compact_pop); 
  
 
   /*     
    for(int ll=0;ll<10;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<compact_pop->P[ll][l];  
     cout<<" "<<fvect[ll]<<endl; 
    }  
   */

   Mixture->MixturesInit(Type,InitTreeStructure,fvect,Complexity,CliqMaxLength,MaxNumCliq,LearningType,Cycles); 
   Mixture->LearningMixture(Type);  
   
   
//   sumprob = Mixture->SumProb(compact_pop,NPoints);             
   // FindBestVal();
    BestInd = selpop->P[0];
   auxprob = Mixture->Prob(BestInd);  
   auxprob  =0;
   //Mixture->SamplingFromMixtureMixt(pop);  
   //Mixture->SamplingFromMixtureHMixing(pop);  
  
   // if(printvals)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;   
      
    if (BestEval==Max) fgen  = i;	   
     else if(NPoints>1)
          { 
          if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);           
           else  selpop->SetElit(Elit,pop);   
	  Mixture->SamplingFromMixture(pop);          
	  } 
       Mixture->Destroy();
       Mixture->RemoveTrees(); 
       Mixture->RemoveProbabilities();
      i++;  
  }  

if(printvals)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
  
  delete Mixture; 
  delete aux_pop;
  delete big_pop;
  DeletePopulations();  
  return fgen;  
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
   switch(ExperimentMode)  
                     {                     
                       case 0: succ = usualinit(Complex);break; // Tree  
                       case 1: succ = MixturesAlgorithm(1,Cardinalities,Complex);break;   
                       case 2: succ = UMDAinit();break; // UMDA 
		       case 3: succ = MixturesAlgorithm(3,Cardinalities,Complex);break;// MT on dependencies  
                       case 4: succ = Markovinit(Complex);break; // Markov Network                    
                       case 5: succ = Intusualinit(Complex) ;break; // Tree of integers                
                       case 6: succ = EBNAAlgorithm(succ); break; // EBNA
                       case 7: succ = MixturesKikuchiAlgorithm(7,Cardinalities,Complex);break;// MT on dependencies  
  
                     } 
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
		   
		    cout<<f<<" "<<ExperimentMode<<" "<<vars<<" "<<func<<" "<<psize<<" "<<Cycles<<" "<<Ntrees<<" "<<Prior<<" "<<MaxMixtProb<<" "<<Complex<<" "<<succexp<<" "<<(auxmeangen+1)<<" "<<meanfit<<" "<<meaneval<<endl;    
                   } 
                  else  
                   {  
		       cout<<f<<" "<<ExperimentMode<<" "<<vars<<" "<<func<<" "<<psize<<" "<<Cycles<<" "<<Ntrees<<" "<<Prior<<" "<<MaxMixtProb<<" "<<Complex<<" "<<0<<" "<<0<<" "<<0<<" "<<meaneval<<endl; 
                   }              
 
} 
 
int main() 
 {  

   FILE *input,*f1; 
   int iexp; 
   char filedetails[25],number[10]; 

   int filenumbers[98] =    { 4,     9,    24,    32,    36,    37,    43,    50,    59,    74,    76,    91, 95,    96,   103,   119,   136,   147,   160,   176,   177,   192,   193,   216, 224,   232,   236,   286,   312,   316,   325,   328,   329,   330,   357,   374, 378,   386,   396,   410,   411,   419,   434,   435,   449,   458,   466,   468, 514,   532,   533,   542,   551,   552,   562,   581,   582,   586,   594,   599, 600,   601,   606,   630,   684,   688,   694,   717,   762,   764,   765,   771, 775,   777,   782,   791,   827,   841,   853,   858,   873,   877,   883,   887,  891,    893,   928,   933,   943,   944,   945,   950,   954,   956,   962,   971, 977};

   // int filenumbers[9] = {434,623,836,7,9,34,984,986,997}; 
   int TC[3]={0,75,100};
   params = new int[3]; 
   int EBNAALG[6] ={0,1,2,5,7,8};
   int i,succ; 
 
   unsigned ta = (unsigned) time(NULL);  
   srand(ta); 
   //srand(1074527919);

   cout<<"seed"<<ta<<endl;

     ReadParameters(); 
     Cardinalities  = new unsigned[vars];
    for(i=0;i<vars;i++) Cardinalities[i] = Card; 
   for (vars=100;vars<=100;vars+=50) 
    for (int f1=3;f1<=4;f1++) 
   // for (int f1=params[0];f1<=98;f1++) 
     { 
	 //cout<<"Instance "<<f<<endl;	
     MatrixFileName[0]=0; 
     //strcat(MatrixFileName,"uf20-0");
     strcat(MatrixFileName,"aim-");
     itoa(vars,number,10); 
     strcat(MatrixFileName,number); 
     strcat(MatrixFileName,"-3_4-yes1-");
     //strcat(MatrixFileName,"aim-200-3_4-yes1-");
     //  strcat(MatrixFileName,"aim-100-6_0-yes1-");
     //f = filenumbers[f1];
     f = f1; 
     itoa(f,number,10); 
     strcat(MatrixFileName,number); 
     //strcat(filedetails,MatrixFileName); 
     strcat(MatrixFileName,".cnf");       
    	 
        input = fopen(MatrixFileName,"r+"); 
	AllClauses = new CNF(input,3); 
	fclose(input); 

	vars = AllClauses->NumberVars; 
	Max = AllClauses->cantclauses; 
        AllClauses->FillMatrix(); 

	//for(ExperimentMode=4;ExperimentMode<=4;ExperimentMode+=2)
	//for(int EMCompl=0;EMCompl<=1;EMCompl+=1)
        //for(ExperimentMode=1;ExperimentMode<=1;ExperimentMode+=2)
	// for(int TCompl=0;TCompl<=4;TCompl+=4)
	//   for(int EMCompl=0;EMCompl<=2;EMCompl+=1)
	//for(Ntrees=1;Ntrees<=2;Ntrees+=1)
    {
      //LEARNEBNA =  EBNAALG[TCompl];
      //LearningType =  EBNAALG[TCompl];
      // LearningType = TCompl;
      //Ntrees =  LearningType;
     
     succ = 0;  succexp = 0;  meangen = 0; meaneval = 0;
     //CliqMaxLength = Ntrees;
     //Cycles = EMCompl;
     //MaxMixtProb = (EMCompl/100.0);
     //Complex = (TC[TCompl]/100.0);
     for (iexp =0;iexp<cantexp;iexp++) 
       { 
	   // printf("Run # %d %d of %d \n", iexp,cantexp,succ); //Alleval[i] 
          runOptimizer(); 
       } 
     PrintStatistics();
    }
     delete AllClauses;    
   }
     delete[] Cardinalities; 
     delete[] params; 
      
 }



 
