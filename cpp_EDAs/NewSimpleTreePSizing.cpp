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
#include "AbstractTree.h"  
#include "FDA.h"  
#include "MixtureTrees.h" 
  
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
int Cycles; 
 
double meaneval;  
double BestEval; 
int TruncMax; 
int NPoints;  
unsigned int *BestInd; 
Popul *pop,*selpop,*elitpop,*compact_pop; 
double *fvect; 
 
void ReadParameters()  
 
{ 
  int T,MaxMixtP,S_alph,Compl; 
 
stream = fopen( "Param.txt", "r+" );  
        params = new int[3];   
 		    
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
} 
 
int Intusualinit(double Complexity)  
{  
  int i,fgen;  
  double auxprob,sumprob;     
  IntTreeModel *IntTree;  
 
  InitPopulations(); 
  IntTree = new IntTreeModel(vars,Complexity,Cardinalities);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
   
   
  while (i<Maxgen && auxprob<1 && BestEval<Max)  
  {  
     pop->EvaluateAll(func);  
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
     auxprob = IntTree->Prob(BestInd);  
     
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
          }  //lsantana@fecsa.es 
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
  MyTree = new BinaryTreeModel(vars,Complexity);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
   
   
  while (i<Maxgen && auxprob<1 && BestEval<Max)  
  {  
     pop->EvaluateAll(func);  
     NPoints = Selection(); 
     //pop->Print(); 
     MyTree->rootnode = MyTree->RandomRootNode();  
     MyTree->CalProbFvect(compact_pop,fvect,NPoints);           
     MyTree->CalMutInf();  
     MyTree->MakeTree(MyTree->rootnode); 
     //MyTree->CalculateILikehood(compact_pop,fvect);  
     //MyTree->MakeTreeLog(); 
     FindBestVal(); 
 
     sumprob = MyTree->SumProb(compact_pop,NPoints); 
               MyTree->PutPriors(Prior,selpop->psize,1); 
     auxprob = MyTree->Prob(BestInd);  
     
     //if(printvals)        cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<MyTree->TreeProb<<" Likehood "<< MyTree->Likehood<<endl;  
       
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
  i=0;  fgen = -1;  auxprob =0; BestEval  = Max -1; NPoints=5; 
 
  while (i<Maxgen && BestEval<Max && NPoints>1)  
  {  
     
      pop->EvaluateAll(func); 
      NPoints = Selection(); 
/* 
  pop->TruncSel(selpop,TruncMax);   
  NSelPoints = TruncMax; 
  selpop->BotzmannDist(1,fvect); 
  MyMarkovNet->SetPop(selpop); 
  NPoints = psize; */ 
   
  MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
  MyMarkovNet->SetPop(compact_pop); 
     
  /*          
    for(int ll=0;ll<10;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<compact_pop->P[ll][l];  
     cout<<" "<<fvect[ll]<<endl; 
    }  
  */ 
  //cout<<"Initial marginals "<<endl; 
   
     MyMarkovNet->UpdateModel(); 
     FindBestVal(); 
     auxprob = MyMarkovNet->Prob(BestInd);  
     
 
     //   if(printvals)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
 
      if (BestEval==Max)    fgen  = i;	  
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else  selpop->SetElit(Elit,pop);   
             
           MyMarkovNet->GenPop(Elit,pop);   
	   /*  
           selpop->UniformProb(pop->psize,fvect); 
           NPoints = pop->CompactPopNew(compact_pop,fvect); 
           MyMarkovNet->SetNPoints(pop->psize,NPoints,fvect); 
           MyMarkovNet->SetPop(compact_pop); 
           cout<<"New marginals "<<endl; 
           MyMarkovNet->CallProb(); 
	   */ 
 
	   if (Mutation) MyMarkovNet->PopMutation(pop,Elit,1,0.01); 	  
           i++; 
          }   
  }  
  if(printvals)  
  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;  
        
  delete[] fvect; 
  DeletePopulations(); 
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
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
  
  while (i<Maxgen && auxprob<1 && BestEval<Max)  
  {     
     pop->EvaluateAll(func);  
     /* if (i==0)  
        { 
         pop->Print(); 
         cout<<endl; 
	 }*/ 
     NPoints = Selection(); 
 
     MyUMDA->CalProbFvect(compact_pop,fvect,NPoints);    
     MyUMDA->PutPriors(Prior,selpop->psize,1); 
     FindBestVal(); 
     auxprob = MyUMDA->Prob(BestInd);  
     /*  
      for(int l=0;l<vars;l++) printf("%d ",selpop->P[0][l]);  
       printf("\n ");     
     */       
        if(printvals)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
      
    
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
        
  delete[] fvect; 
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
  i=0; auxprob = 0; BestEval = Max-1; NPoints = 2; fgen = -1;  
  
 while (i<Maxgen && BestEval<Max && NPoints>1)  //&& oldlikehood != likehood)  
  { 
   pop->EvaluateAll(func);  
   NPoints = Selection(); 
   Mixture->SetNpoints(NPoints,fvect); 
   Mixture->SetPop(compact_pop); 
   Mixture->MixturesInit(Type,InitTreeStructure,fvect,Complexity); 
   Mixture->LearningMixture(Type);  
   
   //Mixture->SamplingFromMixtureMixt(pop);  
   //Mixture->SamplingFromMixtureHMixing(pop);  
  	 
   FindBestVal(); 
// if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;   
      
    if (BestEval==Max) fgen  = i;	   
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else  selpop->SetElit(Elit,pop);   
           Mixture->SamplingFromMixture(pop);  
           if (Mutation) Mixture->EveryTree[0]->PopMutation(pop,Elit,1,0.01); 
	   i++; 
          }   
    Mixture->RemoveTrees(); 
 
      i++;  
  }  
if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl;   
 
   
  delete Mixture;  
  DeletePopulations(); 
  delete[] fvect; 
   
  return fgen;  
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
                     } 
   if (succ>-1)  
   { 
       succexp++; 
       meangen += succ; 
       
   }   
   else nsucc ++;  
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
                    cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" f(x)="<<func<<" N="<<psize<<" Sel="<<Tour<<"  m="<<Ntrees<<"  Prior="<<Prior<<"  LearningT="<<LearningType<<   " Suc.="<<succexp<<"  g="<<(auxmeangen+1)<<"  ave="<<meanfit<<" meaneval "<<meaneval<<endl;                   
                   } 
                  else  
                   {  
		       cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" f(x)="<<func<<" N="<<psize<<" Sel="<<Tour<<"  m="<<Ntrees<<"  Prior="<<Prior<<"  LearningT="<<LearningType<<" Suc.="<<0<<"  g="<<0<<"  ave="<<0<<" meaneval "<<meaneval<<endl; 
                   }              
 
} 
 
  
int main(){  
       
 int i;  
 unsigned ta = (unsigned) time(NULL);  
 //ta =  1046286038;
 srand(ta); 
 cout<<"seed"<<ta<<endl; 
 //int cota; 
 params = new int[3]; 
 ReadParameters(); 
  


 Cardinalities  = new unsigned[vars];  
 for(i=0;i<vars;i++) Cardinalities[i] = Card; 

 int funci[5]={8,15,15,25,19};
 double optim[15]={12,36,36,1191.0,211.0};
 int paramval[5] ={3,3,4,0,6}

for(j=0;j<=8;j+=4)   
 {
  ExperimentMode = j;
  if (j==4) Cycles=0;
  else  if (j==8)
  {
   Cycles=1;
   ExperimentMode = 4;
  }
   
  for(u=0;u<=3;u++) 
   {            
             psize = 200;    
             func = funci[u];
	     Max = optim[u];
	     param[0] = paramval[u];
 
   while(psize < 5000)
     {
       succexp = 0;  meangen = 0; meaneval = 0;  nsucc = 0; i =0;     
       while (i<cantexp  && nsucc<10)  
        { 	   
         runOptimizer(); 
         i++;
        }       
       PrintStatistics();
       psize += 100;
      } 
    }
  }
delete[] Cardinalities; 
delete[] params; 
//fclose(outfile); 
}      
 
//81 
//19600 
//15 
//600 
 
 
 
 
 
 
 
 
