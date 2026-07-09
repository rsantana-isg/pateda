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
//double statistics[100][15];  
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
 
RotProtein* RotamerProtein;
int TotEvaluations;
int sizeProtein;
int EvaluationMode;
int currentexp;
int* indexcomponents;




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
  int k;
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
 	 fscanf( stream, "%lg", &Max); // Max. of the fitness function  
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
//Max = auxMax/double(1000);   
// cout<<"Max "<<Max<<endl;
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
 


void EvalRotamerProtein(Popul* epop,int nelit, int epsize, int atgen)
{
 double CurrentEval=0;
 int start,pos,i;
 

if (atgen==0) start=0;
else start=nelit;
 
for(pos=start; pos < epsize;  pos ++)  
 {
   //cout<<pos<<endl;
   //for(i=0;i<vars;i++) CurrentEval= CurrentEval - 1.0*epop->P[pos][i];
   //if(params[0]==0)  CurrentEval = -1.0*RotamerProtein->CalculateEnergy(epop->P[pos]);
   if(params[0]==0)  CurrentEval = 10000 -1*RotamerProtein->CalculateEnergySCP(epop->P[pos]); //Function for SCP instances
   else  if(params[0]==1)  CurrentEval = -1.0*RotamerProtein->CalculateEnergyWithDEE(epop->P[pos]);
    else  if(params[0]==2)  CurrentEval = -1.0*RotamerProtein->CalculateEnergyDECOMP(epop->P[pos], vars, indexcomponents);
    epop->SetVal(pos,CurrentEval); 
    TotEvaluations ++;    
  }

}




int Markovinit(double Complexity, int typemodel, int sizecliq)  //In this case, complexity is the threshold for chi-square 
{  
  int i,fgen;  
  double auxprob;     
  DynFDA* MyMarkovNet;  
   
 //  init_time();
  InitPopulations(); 
 
  LearningType=3;
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
 

  i=0;  fgen = -1;  auxprob =0; BestEval  = Max -1; NPoints=TruncMax; 

  Popul* aux_pop;
  aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
  
  NPoints = psize;
 
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
    
     init_time();
     EvalRotamerProtein(pop,Elit,psize,i); 
     end_time(); 
     cout<<"0 "<<i<<" "<<BestEval<<" "<<NPoints<<" "<<TotEvaluations<<" "<<auxtime<<endl;  

    //pop->Print(); 
     
       init_time();
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

  
     FindBestVal(); 
     //AllGen[i] += BestEval; 

     auxprob = MyMarkovNet->Prob(BestInd);  
         
 
     //  if(printvals>1)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" Elit"<<Elit<<" TotEval:"<<TotEvaluations<<" DifPoints:"<<NPoints<<endl; 
    
 
 
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
             end_time(); 
             cout<<"1 "<<i<<" "<<BestEval<<" "<<NPoints<<" "<<TotEvaluations<<" "<<auxtime<<endl;    

  
       
             if(params[1]) pop->Mutation(Elit,0.015);
             i++;         
            MyMarkovNet->Destroy();    
  
   }  

  //if(NPoints>10) NPoints = 10;


  //end_time();  

 // if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  
  

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
  NPoints = 100;
   
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
   
     EvalRotamerProtein(pop,Elit,psize,i); 
     //pop->Print(); 
     NPoints = Selection();  
     IntTree->SetNPoints(NPoints);
     IntTree->rootnode =   IntTree->RandomRootNode(); // IntTree->FindRootNode();  
     IntTree->CalProbFvect(selpop,fvect,NPoints,0);   

     //IntTree->CalMutInf(RotamerProtein->RedMatrix); //THIS IS ONLY FOR STRUCTURAL INFO     
     IntTree->CalMutInf();  
     //IntTree->CalMutInfDeception(); 
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
    if(printvals)   cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<sumprob<<" "<<Elit<<endl;    
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

     IntTree->GenPop(Elit,pop);   
     //IntTree->PopMutation(pop,Elit,1,0.015);
     if(params[1]) pop->Mutation(Elit,0.015);
     i++;
     // cout<<i<<" "<<Maxgen<<" "<<BestEval<<" "<<Max<<" "<<NPoints<<endl;
  }  
  //cout<<BestEval<<" ";   selpop->Print(0); 

    end_time(); 
    if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<-1*(BestEval-10000.0)<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<" ";  


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
     cout<<" "<<-1*(selpop->Evaluations[ll]-10000.0)<<endl; 
    }  
   }
 
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
      EvalRotamerProtein(pop,Elit,psize,i); 

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


int  readdata()
{  
  int ctime;
  float bestval;
  int alele,i,j,k,ident;
  int nexper;
   int proteinsizes[325] = {95, 158, 144, 161, 105, 124, 53, 56, 85, 76, 90, 177, 54, 26, 62, 92, 32, 14, 144, 122, 181, 137, 152, 122, 175, 97, 60, 132, 149, 50, 133, 213, 121, 48, 95, 108, 130, 112, 39, 111, 97, 85, 94, 64, 92, 173, 108, 126, 134, 88, 74, 184, 128, 200, 113, 178, 98, 99, 47, 108, 123, 66, 162, 75, 96, 139, 244, 86, 179, 96, 84, 251, 91, 198, 47, 61, 144, 104, 93, 69, 78, 143, 216, 228, 123, 51, 103, 131, 154, 130, 83, 100, 109, 118, 230, 169, 241, 127, 205, 91, 117, 186, 139, 154, 91, 233, 121, 139, 103, 193, 65, 192, 140, 184, 9, 10, 9, 157, 92, 125, 116, 112, 147, 58, 103, 55, 207, 107, 122, 66, 75, 157, 199, 172, 51, 55, 83, 165, 44, 75, 66, 86, 76, 125, 267, 64, 68, 108, 146, 83, 109, 211, 126, 235, 56, 54, 165, 116, 60, 57, 261, 152, 84, 63, 140, 67, 122, 85, 165, 113, 68, 107, 67, 39, 133, 122, 195, 115, 101, 96, 88, 103, 188, 57, 148, 156, 190, 153, 144, 152, 64, 50, 115, 259, 150, 192, 79, 54, 46, 208, 106, 262, 88, 102, 114, 204, 50, 169, 203, 80, 11, 94, 64, 87, 170, 17, 13, 29, 82, 92, 91, 203, 218, 106, 173, 106, 132, 229, 175, 134, 144, 83, 97, 54, 42, 142, 123, 168, 102, 174, 21, 88, 137, 222, 93, 115, 119, 65, 99, 92, 140, 80, 92, 121, 208, 167, 97, 66, 76, 189, 97, 111, 65, 63, 133, 225, 119, 101, 91, 151, 185, 101, 80, 85, 7, 89, 56, 161, 77, 85, 96, 233, 210, 142, 112, 34, 158, 143, 42, 125, 97, 190, 81, 140, 50, 142, 122, 115, 226, 80, 105, 48, 94, 80, 92, 88, 53, 100, 87, 89, 84, 91, 57, 88, 64, 92, 101, 220, 224, 176, 92, 62, 94, 119, 96};

  //int proteinsizes[45] ={704,428,574,315,380,442, 434,618, 374,441,331,340,345,404,338,512,490,397,495,443,360,445,688,337,463,373,332,544,381,424,381,375,358,509,457,510,344,392,332,441,334,641,496,311,594};

  // int proteinsizes[93] = {972,1972,582,397,510,636,441,578,668,1328,413,960, 699,1298,1219,807,750,609,604,964, 880,446, 531,614,630,658, 636,635,1409,650, 750,866, 628,520, 614,804, 520,640, 565,1982, 685,532, 124,544, 744,1671,  314,584, 415,923,919,530, 922,545, 1040,855, 806,454,457,772,739,481,323,522, 536,505, 1060,593, 576,724, 960,1364, 505,629, 283,746, 562,1124, 1108,632, 856,506, 656,465, 610,921,699,926,519,277, 845,576,446};


  

  //int proteinsizes[45] = {367, 147, 160, 75, 103, 146, 185, 246, 90, 127, 69, 110, 172, 99, 85, 166, 136, 136, 137, 124, 125, 208, 318, 145, 174, 144, 81, 241, 144, 207, 176, 113, 66, 155, 189, 175, 113, 146, 157, 141, 192, 234, 191, 62, 292};

  FILE *helpstream; 
 //helpstream = fopen( "rotexperiments200.txt", "r+" );  
 //helpstream = fopen( "roteresultsmissing324.txt", "r+" );  
 //helpstream = fopen( "output.26", "r+" );  
 //helpstream = fopen( "roteNewDEElarge1.txt", "r+" );  
 // helpstream = fopen( "output.31", "r+" );  
  // helpstream = fopen( "output.41", "r+" ); 
  // helpstream = fopen( "output.41", "r+" ); 

helpstream = fopen( "rotSTree.o9416", "r+" ); 

nexper = 50;
 
for(k=0;k<11;k++) 
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
    for(i=0;i<500;i++) 
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




int  shortreaddata()
{  
  int ctime;
  float bestval;
  int alele,i,j,k,f,ident;
  int nexper;
  int it, filenumb;
  double auxval; 
  FILE *helpstream,*namesfiles; 
  char beststr[30]; 

   int proteinsizes[93] = {972,1972,582,397,510,636,441,578,668,1328,413,960, 699,1298,1219,807,750,609,604,964, 880,446, 531,614,630,658, 636,635,1409,650, 750,866, 628,520, 614,804, 520,640, 565,1982, 685,532, 124,544, 744,1671,  314,584, 415,923,919,530, 922,545, 1040,855, 806,454,457,772,739,481,323,522, 536,505, 1060,593, 576,724, 960,1364, 505,629, 283,746, 562,1124, 1108,632, 856,506, 656,465, 610,921,699,926,519,277, 845,576,446};



namesfiles = fopen( "allufiles.txt", "r+" );  

for(f=0;f<30;f++)
 {
  fscanf(namesfiles, "%s",&filedetails);
  helpstream = fopen(filedetails, "r+" );  
  nexper = 1;
  fscanf(helpstream, "%s",&MatrixFileName);
  for(k=0;k<15;k++) 
  { 
     fscanf(helpstream, "%d %d %s %f",&it, &filenumb, &MatrixFileName, &auxval);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%f",&auxval);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%f"  ,&auxval);

   for(j=0;j<nexper;j++) 
   {
     cout<<f<<" "<<proteinsizes[filenumb]<<" "; 
     // cout<<it<<" "<<proteinsizes[notopt[filenumb]-1]<<" "; 
     fscanf(helpstream, "%s",&MatrixFileName);
     cout<<MatrixFileName<<" ";
     fscanf(helpstream, "%s",&MatrixFileName);
     cout<<MatrixFileName<<" ";
     fscanf(helpstream, "%s",&MatrixFileName);
     cout<<MatrixFileName<<" ";   
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s %d", &MatrixFileName, &ctime);  
     //cout<<ident<<" "<<bestval<<" "<<ctime<<" ";
     cout<<ctime<<" ";  

     for(i=0;i<1999;i++) 
     {
       if(i<proteinsizes[filenumb]) 
        {
         fscanf(helpstream, "%d", &alele);
         cout<<alele<<" ";
        }
       else cout<<0<<" ";
      }
    fscanf(helpstream, "%e", &bestval);
    ctime = abs(ctime);
    cout<<filenumb<<" "<<-1.0*bestval<<" "<<ctime<<endl;     
   }
   for(i=0;i<19;i++)  fscanf(helpstream, "%s", &MatrixFileName);
}
 fclose(helpstream);
}
 fclose(namesfiles);
}  











int  otherreaddata()
{  
  int ctime;
  float bestval;
  int alele,i,j,k,ident;
  int nexper;
   
  FILE *helpstream; 
 
  // unsigned int redsizes[11] = {27,21,45,64,43,63,70,51,35,28,35};
  // unsigned int redsizes[14] = {75,146,185,127,166,208,318,144,241,207,155,189,175,292};
  // unsigned int redsizes[14] = {916,281,353,288,454,479,365,123,294,239,240,265,934,206,227,229,728,288,289,329,285,268,244,350,424,164};
 
  //int proteinsizes[325] = {95, 158, 144, 161, 105, 124, 53, 56, 85, 76, 90, 177, 54, 26, 62, 92, 32, 14, 144, 122, 181, 137, 152, 122, 175, 97, 60, 132, 149, 50, 133, 213, 121, 48, 95, 108, 130, 112, 39, 111, 97, 85, 94, 64, 92, 173, 108, 126, 134, 88, 74, 184, 128, 200, 113, 178, 98, 99, 47, 108, 123, 66, 162, 75, 96, 139, 244, 86, 179, 96, 84, 251, 91, 198, 47, 61, 144, 104, 93, 69, 78, 143, 216, 228, 123, 51, 103, 131, 154, 130, 83, 100, 109, 118, 230, 169, 241, 127, 205, 91, 117, 186, 139, 154, 91, 233, 121, 139, 103, 193, 65, 192, 140, 184, 9, 10, 9, 157, 92, 125, 116, 112, 147, 58, 103, 55, 207, 107, 122, 66, 75, 157, 199, 172, 51, 55, 83, 165, 44, 75, 66, 86, 76, 125, 267, 64, 68, 108, 146, 83, 109, 211, 126, 235, 56, 54, 165, 116, 60, 57, 261, 152, 84, 63, 140, 67, 122, 85, 165, 113, 68, 107, 67, 39, 133, 122, 195, 115, 101, 96, 88, 103, 188, 57, 148, 156, 190, 153, 144, 152, 64, 50, 115, 259, 150, 192, 79, 54, 46, 208, 106, 262, 88, 102, 114, 204, 50, 169, 203, 80, 11, 94, 64, 87, 170, 17, 13, 29, 82, 92, 91, 203, 218, 106, 173, 106, 132, 229, 175, 134, 144, 83, 97, 54, 42, 142, 123, 168, 102, 174, 21, 88, 137, 222, 93, 115, 119, 65, 99, 92, 140, 80, 92, 121, 208, 167, 97, 66, 76, 189, 97, 111, 65, 63, 133, 225, 119, 101, 91, 151, 185, 101, 80, 85, 7, 89, 56, 161, 77, 85, 96, 233, 210, 142, 112, 34, 158, 143, 42, 125, 97, 190, 81, 140, 50, 142, 122, 115, 226, 80, 105, 48, 94, 80, 92, 88, 53, 100, 87, 89, 84, 91, 57, 88, 64, 92, 101, 220, 224, 176, 92, 62, 94, 119, 96};

  //  int proteinsizes[45] ={704,428,574,315,380,442, 434,618, 374,441,331,340,345,404,338,512,490,397,495,443,360,445,688,337,463,373,332,544,381,424,381,375,358,509,457,510,344,392,332,441,334,641,496,311,594};

   int proteinsizes[93] = {972,1972,582,397,510,636,441,578,668,1328,413,960, 699,1298,1219,807,750,609,604,964, 880,446, 531,614,630,658, 636,635,1409,650, 750,866, 628,520, 614,804, 520,640, 565,1982, 685,532, 124,544, 744,1671,  314,584, 415,923,919,530, 922,545, 1040,855, 806,454,457,772,739,481,323,522, 536,505, 1060,593, 576,724, 960,1364, 505,629, 283,746, 562,1124, 1108,632, 856,506, 656,465, 610,921,699,926,519,277, 845,576,446};



 // unsigned int notopt[11] = {48,49,110,114,144,157,200,288,296,309,316}; //Solutions for which UMDA did not converged to the known optimum //SMALL

 // unsigned int notopt[33] = {13,48,49,72,82,95,96,97,99,110,114,119,123,144,149,154,157,177,194,200,226,228,251,255,260,282,288,296,299,309,316,318,319}; //Solutions for which  optimum is not known  //SMALL
 
  // unsigned int notopt[14] = {4,6,7,10,16,22,23,26,28,30,34,35,36,45};//LARGE
  // unsigned int notopt[26] = {2,10,12,13,14,15,16,22,24,25,26,31,40,41,42,45,46,55,56,57,61,69,70,71,81,92}; //DIMER
   unsigned int notopt[67] = {1,3,4,5,6,7,8,9,11,17,18,19,20,21,23,27,28,29,30,32,33,34,35,36,37,38,39,43,44,47,48,49,50,51,52,53,54,58,59,60,62,63,64,65,66,67,68,72,73,74,75,76,77,78,79,80,82,83,84,85,86,87,88,89,90,91}; //DIMER
  //helpstream = fopen( "rotSTree.o9416", "r+" ); 
    //helpstream = fopen( "rotSTrST.o9563", "r+" ); 
    //helpstream = fopen( "rotLTrST.o9565", "r+" ); 
   //helpstream = fopen( "rotLTree.o9417", "r+" ); 
   //helpstream = fopen( "AllDimerTree1.txt", "r+" ); 

helpstream = fopen( "alldprot.txt", "r+" ); 
nexper = 30;
int it, filenumb;
 double auxval; 
for(k=0;k<47;k++) 
{ 
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%d %d %s %f",&it, &filenumb, &MatrixFileName, &auxval);
     if (it>11) 
       {
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%f",&auxval);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%f"  ,&auxval);
       } 
    //cout<<it<<" "<<filenumb-1<<" "; 
    
   for(j=0;j<nexper;j++) 
   {
     cout<<j<<" "<<proteinsizes[filenumb]<<" "; 
     // cout<<it<<" "<<proteinsizes[notopt[filenumb]-1]<<" "; 
     fscanf(helpstream, "%d", &ident);
     fscanf(helpstream, "%e", &bestval);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s",&MatrixFileName);
     fscanf(helpstream, "%s %d", &MatrixFileName, &ctime);  
     cout<<ident<<" "<<bestval<<" "<<ctime<<" ";

     for(i=0;i<1999;i++) 
     {
       if(i<proteinsizes[filenumb]) 
        {
         fscanf(helpstream, "%d", &alele);
         cout<<alele<<" ";
        }
       else cout<<0<<" ";
      }
    fscanf(helpstream, "%e", &bestval);
    ctime = abs(ctime);
    cout<<filenumb<<" "<<-1.0*bestval<<" "<<ctime<<endl;     
   }
   for(i=0;i<19;i++)  fscanf(helpstream, "%s", &MatrixFileName);
}
fclose(helpstream);
}  






void  Decomposition()
{
   int i,j,k;
   double valbest;
   int auxpopsize;
   auxpopsize = psize;

    valbest = 0;
    for(i=-1;i<RotamerProtein->ncomp;i++) 
     {
      k = 0;
      for(j=0;j<RotamerProtein->numberresidues;j++) 
        {
	  if ((RotamerProtein->VarPos[j] != -1)  && (RotamerProtein->DissComponents[j]==i))
	    {
	      indexcomponents[k++] = RotamerProtein->VarPos[j];
            }
        }
      if(k>0)
	{
         vars = k;
	 //cout<<"number vars "<<vars<<endl;
	 /*
         for(j=0;j<vars;j++)
	  {
	    cout<<j<<" "<<indexcomponents[j]<<" "<<RotamerProtein->NFinalRot[indexcomponents[j]]<<endl;
          }
	 */
	 if(k<=10) psize = (k*500);
         else psize = auxpopsize;
         for(j=0;j<vars;j++) Cardinalities[j] = RotamerProtein->NFinalRot[indexcomponents[j]];
         valbest = Partialrun(ExperimentMode,0);
         }
     }

    for(int l=0;l<RotamerProtein->numberresidues;l++) cout<<RotamerProtein->Fixed[l]<<" ";    
    cout<<valbest<<endl;

}  





int  main(){  


 int i,j;  
 unsigned ta = (unsigned) time(NULL);  


 unsigned int bestsol[95] = {25, 1, 2, 0, 1, 7, 0, 7, 3, 0, 3, 30, 13, 7, 4, 70, 1, 7, 4, 33, 70, 0, 8, 1, 2, 1, 22, 13, 26, 2, 0, 0, 7, 70, 40, 1, 30, 18, 7, 18, 2, 0, 7, 7, 70, 1, 2, 8, 7, 7, 0, 0, 30, 7, 1, 7, 12, 7, 18, 3, 2, 1, 22, 25, 10, 4, 14, 1, 14, 14, 2, 67, 7, 8, 18, 7, 30, 8, 4, 1, 12, 1, 67, 1, 18, 22, 7, 3, 8, 40, 3, 7, 0, 4, 4}; 
 unsigned int loadedsol[2000];
 unsigned int initassign[2000];

 // unsigned int notopt[11] = {48,49,110,114,144,157,200,288,296,309,316}; //Solutions for which UMDA did not converged to the known optimum //SMALL

 // unsigned int notopt[33] = {13,48,49,72,82,95,96,97,99,110,114,119,123,144,149,154,157,177,194,200,226,228,251,255,260,282,288,296,299,309,316,318,319}; //Solutions for which  optimum is not known  //SMALL
 
 //   unsigned int notopt[14] = {4,6,7,10,16,22,23,26,28,30,34,35,36,45};//LARGE
  unsigned int notopt[26] = {2,10,12,13,14,15,16,22,24,25,26,31,40,41,42,45,46,55,56,57,61,69,70,71,81,92}; //DIMER
 
  otherreaddata();
  return 1;

 //    ta = 1051750320;
 srand(ta); 
 cout<<"seed"<<ta<<endl; 
 params = new int[3]; 
 ReadParameters(); 
 
  
 // otherreaddata(); return 1; //For changing the format of the output data


 
 // stream = fopen( "largeproteinnames.txt", "r+" );  
 //  stream = fopen( "proteinnames.txt", "r+" );  
 //  stream = fopen("dimerproteinnames.txt", "r+" );  
   stream = fopen("designkcs.txt", "r+" );  

    
   //   if(params[0] == 3)  streambestsol = fopen( "Best50SolutionsLarge.txt", "r+" );
   //  if(params[0] == 3)  streambestsol = fopen("Best50SolutionsDimer.txt", "r+" );
   //  if(params[0] == 3)  streambestsol = fopen( "Best50Solutions.txt", "r+" );  
   //  if(params[0] == 3)  streambestsol = fopen( "Random50Solutions.txt", "r+" );  
 //  if(params[0] == 3)  streambestsol = fopen( "Random50SolutionsLarge.txt", "r+" );  
   //  if(params[0] == 3)  streambestsol = fopen( "Random50SolutionsDimer.txt", "r+" );  
 //  if( streambestsol  == NULL )  printf( "The  file  with the initial solutions was not opened\n" );  

   /*
      
    for(int jj=0;jj<params[2];jj++)
      for(int r=0;r<50;r++)
	{
	  //cout<<jj<<"  "<<r<<endl;
           for(int q=0;q<2000;q++) 
            {
              fscanf(streambestsol, "%d", &loadedsol[q]); 
              //cout<<loadedsol[q]<<" ";
            }  
           //cout<<endl;
        } 
        
   */   
 for(int jj=params[2];jj<params[2]+1;jj++)
 { 
   j = jj-1; // ONLY WHEN THE ABSOLUT NUMBER IS KNOWN, ALWAYS FOR designkcs.txt
   //j = notopt[jj]-1;
   //fseek(stream,9*(j),0);
   MatrixFileName[0]=0; 
  // strcat(MatrixFileName,".dat");  
   fscanf(stream, "%s", &MatrixFileName);
   file = fopen(MatrixFileName, "r+" );  
   //file = fopen("1qj4.fix.scp", "r+" );  
 
  if( file  == NULL )  printf( "The protein file  was not opened\n" );  
   
   cout<<j<<" "<<MatrixFileName<<" "<<file<<endl;
   //Max = 0;
   //RotamerProtein = new RotProtein(file);
   RotamerProtein = new RotProtein(file,1);

   if(params[0] == 0)
    {
     vars = RotamerProtein->numberresidues;
     Cardinalities  = new unsigned[vars];  
     for(i=0;i<vars;i++) Cardinalities[i] = RotamerProtein->RotNumber[i];  
    }
   else if(params[0] == 1)
     {
     double  meanvald = 0;
      RotamerProtein->ApplyDEE(0);
      //RotamerProtein->FindDisconnectedComponents();
      vars = RotamerProtein->NResiduesAfterDEE;
      if(RotamerProtein->NResiduesAfterDEE>0)
	{
         Cardinalities  = new unsigned[vars];  
         for(i=0;i<vars;i++)
          {  Cardinalities[i] = RotamerProtein->NFinalRot[i];
             meanvald += Cardinalities[i];
          }
         }
      //cout<<"vars is "<<vars<<" meanval "<<meanvald/vars<<endl;
     }
     else if(params[0] == 2)
     {
       RotamerProtein->ApplyDEE(0);
       RotamerProtein->FindDisconnectedComponents();
       Cardinalities  = new unsigned[RotamerProtein->NResiduesAfterDEE];  
       indexcomponents = new int[RotamerProtein->NResiduesAfterDEE];    
     }
   else if(params[0] == 3) // Local optimizer to improve known best solutions
     {
      RotamerProtein->ApplyDEE(0);
      vars = RotamerProtein->NResiduesAfterDEE;
      RotamerProtein->FindActiveContacts();
      Cardinalities  = new unsigned[vars];  
      for(i=0;i<vars;i++) 
        {
          Cardinalities[i] = RotamerProtein->NFinalRot[i];
     
          //if (i<vars) cout<<Cardinalities[i]<<" ";
          //else  cout<<0<<" ";
        }
      /*    
       for(int r=0;r<50;r++)
	{ 
         for(i=0;i<800;i++) 
           {
             if(i<vars) cout<<randomint( Cardinalities[i])<<" ";
             else cout<<"0 ";
           }
	 cout<<endl;
	}
       return 1;
      */
      // cout<<endl;
      //cout<<"The number of vars is "<<vars<<" and the initial number is "<<RotamerProtein->numberresidues<<endl;  
      // cout<<"The number of original contacts is "<<RotamerProtein->ncells<<" and the number of active contacts is "<<RotamerProtein->NActiveContacts<<endl;
        
      
      for(int r=0;r<50;r++)
	{ 
	  //  for(int q=0;q<500;q++)  fscanf(streambestsol, "%d", &initassign[q]); //For Small Random Solutions
	  //  for(int q=0;q<800;q++)  fscanf(streambestsol, "%d", &initassign[q]); //For Large Random Solutions
	   for(int q=0;q<2000;q++)  fscanf(streambestsol, "%d", &initassign[q]); //For Dimer Random Solutions
          
         //  for(int q=0;q<500;q++)  fscanf(streambestsol, "%d",&loadedsol[q]); //For Small Best Solutions
         // for(int q=0;q<800;q++)  fscanf(streambestsol, "%d",&loadedsol[q]); //For  Large Best Solutions
	 // for(int q=0;q<2000;q++)  fscanf(streambestsol, "%d", &loadedsol[q]); //For Dimer  Solutions
               
	   if(r>30 && r<50)
	    {
	     //  RotamerProtein->FindReducedCodification(loadedsol,initassign); //For Best Solutions
             //RotamerProtein->FindEnlargedCodification(loadedsol,initassign);
	     /* for(int q=0;q<500;q++)  
                {
                  if(q<RotamerProtein->numberresidues) cout<<loadedsol[q]<<" "; //For Dimer Random Solutions
                  else cout<<"0 ";
                }
             cout<<endl;
             */ 
             //for(int q=0;q<800;q++) cout<<loadedsol[q]<<" ";
             //cout<<endl;            
             //for(int q=0;q<500;q++) cout<<initassign[q]<<" ";
	     // cout<<endl;  
       
             double initval = RotamerProtein->CalculateEnergyWithDEE(initassign);  
             //cout<<"initval "<<initval<<endl;
	     init_time();
             cout<<" 0  0 "<<" "<<initval<<" "<<gmt->tm_mday<<" "<<gmt->tm_hour<<" "<<gmt->tm_min<<" "<<gmt->tm_sec<<endl;
             int VNSeval = 0;
             //int nmoves = RotamerProtein->SearchNeighborhood(10000,initassign,&VNSeval);
             int nmoves = RotamerProtein->SearchRandomNeighborhood(2000000,initassign,100000,&VNSeval);
             double finalval = RotamerProtein->CalculateEnergyWithDEE(initassign);  
             init_time();
             cout<<" 0 "<<VNSeval<<" "<<finalval<<" "<<gmt->tm_mday<<" "<<gmt->tm_hour<<" "<<gmt->tm_min<<" "<<gmt->tm_sec<<endl;
             //cout<<j<<" "<<r<<" "<<initval<<" "<<finalval<<" "<<nmoves<<" "<<VNSeval<<endl;
             //if (nmoves>0) cout<<"Solution "<<r<<" of instance "<<j<<" improved after "<<nmoves<<" moves"<<endl;
            }
	}
     }
else if(params[0] == 4) // Local optimizer to improve known best solutions
     {    
      vars = RotamerProtein->numberresidues;
      cout<<" Number of vars    "<<vars<<endl; 
      Cardinalities  = new unsigned[vars];  
      for(i=0;i<vars;i++) Cardinalities[i] = RotamerProtein->RotNumber[i];  
      if (params[1]==0) RotamerProtein->BestLoopySolution(Cardinalities,1.0,2000);
      else if (params[1]==1) RotamerProtein->BestGBPSolution(Cardinalities,1.0,1000);
       else if (params[1]==2)
	{
         int numberconf;
         int finalnumberconf;
         numberconf = 5;

         unsigned** bestconf;
         double* energies; 
       	 bestconf = new unsigned*[numberconf];
         energies = new double[numberconf];  
         for(i=0;i<numberconf;i++)  bestconf[i] = new unsigned[vars];
       
         RotamerProtein->BestLoopySolutionMaxConfigurations(Cardinalities,1.0,2000, bestconf, energies, numberconf, &finalnumberconf);
         
         for(i=0;i<finalnumberconf;i++) 
          {
            for(j=0;j<vars;j++) cout<<bestconf[i][j]<<" ";
            cout<<energies[i]<<endl;
          }
         cout<<"The algorithm converges "<<endl;   
   	 delete[] energies;
	 for(i=0;i<finalnumberconf;i++) delete[] bestconf[i];
         delete[] bestconf;     
        }


     }
else if(params[0] == 5) 
     {  
     
        RotamerProtein->ApplyDEE(0);
        vars = RotamerProtein->NResiduesAfterDEE;
        cout<<" Number of vars    "<<vars<<endl;    
        Cardinalities  = new unsigned[vars];  
        for(i=0;i<vars;i++) Cardinalities[i] = RotamerProtein->NFinalRot[i];  
        if (params[1]==0) RotamerProtein->BestReducedLoopySolution(Cardinalities,1.0,2000);
        else if (params[1]==1) RotamerProtein->BestReducedGBPSolution(Cardinalities,1.0,2000);
        else if (params[1]==2)
	{
         int numberconf;
         int finalnumberconf;
         numberconf = 5;
         unsigned** bestconf;
         double* energies; 
         bestconf = new unsigned*[numberconf];
         energies = new double[numberconf];  
         for(i=0;i<numberconf;i++)  bestconf[i] = new unsigned[vars];
         //cout<<"Everything right until here"<<endl;     
         RotamerProtein->BestReducedLoopyMaxConfigurations(Cardinalities,1.0,2000, bestconf, energies, numberconf, &finalnumberconf);

         for(i=0;i<finalnumberconf;i++) 
          {
            for(j=0;j<vars;j++) cout<<bestconf[i][j]<<" ";
            cout<<energies[i]<<endl;
          }

	 delete[] energies;
	 for(i=0;i<finalnumberconf;i++) delete[] bestconf[i];
         delete[] bestconf;     
        }
     }
         
   if(vars>0 && params[0] < 3 ) //params[0] !=1 &&
    {
     //Max = -1.0*RotamerProtein->CalculateEnergy(bestsol);
     //cout<<"Max is"<<Max<<endl;
     TotEvaluations = 0;  succexp = 0;  meangen = 0; meaneval = 0;  i =0;  nsucc =0; alltime = 0; bestalltime = 0;  

     // RotamerProtein->SimplifyProteinFunction(); //ONLY WHEN THE STRUCTURE OF THE PROTEIN IS USED
     while (i<cantexp) //&& nsucc<1
      { 
       currentexp = i;	 
       TotEvaluations = 0; 
        if(params[0] == 2)   Decomposition();
        else runOptimizer(ExperimentMode,i);
       i++;
       //cout<<i<<"     "<<cantexp<<endl;     
      }  
     //  RotamerProtein->DeleteSimplified(); //ONLY WHEN THE STRUCTURE OF THE PROTEIN IS USED
     PrintStatistics(); 
     delete[] Cardinalities; 
     if(params[0] == 2)  delete[] indexcomponents; 
    }     
 
   if(params[0] >= 3)   
     {
        delete[] Cardinalities; 
      }
   delete RotamerProtein;
 

   fclose(file); 
 }
   fclose(streambestsol);
   fclose(stream);
   delete[] params; 
   return 1;
}      

//fid =  fopen('towrite.txt', 'W')
//for i=1:325,
//	fprintf(fid,'%d, ', MPC(i,1))
//  end,
// fclose(fid)


  
#endif  
