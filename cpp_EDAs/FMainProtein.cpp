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
#include "FProteinClass.h" 
#include "FFDA.h"  
#include "EDA.h" 
#include "AbstractTree.h"  
#include "MixtureTrees.h" 



  
FILE *stream,*streambest;  
FILE *file,*outfile;  	  
  

 
double meanlikehood[500]; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798};  
double SelInt; 
double ModVal = 1;

//double AllGen[100];    
//double statistics[1000][20];  
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
 
HPProtein* FoldingProtein;
int TotEvaluations;
int sizeProtein;
int EvaluationMode;
int currentexp;

double NextMax;
int nNextMax;

int  Optfound;
double   OtherCO;
int   OtherCM;
int   OtherNC;
double   CO;
int   CM;
int   NC;
int BestSol[23];
int* IntConf;
int a;
double** freqvect;
int NContacts;




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



void SymPop(Popul* epop)
{
  int i,j,k;
 
  if(params[1]==4) return;
 if(Card==3)
  {
   for(k=0; k < epop->psize; k ++) 
    {
      i=2;
      while(i<vars && epop->P[k][i]==1) i++;
      	{
         if (i<vars && epop->P[k][i]==2) 
 	  for(j=i;j<vars;j++) epop->P[k][j] = 2-epop->P[k][j];
        }  
    }     
  }

if(Card==5)
  { 
   for(k=0; k < epop->psize; k ++) 
    {

      i=2;
      while(i<vars && (epop->P[k][i]!=2 && epop->P[k][i]!=0) ) i++;
      	{
         if (i<vars && epop->P[k][i]==2) 
 	  for(j=i;j<vars;j++)  if(epop->P[k][j]==0 || epop->P[k][j]==2) epop->P[k][j] = 2-epop->P[k][j];
        }
   
     i=2;
      while(i<vars && (epop->P[k][i]!=3 && epop->P[k][i]!=4) ) i++;
      	{
         if (i<vars && epop->P[k][i]==4) 
 	  for(j=i;j<vars;j++) 
           {
              if(epop->P[k][j]==4)  epop->P[k][j] = 3;
              else if (epop->P[k][j]==3)  epop->P[k][j] = 4;
           }
        }     
    } 
  }
}



void ImproveProtein(Popul* epop,int initp, int nelit, int extent,int howmany)
{
 int k,eval_local;
 double CurrentEval,CurrentEval1;;
 int start,pos;

 for(k=0; k < howmany;  k ++)  //epop->psize
 {
   pos = k; //initp + randomint(nelit);
   cout<<k<<" "<<pos<<" ";
   CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
   eval_local = 1;
   

   //if(k<2*nelit) CurrentEval1 = FoldingProtein->TabuOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,extent,6);
   

   CurrentEval1 = FoldingProtein->TabuOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,extent,10);
   cout<< CurrentEval<<" --> "<<CurrentEval1<<endl;
   epop->SetVal(pos,CurrentEval1);
   TotEvaluations += eval_local;
 }

}

void EvalProtein(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval,CurrentEval1;;
 int start,pos,i;
 int* auxvector;
 int totinc = 0;
 if (atgen==0) start=0;
 else start=nelit;

 // auxvector = new int[sizeProtein];

 
for(k=start; k < epsize;  k ++)  
 {
   pos = k; // randomint(nelit);

          
   FoldingProtein->CallRepair(epop->P[pos],vars);
 
   //CurrentEval =  myrand();                
    CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
    eval_local = 1;   
    //CurrentEval1 = FoldingProtein->TabuOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5,10);
    //epop->SetVal(pos,CurrentEval1);
  
     /*
	{
	   if(myrand()< 0.01 )
             {     
         if(pos>=nelit)    CurrentEval1 = FoldingProtein->ProteinLocalOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5);
         else  CurrentEval1 = FoldingProtein->ProteinLocalOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,2);
               //CurrentEval1 = FoldingProtein->ProteinLocalPerturbation(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5);
               //CurrentEval1 = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
               //if(CurrentEval1<CurrentEval) cout<<"One example here"<<endl;
                totinc += (CurrentEval1>CurrentEval); 
                epop->SetVal(pos,CurrentEval1);
	     }
	       else  epop->SetVal(pos,CurrentEval); 
        }
    */
    //	else 
    // cout<<"CurrentVal is"<<CurrentEval<<endl;
   epop->SetVal(pos,CurrentEval); 
   TotEvaluations += eval_local;
    
   }
 
 //delete[] auxvector; 
//cout<<totinc<<" solutions are better now"<<endl;
 
  
 /*
for(k=start; k < epsize; k ++)  
 {
   // CurrentEval = FoldingProtein->EvalOnlyVectorModel(sizeProtein,epop->P[k]);
CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[k]);
    eval_local = 1;
    TotEvaluations += eval_local;
    epop->SetVal(k,CurrentEval);
 }

 */
/*
for(k=start; k < epsize; k ++)  
 {
   //pos = nelit+randomint(pop->psize-nelit);
   pos = k;
    CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
    eval_local = 1;
    TotEvaluations += eval_local;
    //CurrentEval1 = FoldingProtein->ProteinLocalOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local); 
    //CurrentEval1 = FoldingProtein->ProteinLocalPerturbation(sizeProtein,epop->P[pos],CurrentEval,&eval_local);
    //CurrentEval1 = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
    //cout<<CurrentEval<<"  "<<CurrentEval1<<endl;
    //if(CurrentEval1<CurrentEval) cout<<"One example here"<<endl;
    totinc += (CurrentEval1>CurrentEval); 
    //CurrentEval1 = FoldingProtein->ProteinLocalOptimizerSimple(sizeProtein,epop->P[pos],CurrentEval,&eval_local);
    epop->SetVal(pos,CurrentEval);
   }
//cout<<totinc<<" solutions are better now"<<endl;
*/
}

void EvalProteinLocal(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval,CurrentEval1;
 int start,totinc1,totinc2;

 if (atgen==0) start=0;
 else start=nelit;
 totinc1 = 0;
 totinc2 = 0;

for(k=start; k < epsize; k ++)  
 {

      
   FoldingProtein->CallRepair(epop->P[k],vars);                  
   CurrentEval = FoldingProtein->EvalOnlyVectorModel(sizeProtein,epop->P[k]);
   eval_local = 1;
   TotEvaluations += eval_local;
   epop->SetVal(k,CurrentEval);
   // cout<<k<<endl; 
 }
 
  TotEvaluations += eval_local;
}



void EvalProteinLong(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval,CurrentEval1;;
 int start,pos,i;
 int* auxvector;
 int totinc = 0;
 if (atgen==0) start=0;
 else start=nelit;

 // auxvector = new int[sizeProtein];
 
 
for(k=start; k < epsize;  k ++)  
 {
   pos = k; // randomint(nelit);

          
   FoldingProtein->CallRepair(epop->P[pos],vars);
 
                   
   CurrentEval = FoldingProtein->EvalOnlyVectorLong(sizeProtein,epop->P[pos]);
    eval_local = 1;   
    //CurrentEval1 = FoldingProtein->TabuOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5,10);
    //epop->SetVal(pos,CurrentEval1);
   TotEvaluations += eval_local;
  
     /*
	{
	   if(myrand()< 0.01 )
             {     
         if(pos>=nelit)    CurrentEval1 = FoldingProtein->ProteinLocalOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5);
         else  CurrentEval1 = FoldingProtein->ProteinLocalOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,2);
               //CurrentEval1 = FoldingProtein->ProteinLocalPerturbation(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5);
               //CurrentEval1 = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
               //if(CurrentEval1<CurrentEval) cout<<"One example here"<<endl;
                totinc += (CurrentEval1>CurrentEval); 
                epop->SetVal(pos,CurrentEval1);
	     }
	       else  epop->SetVal(pos,CurrentEval); 
        }
    */
    //	else 
    // cout<<"CurrentVal is"<<CurrentEval<<endl;
   epop->SetVal(pos,CurrentEval); 
   TotEvaluations += eval_local;
    
   }
}
void PerturbPop(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval;
for(k=3; k < epsize; k ++)  
 {
    FoldingProtein->CallBackTracking(epop->P[k]);
    CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[k]);
    epop->SetVal(k,CurrentEval);
 }
}

void MutatePop(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,j, eval_local;
 double CurrentEval;

 eval_local = 1;
for(k=1; k < epsize; k ++)  
 {
   
  for(j=2; j < vars; j ++)  
   {
      if(myrand()>ModVal)   epop->P[k][j] = randomint(3);
      //0.15
   }
  FoldingProtein->CallRepair(epop->P[k],vars); 
  CurrentEval = FoldingProtein->ProteinLocalPerturbation(sizeProtein,epop->P[k],CurrentEval,&eval_local,15);
  CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[k]);
  
  //int repairpoint = 2 + randomint(vars-2);
  //cout<<k<<" "<<repairpoint<<endl;  
  //FoldingProtein->DownCallRepair(repairpoint,epop->P[k],vars);

  //  FoldingProtein->CallRepair(epop->P[k],vars);
  //  
   CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[k]);
  epop->SetVal(k,CurrentEval);
  }
}

void ProteinRandInit(Popul* epop)  
{
  int k;
  epop->RandInit();
  
 for(k=0; k < epop->psize; k++) 
   {
     //epop->P[k][0] =0; epop->P[k][1] = 1;
     //FoldingProtein->CallBackTracking(epop->P[k]);
    //   cout<<k<<endl;   
  }

}

void ReadParameters()  
 
{ 
  int T,MaxMixtP,S_alph,Compl; 
 
stream = fopen( "FParam.txt", "r+" );  
        		    
	if( stream == NULL )  
		printf( "The file FParam.txt was not opened\n" );  
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
 
  //MyMarkovNet->SetProtein(FoldingProtein); // 18/4/2005


  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
    FoldingProtein->totnumboptint = 0;
    //EvaluationMode =  2*randomint(2) + 1;
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,0); //CAMBIAR 0 por i
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);

    double meancontact;
    if (i==0)  meancontact = (FoldingProtein->totnumboptint)/(1.0*psize);
    else meancontact = (FoldingProtein->totnumboptint)/(1.0*(psize));

    // cout<<i<<"->"<<meancontact<<endl; 

    //statistics[currentexp][i] = meancontact;


    //pop->Print(); 
      SymPop(pop);   
    
       NPoints = Selection(); 
       /*   
       int lll;
       for(lll=0;lll<TruncMax;lll++)  FoldingProtein->CollectStat(vars,selpop->P[lll],freqvect);  //Contact order statistics. 
       for(lll=0;lll<NContacts;lll++) 
        {
          cout<<a<<" "<<i<<" "<<freqvect[0][lll]<<" "<<freqvect[1][lll]/TruncMax<<endl;
          freqvect[1][lll] = 0;
	 }
       */

       /*
      ModVal = 0.45; //(NPoints/TruncMax);
    
      ImproveStop = div(i,100); 
    
   Clock = (i>0  &&  ImproveStop.rem==0);
  
       
   if(Clock==1 || NPoints<TruncMax) 
	{
          if(Clock==1) ImproveProtein(pop,Elit,2);
          else ImproveProtein(pop,Elit,50);
          SymPop(pop);   
          NPoints = Selection(); 
	  Clock = 0;
        }
   
       */
   /*
      if(NPoints<TruncMax*0.5 && Clock==0) 
	{
	 
          MutatePop(pop,Elit,pop->psize,i);
          NPoints = Selection(); 
          Clock = 1;
        } 
   

 if(NPoints<TruncMax*0.5) 
	{
          ImproveProtein(pop,Elit,500);
          NPoints = Selection(); 
	 }
   
   */ 
      MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
      MyMarkovNet->SetPop(selpop); 
                                
  if(printvals>1) 
   {           
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
       
   //cout<<"Initial marginals "<<endl; 
  
     MyMarkovNet->UpdateModelProtein(typemodel,sizecliq); 
     FindBestVal(); 
     //AllGen[i] += BestEval; 

     auxprob = MyMarkovNet->Prob(BestInd);  
         
 
     if(printvals>1)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" Elit"<<Elit<<" TotEval:"<<TotEvaluations<<" DifPoints:"<<NPoints<<endl; 
 
      if (BestEval==Max) 
         {
           fgen  = i;
           if(Optfound==0) for(int l=0;l<vars;l++) BestSol[l]=selpop->P[0][l];  
                	 
         } 
       else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {
		selpop->SetElit(Elit,pop);
		//Elit = NPoints; 
                 //SymPop(compact_pop);
                 //compact_pop->SetElit(Elit,pop);  
                 for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }

     
      //         if(NPoints<TruncMax*0.9)    MyMarkovNet->GenPopProtein(Elit,pop,FoldingProtein,1.0+TruncMax);       else   

    
     MyMarkovNet->GenPopProtein(Elit,pop,FoldingProtein,1.0);    
     // MyMarkovNet->GenPopProtein(Elit,pop,1.0);    //18-4-2005
          
           i++;         
          MyMarkovNet->Destroy();    
  
   }  

 if(NPoints>10) NPoints = 10;

 // ImproveProtein(selpop,0,NPoints,10000,NPoints);   
 end_time();  

 if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  
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
    FoldingProtein->totnumboptint = 0;
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,0);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);

    double meancontact;
    if (i==0)  meancontact = (FoldingProtein->totnumboptint)/(1.0*psize);
    else meancontact = (FoldingProtein->totnumboptint)/(1.0*(psize));

    //cout<<i<<"->"<<meancontact<<endl; 

    //     statistics[currentexp][i] = meancontact;

     //ImproveProtein(pop,Elit,4);
     //SymPop(pop); 
     //pop->Print();    
     NPoints = Selection(); 
     
     /*  
   ImproveStop = div(i,1000); 
    
   Clock = (i>0  &&  ImproveStop.rem==0);
  
       
   if(Clock==1 || NPoints<TruncMax) 
	{
          if(Clock==1) ImproveProtein(pop,Elit,10000);
          else ImproveProtein(pop,Elit,50);
          SymPop(pop);   
          NPoints = Selection(); 
	  Clock = 0;
        }
     */
       
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();  
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     //IntTree->CalMutInf();  
     IntTree->CalMutInfDeception();  
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

 
       

      if (BestEval==Max)   fgen  = i;	 
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
   

     i++;
  }  
  //cout<<BestEval<<" ";   selpop->Print(0); 

  //ImproveProtein(selpop,0,NPoints,1000000,NPoints);
  //ImproveProtein(selpop,0,NPoints,250000,NPoints);   
  end_time(); 
  if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  //cout<<BestEval<<endl; 
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
     
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i); 
   
   
    //ImproveProtein(pop,Elit,(psize-Elit),500,5); 
 
     
   //pop->Print(); 

   SymPop(pop); 

   NPoints = Selection(); 
   //if(i>0) ImproveProtein(selpop,0,Elit,100,2);  
  

   /*
   ImproveStop = div(i,5000); 
    
   Clock = (i>0  &&  ImproveStop.rem==0);
  
         
   if(Clock==1 || NPoints<TruncMax) 
	{
          if(Clock==1) ImproveProtein(pop,Elit,10);
          else ImproveProtein(pop,Elit,50);
          SymPop(pop);   
          NPoints = Selection(); 
	  Clock = 0;
        }
   */
   
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


    if (BestEval==Max) fgen  = i;	   
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
 //ImproveProtein(selpop,NPoints,10000);

 if(NPoints>10) NPoints = 10;
 //ImproveProtein(selpop,0,NPoints,2500,NPoints);   

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




int PopulationLocal(double Complexity, int typemodel, int sizecliq)  //In this case, complexity is the threshold for chi-square 
{  
  int i,fgen;  
  double auxprob;     
     
  init_time();
  InitPopulations(); 
 
  
  Popul* aux_pop;
  aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
  i=0;  BestEval = Max-1; NPoints = psize; fgen = -1;   
     
 
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
    
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);
  
       SymPop(pop);   
       NPoints = Selection(); 
                                
  if(printvals>1) 
   {           
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
  
     if(printvals>1)      cout<<"Gen :"<<i<<" Best:"<<BestEval<<" Elit"<<Elit<<" TotEval:"<<TotEvaluations<<" DifPoints:"<<NPoints<<endl; 
 
        
      if (BestEval==Max)   fgen  = i;	 
      else 
          { 
           if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);             
           else 
              {
		selpop->SetElit(Elit,pop);
	        for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
               }
          }
          ImproveProtein(pop,0,NPoints,1000,psize);   
          i++;         
          
   }  

 end_time();  

 if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  
  
  if(printvals>0) 
   {           
    for(int ll=0;ll<NPoints;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 
 
  delete aux_pop; 
  //delete big_pop;
  DeletePopulations(); 
 
  return fgen;  
}  
 

void PrintStatistics() 
{  
  int i;
  double auxmeangen,meanfit,sigma; 
  int j,auxNC;
  //cout<<endl; // This is for the array of contact orders. 
  sigma = 0;
                   meaneval /=  cantexp; 
                   alltime  =  alltime/(1.0*cantexp); 
		   for (i=0;i<cantexp;i++) 
                   {
                    sigma += (meanlikehood[i] - meaneval)*(meanlikehood[i] - meaneval);
                    //cout<<sigma<<endl;
                   } 
                   sigma = sigma/(cantexp-1);
		  /* 
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
		   

          for(int j=1 ;j<cantexp;j++) 
            for(int k=0 ;k<Maxgen;k++)
              {
                if (statistics[j][k]==0) statistics[j][k]=statistics[j][k-1];
                statistics[0][k] +=  statistics[j][k];
	       }
               
	  //for(int k=0 ;k<Maxgen;k++) cout<< statistics[0][k]/cantexp<<" ";
          //cout<<endl;

	  //for(int ll=0 ;ll<Maxgen;ll++)   cout<<AllGen[ll]/(1.0*cantexp)<<" "<<endl; 


		  */

		   cout<<a<<" "; // Instance number
    if (succexp>0)  
                   {  
                    auxmeangen = meangen/succexp;
                    if (BestElitism)  
                         meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;     
                    else meanfit = (auxmeangen+1)*(psize-1) + 1; 
                   for (j=0;j<23;j++) cout<<IntConf[j]<<" ";
                   cout<<FoldingProtein->Max<<" "<<FoldingProtein->NextMax<<" "<<FoldingProtein->nNextMax<<" ";
                   for (j=0;j<23;j++) cout<<BestSol[j]<<" ";
                   cout<<succexp<<" "<<(auxmeangen+1)<<" "<<sigma<<" "<<CO<<" "<<OtherCO/(1.0*nsucc)<<" "<<CM<<" "<<OtherCM/(1.0*nsucc)<<" "<<NC<<" "<<OtherNC/(1.0*nsucc)<<endl;                   
                   } 
                  else  
                   {  	        
                   for (j=0;j<23;j++) cout<<IntConf[j]<<" ";
                   cout<<FoldingProtein->Max<<" "<<FoldingProtein->NextMax<<" "<<FoldingProtein->nNextMax<<" ";
                   for (j=0;j<23;j++) cout<<"0 ";
                   cout<<0<<" "<<0<<" "<<sigma<<" "<<0<<" "<<OtherCO/(1.0*nsucc)<<" "<<0<<" "<<OtherCM/(1.0*nsucc)<<" "<<0<<" "<<OtherNC/(1.0*nsucc)<<endl;              
                  } 


} 



void runOptimizer(int algtype,int nrun)  
{  
    int succ=-1; 
    int auxNC;
          
  switch(algtype)  
                     {                     
                       case 0: succ = Markovinit(Complex,1,Cycles);break;  // Markov Network       1
                       case 1: succ = Intusualinit(Complex);break;
                       case 2: succ = MixturesIntAlgorithm(1,Cardinalities,Complex);break;// MT on dependencies 
                       case 3: succ = PopulationLocal(Complex,1,Cycles);break;  // Markov Network       1
                     } 

  

    if (succ>-1)  
   { 
       
       succexp++; 
       meangen += succ;    
       bestalltime +=auxtime;    
       if (Optfound == 0)
        {
	  CO = FoldingProtein->ContactOrderVector(sizeProtein,&NC,BestInd);
          CM = FoldingProtein->CompactnessVector(sizeProtein,BestInd);        
        }  
        Optfound = 1;    
   } 
   else 
   {
      nsucc++;
      OtherCO += FoldingProtein->ContactOrderVector(sizeProtein,&auxNC,BestInd);
      OtherCM += FoldingProtein->CompactnessVector(sizeProtein,BestInd);
      OtherNC += auxNC;
   }
    //cout<<" i- "<< nrun << " Contact order "<<FoldingProtein->ContactOrderVector(sizeProtein,BestInd)<<endl;     
    //cout<<FoldingProtein->ContactOrderVector(sizeProtein,BestInd)<<" ";       
   alltime += auxtime;  
   meaneval += BestEval; 
   meanlikehood[nrun] = BestEval;  
} 

  

void ReadFileOpt(FILE * stream1,int aa, unsigned int* vector, int* IntConf10)  
{  
  int i,j,auxint;
  long position;        
  double auxfloat;
       IntConf = IntConf10;
       Card = 3;
       
       sizeProtein = 23;
for(i=0; i<2800;i++)   { 
  //fseek(stream1,152*i,0);       
        
       fscanf(stream1, "%d", &auxint);
       for (j=0;j<23;j++) 
        {
         fscanf( stream1, "%d", &auxint); 
         IntConf[j] = auxint;
	 cout<<IntConf[j]<<" ";    
        }
       cout<< endl;
       fscanf(stream1, "%d", &auxint);
       Max = -1.0*auxint;    
       fscanf( stream1, "%d", &auxint); 
       NextMax = -1.0*auxint ; 
       fscanf( stream1, "%d\n", &nNextMax);
       //cout<<Max<<" "<<NextMax<<" "<<nNextMax<<endl; 
       for (j=0;j<23;j++) 
        {
         fscanf(stream1, "%d", &vector[j]);
         cout<<vector[j]<<" ";        
        }
         cout<< endl;
        FoldingProtein = new HPProtein(23,IntConf,Max,NextMax,nNextMax);
        CO = FoldingProtein->ContactOrderVector(sizeProtein,&NC,vector);
        CM = FoldingProtein->CompactnessVector(sizeProtein,vector);       
        cout<<i<<" "<<CO<<" "<<CM<<" "<<NC<<endl;
      delete FoldingProtein;
       fscanf(stream1, "%d", &auxint);

        for (j=0;j<8;j++) 
        {
         fscanf( stream1, "%f", &auxfloat);
         //cout<<auxfloat;        
        }
	//cout<<endl;
	cout<<"ftell "<<ftell(stream1)<<endl;     
  position = ftell(stream1);
 }
}


int  main(){  


 int i;  
 unsigned ta = (unsigned) time(NULL);  


 //    ta = 1051750320;
 srand(ta); 
 //cout<<"seed"<<ta<<endl; 
 params = new int[3]; 
 ReadParameters(); 
Cardinalities  = new unsigned[5000];  
 EvaluationMode = params[0];
int k,j,u;


 
   int  IntConf1[23] = {1,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,1,1,0,0}; // 1 - 20 
   int  IntConf2[23] ={1,0,1,1,0,1,1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,1,0}; // 2 - 17
   int  IntConf3[23] ={0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0,0}; // 3 - 16
   int  IntConf4[23] ={0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0}; // 4 - 20
   int  IntConf5[23] ={1,0,1,1,1,1,1,1,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0}; // 5 - 17
   int  IntConf6[23] ={0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0}; // 6 - 13
   int  IntConf7[23] ={1,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0}; // 7 - 26
   int  IntConf8[23] ={0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,0,0}; // 8 - 16
   int  IntConf9[23] ={1,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,1,1,1,1,0}; // 9 - 15
   int  IntConf10[23] ={0,1,0,1,0,1,1,1,1,1,0,0,1,1,1,0,1,0,1,0,1,0,0}; // 10 - 14
   int  IntConf11[23] ={1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,0}; // 11 - 15
 


   /*
   int  IntConf1[23] = {1,0,1,0,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0}; // 1 - 11
   int  IntConf2[23] = {1,0,1,0,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0}; // 2 - 11 
   int  IntConf3[23] = {1,0,1,0,1,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0}; // 3 - 16
   int  IntConf4[23] = {1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0}; // 4 - 14
   int  IntConf5[23] = {1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0}; // 5 - 14
   int  IntConf6[23] = {0,1,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,0,1,0,0,0,0}; // 6 - 15
   int  IntConf7[23] = {1,0,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}; // 7 - 16
   int  IntConf8[23] = {0,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,0,0}; // 8 - 18
   int  IntConf9[23] = {0,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,1,0}; // 9 - 18
    */


  int  IntConfa[20] = {0,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,1,0}; 
  int  IntConfb[24] = {0,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0}; 
  int  IntConfc[25] ={1,1,0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0};
  int  IntConfd[36] ={1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1};
  int  IntConfe[48] ={1,1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0};
  int  IntConff[50] = {0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0} ;
  int  IntConfg[60] ={1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,1};
  int  IntConfh[64] = {0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0};
  //  int  IntConfi[80] ={1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0};
  int  IntConfj[85] ={0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0};
  int  IntConfk[100]= {1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0};
  int  IntConfl[100] = {1,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,1,1,1,1,1,0,0,0};  // ModelB 
   int  IntConfm[80] ={1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0};
 int  IntConfn[58] = {1,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0};
 int  IntConfo[103] = {1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1};
 int  IntConfp[124] = {1,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0};
 int  IntConfq[136] = {0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,0,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1};

unsigned int* vector;
vector = new unsigned int[23];

//freqvect = new double*[2];
//freqvect[0] = new double[23];
//freqvect[1] = new double[23];

  //  unsigned int vector[64] = {0,0,0,2,0,0,0,2,0,0,0,2,0,0,1,2,0,2,2,2,0,2,2,2,0,2,2,2,2,1,2,1,2,0,2,2,2,0,2,0,0,0,2,0,0,0,2,0,0,0,0,0,0,2,1,2,2,0,2,2,2,0,2,2};

 
  //  unsigned int vector60[60] = {0,0,1,1,1,1,2,1,0,2,2,1,1,0,0,0,1,2,0,0,0,0,2,1,0,0,0,0,2,2,2,1,0,2,2,0,0,1,0,2,1,2,2,0,0,2,1,0,1,1,2,2,1,1,0,2,1,2,0,0};
  double eval;
  char auxchar;

 int ANTrees[5] = {2,4,1,4,3};
 

   FILE *stream;
   //stream = fopen( "2dfmp.dat", "r+" ); 
    stream = fopen( "2dfmp.dat", "r+" ); 
   if( stream == NULL )   printf( "El file2dfmp.dat no fue encontrado \n" );
  
      streambest = fopen( "bestsolutionsnew.dat", "r+" ); 
   if( streambest == NULL )   printf( "El bestsolutions.dat no fue encontrado \n" ); 
   
 //sizeProtein = vars;



 //for(Cycles=2; Cycles<=3;Cycles++)
   // for(int a=params[2]; a<=params[2]+11;a++)

int kk;
for(kk=params[2]; kk<15575;kk++) //15575;a++)
 {
   Optfound = 0;
   OtherCO = 0;
   OtherCM = 0;
   OtherNC = 0;
   CO = 0;
   CM = 0;
   NC = 0;
   a = kk;
   
   // ReadFileOpt(stream,a,vector,IntConf10);

 /*
   if(a==1)
    {
       IntConf = IntConfa;
       sizeProtein = 20;
       Max = 9.0;
    }
   else  if(a==2) 
    {
     IntConf = IntConfb;
     sizeProtein = 24; 
     Max = 9.0;
    }
   else if(a==3) 
    {
    IntConf = IntConfc;
    sizeProtein = 25; 
     Max = 8.0;
    }
   else if(a==4) 
    {
     IntConf = IntConfd;
     sizeProtein = 36; 
     Max = 14.0;
    }
   else if(a==5) 
    {
     IntConf = IntConfe;
     sizeProtein = 48;
     Max = 23.0;
    }
   else if(a==6) 
    {
     IntConf = IntConff;
     sizeProtein = 50; 
     Max = 21.0;
    }
   else if(a==7) 
    {
     IntConf = IntConfg;
     sizeProtein = 60;
     Max = 36.0;
    }
  else if(a==8) 
    {
     IntConf = IntConfh;
     sizeProtein = 64; 
     Max = 42.0;
    }
 else if(a==9) 
    {
     IntConf = IntConfj;
     sizeProtein = 85; 
     Max = 53.0;
    }
   else if(a==10) 
    {
     IntConf = IntConfk;
     sizeProtein = 100;
     Max = 48.0;
    }
  else if(a==11) 
    {
     IntConf = IntConfl;
     sizeProtein = 100; 
     Max = 52.0;
    }
else if(a==12) 
    {
     IntConf = IntConfm;
     sizeProtein = 80; 
     Max = 1000.0;
    }
   else if(a==13) 
    {
     IntConf = IntConfn;
     sizeProtein = 58;
     Max = 1000.0;
    }
  else if(a==14) 
    {
     IntConf = IntConfo;
     sizeProtein = 103; 
     Max = 1000.0;
    }
  else if(a==15) 
    {
     IntConf = IntConfp;
     sizeProtein = 124; 
     Max = 1000.0;
    }
   else if(a==16) 
  {
     IntConf = IntConfq;
     sizeProtein = 136; 
     Max = 1000.0;
    }
 if(a>=20)   sizeProtein = 23;

   */
 /*
 if(a==20)
    {
       IntConf = IntConf1;
        Max = 11.0; 
    }
 else if(a==21)
    {
       IntConf = IntConf2;
        Max = 11.0; 
    }
else if(a==22)
    {
       IntConf = IntConf3;
        Max = 14.0; 
    }
 else if(a==23)
    {
       IntConf = IntConf4;
        Max = 14.0; 
    }
else if(a==24)
    {
       IntConf = IntConf5;
        Max = 14.0; 
    }
 else if(a==25)
    {
       IntConf = IntConf6;
        Max = 15.0; 
    }
else if(a==26)
    {
       IntConf = IntConf7;
        Max = 16.0; 
    }
 else if(a==27)
    {
       IntConf = IntConf8;
        Max = 18.0; 
    }
else if(a==28)
    {
       IntConf = IntConf9;
        Max = 18.0; 
    }
 */

   /*

 if(a==20)
    {
       IntConf = IntConf1;
        Max = 20.0; 
    }
 else if(a==21)
    {
       IntConf = IntConf2;
        Max = 17.0; 
    }
else if(a==22)
    {
       IntConf = IntConf3;
        Max = 16.0; 
    }
 else if(a==23)
    {
       IntConf = IntConf4;
        Max = 20.0; 
    }
else if(a==24)
    {
       IntConf = IntConf5;
        Max = 17.0; 
    }
 else if(a==25)
    {
       IntConf = IntConf6;
        Max = 13.0; 
    }
else if(a==26)
    {
       IntConf = IntConf7;
        Max = 26.0; 
    }
 else if(a==27)
    {
       IntConf = IntConf8;
        Max = 16.0; 
    }
else if(a==28)
    {
       IntConf = IntConf9;
        Max = 15.0; 
    }
else if(a==29)
    {
       IntConf = IntConf10;
        Max = 14.0; 
    }
 else if(a==30)
    {
        IntConf = IntConf11;
        Max = 15.0; 
    }
   */
 //psize =  5000;

 sizeProtein = vars;
 //cout<<a<<" "<<vars<<endl;

  //for(int ll=0;ll<Maxgen;ll++)  AllGen[ll] =   0;

   if(params[1] == 2)
     {        
       Card = 3;
       FoldingProtein = new HPProtein(sizeProtein,IntConf);
     //FoldingProtein = new HPProtein3D(sizeProtein,IntConf);  
     }
   else if(params[1] == 3)
     {
      Card = 5;    
      FoldingProtein = new HPProtein3D(sizeProtein,IntConf);  
      Max = 1000;
    }
    else if(params[1] == 4)
     {
      Card = 3;    
      FoldingProtein = new HPProtein3Diamond(sizeProtein,IntConf);  
      Max = 1000;
    }
  if(params[1] == 5)
     {        
        
       long position;
       IntConf = IntConf10;
       Card = 3;

       /*
       fscanf( streambest, "%d", &a); // InstanceNumber;

        for (j=0;j<22;j++) 
        {
	  fscanf( streambest, "%d", &vector[j]);  //Best solution    
	  // cout<<IntConf[j]<<endl;
        }
       fscanf(streambest, "%d\n", &vector[22]); 
       */
       
       fseek(stream,47*(a),0);
       //position = ftell(stream);
       //cout<<"Position is "<<a<<endl;

     

       for (j=0;j<23;j++) 
        {
         fscanf( stream, "%c", &auxchar);
         IntConf[j] = 1-(auxchar-48);
         //cout<<auxchar<<endl;
        }
       fscanf( stream, "%c", &auxchar);
       fscanf( stream, "%c", &auxchar);
       fscanf( stream, "%d", &j);
       Max = -1.0*j; //OJO QUITAR +1
       fscanf( stream, "%d", &j);
       fscanf( stream, "%d", &j); 
       NextMax = -1.0*j ; 
       fscanf( stream, "%d\n", &nNextMax);
       //cout<<Max<<" "<<NextMax<<" "<<nNextMax<<endl; 
       FoldingProtein = new HPProtein(sizeProtein,IntConf,Max,NextMax,nNextMax);
       /*
     for(u=0;u<23;u++)
       for(i=0;i<23;i++)  FoldingProtein->optmatrix[u][i]=-1;

       NContacts = FoldingProtein->VectorContactMatrix(sizeProtein,vector,freqvect); 
       */
      }

 
 
  for(u=0;u<5000;u++) Cardinalities[u] = Card;  
  
 
  /*     
for( j=1 ;j<cantexp;j++) 
        for( k=0 ;k<Maxgen;k++)
          statistics[j][k] = 0;
  
   

  for(k=0;k<15;k++)
     for(j=0;j<15;j++)  
       {
         FoldingProtein->energymatrix[k][j] = 0;
         FoldingProtein->freqmatrix[k][j] = 0; 
       }
  */

succexp =0; nsucc = 0;

// while( psize <= 20000 && succexp<90)
 {
   //  for(int k=0; k<=1;k++)
     {
       // Ntrees = ANTrees[k]; 
        Cycles = Ntrees;
        TotEvaluations = 0;  
       	succexp = 0;  meangen = 0; meaneval = 0;  i =0;  nsucc =0; alltime = 0; bestalltime = 0;  
	while (i<cantexp) //&& nsucc<1
        { 
          currentexp = i;	  
	  runOptimizer(ExperimentMode,i);
  /*
    for(k=0;k<11;k++)
    {
     for(j=0;j<11;j++)  
       {
         if(FoldingProtein->freqmatrix[k][j]>0) cout<<" "<<(-1.0*FoldingProtein->energymatrix[k][j]/FoldingProtein->freqmatrix[k][j])<<" ";
	  else cout<<" 10.0 ";
                
       }
     cout<<";"<<endl;          
    }
	  */  	  
        i++;    
        }  
        //cout<<i<<"     "<<cantexp<<endl;     
	PrintStatistics();             
      }
     // psize +=  5000;
 }   





   /*
  // for(EvaluationMode=1;EvaluationMode<=2;EvaluationMode++)
    //   for(Cycles=4;Cycles<=4;Cycles++)
  {
    TotEvaluations = 0;  
    succexp = 0;  meangen = 0; meaneval = 0;  nsucc = 0; i =0;     
    while (i<cantexp )  
      { 	 
  	 runOptimizer();
         i++;
       }       
     PrintStatistics();
  } 
   */
  delete FoldingProtein; 
 } 
 delete [] params; 
 delete [] Cardinalities;
 fclose( stream ); 
 fclose(streambest);
 delete[] vector;
 //delete[] freqvect[0];
 //delete[] freqvect[1];
 //delete[] freqvect;
 return 1;
}      



/*



 if(params[1] == 5)
     {        
       long position;
       IntConf = IntConf10;
       Card = 3;
       fseek(stream,47*a,0);
       //position = ftell(stream);
       //cout<<"Position is "<<position<<endl;
       for (j=0;j<23;j++) 
        {
         fscanf( stream, "%c", &auxchar);
         IntConf[j] = 1-(auxchar-48);
         //cout<<IntConf[j]<<endl;
        }
       fscanf( stream, "%c", &auxchar);
       fscanf( stream, "%c", &auxchar);
       fscanf( stream, "%d", &j);
       Max = -1.0*j;
       fscanf( stream, "%d", &j);
       fscanf( stream, "%d", &j); 
       NextMax = -1.0*j;
       fscanf( stream, "%d\n", &nNextMax);
       //cout<<Max<<" "<<NextMax<<" "<<nNextMax<<endl; 
       FoldingProtein = new HPProtein(sizeProtein,IntConf,Max,NextMax,nNextMax);
      }
*/
  /*
eval = FoldingProtein->EvalOnlyVector(sizeProtein,vector);
 cout<< eval<<endl;
 for(i=2;i<vars;i++) vector[i] = 2-vector[i];
eval = FoldingProtein->EvalOnlyVector(sizeProtein,vector);  
 cout<< eval<<endl;
  */

 /*  
 if(a==1)
    {
       IntConf = IntConf1;
       Max = 20;
    }
   else  if(a==2) 
    {
     IntConf = IntConf2;
     Max = 17.0;
    }
   else if(a==3) 
    {
    IntConf = IntConf3;
     Max = 16.0;
    }
   else if(a==4) 
    {
     IntConf = IntConf4;
     Max = 20.0;
    }
   else if(a==5) 
    {
     IntConf = IntConf5;
     Max = 17.0;
    }
   else if(a==6) 
    {
     IntConf = IntConf6;
     Max = 13.0;
    }
   else if(a==7) 
    {
     IntConf = IntConf7;
     Max = 26.0;
    }
   else if(a==8) 
    {
     IntConf = IntConf8;
     Max = 16.0;
    }
   else if(a==9) 
    {
     IntConf = IntConf9;
     Max = 15.0;
    }
  else if(a==10) 
    {
     IntConf = IntConf10;
     Max = 14.0;
    }
   else if(a==11) 
    {
     IntConf = IntConf11;
     Max = 15.0;
    }
   */



