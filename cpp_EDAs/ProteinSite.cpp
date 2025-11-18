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
  
FILE *stream;  
FILE *file,*outfile;  	  
  

 
double meanlikehood[500]; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798};  
double SelInt; 
double ModVal = 1;

//double AllGen[100];    
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
int Dimension;




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
 
 if(Dimension==4) return;
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

void EvalProtein(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval;
 int start,pos;
 if (atgen==0) start=0;
 else start=nelit;
for(k=start; k < epsize;  k ++)  
 {
   pos = k; 
          
   FoldingProtein->CallRepair(epop->P[pos],vars);
 
    CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
    eval_local = 1;   
   epop->SetVal(pos,CurrentEval); 
   TotEvaluations += eval_local;
    
   }
 

}

void EvalProteinLocal(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval;
 int start;

 if (atgen==0) start=0;
 else start=nelit;

for(k=start; k < epsize; k ++)  
 {
      
   FoldingProtein->CallRepair(epop->P[k],vars);                  
   CurrentEval = FoldingProtein->EvalOnlyVectorModel(sizeProtein,epop->P[k]);
   eval_local = 1;
   TotEvaluations += eval_local;
   epop->SetVal(k,CurrentEval);
 }
  TotEvaluations += eval_local;
}



void EvalProteinLong(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval;
 int start,pos;

 if (atgen==0) start=0;
 else start=nelit;


for(k=start; k < epsize;  k ++)  
 {
   pos = k; // randomint(nelit);

          
   FoldingProtein->CallRepair(epop->P[pos],vars);
              
   CurrentEval = FoldingProtein->EvalOnlyVectorLong(sizeProtein,epop->P[pos]);
   eval_local = 1;   
   TotEvaluations += eval_local;
   epop->SetVal(pos,CurrentEval); 
   TotEvaluations += eval_local;
    
   }
}

void ProteinRandInit(Popul* epop)  
{
   epop->RandInit(); 
}

void ReadParameters()  
 
{ 
  int T,MaxMixtP,S_alph,Compl; 
 
stream = fopen( "EDAParams.txt", "r+" );  
        		    
 
	if( stream == NULL )  
		printf( "The file EDAParams.txt was not opened\n" );  
	else  
	{  
         fscanf( stream, "%s", &MatrixFileName);  
         fscanf( stream, "%d", &cantexp); // Number of Experiments  
	 fscanf( stream, "%d", &vars); //  Size of the protein  
 	 fscanf( stream, "%d", &auxMax); // Max. of the fitness function (Maximimization is considered)
  	 fscanf( stream, "%d", &T); // Percent of the Truncation selection or tournament size 
	 fscanf( stream, "%d", &psize); // Population Size  
         fscanf( stream, "%d", &func); // Type of HP model  (1: HP model, 2: functional protein)
	 fscanf( stream, "%d", &ExperimentMode); // Type of Experiment (SEE BELOW case instruction)  
 	 fscanf( stream, "%d", &Ntrees); // Number of Trees for the Mixture
	 fscanf( stream, "%d", &Maxgen);  // Max number of generations  
	 fscanf( stream, "%d", &printvals); // Best value in each generation is printed  
         fscanf( stream, "%d", &Dimension); // Params for function evaluation 
	 fscanf( stream, "%d", &seed); // seed  
        }  

 fclose( stream ); 

if(T>0) 
 {  
   div_t res; 
   res = div(T,5);  
   SelInt = Sel_Int[res.quot]; // Approximation to the selection intensity for truncation. 
 } 
 
Tour = 0;
EvaluationMode = func;
Elit = 0; 
InitTreeStructure = 2;
Nsteps = 50;
VisibleChoiceVar = 0;
BestElitism = 1;
 MaxMixtP = 500;
 S_alph = 0;
 StopCrit = 1;
 Prior = 1;
 Compl = 75;
 Coeftype = 1;
 CliqMaxLength = 10;
 MaxNumCliq = vars;
 OldWaySel = 0;
 Cycles = 1;

Trunc = T/double(100);  
Complex  = Compl/double(100);  
Max = auxMax;   
MaxMixtProb =MaxMixtP/double(100); 
S_alpha = S_alph/double(100); 
 
} 

 
int Selection() 
{ 
   int NPoints=0; 
 
  
           pop->TruncSel(selpop,TruncMax); 
           selpop->UniformProb(TruncMax,fvect); 
           NPoints = selpop->CompactPopNew(compact_pop,fvect); 	     
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
 
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
   
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);

      SymPop(pop); 
      NPoints = Selection(); 
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
       
  
  
     MyMarkovNet->UpdateModelProtein(typemodel,sizecliq); 
     FindBestVal(); 
     auxprob = MyMarkovNet->Prob(BestInd);  
         
 
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

     
         MyMarkovNet->GenPopProtein(Elit,pop,FoldingProtein,1.0);    
          
           i++;         
          MyMarkovNet->Destroy();    
  
   }  
 end_time();  

 if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;  
   
  if(printvals>0) 
   {           
    for(int ll=0;ll<1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 
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
    
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);

     NPoints = Selection(); 
    
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();  
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();  
     IntTree->MakeTree(IntTree->rootnode); 
     FindBestVal(); 
      
     IntTree->PutPriors(Prior,selpop->psize,1);
     sumprob = IntTree->SumProb(selpop,NPoints);  
         
      
     
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
 
  end_time(); 
  if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;   
  
 if(printvals>0) 
   {           
    for(int ll=0;ll<1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
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
     
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i); 
   
   SymPop(pop); 

   NPoints = Selection(); 
   MixtureInt->SetNpoints(NPoints,fvect);
   MixtureInt->SetPop(selpop);
   MixtureInt->MixturesInit(Type,InitTreeStructure,fvect,Complexity,0,0,0,0);
   MixtureInt->LearningMixture(Type);  
   FindBestVal();

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
       i++;  
  }  


 end_time();  

 if(printvals>0)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TotEval:"<<TotEvaluations<<" time "<<auxtime<<endl;    
 
  if(printvals>0) 
   {           
    for(int ll=0;ll<1;ll++)// NPoints 
    { 
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 

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
                    cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" T="<<Trunc<<" N="<<psize<<" Ntrees="<<Ntrees<<"  MaxGen="<<Maxgen<<" Suc.="<<succexp<<"  g="<<(auxmeangen+1)<<"  ave="<<meanfit<<" meaneval="<<meaneval<<" sigma="<<sigma<<" timebest="<<bestalltime<<" fulltime="<<alltime<<endl;                   
                   } 
                  else  
                   {  
		     cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" T="<<Trunc<<" N="<<psize<<"  Ntrees="<<Ntrees<<"  MaxGen="<<Maxgen<<" Suc.="<<0<<"  g="<<0<<"  ave="<<0<<" meaneval="<<meaneval<<" sigma="<<sigma<<" fulltime="<<alltime<<" Eval="<<(TotEvaluations/(1.0*cantexp))<<endl; 
                   } 

	
} 



void runOptimizer(int algtype,int nrun)  
{  
    int succ=-1; 

        
  switch(algtype)  
                     {                     
                       case 0: succ = Markovinit(Complex,1,Ntrees);break;  // Markov Network       1
                       case 1: succ = Intusualinit(Complex);break;
                       case 2: succ = MixturesIntAlgorithm(1,Cardinalities,Complex);break;// MT on dependencies 
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

  
int  main(){  
  int i,u;  
  unsigned ta;
  int* IntConf;
  char ProteinSeq[5000];

 ReadParameters();

       if (seed==0)   ta = (unsigned) time(NULL);  
       else ta = seed;
       srand(ta); 
 
      	file= fopen(MatrixFileName,"r");
       	if( file == NULL )  printf( "Error in the input parameters: the file", MatrixFileName,"  was not opened\n" );        
	else  fscanf (file,  "%s", ProteinSeq);
      	fclose(file);
        IntConf = new int[vars];
        u = 0;  
	i=0;
        while (i<2*vars && u<vars) 
          {
            if(ProteinSeq[i] == 'H') IntConf[u] = 0; 
            else if (ProteinSeq[i] == 'P') IntConf[u] = 1;
            else u--;
             u++;
            i++;
          }
        
 Cardinalities  = new unsigned[5000];  

 // The Protein configuration must be given as a sequence of zeros (H)  and ones (P)

// The program assumes the best configuration is not known and performs as many generations as determined
// sizeProtein  must be given

   sizeProtein = vars;
   if(Dimension == 2)
     {
       Card = 3;
       FoldingProtein = new HPProtein(sizeProtein,IntConf);
     }
   else if(Dimension == 3)
     {
      Card = 5;    
      FoldingProtein = new HPProtein3D(sizeProtein,IntConf);  
      Max = 1000;
    }
   
        
   for(u=0;u<5000;u++) Cardinalities[u] = Card;  


        TotEvaluations = 0;  
       	succexp = 0;  meangen = 0; meaneval = 0;  i =0;  nsucc =0; alltime = 0; bestalltime = 0;  
	while (i<cantexp) //&& nsucc<1
        { 	  
	  runOptimizer(ExperimentMode,i); 
         i++;
        }  
	PrintStatistics();

  delete[] IntConf;
  delete FoldingProtein; 
  delete [] Cardinalities; 
 return 1;
}      
