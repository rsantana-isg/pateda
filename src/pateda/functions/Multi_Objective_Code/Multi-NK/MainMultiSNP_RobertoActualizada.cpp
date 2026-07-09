#ifndef __MAIN_H 
#define __MAIN_H 
#include <math.h> 
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <iostream> 
#include <fstream> 
#include "auxfunc.h"  
#include "Popul.h"  
//#include "Treeprob.h"  
//#include "IntTreeprob.h" 
#include "EDA.h" 
#include "AbstractTree.h"  
#include "FDA.h"  
#include "MixtureTrees.h" 
#include "RotamerClass.h" 
#define itoa(a,b,c) sprintf(b, "%d", a) 
  
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
char MatrixFileName[60]; 
double meaneval;  
double BestEval; 
int TruncMax; 
int NPoints;  
unsigned int *BestInd; 
MultiPopul *pop,*selpop,*elitpop,*compact_pop; 
double *fvect; 
int  nsucc;
int MAXD;
double MINC,MINM;
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
int sizeProtein;
int EvaluationMode;
int currentexp;
int currentinstance;
int* indexcomponents;
double temp = 1.0;
int MaxIter = 200; //Iterations for loopy BP
int T,MaxMixtP,S_alph,Compl; 

SNPs* AllSNPs;
unsigned** matrixSNPs;
int NObj;
SNPSets* ContainerSNPSets;



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


void ProteinRandInit(MultiPopul* epop)  
{
  //  int k;
  epop->RandInit();
  
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
     else if (Tour==4)  
         {  

          
           pop->ParetoRankingSel(selpop,TruncMax);
           selpop->UniformProb(TruncMax,fvect); 
           NPoints = selpop->CompactPopNew(compact_pop,fvect);  
	 
	 } 



  if (Tour==1 || Tour==2 || Tour==3 || (Tour==0 && Elit>TruncMax)) pop->TruncSel(elitpop,Elit);  
    
   return NPoints; 
} 
 
inline void FindBestVal() 
{    
  int i;
 
      if(Elit && Tour != 0 && Tour != 4)  
          { 
            BestEval =elitpop->Evaluations[0]; 
            for(i=0;i<vars;i++) BestInd[i] = elitpop->P[0][i]; 
            //BestInd = elitpop->P[0]; 
	  } 
      else if(Tour==0 || Tour==4) 
      {  
        BestEval = selpop->Evaluations[0];
        for(i=0;i<vars;i++) BestInd[i] = selpop->P[0][i];  
        //BestInd = selpop->P[0]; 
      } 
      else  
          { 
	   int auxind =  pop->FindBestIndPos(0);  
           //BestInd =  pop->P[auxind]; 
           for(i=0;i<vars;i++) BestInd[i] = pop->P[auxind][i]; 
           BestEval = pop->Evaluations[auxind]; 
          } 
} 
 
inline void InitPopulations() 
{
  //  int i; 
 

 if (Tour==0 ||Tour==4) 
   { 
     TruncMax = int(psize*Trunc);  
   
     if (BestElitism)  Elit = TruncMax;   //Only for Trunc Selection  

   

     selpop = new MultiPopul(TruncMax,vars,Elit,Cardinalities,NObj);  
   }  
  else selpop = new MultiPopul(psize,vars,Elit,Cardinalities,NObj); 
 
 
  
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) elitpop = new MultiPopul(Elit,vars,Elit,Cardinalities,NObj); 

  pop = new MultiPopul(psize,vars,Elit,Cardinalities,NObj);  


 compact_pop = new MultiPopul(psize,vars,Elit,Cardinalities,NObj);  


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
 


void MultiEvalSNPs(MultiPopul* epop,int nelit, int epsize, int atgen)
{
 double CurrentEval[10];
 int i,start,pos; 

if (atgen==0) start=0;
else start=nelit;


for(pos=start; pos < epsize;  pos ++)  
 {
   //cout<<pos<<endl;
   //for(i=0;i<vars;i++) CurrentEval= CurrentEval - 1.0*epop->P[pos][i];
  
   ContainerSNPSets->MultiCalculateNumberTaggedSNPs(epop->P[pos],CurrentEval);
   epop->SetVals(pos,CurrentEval); 
   //cout<<"Pos "<<pos<<" :";
   //for(i=0;i<3;i++) cout<<CurrentEval[i]<<" ";
   //cout<<endl; 

   TotEvaluations ++;    
  }

}


int Markovinit(double Complexity, int typemodel, int sizecliq)  //In this case, complexity is the threshold for chi-square 
{  
  int i,fgen;  
  double auxprob;     
  DynFDA* MyMarkovNet;  
   
 
  InitPopulations(); 
 
  LearningType=3;
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
 

  i=0;  fgen = -1;  auxprob =0; BestEval  = Max -1; NPoints=TruncMax; 

  MultiPopul* aux_pop;
  aux_pop = new MultiPopul(psize,vars,Elit,Cardinalities,NObj);  
  
  NPoints = psize;
 
  while (i<Maxgen && BestEval<Max && NPoints>100)  
  {  
    
    MultiEvalSNPs(pop,Elit,psize,i); 
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
 
 
  InitPopulations(); 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  
  NPoints = 10000;

  //cout<<"Pass 1"<<"  "<<TruncMax<<"  "<< Elit<<endl;
  
   while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  

 

   MultiEvalSNPs(pop,Elit,psize,i); 

    
  //pop->Print(); 
   
     NPoints = Selection();  

     

     IntTree->SetNPoints(NPoints);
 
     IntTree->rootnode =   IntTree->RandomRootNode(); // IntTree->FindRootNode();  
    
  IntTree->CalProbFvect(selpop,fvect,NPoints,0);        
  
     //IntTree->CalMutInf(); 
     IntTree->CalMutInf(matrixSNPs); //THIS IS ONLY FOR STRUCTURAL INFO
 
     IntTree->MakeTree(IntTree->rootnode);
 
     //IntTree->PrintModel(); // This is only to analyze the structure 
     FindBestVal();   
   
     // IntTree->PrintMut();
     
     IntTree->PutPriors(Prior,selpop->psize,1);
     sumprob = IntTree->SumProb(selpop,NPoints);  
     //auxprob = IntTree->Prob(BestInd); 
     //selpop->Print(0); 
   
               
if(printvals>1) 
   {           
 
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
      cout<<"Gen :"<<i<<"  BestInd:"<<ll<<"  :: ";
      for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
      for(int l=0;l<NObj;l++) cout<<selpop->MultiEvaluations[ll][l]<<" "; 
      cout<<endl;
    }
 
   }

      if (BestEval>=Max)   fgen  = i;	 
      else 
          { 
           if (Tour==1 ||  Tour==2 || Tour==3 || (Tour==0 && Elit>TruncMax)) elitpop->SetElit(Elit,pop);                       else 
              {
		selpop->SetElit(Elit,pop);
	        //for(int ll=0;ll<Elit;ll++)  
		//for(int mm =0;mm <NObj;mm++)  
		//pop->MultiEvaluations[ll][mm]=selpop->MultiEvaluations[ll][mm];               
              }
          }
  
     IntTree->GenPop(Elit,pop); 
     

     if(params[1]) pop->Mutation(Elit,0.015);
     i++;
 
  }  
 
 if(printvals>0) 
   {           
    for(int ll=0;ll<printvals;ll++)// NPoints 
    { 
     cout<<"CurrentInstance: " << currentinstance<<". CurrenExp: "<<currentexp<<". Generation:  "<<i<<". "; 
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

  InitPopulations(); 
  MixtureInt = new MixtureIntTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior,Cardinalities);
  i=0; auxprob = 0; BestEval = Max-1; NPoints = 100; fgen = -1;  


 while (i<Maxgen && BestEval<Max && NPoints>10)  //&& oldlikehood != likehood)  
  { 
      MultiEvalSNPs(pop,Elit,psize,i); 

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


void runOptimizer(int algtype,int nrun)  
{  
    int succ=-1; 

        
  switch(algtype)  
                     {                     
                       case 0: succ = Markovinit(Complex,1,Cycles);break;  // Markov Network       1                
                       case 1: succ = Intusualinit(Complex);break;
                       case 2: succ = MixturesIntAlgorithm(1,Cardinalities,Complex);break;// MT on dependencies 
                      } 

   end_time();  

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

// ./snps ENm010.CEU.map.txt ENm010.CEU.hap 1 1 1 ENm010.CEU.tags.pairs ENm010.CEU.tags.trios Prueb 100 15 10  1 2 2333

int main(int argc, char *argv[ ])
{
 int i,j,k;  
 unsigned ta;
  //ta = (unsigned) time(NULL);  
 ta = 3;

 //ALEX 
 //srand(ta); 
 //ALEX

  //char auxstring[60];
char* PairsFilename; 
char* TriplesFilename; 
char* TagsFilename;
char* MapFilename;
char* HapFilename;

//string PairsFilename= ""; 
//string TriplesFilename = ""; 
//string TagsFilename = "";

FILE *streamPairs, *streamTriples, *streamTags; 
char *aux=  new char[10];
params = new int[3];  
//MatrixFileName = "newviewtrees.txt";  

 // params[0] = 1;
 // params[1] = 2;

// PairsFilename[0] = 0;
// TriplesFilename[0] = 0;
// TagsFilename[0] = 0;
 
 if( argc != 15 ) {
    //std::cout << "Usage: " <<"./snps PairsFilename TriplesFilename TagsFilename(prefix) cantexp  EDA{0:Markov, 1:Tree  2:Mixture} TypeFunction{0: MinCover} psize T Maxgen BestElitism VerboseLevel Seed" << std::endl;
    std::cout << "Usage: " <<"./snps MapFilename HapFilename MAXD MINC MINM PairsFilename TriplesFilename TagsFilename(prefix) psize T Maxgen BestElitism VerboseLevel Seed" << std::endl;
    std::cout << "       Please read the README file." << std::endl;
    exit(1);
}

 MapFilename = argv[1];
 HapFilename = argv[2];
 MAXD = atoi(argv[3]);
 MINC = atof(argv[4]);
 MINM = atof(argv[5]);
 PairsFilename = argv[6];
 TriplesFilename = argv[7];
 TagsFilename = argv[8];

 ExperimentMode = 1; // Type of EDA. Fixed to Tree
 func = 0;       // Number of function. Fixed to 0  
 psize = atoi(argv[9]);          // Population size
 T = atoi(argv[10]);              // Percentage of truncation integer number (1:99)
 Maxgen =  atoi(argv[11]);        // Max number of generations 
 BestElitism = atoi(argv[12]);         // If there is or not BestElitism, if thereisnot BestElitism, Elitism = 1 by default;
 printvals =  atoi(argv[13]);  

 // New Parameter that should be entered
 //NObj = 3; //Number of Objectives


 srand(ta+atoi(argv[14]));

cout << "Parameters:"<<endl;
cout << "MapFilename: " << MapFilename << ". HapFilename: " << HapFilename << "MAXD: " << MAXD << ". MINC: " << MINC << ". MINM: " << MINM << endl;
cout << "PairsFilename: " << PairsFilename << endl;
cout << "TriplesFilename: " << TriplesFilename << endl;
cout << "TagsFilename: " << TagsFilename << endl;
cout << "Population size: " << psize << endl;
cout << "Truncation(percentage): " << T << endl;
cout << "Max number of generations: " << Maxgen << endl;
cout << "Best Elitism (1) or not (0): " << BestElitism << endl;
cout << "Verbose level: " << printvals << endl;
cout << "Seed: " << ta+atoi(argv[14]) << endl;


 Tour = 4;                       // Pareto Ranking Selection is used  //Truncation Selection is used
 Ntrees = 2;                     // Number of Trees  for MT-EDA
 Elit = 1;                       // Elitism
 Nsteps = 50;                    // Learning steps of the Mixture Algorithm  
 InitTreeStructure = 1;    // 0 for a random init tree structures, 1 for a Chu&Liu learned init Tree Structure  
 VisibleChoiceVar = 0;     // 0 The Mixture choice var is hidden, 1 & 2 depends on a variable and the  unitation respectively  
 //printvals = 1;            // The printvals-1 best values in each generation are printed 
 MaxMixtP = 500;           // Maximum learning parameter mixture 
 S_alph = 0;               // Value alpha for smoothing 
 StopCrit = 1;             // Stop Criteria for Learning of trees alg.  
 Prior = 1;                // Type of prior. 
 Compl=75;                 // Complexities of the trees. 
 Coeftype=2;               // Type of coefficient calculation for Exact Learning. 
 params[0] = 1 ;           //  Params for function evaluation 
 params[1] = 2;  
 params[2] = 10;  

 Trunc = T/double(100);  
 Complex  = Compl/double(100);  
 //Max = auxMax/double(1000);   
 MaxMixtProb =MaxMixtP/double(100); 
 S_alpha = S_alph/double(100); 

 char linea[100];

 //Find Correlation between pairs if PairsFilename does not exist
 FILE *fp;
 fp  = fopen(PairsFilename,"r");
 if (fp){
   fclose(fp);
   cout << "Pairwise correlations are not calculated. Using existing file: " << PairsFilename << endl;
 }
 else{
   sprintf(linea, "perl find2Correlations.pl %s %s %d %f %f > %s",MapFilename,HapFilename,MAXD,MINC,MINM,PairsFilename);
   system(linea);
 }
 //Find Correlation between triples if TriplesFilename does not exist
 fp = fopen(TriplesFilename,"r");
 if (fp){
   fclose(fp);
   cout << "Triple correlations are not calculated. Using existing file: " << TriplesFilename << endl;
 }
 else{
   sprintf(linea, "perl find3Correlations.pl %s %s %d %f %f > %s",MapFilename,HapFilename,MAXD,MINC,MINM,TriplesFilename);
   system(linea);
 }

 // THIS PART HAS BEEN MODIFIED TO INCLUDE MULTIPLE SNP SETS
 // AT THIS POINT WE SHOULD HAVE  THE NUMBER OF SNP SETS (NumberSNPSets)
 // AND THE NAMES OF EACH SNP FILE
 
 // TO TEST THE PROBLEM I WILL ASSUME TWO (EQUAL) SNP SETS
 
 int numberSNPSets=3; 
 ContainerSNPSets = new SNPSets(numberSNPSets); 
 //for (i=0;i<numberSNPSets;i++)
   { 
    streamPairs = fopen(PairsFilename, "r+" );  
    if( streamPairs == NULL ) printf( "The file with the SNPs pairs was not opened\n" );  
    streamTriples = fopen(TriplesFilename, "r+" );  
    if( streamTriples == NULL ) printf( "The file with the SNPs triples was not opened\n" );
   
    AllSNPs = new SNPs(streamPairs,streamTriples);
    ContainerSNPSets->TheSNPSets[0] = AllSNPs;

    fclose(streamPairs);
    fclose(streamTriples);
    
    streamPairs = fopen("ENm010.CHB.tags.pairs", "r+" );  
    if( streamPairs == NULL ) printf( "The file with the SNPs pairs was not opened\n" );  
    streamTriples = fopen("ENm010.CHB.tags.trios", "r+" );  
    if( streamTriples == NULL ) printf( "The file with the SNPs triples was not opened\n" );
   
    AllSNPs = new SNPs(streamPairs,streamTriples);
    ContainerSNPSets->TheSNPSets[1] = AllSNPs;


    streamPairs = fopen("ENm010.JPT.tags.pairs", "r+" );  
    if( streamPairs == NULL ) printf( "The file with the SNPs pairs was not opened\n" );  
    streamTriples = fopen("ENm010.JPT.tags.trios", "r+" );  
    if( streamTriples == NULL ) printf( "The file with the SNPs triples was not opened\n" );
   
    AllSNPs = new SNPs(streamPairs,streamTriples);
    ContainerSNPSets->TheSNPSets[2] = AllSNPs;
    
       
   }

 ContainerSNPSets->Initialize();


 vars = ContainerSNPSets->numberSNPs;
 Max = ContainerSNPSets->numberSNPs;
 NObj = ContainerSNPSets->numberSNPSets+1; //The additional objective is the number of tags

 matrixSNPs = new unsigned*[vars];
 for (i=0;i<vars;i++) matrixSNPs[i] = new unsigned[vars];
 ContainerSNPSets->CreateMatrix(matrixSNPs);

 cout<<"Details of the SNP sets simultaneosly optimized"<<endl;
 for(j=0;j<ContainerSNPSets->numberSNPSets;j++)
   {
     cout<<"SNPFile: "<<j<<" NSNPs "<<ContainerSNPSets->TheSNPSets[j]->numberSNPs<<" NSNPsToCover(vars) "<<vars<<" NPairs "<<ContainerSNPSets->TheSNPSets[j]->npairs<<" NTriples " <<ContainerSNPSets->TheSNPSets[j]->ntriples<<endl;
   }


 BestInd = new unsigned[vars];
 Cardinalities  = new unsigned[vars];  
 for(i=0;i<vars;i++) Cardinalities[i] = 2;   
  TotEvaluations = 0;  succexp = 0;  meangen = 0; meaneval = 0;  i =0;  nsucc =0; alltime = 0; bestalltime = 0; auxtime = 0;
  init_time();  
          
  runOptimizer(ExperimentMode,i);
  end_time();

  cout<<"Run: "<<i<<" Time "<<auxtime<<" BestSol: ";
  for(int l=0;l<vars;l++) cout<<BestInd[l]<<" ";  

  cout<<BestEval<<endl;

  streamTags = fopen(TagsFilename, "w+" );  
  ContainerSNPSets->SaveTags(streamTags,BestInd); 
  fclose(streamTags);

  //PrintStatistics();
  for (i=0;i<vars;i++) delete[] matrixSNPs[i]; 
   delete[] matrixSNPs; 
  delete[]  BestInd;
  delete[] Cardinalities;
  delete ContainerSNPSets;      
  delete[]  params;
 return 1;
}       
 
#endif
