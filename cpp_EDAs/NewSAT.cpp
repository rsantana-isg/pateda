#include <stdio.h>
#include <time.h> 
#include <stdlib.h> 
#include <string.h> 
#include "auxfunc.h" 
#include "Popul.h" 
#include "AbstractTree.h" 
#include "MixtureTrees.h"
#include "FDA.h"
#include "CNF.h" 
#define itoa(a,b,c) sprintf(b, "%d", a) 
 
 CNF *AllClauses;
//CNF_Generator *AllClauses; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798}; 
double SelInt;

int* params;  
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
int extraedges; 
int* indexfunc; 
int limit; 
int reward; 
int Mutation;
int CliqMaxLength;
int MaxNumCliq;


 
void ReadParameters() { 
  int T,MaxMixtP,S_alph,Compl;
	FILE *stream; 
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
	 fscanf( stream, "%d", &vars); // Cant of Vars in the vector 
 	 fscanf( stream, "%d", &auxMax); // Max. of the fitness function 
  	 fscanf( stream, "%d", &T); // Percent of the Truncation selection 
	 fscanf( stream, "%d", &psize); // Population Size 
	 fscanf( stream, "%d", &Tour);  // Tournament size 
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
    	 fscanf( stream, "%d", &Card); // Cardinal for all variables 
	 fscanf( stream, "%d", &Mutation); // Population based mutation 
	 fscanf( stream, "%d", &reward); // Reward for clause weighting
         fscanf( stream, "%d", &CliqMaxLength); // Maximum size of the cliques for Markov 
	 fscanf( stream, "%d", &MaxNumCliq); // Maximum number of cliques for Markov
    } 
fclose(stream); if(T>0)
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

int usualinit(double Complexity) 
{ 
  int i,k,TruncMax,NSelPoints,NPoints,fgen; 
  Popul *pop,*selpop,*elitpop,*compact_pop;
  double BestEval,auxprob,sumprob;    
  double* fvect;
  int* AllIndex; 
  BinaryTreeModel *MyTree; 
  NPoints = 0; //rinit
  elitpop = (Popul*)0;  
  TruncMax = int(psize*Trunc); 
  if (BestElitism)  Elit = TruncMax;  
  pop = new Popul(psize,vars,Elit); 
  fgen = -1; 

 if (Tour>0)  
     { 
	 selpop = new Popul(psize,vars,Elit); 
	 elitpop = new Popul(TruncMax,vars,Elit); 
         compact_pop = new Popul(psize,vars,Elit); 
         fvect = new double[psize];
     } 
  else  
    {
     selpop = new Popul(TruncMax,vars,Elit); 
     compact_pop = new Popul(TruncMax,vars,Elit);
     fvect = new double[TruncMax];
    }
    
  AllIndex = new int[selpop->psize]; 
  SetIndexNormal(AllIndex,selpop); 
  MyTree = new BinaryTreeModel(vars,Complexity); 
  i=0; 
  pop->RandInit(); 
  auxprob =0;
  AllClauses->Satisfied = 0; 
  BestEval  = Max -1;
  while (i<Maxgen && auxprob<1 && BestEval<Max) 
  { 
    
     pop->InitIndex();
     for(k=0; k < psize; k ++)  pop->SetVal(k,AllClauses->SatClauses(pop->P[k])); 
     //pop->EvaluateAll(func); 
     if (Tour==0) 
         { 
           pop->TruncSel(selpop,TruncMax);         
           NSelPoints = TruncMax;
	 }
	 else 
	 { 
	   pop->TruncSel(elitpop,TruncMax);  
	   pop->TournSel(selpop,Tour);  
           NSelPoints = psize;
	 } 
     
    
     NPoints = selpop->CompactPop(compact_pop,fvect);
     MyTree->rootnode = MyTree->RandomRootNode(); 
     MyTree->CalProbFvect(compact_pop,fvect,NPoints);    
     MyTree->SetNPoints(NSelPoints);
     if(MyTree->Complexity == 0)
     { 
      MyTree->CalMutInf(); 
      MyTree->MakeTree(MyTree->rootnode);
     }
     else
     {
      MyTree->CalculateILikehood(compact_pop,fvect); 
      MyTree->MakeTreeLog();
     }
     
      AllClauses->AdaptWeights(reward,selpop->P[0]);   
      //AllClauses->UpdateWeights(MyTree->AllProb);

     sumprob = MyTree->SumProb(compact_pop,NPoints);
               MyTree->PutPriors(Prior,TruncMax,1);
     auxprob = MyTree->Prob(selpop->P[0]); 
      
     BestEval = AllClauses->Satisfied;
       
      
      if (BestEval==Max) fgen  = i;	  
      else
          {
           if (Tour>0) 
            {
	      elitpop->SetElit(Elit,pop); 
	      //MyTree->CollectKConf(Elit,vars,MyTree,pop);
            }
	   else  selpop->SetElit(Elit,pop);             
           MyTree->GenPop(Elit,pop);  
	   if (Mutation) MyTree->PopMutation(pop,Elit,1,0.01); 	 
           i++;
          }  
  } 
 if(printvals) 
  cout<<"Gen :"<<i<<" Best:"<<selpop->Evaluations[0]<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
       
 

  delete[] AllIndex; 
  delete[] fvect;
  delete compact_pop;
  delete pop; 
  delete selpop; 
  if (Tour>0) delete elitpop; 
  delete MyTree; 
  return fgen; 
} 

int Markovinit(double Complexity) //In this case, complexity is the threshold for chi-square
{ 
  int i,k,TruncMax,NSelPoints,NPoints,fgen; 
  Popul *pop,*selpop,*elitpop,*compact_pop;
  double BestEval,auxprob,sumprob;    
  double* fvect;
  int* AllIndex; 
  DynFDA* MyMarkovNet; 
  NPoints = 0; //rinit
  elitpop = (Popul*)0;  
  TruncMax = int(psize*Trunc); 
  if (BestElitism)  Elit = TruncMax;  
  pop = new Popul(psize,vars,Elit); 
  fgen = -1; 

 if (Tour>0)  
     { 
	 selpop = new Popul(psize,vars,Elit); 
	 elitpop = new Popul(TruncMax,vars,Elit); 
         compact_pop = new Popul(psize,vars,Elit); 
         fvect = new double[psize];
     } 
  else  
    {
     selpop = new Popul(TruncMax,vars,Elit); 
     compact_pop = new Popul(TruncMax,vars,Elit);
     fvect = new double[TruncMax];
    }
    
  AllIndex = new int[selpop->psize]; 
  SetIndexNormal(AllIndex,selpop); 
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity); 
  i=0; 
  pop->RandInit(); 
  auxprob =0;
  AllClauses->Satisfied = 0; 
  BestEval  = Max -1;
  while (i<Maxgen && BestEval<Max) 
  { 
    
     pop->InitIndex();
     for(k=0; k < psize; k ++)  pop->SetVal(k,AllClauses->SatClauses(pop->P[k])); 
     //pop->EvaluateAll(func); 
     if (Tour==0) 
         { 
           pop->TruncSel(selpop,TruncMax);         
           NSelPoints = TruncMax;
	 }
	 else 
	 { 
	   pop->TruncSel(elitpop,TruncMax);  
	   pop->TournSel(selpop,Tour);  
           NSelPoints = psize;
	 } 
     
    
     NPoints = selpop->CompactPop(compact_pop,fvect);
      
     MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect);
     MyMarkovNet->UpdateModel();
     
      AllClauses->AdaptWeights(reward,selpop->P[0]);   
      //AllClauses->UpdateWeights(MyTree->AllProb);

     
     auxprob = MyMarkovNet->Prob(selpop->P[0]); 
      
     BestEval = AllClauses->Satisfied;
       
      
      if (BestEval==Max) fgen  = i;	  
      else
          {
           if (Tour>0)  elitpop->SetElit(Elit,pop); 
	   else  selpop->SetElit(Elit,pop);             
           MyMarkovNet->GenPop(Elit,pop);  
	   if (Mutation) MyMarkovNet->PopMutation(pop,Elit,1,0.01); 	 
           i++;
          }  
      //MyMarkovNet->
   
  } 
 if(printvals) 
  cout<<"Gen :"<<i<<" Best:"<<selpop->Evaluations[0]<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
       
 

  delete[] AllIndex; 
  delete[] fvect;
  delete compact_pop;
  delete pop; 
  delete selpop; 
  if (Tour>0) delete elitpop; 
  delete MyMarkovNet; 
  return fgen; 
} 


 
int UMDAinit() 
{ 
  int i,k,TruncMax,NSelPoints,NPoints,fgen; 
  Popul *pop,*selpop,*elitpop,*compact_pop;
  double BestEval,auxprob;    
  double* fvect;
  int* AllIndex;  
  UnivariateModel *MyUMDA; 
  NPoints = 0; //rinit
    elitpop = (Popul*)0;   //rinit
  TruncMax = int(psize*Trunc); 
  if (BestElitism)  Elit = TruncMax;  
  pop = new Popul(psize,vars,Elit); 
  fgen = -1; 

 if (Tour>0)  
     { 
	 selpop = new Popul(psize,vars,Elit); 
	 elitpop = new Popul(TruncMax,vars,Elit); 
         compact_pop = new Popul(psize,vars,Elit); 
         fvect = new double[psize];
     } 
  else  
    {
     selpop = new Popul(TruncMax,vars,Elit); 
     compact_pop = new Popul(TruncMax,vars,Elit);
     fvect = new double[TruncMax];
    }
    
  AllIndex = new int[selpop->psize]; 
  SetIndexNormal(AllIndex,selpop); 
  MyUMDA = new UnivariateModel(vars,AllIndex,1); 
  i=0; 
  pop->RandInit(); 
  auxprob =0;
  AllClauses->Satisfied = 0; 
  BestEval  = Max -1;
  while (i<Maxgen && auxprob<1 && BestEval<Max) 
  { 
    
     pop->InitIndex();
     for(k=0; k < psize; k ++)  pop->SetVal(k,AllClauses->SatClauses(pop->P[k])); 
     //pop->EvaluateAll(func); 
     if (Tour==0) 
         { 
           pop->TruncSel(selpop,TruncMax);         
           NSelPoints = TruncMax;
	 }
	 else 
	 { 
	   pop->TruncSel(elitpop,TruncMax);  
	   pop->TournSel(selpop,Tour);  
           NSelPoints = psize;
	 } 
     
    
     NPoints = selpop->CompactPop(compact_pop,fvect);
     MyUMDA->CalProbFvect(compact_pop,fvect,NPoints);   
     MyUMDA->PutPriors(Prior,TruncMax,1);
     auxprob = MyUMDA->Prob(selpop->P[0]); 
      AllClauses->AdaptWeights(reward,selpop->P[0]);   
      //AllClauses->UpdateWeights(MyUMDA->AllProb);      
     BestEval = AllClauses->Satisfied;
       
      
      if (BestEval==Max) fgen  = i;	  
      else
          {
           if (Tour>0) 
            {
	      elitpop->SetElit(Elit,pop); 
	      //MyTree->CollectKConf(Elit,vars,MyTree,pop);
            }
	   else  selpop->SetElit(Elit,pop);             
           MyUMDA->GenPop(Elit,pop);  
	   if (Mutation) MyUMDA->PopMutation(pop,Elit,1,0.01); 	 
           i++;
          }  
  } 
 if(printvals) 
  cout<<"Gen :"<<i<<" Best:"<<selpop->Evaluations[0]<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" Satisfied:"<<BestEval<<endl; 
       
 

  delete[] AllIndex; 
  delete[] fvect;
  delete compact_pop;
  delete pop; 
  delete selpop; 
  if (Tour>0) delete elitpop; 
  delete MyUMDA; 
  return fgen; 
} 

int  MixturesAlgorithm(int Type,unsigned *Cardinalities,double Complexity) 
{ 
  int i,k,fgen; 
  Popul *pop,*selpop,*elitpop,*compact_pop; 
  int TruncMax,NPoints; 
  double auxprob,BestEval; 
  double *pvect;
  MixtureTrees *Mixture; 
   elitpop = (Popul*)0;   //rinit

  fgen  = -1;
  TruncMax = int(psize*Trunc); 
  if (BestElitism) Elit = TruncMax;  
  pop = new Popul(psize,vars,Elit,Cardinalities); 
   
 
 if (Tour>0)  
  { 
	 selpop = new Popul(psize,vars,Elit,Cardinalities); 
	 elitpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
         compact_pop = new Popul(psize,vars,Elit,Cardinalities); 
         pvect = new double[psize];
  } else 
  {
   selpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
   compact_pop = new Popul(TruncMax,vars,Elit,Cardinalities);  
   pvect = new double[TruncMax];
  }
   
  Mixture = new MixtureTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior);//NSteps+1 
 
  pop->RandInit(); 
  i=0; 
  auxprob = 0; 
  BestEval = Max-1;
  NPoints = 2;
  AllClauses->Satisfied=0;

  while (i<Maxgen && BestEval<Max && NPoints>1)  //&& oldlikehood != likehood) 
  {
   pop->InitIndex();
   for(k=0; k < psize; k ++)  pop->SetVal(k,AllClauses->SatClauses(pop->P[k])); 
   //pop->EvaluateAll(func); 
   if (Tour==0) pop->TruncSel(selpop,TruncMax);  
	 else 
	 { 
	   pop->TruncSel(elitpop,TruncMax);  
	   pop->TournSel(selpop,Tour);  
	 } 
    AllClauses->AdaptWeights(reward,selpop->P[0]);    
    BestEval = AllClauses->Satisfied;

   NPoints = selpop->CompactPop(compact_pop,pvect);
   Mixture->SetNpoints(NPoints,pvect);
   Mixture->SetPop(compact_pop);
   Mixture->MixturesInit(Type,InitTreeStructure,pvect,Complexity);
   Mixture->LearningMixture(Type); 
   Mixture->SamplingFromMixture(pop); 
   //Mixture->SamplingFromMixtureMixt(pop); 
   //Mixture->SamplingFromMixtureHMixing(pop); 
   if (Mutation) Mixture->EveryTree[0]->PopMutation(pop,Elit,1,0.01);
   Mixture->RemoveTrees();
	 
   if (Tour>0) elitpop->SetElit(Elit,pop); 
   else selpop->SetElit(Elit,pop); 
   
   if (BestEval==Max) fgen= i;    
   //if(printvals)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" DifPoints:"<<NPoints<<endl; 
   
   i++; 
  } 
if(printvals)  cout<<"Gen :"<<i<<" Best:"<<BestEval<<" DifPoints:"<<NPoints<<endl; 
  delete Mixture; 
  delete pop; 
  delete selpop; 
  delete compact_pop;
  delete[] pvect;
  if (Tour>0) delete elitpop;  
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
                     }
   if (succ>-1) 
   {
       succexp++;
       meangen += succ;
   }  
}

void PrintStatistics(int f)
{ 
  double auxmeangen, meanfit;
  //cout<<"G="<<f<<"  ";
                  if (succexp>0) 
                   {
                    auxmeangen = meangen/succexp;
                    if (BestElitism) 
                         meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;    
                    else meanfit = (auxmeangen+1)* (psize-1) + 1;
                    cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" f(x)="<<func<<" N="<<psize<<"  m="<<Ntrees<<"  Prior="<<Prior<<" Suc.="<<succexp<<"  g="<<(auxmeangen+1)<<"  ave="<<meanfit<<endl;      
                    //cout<<f<<" "<<vars<<"  "<<Ntrees<<"  "<<Mutation<<"  "<<Complex<<"  "<<succexp<<"  "<<(auxmeangen+1)<<"  "<<meanfit<<endl;                  
                   }
                  else 
                   { 
		       cout<<"TypeExp="<<ExperimentMode<<"  n="<<vars<<" f(x)="<<func<<" N="<<psize<<"  m="<<Ntrees<<"  Prior="<<Prior<<" Suc.="<<0<<"  g="<<0<<"  ave="<<0<<endl;
		       //cout<<f<<" "<<vars<<"  "<<Ntrees<<"  "<<Mutation<<"  "<<Complex<<"  "<<succexp<<"  "<< 0 <<"  "<<0<<endl;  
                   }             

}
 
int main() 
{ 
	 
	FILE *input; 
	int i,f,Compl,jmax; 
	char number[10]; 
	//int filenumbers[15] = {21,53,61,74,92,95, 36,37,43,93,160}; //Good for eachone graphs 20
        //int filenumbers[15] = {9,31,32,37,96,192,231,232,248,286}; // hard graphs 20
	//int filenumbers[30] = {4,8,10,13,21,24,28,35,37,41,44,45,50,51,55,62,63,64,65,67,71,74}; //good MT1-50
        int filenumbers[30] = {7,9,20,23,34,38,43,47,48,54,56,68,69,73}; //good Mt3-50

       unsigned ta = (unsigned) time(NULL); 
       srand(ta);
       cout<<"seed"<<ta<<endl;
       ReadParameters();     
   
    for(f=9;f<=13;f++)
	//for(Compl=20;Compl<=80;Compl+=20) 
	{ 
	    Complex  = 0; //Compl/double(100);
            //MaxMixtProb = Compl/double(100);
       
       for (ExperimentMode =0;  ExperimentMode<=3;ExperimentMode+=1)
       { 
          func = filenumbers[f];
	
	   if(ExperimentMode==1 || ExperimentMode==3)  jmax = 3;
	   else jmax = 0;
	   for (int j=0; j<=jmax;j++)       
	 { 
             Ntrees = (j+2);
             MatrixFileName[0]=0;
             
         strcat(MatrixFileName,"uf50-0"); 
	 itoa(func,number,10); 
	 strcat(MatrixFileName,number);
	 strcat(MatrixFileName,".cnf");       
         input = fopen(MatrixFileName,"r+"); 
	AllClauses = new CNF(input,3); 
	fclose(input); 
  	vars = AllClauses->NumberVars; 
	Max = AllClauses->cantclauses; 
        Cardinalities  = new unsigned[vars]; 
        for(i=0;i<vars;i++) Cardinalities[i] = Card;
        
        AllClauses->FillMatrix();   
        succexp = 0;  meangen = 0;
        for(i=0;i<cantexp;i++) runOptimizer();
       
        PrintStatistics(func); 
	
      	delete AllClauses; 
	 delete[] Cardinalities;
	 }	
	}  // Experiment mode   
       
	} // f 
  } 



