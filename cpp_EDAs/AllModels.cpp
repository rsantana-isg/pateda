#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "auxfunc.h"
#include "Popul.h"
#include "AbstractTree.h"
#include "Constraints.h"

FILE *f1, *stream;
char MatrixFileName[25];
int auxtime;
int* params;
int Ntrees;
int Card;
int VisibleChoiceVar;
int Nsteps;
int InitTreeStructure;
int r;
int Tour;
int auxMax;
int cantexp;
int vars;
int func;
int seed;
int succexp;
double meangen;   
double Max;
int T;
double  Trunc;
int psize;
int ExperimentMode;
int Elit;
int Maxgen;
int printvals; 
double BestValue;
int minunit;
int maxunit;
int gen;

double ConstDecp(Popul* pop,int pos,int cant)
{
double sum;
int i,s, aux;
sum = 0;

for(i=0; i < cant; i += 4) 
{
	s =  ( (pop->P[pos][i]+ 1-pop->P[pos][i+1]==0) || (pop->P[pos][i]+ 1-pop->P[pos][i+1]==2));
	s *= ((pop->P[pos][i+1]+ pop->P[pos][i+2]==0) || (pop->P[pos][i+1]+  pop->P[pos][i+2]==2));
	s *= ((pop->P[pos][i+2]+ 1-pop->P[pos][i+3]==0) || (pop->P[pos][i+2]+ 1-pop->P[pos][i+3]==2));
	if (s) sum +=s;
	else if (pop->P[pos][i]+pop->P[pos][i+3]==1) sum += 0.9;
}		

//	if ((pop->P[pos][i+2]==1) && (s==1) ) sum += 5;
	//else if (s!=1) sum += 4.9;
return 4*sum;

}

/*
double Onemax(Popul* pop,int pos,int cant)
{
double sum;
int i;
sum = 0;

for(i=0; i < cant; i ++)  sum += pop->P[pos][i]; 

return sum;

}

double S_Peak(Popul* pop,int pos,int cant){
double sum = 0;
int m,k;
unsigned* buff;
int n;


n=cant;
buff = pop->P[pos];

k = int(params[0]);
m = int(params[1]);

sum = Onemax(pop,pos,cant);
int s = 0;
for(int i = 0; i < m; i++)
  s +=  (1-buff[i]);
for(i = m; i < n; i++)
  s  +=  buff[i];
return sum+s*k*(m+1);
} 

double PosFunction(Popul* pop,int pos,int cant)
{
double sum;
int i;
sum = 0;

for(i=0; i < cant; i ++)  sum += i*pop->P[pos][i]; 

return sum;

}

double PosFunctionPenalty(Popul* pop,int pos,int cant)
{
double sum;
int i,s;
sum = 0;
s = 0;

for(i=0; i < cant; i ++) 
{
 s += pop->P[pos][i];
 sum += i*pop->P[pos][i]; 
}
  
return sum/double(abs(r-s)+1);

}


double FindMax()
{
double sum;
int i;
sum = 0;

for(i=vars-1; i >= vars-minunit; i--)  sum += i; 

return sum;

}
*/
void evolve (AbstractProbModel* MyModel,Popul* pop, Popul* selpop )
{
  int i,j,k;
  int stopcond;
  double auxprob;     
   
  double *auxstruct;
  auxstruct = new double[vars];
  for(k=0; k < vars; k ++)   auxstruct[k] = 0.5;
  MyModel->InitPop(pop);
  //if(ExperimentMode==4 || ExperimentMode==1 ) pop->Repair(minunit,maxunit,auxstruct); 
  if(ExperimentMode==4 || ExperimentMode==1 ) pop->Repair(minunit,maxunit); 
  delete[] auxstruct;

  i = 0;
  auxprob = 0;
  BestValue = 0;

  while (i<Maxgen && auxprob<1 && BestValue<Max)    {
     pop->InitIndex();
	 //for(k=0; k < psize; k ++)  pop->SetVal(k,PosFunction(pop,k,vars));
	 //for(k=0; k < psize; k ++)  pop->SetVal(k,PosFunctionPenalty(pop,k,vars));
     //for(k=0; k < psize; k ++)  pop->SetVal(k,ConstDecp(pop,k,vars));
//	 for(k=0; k < psize; k ++)  pop->SetVal(k,Onemax(pop,k,vars));
     pop->TruncSel(selpop,selpop->psize); 
     MyModel->UpdateModel(); 
     auxprob = MyModel->Prob(selpop->P[0]);
	 BestValue = selpop->Evaluations[0];
	 
	 if(printvals) 
	 {
		 printf("%d %f %f \n ",i,BestValue,auxprob);
   		  for(k=0; k < vars; k ++) 
           if(selpop->P[0][k]) printf("%d ",k);
	       printf("\n");
     }
	 
     if(auxprob<1)
	 {
		 selpop->SetElit(Elit,pop);
         MyModel->GenPop(Elit,pop); 
	     //if(ExperimentMode==4) pop->Repair(minunit,maxunit,((SConstraintUnivariateModel*)MyModel)->AllProb); 
		 //if(ExperimentMode==1) pop->Repair(minunit,maxunit,((BinaryTreeModel*)MyModel)->AllProb); 
		 if (ExperimentMode==4) pop->Repair(minunit,maxunit); 
		 if (ExperimentMode==1) pop->Repair(minunit,maxunit); 
	 }
    i++;
  }

 gen = i-1;  
}

void runOptimizer ()
{
  int Ti;
  int* AllIndex;
  Popul *pop,*selpop;
    
  Ti = Trunc*psize;
  pop = new Popul(psize,vars,Elit);
  selpop = new Popul(Ti,vars,Elit);
  AllIndex = new int[selpop->psize];
  SetIndexNormal(AllIndex,selpop);
  
 switch(ExperimentMode)
    {
 case 0: {
	       UnivariateModel *MyUMDA;
	       MyUMDA = new UnivariateModel(vars,AllIndex,1,selpop);
		   evolve(MyUMDA,pop,selpop);
		   delete MyUMDA;
		 };   break; // UMDA
 case 1: {
	       BinaryTreeModel *MyTree; 
	       MyTree = new BinaryTreeModel(vars,AllIndex,1,selpop);
		   evolve(MyTree,pop,selpop);
		   delete MyTree;
		 };   break; // Tree
 case 2: {
	       SConstraintUnivariateModel *SCUMDA; 
	       SCUMDA = new SConstraintUnivariateModel(vars,AllIndex,1,selpop,minunit,maxunit,0);
		   evolve(SCUMDA,pop,selpop);
		   delete SCUMDA;
		 };   break; // Constraint UMDA, Type Simplex
 case 3: {
	       CConstraintUnivariateModel *CCUMDA; 
	       CCUMDA = new CConstraintUnivariateModel(vars,AllIndex,1,selpop,minunit,maxunit,0);
		   evolve(CCUMDA,pop,selpop);
		   delete CCUMDA;
		 };   break; // Constraint UMDA, Type Complex
 
 case 4: {
	       UnivariateModel *MyUMDA;
	       MyUMDA = new UnivariateModel(vars,AllIndex,1,selpop);
		   evolve(MyUMDA,pop,selpop);
		   delete MyUMDA;
		 };   break; // UMDA
 	 
 
 case 5: {
	       ConstraintBinaryTreeModel *MyTree; 
	       MyTree = new ConstraintBinaryTreeModel(vars,AllIndex,1,selpop,minunit,maxunit);
		   evolve(MyTree,pop,selpop);
		   delete MyTree;
		 };   break; // Tree

 }  
 if (BestValue==Max)
 { 
	 succexp++;
     meangen += gen;
 }
 delete pop;
  delete selpop;
  delete[] AllIndex;
}

void main()
 { 
   	char filedetails[50],number[10];
    int toteval, iexp;
	int i;
    srand((unsigned)time(NULL) );
    stream = fopen( "ParamModels.txt", "r+" );
    params = new int[3]; 
	unsigned *Cardinalities;
	
	
	if( stream == NULL )
		printf( "The file Param.txt was not opened\n" );
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
	 fscanf( stream, "%d", &Elit); // Elistism (Now best elitism is fixed)
	 fscanf( stream, "%d", &Nsteps); // Learning steps of the Mixture Algorithm
 	 fscanf( stream, "%d", &InitTreeStructure); // 0 for a random init tree structures, 1 for a Chu&Liu learned init Tree Structure
	 fscanf( stream, "%d", &VisibleChoiceVar); // 0 The Mixture choice var is hidden, 1 & 2 depends on a variable and the  unitation respectively
	 fscanf( stream, "%d", &Maxgen);  // Max number of generations
	 fscanf( stream, "%d", &printvals); // Best value in each generation is printed
	 fscanf( stream, "%d", &params[0]); // Params for function evaluation
	 fscanf( stream, "%d", &params[1]);
	 fscanf( stream, "%d", &params[2]);
	 fscanf( stream, "%d", &Card); // Cardinal for all variables
	 fscanf( stream, "%d", &seed); // seed
	 fclose( stream );
    }


Trunc = T/double(100);
Max = auxMax/double(100); 
succexp = 0;
meangen = 0;   
Cardinalities  = new unsigned[vars];
for(i=0;i<vars;i++) Cardinalities[i] = Card;

    filedetails[0]=0;
    strcat(filedetails,"OneMExp");
  	itoa(vars,number,10);
	strcat(filedetails,number);
    strcat(filedetails,"_");
	itoa(T,number,10);
	strcat(filedetails,number);
    strcat(filedetails,".txt");
    f1 = fopen(filedetails, "w+" ); 
 
	for (iexp=0;iexp<cantexp;iexp++)
	{
     runOptimizer();
     if (succexp>0)
		 printf(" Run # %d  meangen %e succexp %d \n",iexp,meangen/succexp,succexp);
	 else printf(" Run # %d  meangen %e succexp %d \n",iexp,0,0);
     fprintf(f1, "%d %d %e %d \n",psize,iexp,BestValue,gen);	   
	}
fclose(f1); 
 delete[] params;
delete[] Cardinalities;
    return;
	
 }
