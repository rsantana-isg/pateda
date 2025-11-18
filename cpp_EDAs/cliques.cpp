#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include "auxfunc.h"
#include "Popul.h"
#include "AbstractTree.h"
#include "Constraints.h"
#include "TriangSubgraph.h"
#include "FDA.h"


// En esta implementacion el elitismo es cero
FILE* f1;

int *Alls;
int cantexp;
int vars;
double Max;
int auxMax;
int T;
double  Trunc;
int psize;
int ExperimentMode;
int Elit;
int maxclique;
int extraedges;
int Maxgen;
int printvals; 
int number_nodes;
int BestValue;
unsigned int** CliquesMatrix;
char MatrixFileName[25];
int* index;
int* indexfunc;
int* params;
int limit;
int minunit;
int maxunit;
int func;
int succ;



void ReadParameters() {

	FILE *stream;
    stream = fopen( "CliquesParams.txt", "r+" );
    if( stream == NULL )
	{
		printf( "The file CliquesParams.txt.txt was not opened\n" );
		return;
	}
	else
	{
     fscanf( stream, "%s", &MatrixFileName); 
	 fscanf( stream, "%d", &number_nodes); 
	 fscanf( stream, "%d", &cantexp); // Number of Experiments
	 fscanf( stream, "%d", &maxclique); // Maximum known clique in the graph
	 fscanf( stream, "%d", &extraedges); 
	 fscanf( stream, "%d", &T); // Percent of the Truncation selection
	 fscanf( stream, "%d", &psize); // Population Size
	 fscanf( stream, "%d", &ExperimentMode); // Type of Experiment (SEE BELOW case instruction)
	 fscanf( stream, "%d", &Elit); // Elistism (Now best elitism is fixed)
	 fscanf( stream, "%d", &Maxgen);  // Max number of generations
	 fscanf( stream, "%d", &printvals); // Best value in each generation is printed
	 fscanf( stream, "%d", &minunit); // Minimum unitation of solutions
	 fscanf( stream, "%d", &maxunit); // Maximum unitation of solutions
	 fscanf( stream, "%d", &func); // Fitness Function
	 fscanf( stream, "%d", &auxMax); // Maximum
	 fclose( stream );
    }
fclose(stream);
Trunc = T/double(100);
vars = number_nodes;
Max = double(auxMax);
}

void init_CliquesMatrix ()
{   int i,j;
	
    CliquesMatrix = new unsigned*[number_nodes];
	 for (i=0;i<number_nodes;i++) CliquesMatrix[i] = new unsigned[number_nodes];
	
	FILE *stream;
	stream = fopen( MatrixFileName, "r+" ); //mlarge.DAT T269x42.txt
   if( stream == NULL )
      printf( "La Matriz con los cliques no fue encontrada\n" );
   else
   {
     for (i=0;i<number_nodes;i++)
		 for (j=0;j<number_nodes;j++) 
			 fscanf( stream, "%d", &CliquesMatrix[i][j]);
   }
   fclose( stream );
}


double fcliques(Popul* pop,int pos,int cant)
{
int i,j,s,sum;

s =0;
sum = 0;

for(i=0; i < cant; i ++) 
{
 if(pop->P[pos][i]) index[s++]=i; // Posiciones de los 1s
}
if (s==0)  return 0;

for(i=0; i < s-1; i ++)
  for(j=i+1; j < s; j ++) 
	sum += (1-CliquesMatrix[index[i]][index[j]]);
 Alls[pos] = sum; 	
 return (number_nodes*number_nodes - sum + s);
}


double fcliques1(Popul* pop,int pos,int cant)
{
int i,j,s,sum,shared;

s =0;
sum = 0;


for(i=0; i < cant; i ++) 
{
 if(pop->P[pos][i]) index[s++]=i; // Posiciones de los 1s
 indexfunc[i] = 0;
}
if (s==0)  return 0;

for(i=0; i < s-1; i ++)
  for(j=i+1; j < s; j ++) 
  {
	sum += (1-CliquesMatrix[index[i]][index[j]]);
	if(CliquesMatrix[index[i]][index[j]] == 1) 
		indexfunc[index[i]]++;
  }
  
  shared = 0;

  for(i=0; i < s-1; i ++) 
  {
	shared += (s-indexfunc[index[i]]-1)*(s-indexfunc[index[i]]-1);  
 
  }
 
 Alls[pos] = sum; 	
 return ((number_nodes*number_nodes*number_nodes- shared+1)/double((maxclique-s)*(maxclique-s)+1));
}


/*
void SaveMatrices(int col)
{
  int h,i;
  for(i=0;i<=now;i++)
  {
   for(h=0;h<=col;h++)   fprintf(f1, "%d ",V[i][h]);
  fprintf(f1,"\n"); //Alleval[i]
  }
  
}	 
*/
void evolve (AbstractProbModel* MyModel,Popul* pop, Popul* selpop )
{
  int i,j,k,l;
  int switch_fun;
  int stopcond;
  double auxprob,auxfun;     
  
  MyModel->InitPop(pop);
  i = 0;
  auxfun = 0;
  // && stopcond<limit stopcond = limit; momentaneamente OJO
  //switch_fun = myrand()>0.5;
  while (i<Maxgen && auxprob<1 && auxfun<Max)
  {
     pop->InitIndex();
	 //if (switch_fun) for(k=0; k < psize; k ++)  pop->SetVal(k,fcliques1(pop,k,number_nodes));
     //for(k=0; k < psize; k ++)  pop->SetVal(k,fcliques(pop,k,number_nodes));
	 pop->EvaluateAll(func);
     pop->TruncSel(selpop,selpop->psize); 
     MyModel->UpdateModel(); 
     auxprob = MyModel->Prob(selpop->P[0]);
	 /*
	  if (selpop->Evaluations[selpop->index[0]] ==1)
	  {
       j=0;
	   stopcond++;
       while ( (j<selpop->psize) && (selpop->Evaluations[selpop->index[j]] == 1))
	   {
        AddTestor(selpop,selpop->index[j],number_columns);
        j++;
	   }
	  }
	  */
		 auxfun = selpop->Evaluations[0];
     if(printvals)
	 {   
		//auxfun = fcliques(selpop,0,number_nodes);
		 //if (switch_fun) else auxfun = fcliques1(selpop,0,number_nodes);
		 printf("%d %f %d %f \n ",i,auxfun,Alls[0],auxprob);
	     for(l=0;l<vars;l++) printf("%d ",selpop->P[0][l]);
        printf("\n ");
 
		 if((auxfun-number_nodes*number_nodes)>=maxclique && Alls[0]==extraedges)
		 {
  		  for(j=0; j < number_nodes; j ++) 
           if(selpop->P[0][j]) printf("%d ",j);
	       printf("\n");
		 }
	 }
     selpop->SetElit(Elit,pop);
     MyModel->GenPop(Elit,pop); 
    i++;
  }
  
  if (auxfun==Max) succ++;
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
	       SCUMDA = new SConstraintUnivariateModel(vars,AllIndex,1,selpop,minunit,maxunit);
		   evolve(SCUMDA,pop,selpop);
		   delete SCUMDA;
		 };   break; // Constraint UMDA, Type Simplex
 case 3: {
	       CConstraintUnivariateModel *CCUMDA; 
	       CCUMDA = new CConstraintUnivariateModel(vars,AllIndex,1,selpop,minunit,maxunit);
		   evolve(CCUMDA,pop,selpop);
		   delete CCUMDA;
		 };   break; // Constraint UMDA, Type Complex
 
 case 4: {
	       ConstraintBinaryTreeModel *MyTree; 
	       MyTree = new ConstraintBinaryTreeModel(vars,AllIndex,1,selpop,minunit,maxunit);
		   evolve(MyTree,pop,selpop);
		   delete MyTree;
		 };   break; 
 
 case 5: {
	       FDA *MyFDA; 
		   MyFDA = new FDA(vars,AllIndex,1,selpop,CliquesMatrix,maxclique);
	       evolve(MyFDA,pop,selpop);
		   delete MyFDA;
		 };   break; 
 case 6: {
	       ConstraintFDA *CFDA; 
		   CFDA = new ConstraintFDA(vars,AllIndex,1,selpop,CliquesMatrix,maxclique,minunit,maxunit);
	       evolve(CFDA,pop,selpop);
		   delete CFDA;
		 };   break; 

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

     
     ReadParameters();
     init_CliquesMatrix();
	 index = new int[number_nodes];
	 indexfunc = new int[number_nodes];
	 Alls = new int[psize];
     srand((unsigned)time(NULL) );

    /*
	 filedetails[0]=0;
	 strcat(filedetails,MatrixFileName);
	 itoa(T,number,10);
	 strcat(filedetails,number);
     strcat(filedetails,"_");
     itoa(psize,number,10);
	 strcat(filedetails,number);
     strcat(filedetails,".txt");
   */
	 succ = 0;
	for (iexp =1;iexp<cantexp;iexp++)
	 {
       printf("Run # %d %d of %d \n", iexp,cantexp,succ); //Alleval[i]
       runOptimizer();
	 }
	 /*
	 f1 = fopen(filedetails, "w+" );
	 SaveMatrices(number_nodes);
	 fclose(f1); 
    */
	 for (i=0;i<number_nodes;i++) delete[] CliquesMatrix[i];
     delete[] CliquesMatrix; 
	 delete[] index;
	 delete[] indexfunc;
	 delete[] Alls;
     
 }
