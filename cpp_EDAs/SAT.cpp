#include <stdio.h>
#include <time.h> 
#include <stdlib.h> 
#include <string.h> 
#include "auxfunc.h" 
#include "Popul.h" 
#include "AbstractTree.h" 
#include "Constraints.h" 
#include "TriangSubgraph.h" 
#include "FDA.h" 
#include "CNF.h" 
#define itoa(a,b,c) sprintf(b, "%d", a) 
 
 CNF *AllClauses;
//CNF_Generator *AllClauses; 
int cantexp; 
int vars; 
double Max; 
int T; 
double  Trunc; 
int psize; 
int ExperimentMode; 
int Elit; 
int extraedges; 
int Maxgen; 
int printvals;  
int BestValue; 
char MatrixFileName[25]; 
//int* index; 
int* indexfunc; 
int* params; 
int limit; 
int maxcliqueFDA; 
int minunit; 
int maxunit; 
int reward; 
int Mutation; 
int succexp; 
double meangen; 
int thresh; 
 
void ReadParameters() { 
 
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
	 fscanf( stream, "%d", &T); // Percent of the Truncation selection 
	 fscanf( stream, "%d", &psize); // Population Size 
	 fscanf( stream, "%d", &ExperimentMode); // Type of Experiment (SEE BELOW case instruction) 
	 fscanf( stream, "%d", &Elit); // Elistism (Now best elitism is fixed) 
	 fscanf( stream, "%d", &Maxgen);  // Max number of generations 
	 fscanf( stream, "%d", &printvals); // Best value in each generation is printed 
     fscanf( stream, "%d", &maxcliqueFDA); 
	 fscanf( stream, "%d", &minunit); // Minimum unitation of solutions 
	 fscanf( stream, "%d", &maxunit); // Maximum unitation of solutions 
	 fscanf( stream, "%d", &reward); // Reward for clause weighting 
	 fscanf( stream, "%d", &Mutation); // Mutation (1), No Mutation (0) 
	 fscanf( stream, "%d", &thresh); // parametro para vecindad del constraint 
	 
    } 
fclose(stream); 
Trunc = T/double(100); 
} 
 
void evolve (AbstractProbModel* MyModel,Popul* pop, Popul* selpop ) 
{ 
  int i,j,k; 
  int stopcond; 
  double auxprob,auxfun;      
  Popul *auxpop; 
  auxpop = new Popul(Elit,vars,Elit); 
  double* auxvect; 
  UnivariateModel* Univ; 
  BinaryTreeModel* Biv; 
  MyModel->InitPop(pop); 
  i = 0; 
  AllClauses->Satisfied = 0; 
 
  //&& auxprob<1 
  while (i<Maxgen  && AllClauses->Satisfied<Max) 
  { 
     pop->InitIndex(); 
     for(k=0; k < psize; k ++)  pop->SetVal(k,AllClauses->SatClauses(pop->P[k])); 
     pop->TruncSel(selpop,selpop->psize);  
	 pop->TruncSel(auxpop,auxpop->psize);  
     MyModel->UpdateModel();  
     auxprob = MyModel->Prob(selpop->P[0]); 
     auxfun = AllClauses->SatClauses(selpop->P[0]); 
      AllClauses->AdaptWeights(reward,selpop->P[0]); 
      
	if(ExperimentMode==0) 
	{ 
		 Univ = (UnivariateModel*) MyModel; 
		 AllClauses->UpdateWeights(Univ->AllProb); 
		 //AllClauses->EqualUpdateWeights(Univ->AllProb); 
	}  
	else if(ExperimentMode==1) 
	{ 
		 Biv = (BinaryTreeModel*) MyModel; 
		 AllClauses->UpdateWeights(Biv->AllProb); 
	} 	 
     
    if(printvals) 
	 {    
	printf("%d %f %f %d \n ",i,auxfun,auxprob,AllClauses->Satisfied); 
		  
/*		 for(j=0; j < vars; j ++)  
           if(selpop->P[0][j]) printf("%d ",j); 
	       printf("\n"); 
*/		  
	 } 
     auxpop->SetElit(Elit,pop); 
     MyModel->GenPop(Elit,pop);  
	 if (Mutation)  
	 {  
		 if(ExperimentMode==5)  
			 MyModel->PopMutation(pop,Elit,1,0.1); 
		 else 
		 { 
		  if( auxprob > 0.99) //|| auxfun < 2 
			  MyModel->PopMutation(pop,Elit,1,0.005); 
		  else MyModel->PopMutation(pop,Elit,1,0.005); 
		 }  
	 } 
    i++; 
  } 
   
  delete auxpop; 
 
if(auxfun>=Max) 
{ 
  succexp++; 
  meangen+= i; 
} 
 
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
	       SCUMDA = new SConstraintUnivariateModel(vars,AllIndex,1,selpop,minunit,maxunit,thresh); 
		   evolve(SCUMDA,pop,selpop); 
		   delete SCUMDA; 
		 };   break; // Constraint UMDA, Type Simplex 
 case 3: { 
	       CConstraintUnivariateModel *CCUMDA;  
	       CCUMDA = new CConstraintUnivariateModel(vars,AllIndex,1,selpop,minunit,maxunit,1); 
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
	       ConstraintFDA *CFDA;  
		   CFDA = new ConstraintFDA(vars,AllIndex,1,selpop,AllClauses->adjmatrix,maxcliqueFDA,minunit,maxunit); 
	       evolve(CFDA,pop,selpop); 
		   delete CFDA; 
		 };   break;  
 
 case 6: { 
	       FDA *MyFDA;  
		   MyFDA = new FDA(vars,AllIndex,1,selpop,AllClauses->adjmatrix,maxcliqueFDA); 
	       evolve(MyFDA,pop,selpop); 
		   delete MyFDA; 
		 };   break;  
}	 
    
 delete pop; 
  delete selpop; 
  delete[] AllIndex; 
} 
 
int main() 
{ 
	 
	FILE *input,*f1; 
	int iexp,f; 
	char filedetails[25],number[10]; 
	int filenumbers[9] = {434,623,836,7,9,34,984,986,997}; 
     
	srand((unsigned)time(NULL) );     
	ReadParameters();     
	
    for(f=0;f<1;f++) 
	{ 
 
	for (ExperimentMode =0;ExperimentMode<=0;ExperimentMode++) 
	{ 
	    /*
	 MatrixFileName[0]=0; 
         strcat(MatrixFileName,"uf20-0"); 
	 itoa(filenumbers[f],number,10); 
	 strcat(MatrixFileName,number);
	    */ 
	 filedetails[0]=0; 
     strcat(filedetails,MatrixFileName); 
     //strcat(MatrixFileName,".cnf");       
 
        input = fopen(MatrixFileName,"r+"); 
	AllClauses = new CNF(input,3); 
	fclose(input); 
 

	 strcat(filedetails,"_f"); 
 	 itoa(f,number,10); 
	 strcat(filedetails,number); 
	 strcat(filedetails,"_"); 
	 itoa(ExperimentMode,number,10); 
	 strcat(filedetails,number); 
     strcat(filedetails,".txt"); 
	
  	
        f1 = fopen(filedetails, "w+" );  
 
	//AllClauses = new CNF_Generator(5,30,20,100,50,3); 
	//AllClauses->Create(); 
	//AllClauses->SaveInstance("CNF_5_30_20_100_50.cnf");  
	//iexp = AllClauses->SatClauses(AllClauses->solution); 
	vars = AllClauses->NumberVars; 
	Max = AllClauses->cantclauses; 
    AllClauses->FillMatrix(); 
 
	 // Initialization for algorithms that use unitation constraints 
	 
 	if(ExperimentMode>=2 && ExperimentMode<=5) 
	{ 
	  AllClauses->AssignUnitations(); 
      minunit = AllClauses->MinUnit; 
      maxunit = AllClauses->MaxUnit; 
	 //fprintf(f1,"%d %d %d \n ",f, minunit,maxunit); 
	 //printf("%d %d %d \n ",f, minunit,maxunit); 
	} 
  
	 
    meangen = 0; 
	succexp = 0; 
 
	for (iexp =0;iexp<cantexp;iexp++) 
	 { 
       printf("Run # %d of %d \n", iexp,cantexp); //Alleval[i] 
       runOptimizer(); 
	   if (succexp>0)   
	   { 
		   printf("%d  %d   %e \n", iexp+1, succexp,  (meangen)/succexp ); 
		   fprintf(f1,"%d  %d   %e \n", iexp+1, succexp,  (meangen)/succexp ); 
	   } 
       else 
	   { 
		   printf("%d  %d   \n", iexp+1, succexp); 
		   fprintf(f1,"%d  %d   \n", iexp+1, succexp); 
	   } 
	 } 
      	delete AllClauses; 
		fclose(f1);  
	} // Experiment mode 
 
	} // f 
  } 
/*=================================================================*/ 
/*=================================================================*/ 
 
/* 
	 
   
 
  for (f =1;f<=999;f++) 
 { 
	  
	 MatrixFileName[0]=0; 
     strcat(MatrixFileName,"uf20-0"); 
	 itoa(filenumbers[f],number,10); 
	 strcat(MatrixFileName,number); 
     strcat(filedetails,MatrixFileName); 
     strcat(MatrixFileName,".cnf");       
 
	 filedetails[0]=0; 
	 
	 strcat(filedetails,"_"); 
	 itoa(ExperimentMode,number,10); 
	 strcat(filedetails,number); 
     strcat(filedetails,".txt"); 
	 f1 = fopen(filedetails, "w+" ); 
 
	 } 
fclose(f1);  
 
 */
