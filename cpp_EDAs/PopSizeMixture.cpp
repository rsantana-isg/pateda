#include <math.h> 
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "auxfunc.h" 
#include "Popul.h" 
//#include "Treeprob.h" 
//#include "IntTreeprob.h" 
#include "AbstractTree.h" 
#include "MixtureTrees.h" 

 
 
FILE *stream; 
FILE *file,*outfile;  	 
 

   
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
 
void Integers_usualinit(unsigned *Cardinalities) 
{ 
  int i,l,TruncMax; 
  Popul *pop,*selpop; 
  double auxprob;    
  int *AllIndex; 
 
  IntegerTreeModel *MyIntTree; 

  TruncMax = psize*Trunc; 
  Elit = TruncMax;  
  pop = new Popul(psize,vars,Elit,Cardinalities); 
  selpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
  AllIndex = new int[selpop->psize]; 
  SetIndexNormal(AllIndex,selpop); 
  MyIntTree = new IntegerTreeModel(vars,AllIndex,1,selpop); 
  //MyTree = new BinaryTreeModel(vars,AllIndex,1,selpop); 
  i=0; 
  pop->RandInit(); 
  auxprob =0; 
 
  while (i<Maxgen && auxprob<1) 
  { 
     pop->InitIndex(); 
     pop->EvaluateAll(func); 
     pop->TruncSel(selpop,TruncMax);  
     MyIntTree->rootnode = MyIntTree->RandomRootNode(); 
/* 
 	  MyTree->rootnode = MyIntTree->rootnode; 
      MyTree->CalProb(); //Debe ir newpop 
      MyTree->CalMutInf(); 
      MyTree->MakeTree(MyTree->rootnode); 
*/ 
	 MyIntTree->SetPop(selpop); 
     MyIntTree->MakeProbStructures(); 
     MyIntTree->CalMutInf(); 
     MyIntTree->MakeTree(MyIntTree->rootnode); 
 
 
 
         auxprob = MyIntTree->Prob(selpop->P[0]); 
      if(printvals)  printf("%d %f %f \n ",i,selpop->Evaluations[0],auxprob); 
	  // for(l=0;l<vars;l++) printf("%d ",selpop->P[0][l]); 
      // printf("\n "); 
	  // for(l=0;l<vars;l++) printf("%f ",MyIntTree->UnivProb(l, selpop->P[0][l])); 
      // printf("\n "); 
      if (selpop->Evaluations[0]==Max) 
	  {  
 
	   for(l=0;l<Maxgen ;l++)  
		if (l<=i) fprintf(file,"%d ",timevector[l]); 
		else fprintf(file,"%d ",0); 
       fprintf(file,"%e ",selpop->Evaluations[0]); 
       fprintf(file,"%d ",i); 
       fprintf(file,"\n "); 
 
	   succexp++; 
       meangen+= i; 
	   delete[] AllIndex; 
       delete pop; 
       delete selpop; 
       delete MyIntTree; 
       return; 
	  } 
    selpop->SetElit(Elit,pop); 
    MyIntTree->GenPop(Elit,pop);  
 
 
    i++; 
  } 
 
/*  for(l=0;l<Maxgen ;l++)  
		if (l<=i) fprintf(file,"%d ",timevector[l]); 
		else fprintf(file,"%d ",0); 
       fprintf(file,"%e ",selpop->Evaluations[0]); 
       fprintf(file,"%d ",i); 
       fprintf(file,"\n "); 
*/ 
 if (selpop->Evaluations[0]==Max) 
{ succexp++; 
  meangen+= i; 
} 
  delete[] AllIndex; 
  delete pop; 
  delete selpop; 
  delete MyIntTree; 
  return; 
} 
 
void usualinit() 
{ 
  int i,l,TruncMax; 
  Popul *pop,*selpop,*elitpop; 
  double auxprob;    
  int* AllIndex; 
  BinaryTreeModel *MyTree; 
   
  TruncMax = psize*Trunc; 
  if (BestElitism)  Elit = TruncMax;  
  pop = new Popul(psize,vars,Elit); 
   
  if (Tour>0)  
  { 
	 selpop = new Popul(psize,vars,Elit); 
	 elitpop = new Popul(TruncMax,vars,Elit); 
  } else  selpop = new Popul(TruncMax,vars,Elit); 
   
  AllIndex = new int[selpop->psize]; 
  SetIndexNormal(AllIndex,selpop); 
  MyTree = new BinaryTreeModel(vars,AllIndex,1,selpop); 
  i=0; 
  pop->RandInit(); 
  auxprob =0; 
 
  while (i<Maxgen && auxprob<1) 
  { 

     pop->InitIndex(); 
     pop->EvaluateAll(func); 
     if (Tour==0) pop->TruncSel(selpop,TruncMax);  
	 else 
	 { 
	   pop->TruncSel(elitpop,TruncMax);  
	   pop->TournSel(selpop,Tour);  
	 } 
      
	 MyTree->rootnode = MyTree->RandomRootNode(); 
     MyTree->CalProb(); //Debe ir newpop 
  /*for(k=0;k<psize;k++)  
  {  
    for(l=0;l<vars;l++) printf("%d ",pop->P[k][l]); 
	printf("\n "); 
  }*/ 
      MyTree->CalMutInf(); 
      MyTree->MakeTree(MyTree->rootnode); 
      auxprob = MyTree->Prob(selpop->P[0]); 
      if(printvals)  printf("%d %f %f \n ",i,selpop->Evaluations[0],auxprob); 
      if (selpop->Evaluations[0]==Max) 
	  {  
	    //for(l=0;l<vars;l++) printf("%d ",selpop->P[0][l]); 
	    //printf("\n "); 
            //fprintf(file,"%e ",selpop->Evaluations[0]); 
            //fprintf(file,"%d ",i); 
            //fprintf(file,"\n "); 
 
       succexp++; 
       meangen+= i; 
	   delete[] AllIndex; 
       delete pop; 
       delete selpop; 
	   if (Tour==1) delete elitpop; 
       delete MyTree; 
       return; 
	  } 
    if (Tour>0) elitpop->SetElit(Elit,pop); 
	else  selpop->SetElit(Elit,pop); 
 
    MyTree->GenPop(Elit,pop);  
	//MyTree->PopMutation(pop,Elit,1,0.01); 	 
    i++; 
  } 
 
 if (selpop->Evaluations[0]==Max) 
{ succexp++; 
  meangen+= i; 
} 
  //for(l=0;l<Maxgen ;l++)  
	//	if (l<=i) fprintf(file,"%d ",timevector[l]); 
	//	else fprintf(file,"%d ",0); 
 //       fprintf(file,"%e ",selpop->Evaluations[0]); 
 //    fprintf(file,"%d ",i); 
 //    fprintf(file,"\n "); 
 
  delete[] AllIndex; 
  delete pop; 
  delete selpop; 
  if (Tour>0) delete elitpop; 
  delete MyTree; 
  return; 
} 
 
 
 
void MixturesInit(Popul *pop,Popul *newpop,MixtureTrees *Mixture, int SC1,int SC2,double SC3) 
{ 
  int i; 
   
  BinaryTreeModel **TreeArray; 
  int* AllIndex;  
     
  TreeArray = new BinaryTreeModel*[Mixture->NumberTrees]; 
  Mixture->SetPop(pop); 
  AllIndex = new int[pop->psize]; 
  SetIndex(VisibleChoiceVar,AllIndex,pop,0); 
   
  for(i=0; i<Mixture->NumberTrees;i++) 
  { 
    if(VisibleChoiceVar==0) TreeArray[i] = new BinaryTreeModel(pop->vars,AllIndex,1,pop); 
	else TreeArray[i] = new BinaryTreeModel(pop->vars,AllIndex,i,pop); 
    TreeArray[i]->rootnode = TreeArray[i]->RandomRootNode(); 
    TreeArray[i]->CalProb(); //Debe ir newpop 
    TreeArray[i]->CalMutInf(); 
	if (InitTreeStructure==0 )TreeArray[i]->MakeRandomTree(TreeArray[i]->rootnode); 
	else   
	{ 
		TreeArray[i]->MakeTree(TreeArray[i]->rootnode); 
        TreeArray[i]->MutateTree(); 
	} 
    Mixture->SetTrees(i,TreeArray[i]); 
  } 
   
  Mixture->SetStopCriteria(SC1,SC2,SC3); 
  Mixture->LearningMixture(); 
  //newpop->SetElit(newpop->elit,); 
  Mixture->SamplingFromMixture(newpop); 
   
  for(i=0; i<Mixture->NumberTrees;i++) delete TreeArray[i]; 
  delete[] AllIndex; 
  delete[] TreeArray; 
  return; 
} 
 
void Mixtures_Integers_Init(Popul *pop,Popul *newpop,MixtureTrees *Mixture, int SC1,int SC2,double SC3) 
{ 
  int i; 
  IntegerTreeModel **TreeArray; 
  int* AllIndex;  
  TreeArray = new IntegerTreeModel*[Mixture->NumberTrees]; 
  Mixture->SetPop(pop); 
  AllIndex = new int[pop->psize]; 
  SetIndex(VisibleChoiceVar,AllIndex,pop,0); 
   
  for(i=0; i<Mixture->NumberTrees;i++) 
  { 
    if(VisibleChoiceVar==0) TreeArray[i] = new IntegerTreeModel(pop->vars,AllIndex,1,pop); 
	else TreeArray[i] = new IntegerTreeModel(pop->vars,AllIndex,i,pop); 
    TreeArray[i]->rootnode = TreeArray[i]->RandomRootNode(); 
	TreeArray[i]->MakeProbStructures(); 
    TreeArray[i]->CalMutInf(); 
	if (InitTreeStructure==0) TreeArray[i]->MakeRandomTree(TreeArray[i]->rootnode); 
	else  TreeArray[i]->MakeTree(TreeArray[i]->rootnode); 
    Mixture->SetTrees(i,TreeArray[i]); 
  } 
   
  Mixture->SetStopCriteria(SC1,SC2,SC3); 
  Mixture->LearningMixture(); 
  Mixture->SamplingFromMixture(newpop); 
   
  for(i=0; i<Mixture->NumberTrees;i++) delete TreeArray[i]; 
  delete[] AllIndex; 
  delete[] TreeArray; 
  return; 
} 

 
void MixturesAlgorithm(int Type,unsigned *Cardinalities) 
{ 
  int i; 
  Popul *pop,*selpop,*elitpop; 
  int *TOrig; 
  int TruncMax; 
  int auxNsteps; 
  double auxprob,BestEval; 
  double likehood, oldlikehood; 
 
  MixtureTrees *Mixture; 
 
  TruncMax = psize*Trunc; 
  if (BestElitism) Elit = TruncMax;  
  auxNsteps = Nsteps; 
  
  pop = new Popul(psize,vars,Elit,Cardinalities); 
   
 
 if (Tour>0)  
  { 
	 selpop = new Popul(psize,vars,Elit,Cardinalities); 
	 elitpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
  } else selpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
   
  pop->RandInit(); 
  TOrig = new int[Ntrees]; 
  for(i=0; i<Ntrees;i++)  TOrig[i]=i;   
  oldlikehood = 0; 
  likehood = 1; 
 
  Mixture = new MixtureTrees(vars,Ntrees,selpop->psize,0,TOrig,40);//NSteps+1 
 
  i=0; 
  auxprob = 0; 
  BestEval = Max-1;

  while (i<Maxgen && BestEval<Max  && oldlikehood != likehood) 
  { 
   pop->InitIndex(); 
   pop->EvaluateAll(func); 
   if (Tour==0) pop->TruncSel(selpop,TruncMax);  
	 else 
	 { 
	   pop->TruncSel(elitpop,TruncMax);  
	   pop->TournSel(selpop,Tour);  
	 } 
 
   switch(Type) 
   { 
     case 1: MixturesInit(selpop,pop,Mixture,0,auxNsteps,0.5);break; // Mixture of Binary Trees 
     case 3: Mixtures_Integers_Init(selpop,pop,Mixture,0,auxNsteps,0.5);break; // Mixture of Integer Trees  
   }	 
//    auxprob = Mixture->Prob(selpop->P[0]); 
   if (Tour>0) elitpop->SetElit(Elit,pop); 
   else selpop->SetElit(Elit,pop); 

   BestEval = selpop->Evaluations[0];
   if (auxNsteps > 5) auxNsteps -= 5; 
   /* if (auxNsteps < 15) auxNsteps += 3; 
   { 
    oldlikehood = likehood; 
    likehood = Mixture->LikehoodValues[auxNsteps-3]; 
   } 
   */  
   oldlikehood = likehood; 
   likehood = Mixture->LikehoodValues[auxNsteps];
   if(printvals)  printf(" %f %f  %f \n ",BestEval,Mixture->LikehoodValues[auxNsteps],auxprob); 
   //fprintf(file,"%f ",BestEval); 
   //fprintf(file,"%d ",i); 
   //fprintf(file,"\n "); 
   i++; 
  } 
  if (BestEval==Max) 
  {  
      succexp++; 
      meangen+= i-1; 
  } 
 
  delete[] TOrig; 
  delete Mixture; 
  delete pop; 
  delete selpop; 
  if (Tour>0) delete elitpop;  
  return; 
} 
 
 
void MixturesAlgorithmTourn(int Type,unsigned *Cardinalities) 
{ 
  int i; 
  Popul *pop,*selpop,*auxpop; 
  int *TOrig; 
  int TruncMax; 
  int auxNsteps; 
  double auxprob; 
  double likehood, oldlikehood; 
 
  MixtureTrees *Mixture; 
 
  TruncMax = psize*Trunc; 
  Elit = TruncMax;  
  auxNsteps = Nsteps; 
  
  pop = new Popul(psize,vars,Elit,Cardinalities); 
  selpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
  auxpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
  pop->RandInit(); 
   
  TOrig = new int[Ntrees]; 
  for(i=0; i<Ntrees;i++)  TOrig[i]=i;   
  oldlikehood = 0; 
  likehood = 1; 
 
  Mixture = new MixtureTrees(vars,Ntrees,selpop->psize,0,TOrig,Nsteps+1); 
 
  i=0; 
  auxprob = 0; 
  auxpop->Evaluations[0] = 0; 
  while (i<Maxgen && auxpop->Evaluations[0]<Max  && oldlikehood != likehood) 
  { 
   pop->InitIndex(); 
   pop->EvaluateAll(func); 
   pop->TournSel(selpop,Tour); 
   pop->TruncSel(auxpop,TruncMax); 
    
  // auxprob = Mixture->Prob(selpop->P[0]); 
   switch(Type) 
   { 
     case 1: MixturesInit(selpop,pop,Mixture,0,auxNsteps,0.5);break; // Mixture of Binary Trees 
     case 3: Mixtures_Integers_Init(selpop,pop,Mixture,0,auxNsteps,0.5);break; // Mixture of Integer Trees  
   }	 
//    auxprob = Mixture->Prob(selpop->P[0]); 
   auxpop->SetElit(Elit,pop); 
   if (auxNsteps > 3) auxNsteps -= params[2]; 
   oldlikehood = likehood; 
 
   likehood = Mixture->LikehoodValues[auxNsteps]; 
   if(printvals)  printf("        %d  %f %f  %f \n ",i,auxpop->Evaluations[0],Mixture->LikehoodValues[auxNsteps],auxprob); 
   fprintf(file,"%e ",auxpop->Evaluations[0]); 
   fprintf(file,"%d ",i); 
   fprintf(file,"\n "); 
   i++; 
  } 
  if (auxpop->Evaluations[0]==Max) 
  {  
	  succexp++; 
      meangen+= i; 
  } 
 
  delete[] TOrig; 
  delete Mixture; 
  delete pop; 
  delete auxpop; 
  delete selpop; 
   
  return; 
} 
 
 
void main() 
{ 
	int i,T; 
 
	stream = fopen( "Param.txt", "r+" ); 
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
         fscanf( stream, "%d", &BestElitism); // If there is or not BestElitism
	 fscanf( stream, "%d", &params[0]); // Params for function evaluation 
	 fscanf( stream, "%d", &params[1]); 
	 fscanf( stream, "%d", &params[2]); 
	 fscanf( stream, "%d", &Card); // Cardinal for all variables 
	 fscanf( stream, "%d", &seed); // seed 
	 fclose( stream ); 
    } 
 
 
Trunc = T/double(100); 
Max = auxMax/double(100);  
  
srand((unsigned) time( NULL ) ); 

 
 int auxint[16]={200,400,600,800,1000,1200,1500,2000,2500,3000,3500,4000,6000,8000};
 int varvals[6] = {21,37,69,101};
 double optvals[6] = {14.8,25.6,47.2,68.8};

 int alph,number,beg;
 double auxmeangen, meanfit;
beg = 0;
 outfile = fopen(MatrixFileName, "wr+" );
 //for(vars=21;vars<=21;vars+=21)
 for(int v=0;v<3;v++)
  {
    vars = varvals[v];
    Max = optvals[v];
    Cardinalities  = new unsigned[vars]; 
    for(i=0;i<vars;i++) Cardinalities[i] = Card;
    //Max = vars;
	for(alph=beg;alph<14;alph++)
	  { 
            psize = auxint[alph];
	    for(Ntrees=1;Ntrees<3;Ntrees++)	
	      { 
               
                ExperimentMode = Ntrees-1;        
                
                succexp = 0; 
                meangen = 0;
                for(i=0;i<cantexp;i++) 
                  { 
	           switch(ExperimentMode) 
                     { 
                       case 0: usualinit();break; // Tree 
                       case 1: MixturesAlgorithm(ExperimentMode,Cardinalities);break; // Mixture of Binary Trees 
                       case 2: Integers_usualinit(Cardinalities); break; // Integer Tree 
                       case 3: MixturesAlgorithm(ExperimentMode,Cardinalities);break; // Mixture of Integer Trees  
                     }
                     if( (i-succexp)>10) i=cantexp;	 
                   }
                 
                 if (succexp>0) 
                   {
                    auxmeangen = meangen/succexp;
                    if (BestElitism) meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;       else meanfit = (auxmeangen+1)* (psize-1) + 1;
                    printf("%d  %d %d %d %d  %e %e \n",vars, psize, Ntrees, i, succexp,  auxmeangen, meanfit ); 
                    fprintf(outfile, "%d  %d  %d %d %d  %e %e \n",vars, psize, Ntrees, i, succexp,  auxmeangen, meanfit );
                   }
                  else 
                   {
                    printf("%d  %d %d %d %d   \n",vars, psize, Ntrees, i+1, succexp); 
                    fprintf(outfile, "%d %d %d %d  %d   \n",vars, psize, Ntrees,i+1, succexp); 
                    printf("%d  %d %d %d %d   \n",vars, psize, Ntrees, i+1, succexp);

                   }
              
	   }//Ntrees
         if (succexp>90) 
	   { beg = alph; alph = 20;}
     	 }//alph 
        delete[] Cardinalities; 
      }//vars 
 delete[] params;
fclose(outfile);
}     

 /*
filedetails[0]=0;
 strcat(filedetails,MatrixFileName);
           itoa(vars,number,10);
	   strcat(filedetails,number);
           strcat(filedetails,"_");
	   itoa(psize,number,10);
	   strcat(filedetails,number);
           strcat(filedetails,"_");
           itoa(Ntrees,number,10);
	   strcat(filedetails,number);
           strcat(filedetails,".txt");
  */







