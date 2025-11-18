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

 
void Integers_usualinit(unsigned *Cardinalities) 
{ 
  int i,l,TruncMax; 
  Popul *pop,*selpop; 
  double auxprob;    
  int *AllIndex; 
 
  IntegerTreeModel *MyIntTree; 

  TruncMax = int(psize*Trunc);
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
 
void usualinit(double Complexity) 
{ 
  int i,l,TruncMax,NSelPoints,NPoints; 
  Popul *pop,*selpop,*elitpop,*compact_pop; 
  double auxprob,sumprob;    
  double univ_prior, biv_prior;
  double* fvect;
  int* AllIndex; 
  BinaryTreeModel *MyTree; 
   
  TruncMax = int(psize*Trunc); 
  if (BestElitism)  Elit = TruncMax;  
  pop = new Popul(psize,vars,Elit); 
   
 
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
   
  while (i<Maxgen && auxprob<1) 
  { 
    
     pop->InitIndex();
     pop->EvaluateAll(func); 
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
     // MyTree->CalMutInf(); 
     //MyTree->MakeTree(MyTree->rootnode);
     MyTree->CalculateILikehood(compact_pop,fvect); 
     MyTree->MakeTreeLog();
     

    for (int l=0;l<NPoints;l++)
      {  
	 for(int ll=0;ll<vars;ll++) cout<<compact_pop->P[l][ll]<<" "; 
	 cout<<MyTree->Prob(compact_pop->P[l])<<" Real "<<fvect[l];
         cout<<endl;
      }
      cout<<endl;
     //MyTree->PrintMut();
  

     MyTree->CollectKConf(20,vars,MyTree,pop);
 
     sumprob = MyTree->SumProb(compact_pop,NPoints);
     

        switch(Prior) 
                     { 
                       case 0: break; // No prior 
                       case 1: univ_prior = Calculate_Best_Prior(TruncMax,vars,1,3*SelInt);
                               biv_prior = Calculate_Best_Prior(TruncMax,vars,2,3*SelInt);
                               MyTree->SetPrior(univ_prior,biv_prior,TruncMax);
                               break; // Recommended prior 

                       case 2: 
                                 univ_prior = Calculate_Best_Prior(TruncMax,vars,1,SelInt*(1.0+2*sumprob));
                                 biv_prior = Calculate_Best_Prior(TruncMax,vars,2,SelInt*(1.0+2*sumprob));
                                 MyTree->SetPrior(univ_prior,biv_prior,TruncMax);
                                
                               break; // Adaptive Prior                       
                     } 



     MyTree->CalculateILikehood(compact_pop,fvect);
    
    
     MyTree->PrintModel();
     cout<<endl;
     /* MyTree->PrintProbMod();
     cout<<endl;
    */

     /*    
     if(MyTree->TreeProb==1)
     for (l=0;l<NPoints;l++)
      {  
	 for(int ll=0;ll<vars;ll++) cout<<compact_pop->P[l][ll]<<" "; 
	 cout<<MyTree->Prob(compact_pop->P[l])<<" ";
         cout<<endl;
      }
      */     

     
     
      auxprob = MyTree->Prob(selpop->P[0]); 
       
      if(printvals) 
      { 
       for(l=0;l<vars;l++) cout<<" "<<selpop->P[0][l]; 
        cout<<endl; 
        cout<<"Gen :"<<i<<" Best:"<<selpop->Evaluations[0]<<" ProbBest:"<<auxprob<<" DifPoints:"<<NPoints<<" TreProb:"<<MyTree->TreeProb<<" Likehood "<< MyTree->Likehood<<endl; 
      }
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
 
      //MyTree->GenPop(Elit,pop);  
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
  delete[] fvect;
  delete compact_pop;
  delete pop; 
  delete selpop; 
  if (Tour>0) delete elitpop; 
  delete MyTree; 
  return; 
} 
 
void MixturesInit(Popul *pop,Popul *newpop,MixtureTrees *Mixture, int SC1,int SC2,double SC3, double* pvect, double Complexity) 
{ 
  int i; 
   
  BinaryTreeModel *OneTree; 
 
      
  Mixture->SetPop(pop); 
 
 
  for(i=0; i<Mixture->NumberTrees;i++) 
  { 
    InitTreeStructure=0; //CAMBIAR
        OneTree =  new BinaryTreeModel(pop->vars, Complexity); 
        OneTree->SetGenPoolLimit(Mixture->CNumberPoints);
        OneTree->rootnode = OneTree->RandomRootNode();    
        OneTree->SetNPoints(Mixture->NumberPoints);	  
        InitTreeStructure=0;
	if(InitTreeStructure==0 )
         {
          Mixture->RandomLambdas();	
          OneTree->MakeRandomTree(OneTree->rootnode);
          OneTree->RandParam(); 
         }
        else if(InitTreeStructure==1)
         { 
          Mixture->RandomLambdas();	
          OneTree->MakeRandomTree(OneTree->rootnode);
          OneTree->CalProbFvect(pop,pvect,Mixture->CNumberPoints); 
         }
     	else   
	{  
          Mixture->UniformLambdas();	
          OneTree->CalProbFvect(pop,pvect,Mixture->CNumberPoints);   
          OneTree->CalMutInf(); 
	  OneTree->MakeTree(OneTree->rootnode); 
          OneTree->MutateTree(); 
	} 
        Mixture->EveryTree[i]=OneTree; 
  } 

   Mixture->LearningMixture(); 
   Mixture->SamplingFromMixture(newpop);
   
 
   for(i=0; i<Mixture->NumberTrees;i++)
     {  
      delete Mixture->EveryTree[i]; 
     }
   return; 
} 
 

void testrandom(int v,int iter, unsigned* Cardinalities )
{
 int i,j,psize;
 Popul *pop; 
 double sumprob;   

  psize = int(pow(2,v));
  Elit = 0;
  pop = new Popul(psize,v,0,Cardinalities);  

  pop->InitIndex(); 
  pop->ProbInit(); 

  for(i=0; i<iter;i++) 
    {
      cout<<i<<endl;    
      BinaryTreeModel *OneTree; 

 
  OneTree = new BinaryTreeModel(v,1); 	  
  OneTree->rootnode = OneTree->RandomRootNode();
  OneTree->MakeRandomTree(OneTree->rootnode);
  OneTree->RandParam(); 
  sumprob = 0;
  for(j=0; j<psize;j++) sumprob += OneTree->Prob(pop->P[j]);
  //cout<<"sumprob="<<sumprob<<endl;
  delete OneTree;
    }

 delete pop;
}

void MixturesInitExact(Popul *pop,Popul *newpop,MixtureTrees *Mixture, int SC1,int SC2,double SC3, double* pvect,double Complexity) 
{ 
 int i;   
 BinaryTreeModel *OneTree;
 double Likeh;

  Mixture->SetPop(pop);   
  Likeh = Mixture->CalculateOptimalLikehood();  
  //cout<<"  Likehood Optimum "<<Likeh<<endl; 
  for(i=0; i<Mixture->NumberTrees;i++) 
  {  
    OneTree = new BinaryTreeModel(pop->vars,Complexity); 
    OneTree->SetNPoints(Mixture->NumberPoints);	  
    OneTree->SetGenPoolLimit(Mixture->CNumberPoints);
    OneTree->rootnode = OneTree->RandomRootNode();
    OneTree->CalProbFvect(pop,pvect,Mixture->CNumberPoints); 
   
    //OneTree->CalculateILikehood(pop,pvect);  
    //cout<<"  LikehoodUnivariate "<<OneTree->Likehood<<endl; 
    //OneTree->CalMutInf();
    //OneTree->MakeTree(OneTree->rootnode);
    OneTree->ConstructTree();
    //OneTree->CalculateILikehood(pop,pvect);  
    //cout<<"  LikehoodBestTree "<<OneTree->Likehood<<endl; 
    if(i==1)
     {
       OneTree->CleanTree();
       //OneTree->CalProbUnif();
     }
    Mixture->EveryTree[i] = OneTree;
  } 

  Mixture->LearningMixtureExact(Coeftype,Complexity); 
  Mixture->SamplingFromMixture(newpop); 
  //Mixture->LearningMixtureGreedy(); 

  for(i=0; i<Mixture->NumberTrees;i++) 
    {     
      delete Mixture->EveryTree[i]; 
      // cout<<Mixture->NumberTrees<<"  "<<i;
    }
  
  //cout<<endl;
  return; 
} 


void MixturesInitGreedy(Popul *pop,Popul *newpop,MixtureTrees *Mixture, int SC1,int SC2,double SC3, double* pvect, double Complexity) 

{ 
  int i;    
  BinaryTreeModel *OneTree; 
 
  
  Mixture->SetPop(pop); 
 
  OneTree = new BinaryTreeModel(pop->vars,Complexity); 	  
  OneTree->SetGenPoolLimit(Mixture->CNumberPoints);
  OneTree->rootnode = OneTree->RandomRootNode();
  //InitTreeStructure=0; 
  if(InitTreeStructure==0 )
   {

    OneTree->MakeRandomTree(OneTree->rootnode);
    OneTree->RandParam(); 
   }
  else if(InitTreeStructure==1)
    { 
     OneTree->MakeRandomTree(OneTree->rootnode);
     OneTree->CalProbFvect(pop,pvect,Mixture->CNumberPoints); 
    }
  else 
    {
     OneTree->CalProbFvect(pop,pvect,Mixture->CNumberPoints); 
     OneTree->CalMutInf();
     OneTree->MakeTree(OneTree->rootnode);
    }

  Mixture->EveryTree[0] = OneTree;

  for(i=1; i<Mixture->NumberTrees;i++) 
  { 
    OneTree = new BinaryTreeModel(pop->vars,Complexity); 	  
    OneTree->SetGenPoolLimit(Mixture->CNumberPoints);
    Mixture->EveryTree[i] = OneTree;
  } 
  

  Mixture->LearningMixtureGreedy1(); 
  Mixture->SamplingFromMixture(newpop); 

  for(i=0; i<Mixture->NumberTrees;i++) 
    {     
      delete Mixture->EveryTree[i]; 
      // cout<<Mixture->NumberTrees<<"  "<<i;
    }
  //cout<<endl;
  return; 
} 


void Mixtures_Integers_Init(Popul *pop,Popul *newpop, MixtureTrees *Mixture, int SC1,int SC2,double SC3, double* pvect) 
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
   
 
  delete[] AllIndex; 
  delete[] TreeArray; 
  return; 
} 



void MixturesAlgorithm(int Type,unsigned *Cardinalities,double Complexity) 
{ 
  int i; 
  Popul *pop,*selpop,*elitpop,*compact_pop; 
  int TruncMax,NPoints; 
  double auxprob,BestEval; 
  double *pvect;
 
  MixtureTrees *Mixture; 
 
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
   
  
 pop->RandInit(); 
  
 
  Mixture = new MixtureTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior);//NSteps+1 
 
  i=0; 
  auxprob = 0; 
  BestEval = Max-1;
  NPoints = 2;

  while (i<Maxgen && BestEval<Max && NPoints>1)  //&& oldlikehood != likehood) 
  { 

   pop->InitIndex(); 
   pop->EvaluateAll(func); 
   if (Tour==0) pop->TruncSel(selpop,TruncMax);  
	 else 
	 { 
	   pop->TruncSel(elitpop,TruncMax);  
	   pop->TournSel(selpop,Tour);  
	 } 


    NPoints = selpop->CompactPop(compact_pop,pvect);

    
    // cout<<"Npoints is "<<NPoints<<endl;
    Mixture->SetNpoints(NPoints,pvect);

    /*           
for(int k=0;k<NPoints;k++)  
  {  
    for(int l=0;l<vars;l++) printf("%d ",compact_pop->P[k][l]); 
	printf("\n "); 
  } 
    */
   switch(Type) 
   { 
     case 1: MixturesInit(compact_pop,pop,Mixture,StopCrit,Nsteps,0.5,pvect,Complexity);break; // Mixture of Binary Trees 
     case 2: MixturesInitGreedy(compact_pop,pop,Mixture,StopCrit,Nsteps,0.5,pvect,Complexity);break; // Mixture with Greedy Learning 
     case 3: Mixtures_Integers_Init(compact_pop,pop,Mixture,StopCrit,Nsteps,0.5,pvect);break; // Mixture of Integer Trees
     case 4: MixturesInitExact(compact_pop,pop,Mixture,StopCrit,Nsteps,0.5,pvect,Complexity); break; 
   }
	 
  // auxprob = Mixture->Prob(selpop->P[0]); 
   if (Tour>0) elitpop->SetElit(Elit,pop); 
   else selpop->SetElit(Elit,pop); 

   BestEval = selpop->Evaluations[0];
   //auxprob = Mixture->Prob(selpop->P[0]); 

   //for(int l=0;l<vars;l++) cout<<" "<<selpop->P[0][l]; 
   // cout<<endl;  
      if(printvals)  
        cout<<"Gen :"<<i<<" Best:"<<BestEval<<" DifPoints:"<<NPoints<<endl; 
   //if(printvals)  printf(" %d %f %f  %f \n ",i,BestEval,Mixture->LikehoodValues[Nsteps],auxprob); 
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
 
 
  delete Mixture; 
  delete pop; 
  delete selpop; 
  delete compact_pop;
  delete[] pvect;
  if (Tour>0) delete elitpop;  
  return; 
} 
 


void MixtureStats(int Type,unsigned *Cardinalities, int nexp, double beta, double *meanl, double *varl, double *lsup,double *nsteps, double *eprob, double *aprob, double Complexity) 
{ 
  int i,TruncMax; 
  Popul *pop,*selpop; 
  double *pvect, *likehood;
  MixtureTrees *Mixture; 
  double* Alllikehoods;

  
  psize = int(pow(2,vars));
  Elit = 0;
  pop = new Popul(psize,vars,Elit,Cardinalities); 
 
  pop->InitIndex(); 
  pop->ProbInit(); 
  pop->EvaluateAll(func); 

  TruncMax = int(psize*Trunc);
  selpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
  

  pop->TruncSel(selpop,TruncMax);  
  psize = TruncMax;
  pvect = new double[psize];
  selpop->BotzmannDist(beta,pvect);

  //for(int k=0;k<psize;k++)  pvect[k] = 1/double(psize); // Inicializacion uniforme
  
  //selpop->ProporDist(pvect);
  //pop->EvaluateAll(func); 
  *eprob = pvect[0]; 

  

  /* 
              
for(int k=0;k<psize;k++)  
  {  
  for(int l=0;l<vars;l++)  cout<<selpop->P[k][l]<<" "; 
  cout<<"Prob "<<pvect[k]<<endl; 
  } 
  */

 Alllikehoods = new double[nexp];
 (*lsup) = 0;
 (*meanl) = 0;
  (*nsteps) =0;
  (*aprob) = 0;
 for(i=0;i<nexp;i++)  
  {
    
    //   cout<<"t"<<i<<endl;  
   Mixture = new MixtureTrees(vars,Ntrees,psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior);
   Mixture->SetNpoints(psize,pvect);
   
   switch(Type) 
   { 
     case 1: MixturesInit(selpop,pop,Mixture,StopCrit,Nsteps,0.5,pvect,Complexity);break; // Mixture of Binary Trees 
     case 2: MixturesInitGreedy(selpop,pop,Mixture,StopCrit,Nsteps,0.5,pvect,Complexity);break; // Mixture with Greedy Learning  
     case 3: MixturesInitExact(selpop,pop,Mixture,StopCrit,Nsteps,0.5,pvect,Complexity);break; // Mixture with edge learning   
   }	 

   Alllikehoods[i] = Mixture->BestLikehood;
   if (Type==2) (*lsup) += Mixture->lsup;
   //cout<<Alllikehoods[i]<<endl;
   (*nsteps) += double(Mixture->Count);
   (*meanl) += Alllikehoods[i];
   (*aprob) += (Mixture->BestProb);
   delete Mixture; 
  } 

 (*meanl) = (*meanl) / nexp;  //Average of the likehood;
 (*aprob) = (*aprob) / nexp; 
 (*varl) = 0;

 for(i=0;i<nexp;i++)  (*varl) += fabs(Alllikehoods[i]- (*meanl));  // variance of the likehood;
 (*varl) = (*varl)/nexp;
 (*nsteps) /= double(nexp);
 (*lsup) /= nexp;

 //  cout<<endl;
  delete[] Alllikehoods;
  delete pop; 
  delete selpop;
  delete[] pvect;
  
  return; 
} 


/* 
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
 
  TruncMax = int(psize*Trunc); 
  Elit = TruncMax;  
  auxNsteps = Nsteps; 
  
  pop = new Popul(psize,vars,Elit,Cardinalities); 
  selpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
  auxpop = new Popul(TruncMax,vars,Elit,Cardinalities); 
  pop->RandInit(); 
   
  
  oldlikehood = 0; 
  likehood = 1; 
 
  Mixture = new MixtureTrees(vars,Ntrees,selpop->psize,0,Nsteps+1,MaxMixtProb,S_alpha,SelInt,Prior); 
 
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
     case 1: MixturesInit(selpop,pop,Mixture,StopCrit,auxNsteps,0.5);break; // Mixture of Binary Trees 
     case 3: Mixtures_Integers_Init(selpop,pop,Mixture,StopCrit,auxNsteps,0.5);break; // Mixture of Integer Trees  
   }	 
//    auxprob = Mixture->Prob(selpop->P[0]); 
   auxpop->SetElit(Elit,pop); 
    
 
 
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
 
 
  delete Mixture; 
  delete pop; 
  delete auxpop; 
  delete selpop; 
   
  return; 
} 
*/ 
 
int main(){ 
	
        int i,k,T,MaxMixtP,S_alph,u,uu,Compl; 
 
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
	 fclose( stream ); 
    } 
	
Trunc = T/double(100); 
Complex  = Compl/double(100); 

if(T>0)
 { 
   div_t res;
   res = div(T,5); 
   SelInt = Sel_Int[res.quot]; // Approximation to the selection intensity for truncation.
 }

 
Max = auxMax/double(100);  
MaxMixtProb =MaxMixtP/double(100);
S_alpha = S_alph/double(100);

 unsigned ta = (unsigned) time(NULL); 
 srand(ta); //1005303313
 //srand(1022260481);
 //srand(1021560914); //Valor de experimentos
 // srand(1022275142);
 //srand(1022703130);
 //srand(1022714261); 
 //srand(1022717223);
 //srand(1022766681);   
// srand(1022777877); 
 //srand(1022780940);
 srand(1022789199);

 cout<<"seed"<<ta<<endl;


 
 // int funci[4]={8,15,25};

 int alph,number,j;
 double auxmeangen, meanfit;
 

 outfile = fopen(MatrixFileName, "wr+" );

/*     
 double meanl,varl,lsup;
 double nsteps;
 double eprob, aprob;

 //for(int kk=0;kk<=2;kk+=1)
  for(u=6;u<=6;u+=10)
   for(j=6;j<=6;j+=1)
    for(k=3;k<=3;k+=1)
   {
  
       //vars = j;
    Ntrees=2;
    //func = 23; //15funci[kk];
    Cardinalities  = new unsigned[vars]; 
    for(i=0;i<vars;i++) Cardinalities[i] = Card;
        
        
/*
    MixtureStats(1,Cardinalities,1,u*0.5,&meanl,&varl,&lsup,&nsteps,&eprob,&aprob);
    cout<<"Meila "<<" t="<< u<<" Nt= "<<Ntrees<<" n = "<<vars<< "  mean "<<meanl<<" varl  "<<varl<<" sup "<<lsup<<" eprob "<<eprob <<" aprob "<<aprob<<" nsteps "<<nsteps<<" f= "<<func<<endl;
     
   
    MixtureStats(2,Cardinalities,100,u*0.5,&meanl,&varl,&lsup,&nsteps,&eprob,&aprob);
    cout<<"Other  "<<" t="<< u<<" Nt= "<<Ntrees<<" n = "<<vars<< "  mean "<<meanl<<" varl  "<<varl<<" sup "<<lsup<<" eprob "<<eprob <<" aprob "<<aprob<<" nsteps "<<nsteps<<" f= "<<func<<endl;
*/
 /* 
  MixtureStats(3,Cardinalities,1,u*0.5,&meanl,&varl,&lsup,&nsteps,&eprob,&aprob,1);
   cout<<"Other  "<<" t="<< u<<" Nt= "<<Ntrees<<" n = "<<vars<< "  mean "<<meanl<<" varl  "<<varl<<" sup "<<lsup<<" eprob "<<eprob <<" aprob "<<aprob<<" nsteps "<<nsteps<<" f= "<<func<<endl;
    delete[] Cardinalities;
   }
 } 
 */

 int funci[5]={23,15,19,25,8};
 int popul[15]={1000,1500,2000,1000,1000,5000,5000,1000,1000,5500};
 int ntt[5]={8,12};
 double optim[15]={721.0,1717.0,3361.0,211.0,1191.0,99.0,100.0,911.0,1191.0,33.0};

  
              
                //if (Ntrees>1) ExperimentMode = 1; else ExperimentMode = 0;  
 //for(int uu=0;uu<=0;uu++)     
 //    for(u=0;u<3;u++) 
           { 		
               //Complex = u/100.0;
               //func = funci[u];
	       /*
                vars = ((u+3)*3)*((u+3)*3);                       
	        psize =  popul[u];
                Max = optim[u];
	        Ntrees = ntt[uu];
                */
                Cardinalities  = new unsigned[vars]; 
                for(i=0;i<vars;i++) Cardinalities[i] = Card;
                succexp = 0; 
                meangen = 0;
                for(i=0;i<cantexp;i++) 
                  { 		   
	           switch(ExperimentMode) 
                     {                    
                       case 0: usualinit(Complex);break; // Tree 
                       case 1: MixturesAlgorithm(1,Cardinalities,Complex);break; // Mixture, Meila Algorithm 
                       case 2: Integers_usualinit(Cardinalities); break; // Integer Tree 
                       case 3: MixturesAlgorithm(3,Cardinalities,Complex);break; // Mixture of Integer Trees  
                       case 4: MixturesAlgorithm(2,Cardinalities,Complex);break; // Mixture, Incremental Algorithm  
                       case 5: MixturesAlgorithm(4,Cardinalities,Complex);break; // Mixture, Quadratic Algorithm  
                     }
                     
                  }                
                
                 if (succexp>0) 
                   {
                    auxmeangen = meangen/succexp;
                    if (BestElitism) meanfit = (auxmeangen+1)*(1-Trunc)*psize + psize*Trunc;       else meanfit = (auxmeangen+1)* (psize-1) + 1;
                    printf("%d %d %d %d %d  %d  %e %d %d %d  %e %e  \n",ExperimentMode, Prior,func,vars, psize,Nsteps,MaxMixtProb, Ntrees, i, succexp,  auxmeangen, meanfit);                   
                    fprintf(outfile, "%d %d %d  %d  %d %d %e %d %d  %e %e \n",ExperimentMode, Prior,func,vars, psize, Nsteps, Ntrees,MaxMixtProb, i, succexp,  auxmeangen, meanfit);
                   }
                  else 
                   { 
                   printf("%d %d %d  %d %d %d %e %d %d %d  %e %e %d  %d \n",ExperimentMode, Prior,func, vars, psize,Nsteps,MaxMixtProb, Ntrees, i, 0,  0.0, 0.0); 
                   fprintf(outfile," %d %d  %d %d %d %d %e %d %d %d  %e %e %d  %d \n",ExperimentMode, Prior, func,vars, psize,Nsteps,MaxMixtProb, Ntrees, i, 0,  0.0, 0.0); 
		   //printf("%d  %d %d %e %d %d   %d  \n",vars, psize, Ntrees,MaxMixtProb, i, succexp,u); 
                   // fprintf(outfile, "%d %d %d %d  %d   %d \n",vars, psize, Ntrees,i+1, succexp, u); 
                    //printf("%d  %d %d %d %d   \n",vars, psize, Ntrees, i, succexp);

                   }                

		 delete[] Cardinalities;

	   }// for u
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









