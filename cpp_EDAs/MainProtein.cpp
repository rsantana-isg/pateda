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
#include "affinity.h" 
#include "JtreeTable.h" 
#include "jtPartition.h" 

  
FILE *stream;  
FILE *file,*outfile;  	  
  

 
double meanlikehood[500]; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798};  
double SelInt; 


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
double BestEval,AbsBestEval,AuxBest; 
int TruncMax; 
int NPoints;  
unsigned int  *BestInd, *AbsBestInd;  
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



int  FindMaxConf (int elit, JtreeTable* Table, Popul* Mypop, Popul* Otherpop, int cantconf ) 
 {
     Table->Compute(Mypop); 
         
     Table->PassingFluxesTopDown(); 
 
     Table->PassingFluxesBottomUp(); 
 
     Table->FindBestConf(); 
 
     double  aux = Table->GetMaxProb(); 
 
     JtPartition*  Pt = new JtPartition(cantconf, Table->getNoVars(),Table->getNoCliques()); 
        
     Pt->Add(Table,aux); 
   
     Pt->Cycle(); 
 
     cantconf = (Pt->GetCantConf()+1 > Otherpop->psize) ? Otherpop->psize :Pt->GetCantConf()+1;  
  
     //printf("Popsizes %d , %d\n",cantconf ,Otherpop->psize);      
 
     Pt->SetPop(elit, Otherpop,cantconf); 
 
     //Otherpop->Print(); 
         
     delete Pt; 
     
     return cantconf; 
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

      i=2;
      while(i<vars && (epop->P[k][i]!=0 && epop->P[k][i]!=3) ) i++;
      	{
         if (i<vars && epop->P[k][i]==3) 
 	  for(j=i;j<vars;j++) 
           {
              if(epop->P[k][j]==3)  epop->P[k][j] = 0;
              else if (epop->P[k][j]==0)  epop->P[k][j] = 3;
              if(epop->P[k][j]==2)  epop->P[k][j] = 4;
              else if (epop->P[k][j]==4)  epop->P[k][j] = 2;
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


void MutatePop(Popul* epop,int nelit, int epsize,double ModVal)
{
 int k,j;
 double CurrentEval;

 
for(k=nelit; k < epsize; k ++)  
 {
   
  for(j=2; j < vars; j ++)  if(myrand()<ModVal)   epop->P[k][j] = randomint(3);
   
  /*
  FoldingProtein->CallRepair(epop->P[k],vars); 
  CurrentEval = FoldingProtein->ProteinLocalPerturbation(sizeProtein,epop->P[k],CurrentEval,&eval_local,15);
  CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[k]);
  CurrentEval = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[k]);
  epop->SetVal(k,CurrentEval);
  */
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
   //CurrentEval = FoldingProtein->EvalOnlyVectorModel(sizeProtein,epop->P[pos]); // For functional protein model
    eval_local = 1;   
    //CurrentEval1 = FoldingProtein->TabuOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5,10);
    //epop->SetVal(pos,CurrentEval1);
  
    //CurrentEval1 = FoldingProtein->ProteinLocalOptimizer(sizeProtein,epop->P[pos],CurrentEval,&eval_local,5);
    // if(CurrentEval1<CurrentEval) cout<<"One example here"<<endl;
    // totinc += (CurrentEval1>CurrentEval); 
    // epop->SetVal(pos,CurrentEval1);
   
     epop->SetVal(pos,CurrentEval); 
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








void EvalProteinNew(Popul* epop,int nelit, int epsize, int atgen)
{

 double bestvalprot,valprot,origval,globalb;
 int i,a,l,j,k,mm, bef,bestpos,bestval,improvement,extent;
 int eval_local,candidates,auxpos;
 unsigned int auxbuff[200];
 unsigned int bestbuff[200];
 unsigned int candbestpos[400];
 unsigned int candbestval[400];
 
 double bestb,auxbest;
 double* auxglobal = &bestb;
 int start, pos;
 if (atgen==0) start=0;
 else start=nelit;
 eval_local = 1;   

for(l=start; l < epsize;   l++)  
 {
   pos = l; 
      
    FoldingProtein->CallRepair(epop->P[pos],vars);              
    bestvalprot = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
    TotEvaluations += eval_local;    
  
 
 if(pos>Elit && pos<=Elit+100)  
    {  
      for(i=0;i<sizeProtein;i++) auxbuff[i] =   epop->P[pos][i];
      globalb = 0;
      //cout<<" atgen "<<atgen<<" pos "<<pos<<" best  "<<bestb<<"  ABS "<<AbsBestEval<<" valprot "<<bestvalprot<<endl;      
      k = 0;
      improvement = 0;   
      for(i=0;i<sizeProtein;i++) bestbuff[i] =   auxbuff[i];
      while (k==0 || improvement==1)
      {
       candidates = 0;
       origval = bestvalprot;
       for(i=2;i<sizeProtein;i++)
	 {
	  bef = bestbuff[i];
          
          for(j=0;j<FoldingProtein->nmoves;j++)
           {
	     if (j!=bef) bestbuff[i] = j;
	        valprot = FoldingProtein->EvalOnlyVector(sizeProtein,bestbuff);    
                 TotEvaluations += eval_local;    
               //bestb = valprot;   
             
          
             if(valprot >= bestvalprot) 
             {      
               	      
               if(valprot > bestvalprot)
                {
                 candbestpos[0] = (unsigned int) i;
                 candbestval[0] = (unsigned int) j;
                 candidates = 1; 
                }        
               else 
                {
                  candbestpos[candidates] = (unsigned int) i;
                  candbestval[candidates] = (unsigned int) j;
                  candidates++;
                }
	       // cout<<pos<<" "<<k<<"  "<<origval<<" "<< valprot<<" "<<bestvalprot<<" "<<i<<"  "<<j<<" "<<candidates<<endl;
 	       bestvalprot = valprot;                            
             }
            
            
	   }
           bestbuff[i] = (unsigned int) bef;
	 }
        improvement = bestvalprot > origval;
             
          if(improvement==1)
	  {       
            auxpos = randomint(candidates);
            bestpos = candbestpos[auxpos];
            bestval = candbestval[auxpos];
	    bestbuff[bestpos] = bestval;             
	    //cout<<" "<<k<<"  "<<origval<<" "<<bestvalprot<<" "<<bestpos<<" "<<bestval<<" "<<globalb<<endl;
           
          }
      
       k++;         
      }
      //TotEvaluations+= k*(sizeProtein-2);
      
       for(i=0;i<sizeProtein;i++)  epop->P[pos][i] = bestbuff[i];
    }
     
      epop->SetVal(pos,bestvalprot); 
      if(bestvalprot>=AbsBestEval)
	{
	  for(i=0;i<sizeProtein;i++)    AbsBestInd[i] = epop->P[pos][i];
          AbsBestEval = bestvalprot;          
        }
 }

    
}








void EvalProteinWithCO(Popul* epop,int nelit, int epsize, int atgen)
{
 int k,eval_local;
 double CurrentEval,CurrentEval1;;
 int start,pos,i;
 int* auxvector;
 int totinc = 0;
 if (atgen==0) start=0;
 else start=nelit;

for(k=start; k < epsize;  k ++)  
 {
   pos = k; // randomint(nelit);         
   //FoldingProtein->CallRepair(epop->P[pos],vars);     
   CurrentEval = FoldingProtein->EvalWithCO(sizeProtein,epop->P[pos]);
   eval_local = 1;   
   epop->SetVal(pos,CurrentEval); 
   TotEvaluations += eval_local;  
   }
 

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

 // if (atgen>0) MutatePop(epop,nelit,epsize,0.1);
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
   //TotEvaluations += eval_local;
    
   }
}


void EvalProteinWeightinNew(Popul* epop,int nelit, int epsize, int atgen)
{
 double bestvalprot,valprot,origval,globalb;
 int i,a,l,j,k,mm, bef,bestpos,bestval,improvement,extent;
 int eval_local,candidates,auxpos;
 unsigned int auxbuff[200];
 unsigned int bestbuff[200];
 unsigned int candbestpos[400];
 unsigned int candbestval[400];
 
 double bestb,auxbest;
 double* auxglobal = &bestb;
 int start, pos;
 
 extent = 10;

 if (atgen==0) start=0;
 else start=nelit;
 
 FoldingProtein->ContactsInSet = 0;

 for(mm=start; mm < epsize;  mm++)  
 {
   pos = mm; // randomint(nelit);
   FoldingProtein->CallRepair(epop->P[pos],sizeProtein); 
   bestvalprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,epop->P[pos],auxglobal);
   eval_local = 1;   
   TotEvaluations += eval_local; 
   //cout<<pos<<"  "<<bestb<<" "<<bestvalprot<<endl;
   //if (pos<Elit)
   // {
       //   FoldingProtein->sum_all_contacts(epop->P[pos]); 
   // }
    
   if(bestb>=AbsBestEval)
    {
       for(i=0;i<sizeProtein;i++)    AbsBestInd[i] = epop->P[pos][i];
       AbsBestEval = bestb;   
    }
   

  if(pos>Elit && pos<=Elit+100)  // (bestvalprot>=BestEval) || (atgen==0) )
     {  
      for(i=0;i<sizeProtein;i++) auxbuff[i] =   epop->P[pos][i];
      globalb = 0;
      //cout<<" atgen "<<atgen<<" pos "<<pos<<" best  "<<bestb<<"  ABS "<<AbsBestEval<<" valprot "<<bestvalprot<<endl;      
      k = 0;
      improvement = 0;   
      for(i=0;i<sizeProtein;i++) bestbuff[i] =   auxbuff[i];
      while (k==0 || improvement==1)
      {
       candidates = 0;
       origval = bestvalprot;
       for(i=2;i<sizeProtein;i++)
	 {
	  bef = bestbuff[i];
          
          for(j=0;j<FoldingProtein->nmoves;j++)
           {
	     if (j!=bef) bestbuff[i] = j;
	       valprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,bestbuff,auxglobal);   
                TotEvaluations += eval_local; 
               //valprot = FoldingProtein->EvalOnlyVector(sizeProtein,bestbuff);       
               //bestb = valprot;   
             
          
             if(valprot >= bestvalprot) 
             {      
               	      
               if(valprot > bestvalprot)
                {
                 candbestpos[0] = (unsigned int) i;
                 candbestval[0] = (unsigned int) j;
                 candidates = 1; 
                }        
               else 
                {
                  candbestpos[candidates] = (unsigned int) i;
                  candbestval[candidates] = (unsigned int) j;
                  candidates++;
                }
	       // cout<<pos<<" "<<k<<"  "<<origval<<" "<< valprot<<" "<<bestvalprot<<" "<<i<<"  "<<j<<" "<<candidates<<endl;
 	       bestvalprot = valprot;                            
             }
            
             if(bestb>globalb)
	      {
		globalb = bestb;
                for(l=0;l<sizeProtein;l++) auxbuff[l] =   bestbuff[l];
		//cout<<pos<<" "<<k<<"  "<<origval<<" "<<bestvalprot<<" "<<bestpos<<" "<<bestval<<" "<<globalb<<endl;
	      }
	   }
           bestbuff[i] = (unsigned int) bef;
	 }
        improvement = bestvalprot > origval;
             
          if(improvement==1)
	  {
            //cout<<"cand "<<candidates<<endl;
            auxpos = randomint(candidates);
            //cout<<"ind pos "<<bestpos<<endl;
	    bestpos = candbestpos[auxpos];
            bestval = candbestval[auxpos];
	    bestbuff[bestpos] = bestval;             
	    //cout<<" "<<k<<"  "<<origval<<" "<<bestvalprot<<" "<<bestpos<<" "<<bestval<<" "<<globalb<<endl;
          }
	  /*
         else
	    {
               auxbest = FoldingProtein->TabuOptimizer(sizeProtein,epop->P[pos],globalb,&eval_local,extent,10);
	       // cout<<auxbest<<" "<<globalb<<endl; 
              if(auxbest>globalb)
		 {
                   improvement = 1;
                   for(i=0;i<sizeProtein;i++)  bestbuff[i] = epop->P[pos][i]; 
                   valprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,bestbuff,auxglobal);  
                   if(auxbest>globalb)
	           {
		      globalb = auxbest;
                      for(l=0;l<sizeProtein;l++) auxbuff[l] =   bestbuff[l];
                   }  
		 
                 }
                improvement =  valprot >  origval;
                 
            }
	  */
       
       k++;         
      }
      //TotEvaluations+= k*(sizeProtein-2);
     
      if(globalb>=AbsBestEval)
	{
         for(i=0;i<sizeProtein;i++)    AbsBestInd[i] = auxbuff[i];
         AbsBestEval = globalb;   
         for(i=0;i<sizeProtein;i++)    epop->P[pos][i] = auxbuff[i];
        }
      else for(i=0;i<sizeProtein;i++)  epop->P[pos][i] = bestbuff[i];
      
     }

     epop->SetVal(pos,bestvalprot); 
 
 }

 return;
} 



void EvalProteinWeightin(Popul* epop,int nelit, int epsize, int atgen)
{
 double bestvalprot,valprot,origval,globalb;
 int i,a,l,j,k,mm, bef,bestpos,bestval,improvement,extent;
 int eval_local,candidates,auxpos;
 unsigned int auxbuff[200];
 unsigned int bestbuff[200];
 unsigned int candbestpos[400];
 unsigned int candbestval[400];
 
 double bestb,auxbest;
 double* auxglobal = &bestb;
 int start, pos;
 
 extent = 10;

 if (atgen==0) start=0;
 else start=nelit;
 

 for(mm=start; mm < epsize;  mm++)  
 {
   pos = mm; // randomint(nelit);
   //FoldingProtein->CallRepair(epop->P[pos],sizeProtein); 
   bestvalprot = FoldingProtein->EvalOnlyVector(sizeProtein,epop->P[pos]);
   //bestvalprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,epop->P[pos],auxglobal);
   //cout<<bestb<<endl;
   bestb = bestvalprot; //Just when weighting is not considered
   if( (bestb >= AbsBestEval-5) || (bestvalprot>=BestEval && atgen!=0))  // (bestvalprot>=BestEval) || (atgen==0) )
     {
    
      for(i=0;i<sizeProtein;i++) auxbuff[i] =   epop->P[pos][i];
      globalb = 0;
      //cout<<" atgen "<<atgen<<" pos "<<pos<<" best  "<<bestb<<"  ABS "<<AbsBestEval<<" valprot "<<bestvalprot<<endl;      
      k = 0;
      improvement = 0;
   
      for(i=0;i<sizeProtein;i++) bestbuff[i] =   auxbuff[i];
      while (k==0 || improvement==1)
      {
       candidates = 0;
       origval = bestvalprot;
       for(i=2;i<sizeProtein;i++)
	 {
	  bef = bestbuff[i];
          
          for(j=0;j<FoldingProtein->nmoves;j++)
           {
	     if (j!=bef) bestbuff[i] = j;
	     //valprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,bestbuff,bestb);   
               valprot = FoldingProtein->EvalOnlyVector(sizeProtein,bestbuff);       
               bestb = valprot;   
             
          
             if(valprot >= bestvalprot) 
             {      
               	      
               if(valprot > bestvalprot)
                {
                 candbestpos[0] = (unsigned int) i;
                 candbestval[0] = (unsigned int) j;
                 candidates = 1; 
                }        
               else 
                {
                  candbestpos[candidates] = (unsigned int) i;
                  candbestval[candidates] = (unsigned int) j;
                  candidates++;
                }
               //cout<<pos<<" "<<k<<"  "<<origval<<" "<< valprot<<" "<<bestvalprot<<" "<<i<<"  "<<j<<" "<<candidates<<endl;
 	       bestvalprot = valprot;                            
             }
            
             if(bestb>globalb)
	      {
		globalb = bestb;
                for(l=0;l<sizeProtein;l++) auxbuff[l] =   bestbuff[l];
		// cout<<pos<<" "<<a<<" "<<k<<"  "<<origval<<" "<<bestvalprot<<" "<<bestpos<<" "<<bestval<<" "<<globalb<<endl;
	      }
	   }
           bestbuff[i] = (unsigned int) bef;
	 }
        improvement = bestvalprot > origval;
             
          if(improvement==1)
	  {
            //cout<<"cand "<<candidates<<endl;
            auxpos = randomint(candidates);
            //cout<<"ind pos "<<bestpos<<endl;
	    bestpos = candbestpos[auxpos];
            bestval = candbestval[auxpos];
	    bestbuff[bestpos] = bestval;             
	    //cout<<" "<<k<<"  "<<origval<<" "<<bestvalprot<<" "<<bestpos<<" "<<bestval<<" "<<globalb<<endl;
          }
	  /*
         else
	    {
               auxbest = FoldingProtein->TabuOptimizer(sizeProtein,epop->P[pos],globalb,&eval_local,extent,10);
	       // cout<<auxbest<<" "<<globalb<<endl; 
              if(auxbest>globalb)
		 {
                   improvement = 1;
                   for(i=0;i<sizeProtein;i++)  bestbuff[i] = epop->P[pos][i]; 
                   valprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,bestbuff,auxglobal);  
                   if(auxbest>globalb)
	           {
		      globalb = auxbest;
                      for(l=0;l<sizeProtein;l++) auxbuff[l] =   bestbuff[l];
                   }  
		 
                 }
                improvement =  valprot >  origval;
                 
            }
	  */
       
       k++;         
      }
      TotEvaluations+= k*(sizeProtein-2);
     
      if(globalb>=AbsBestEval)
	{
         for(i=0;i<sizeProtein;i++)    AbsBestInd[i] = auxbuff[i];
         AbsBestEval = globalb;   
         for(i=0;i<sizeProtein;i++)    epop->P[pos][i] = auxbuff[i];
        }
      else for(i=0;i<sizeProtein;i++)  epop->P[pos][i] = bestbuff[i];
      
     }
    epop->SetVal(pos,bestvalprot); 
 
 }

 return;
 /*
 for(mm=start; mm < epsize;  mm++)  
 {
   pos = mm; // randomint(nelit); 
   
   for(i=0;i<sizeProtein;i++) auxbuff[i] =  epop->P[pos][i];
  
   FoldingProtein->init_contact_weights(10);
  
   globalb = 0;
   for(a=0;a<5;a++)
   { 
     valprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,auxbuff,auxglobal);
     bestvalprot = valprot;
     k = 0;
     improvement = 0;
     for(i=0;i<sizeProtein;i++) bestbuff[i] =   auxbuff[i];
 
     while (k==0 || improvement==1)
      {
      
       origval = bestvalprot;
       for(i=2;i<sizeProtein;i++)
	 {
	  bef = bestbuff[i];
          for(j=0;j<3;j++)
           {
	     if (j!=bef) bestbuff[i] = j;
                valprot = FoldingProtein->EvalVectorWithWeights(sizeProtein,bestbuff,auxglobal);       
                //cout<<i<<"  "<<bestb<<endl;
          
             if(valprot > bestvalprot) 
             {
	      bestvalprot = valprot;
              bestpos = i;
              bestval = j;                    
             }
             if(bestb>globalb)
	      {
		globalb = bestb;
                //for(l=0;l<sizeProtein;l++) auxbuff[l] =   bestbuff[l];
		// cout<<pos<<" "<<a<<" "<<k<<"  "<<origval<<" "<<bestvalprot<<" "<<bestpos<<" "<<bestval<<" "<<globalb<<endl;
	      }
           }
           bestbuff[i] = bef;
	 }
        improvement = bestvalprot > origval;
         if(improvement==1)
	  {
       	    bestbuff[bestpos] = bestval;             
	    // cout<<a<<" "<<k<<"  "<<origval<<" "<<bestvalprot<<" "<<bestpos<<" "<<bestval<<" "<<globalb<<endl;
          }
       
       k++;         
     }
     for(i=0;i<sizeProtein;i++) auxbuff[i] =   bestbuff[i];
     FoldingProtein->update_contact_weights(10,auxbuff); 
   }
   
   //FoldingProtein->CallRepair(auxbuff,sizeProtein); 
  bestvalprot = FoldingProtein->EvalOnlyVector(sizeProtein,auxbuff);
  
  epop->SetVal(pos, bestvalprot);
  for(int i=0;i<sizeProtein;i++) epop->P[pos][i] =  auxbuff[i];
 }
 return; 
 */
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
 
stream = fopen( "Param.txt", "r+" );  
        		    
	if( stream == NULL )  
		printf( "The file Param.txt was not opened\n" );  
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
           //selpop->BotzmannDist(1.0,fvect);          
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
  int i,j, fgen,gap;  
  double auxprob,OldBest;     
  DynFDA* MyMarkovNet;  
  
    
  init_time();
  InitPopulations(); 
 
  //pop->Print();
  LearningType=3;
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  MyMarkovNet->SetProtein(FoldingProtein); 

  i=0;  fgen = -1;  auxprob =0; BestEval  = Max -1; NPoints=TruncMax; 

  Popul* aux_pop;
  aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
 
 NPoints = psize;
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 1;
 FoldingProtein->init_contact_weights(10);
 TotEvaluations = 0; 

  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i); 
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);
    else if(EvaluationMode==4 || EvaluationMode==6)
        {
	  if(gap==1) EvalProteinWeightinNew(pop,Elit,psize,0);
          else EvalProteinWeightinNew(pop,Elit,psize,i);
        }  
   else if(EvaluationMode==5) EvalProteinNew(pop,Elit,psize,i);
 
        NPoints = Selection(); 
        SymPop(selpop);  

        MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
        MyMarkovNet->SetPop(selpop); 
   
  FindBestVal(); 

  if(printvals>1) 
    {     
     cout<<"Abs : ";
     for(int l=0;l<vars;l++) cout<<AbsBestInd[l]<<" ";
     cout<<" "<<AbsBestEval<<endl; 
     cout<<"Best : ";
     for(int l=0;l<vars;l++) cout<<BestInd[l]<<" "; 
     cout<<" "<<FoldingProtein->EvalOnlyVector(sizeProtein,BestInd); 
     cout<<" "<<BestEval<<endl; 
     if(printvals>0)     cout<<"Gen : "<<i<<" Best: "<<BestEval<<" Elit "<<Elit<<" TotEval: "<<TotEvaluations<<" DifPoints: "<<NPoints<<" sizecliq "<<sizecliq<<" "<<OldBest<<" "<<BestEval<<" "<<gap<<endl; 
    }

     MyMarkovNet->UpdateModelProtein(typemodel,sizecliq); 
    
     auxprob = MyMarkovNet->Prob(BestInd);  
          

           
     //  if(printvals>1)     cout<<"Gen : "<<i<<" Best: "<<BestEval<<" Elit "<<Elit<<" TotEval: "<<TotEvaluations<<" DifPoints: "<<NPoints<<" sizecliq "<<sizecliq<<endl; 
         
           FoldingProtein->init_contact_weights(10);
               for(j=0;j<Elit;j++) FoldingProtein->sum_all_contacts(selpop->P[j]);        
               FoldingProtein->global_update_contact_weights(selpop->psize);

	       /*
if(EvaluationMode==4 || EvaluationMode==6)
  {
    if(BestEval == OldBest && gap==10) //10
	{
          if(EvaluationMode==4) FoldingProtein->update_contact_weights(10,BestInd);   
          // FoldingProtein->update_contact_weights(10,AbsBestInd);
          else
            { 
               FoldingProtein->init_contact_weights(10);
               for(j=0;j<Elit;j++) FoldingProtein->sum_all_contacts(selpop->P[j]);        
               FoldingProtein->global_update_contact_weights(selpop->psize);
               //cout<<"For "<<Elit<<" Individuals there are "<< FoldingProtein->ContactsInSet<<" contacts "<<endl;
	       // FoldingProtein->print_all_contacts(); 
            }
          gap = 1; 
          BestEval = FoldingProtein->EvalVectorWithWeights(sizeProtein,BestInd,&AuxBest); //For AuxBest
          BestEval = -100;
        }
    else if (BestEval != OldBest && gap > 5) //5
        {          
         if(EvaluationMode==4) FoldingProtein->init_contact_weights(10);
          OldBest = BestEval;
	  gap = 1;
        }
      else gap++;
  }
	       */


      if (BestEval>=Max)   fgen  = i;	 
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
     
     if(NPoints>10)  MyMarkovNet->GenPopProtein(Elit,pop,1.0);   
  
          

           i++;         
          MyMarkovNet->Destroy();    
  
  }  

  if(NPoints>10) NPoints = 10;

 // ImproveProtein(selpop,0,NPoints,10000,NPoints);   
 end_time();  

 if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" Abs: "<<AbsBestEval<<" ProbBest: "<<auxprob<<" DifUhyPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" Best Solution:  ";  
  
  if(printvals>0) 
   {           
    for(int ll=0;ll<printvals-1;ll++)// NPoints 
    { 
     for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
     cout<<" "<<selpop->Evaluations[ll]<<endl; 
    }  
   }
 
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
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);

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
     IntTree->CalMutInf();  
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
if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TreProb:"<<sumprob<<" "<<Elit<<endl;    
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

      if(NPoints>10) IntTree->GenPop(Elit,pop);   
     /*else  //This was used for random re-initialization in the functional protein instances
        {
         NPoints = TruncMax;
         ProteinRandInit(pop);  
	 selpop->SetElit(Elit,pop);
	 for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
        }
      */
     i++;
  }  
  //cout<<BestEval<<" ";   selpop->Print(0); 

  //ImproveProtein(selpop,0,NPoints,1000000,NPoints);
  //ImproveProtein(selpop,0,NPoints,250000,NPoints);   
  end_time(); 
  if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<endl;  //cout<<BestEval<<endl; 
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
if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" "<<Elit<<endl;    
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
       if(NPoints>10)    MixtureInt->SamplingFromMixture(pop);  
      else 
        {
	  NPoints = TruncMax;
         ProteinRandInit(pop);  
	 selpop->SetElit(Elit,pop);
	 for(int ll=0;ll<Elit;ll++)   pop->Evaluations[ll]=selpop->Evaluations[ll];               
        }          
 
                 
      	  }   

       MixtureInt->RemoveTrees(); 
       MixtureInt->RemoveProbabilities();
       // cout<<"Pass 7 "<<endl;    
      i++;  

 
  }  
 //EvalProtein(selpop,0,Elit,i);   
 //ImproveProtein(selpop,NPoints,10000);

 // if(NPoints>10) NPoints = 10;
 //ImproveProtein(selpop,0,NPoints,2500,NPoints);   

 end_time();  

 if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<endl;   // cout<<BestEval<<endl;
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


int AffEDA(double Complexity,int typemodel)  
{  
  int i,fgen,gap;  
  double auxprob,sumprob,OldBest;     
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  

  double lam;   //Parameters used by Affinity 
  int maxits, convits, deph;
  unsigned long **listclusters;
  double SimThreshold;  
  unsigned long k,j;
  unsigned long nclust; 
  unsigned long* allvars;
  int ncalls;
  
  init_time(); 
  InitPopulations(); 
 

  CliqMaxLength = (int) (log(selpop->psize)/log(Cardinalities[0])) ;
 
  MaxNumCliq = vars;

 
  LearningType=5; // A Marginal Product  Model 
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  MyMarkovNet->SetProtein(FoldingProtein); 
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
  
  AffPropagation* AffProp; 
  AffProp = new AffPropagation(vars);
  
  listclusters = new  unsigned long* [vars];
  allvars = new  unsigned long [vars];
  for(i=0;i<vars;i++) allvars[i] = i;
  
  lam = 0.5; maxits = 1000; convits = 50; deph = 30; SimThreshold = 0.00001;
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1; NPoints = 100;  

  //cout<<CliqMaxLength<<" "<<MaxNumCliq<<" "<<Complexity<<" "<<selpop->psize<<" "<<EvaluationMode<<" "<<BestEval<<"  "<<NPoints<<endl; 
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 0;
 FoldingProtein->init_contact_weights(10);
 TotEvaluations = 0;
 while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
  
 
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);
    else if(EvaluationMode==4)
        {
	  //if(i==0) EvalProtein(pop,Elit,psize,i);  
	  //else
	  if(gap==1) EvalProteinWeightin(pop,Elit,psize,0);
          else EvalProteinWeightin(pop,Elit,psize,i);
        } 
   // pop->Print();

     NPoints = Selection(); 

     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();  
     

     IntTree->MakeTree(IntTree->rootnode); 
     FindBestVal();

     MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(selpop); 


     AffProp->Matrix = IntTree->MutualInf;

     ncalls = 0; nclust = 0;   
     AffProp->CallAffinity(lam, maxits,convits,vars, allvars, deph, listclusters, CliqMaxLength,SimThreshold,&nclust,&ncalls);
     /*
        for(k=0; k<nclust; k++)
        {
        for(j=1; j<listclusters[k][0]+1; j++ ) cout<<listclusters[k][j]<<" ";
        cout<<endl;
       }
     */

     MyMarkovNet->UpdateModelProteinMPM(typemodel,nclust,listclusters); //Non overlapping sets for the factorization
     FindBestVal();   
     auxprob = MyMarkovNet->Prob(BestInd);  
           
       
   if(printvals>1) 
    {     
     cout<<"Abs :";
     for(int l=0;l<vars;l++) cout<<AbsBestInd[l]<<" ";
     cout<<" "<<AbsBestEval<<endl; 
     for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
       cout<<" "<<FoldingProtein->EvalOnlyVector(sizeProtein,BestInd); 
       cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }
     if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" Nclusters "<<nclust<<endl;    
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

      /*
      if(NPoints<selpop->psize)
         {
           FoldingProtein->init_contact_weights(10);
           FoldingProtein->update_contact_weights(10,BestInd);  
           BestEval = FoldingProtein->EvalVectorWithWeights(sizeProtein,BestInd,&AuxBest); 
           gap = 1;
           BestEval = 10; 
         }
      else gap++;
      */

      
      if(BestEval == OldBest && gap==20)
	{
          FoldingProtein->update_contact_weights(10,BestInd);    
          gap = 1; 
          BestEval = FoldingProtein->EvalVectorWithWeights(sizeProtein,BestInd,&AuxBest); //For AuxBest
          BestEval = 10;
        }
      else if (BestEval != OldBest)
        {
          FoldingProtein->init_contact_weights(10);
          OldBest = BestEval;
	  gap = 1;
        }
      else gap++;
       
     
      //if(NPoints>10) IntTree->GenPop(Elit,pop);
       if(NPoints>10) MyMarkovNet->GenPop(Elit,pop);     
      MyMarkovNet->Destroy();    
      i++;

  } 
   end_time(); 
 
   if(printvals>0)  cout<<"LastGen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" "<<TotEvaluations<<endl;  //cout<<BestEval<<endl; 
  if(NPoints>10) NPoints = 10;
 
  DeletePopulations(); 
  delete IntTree;
  delete MyMarkovNet; 
  delete AffProp;
  for(i=0;i<nclust;i++) delete[] listclusters[i];
  delete[] listclusters;       
  delete[] allvars; 

  return fgen;
} 


int AffEDAMaxConf(double Complexity,int typemodel)  
{  
  int i,fgen,gap;  
  double auxprob,sumprob,OldBest;     
  IntTreeModel *IntTree;  
  DynFDA* MyMarkovNet;  
  JtreeTable *PropagationJtree;

  int ncMPM, ncTree, Kmpc; //Parameters used for the K most prob. conf. algorithm

  double lam;   //Parameters used by Affinity 
  int maxits, convits, deph;
  unsigned long **listclusters;
  double SimThreshold;  
  unsigned long k,j;
  unsigned long nclust; 
  unsigned long* allvars;
  int ncalls;
  
  init_time(); 
  InitPopulations(); 
 

  CliqMaxLength = (int) (log(selpop->psize)/log(Cardinalities[0])) ;
 
  MaxNumCliq = vars;
  Kmpc = psize-1; 
  ncMPM = 0; ncTree = 0;
 
  LearningType=5; // A Marginal Product  Model 
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  MyMarkovNet->SetProtein(FoldingProtein); 
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
  
  AffPropagation* AffProp; 
  AffProp = new AffPropagation(vars);
  
  listclusters = new  unsigned long* [vars];
  allvars = new  unsigned long [vars];
  for(i=0;i<vars;i++) allvars[i] = i;
  
 
  lam = 0.5; maxits = 1000; convits = 50; deph = 30; SimThreshold = 0.00001;
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1; NPoints = 100;  

 cout<<CliqMaxLength<<" "<<MaxNumCliq<<" "<<Complexity<<" "<<selpop->psize<<" "<<EvaluationMode<<" "<<BestEval<<"  "<<NPoints<<endl; 
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 0;
 FoldingProtein->init_contact_weights(10);
 TotEvaluations = 0;
 while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
  
 
    if(EvaluationMode==1) EvalProtein(pop,Elit,psize,i);      
    else if(EvaluationMode==2) EvalProteinLocal(pop,Elit,psize,i);
    else if(EvaluationMode==3) EvalProteinLong(pop,Elit,psize,i);
    else if(EvaluationMode==4)
        {
	  //if(i==0) EvalProtein(pop,Elit,psize,i);  
	  //else
	  if(gap==1) EvalProteinWeightin(pop,Elit,psize,0);
          else EvalProteinWeightin(pop,Elit,psize,i);
        } 
   
    // pop->Print();

     
    // Selection is done 
     NPoints = Selection(); 

     
     // The tree structure is learned 
     IntTree->rootnode = IntTree->FindRootNode();   //IntTree->RandomRootNode();
     IntTree->CalProbFvect(selpop,fvect,NPoints);        
     IntTree->CalMutInf();  
     IntTree->MakeTree(IntTree->rootnode);
     
 
      

     // The structure of the Junction tree for propagation is copied from the tree

     // PropagationJtree  = new JtreeTable(vars, vars, Cardinalities); 
     //PropagationJtree->convert(IntTree); 

   
      

     //FindBestVal();

     MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
     MyMarkovNet->SetPop(selpop); 


     AffProp->Matrix = IntTree->MutualInf;

     ncalls = 0; nclust = 0;   
     AffProp->CallAffinity(lam, maxits,convits,vars, allvars, deph, listclusters, CliqMaxLength,SimThreshold,&nclust,&ncalls);
    

     /*
       for(k=0; k<nclust; k++)
        {
        for(j=1; j<listclusters[k][0]+1; j++ ) cout<<listclusters[k][j]<<" ";
        cout<<endl;
       }   
    */
     MyMarkovNet->UpdateModelProteinMPM(typemodel,nclust,listclusters); //Non overlapping sets for the factorization
     FindBestVal();   
     auxprob = MyMarkovNet->Prob(BestInd);  
           

       
   if(printvals>1) 
    {     
     cout<<"Abs :";
     for(int l=0;l<vars;l++) cout<<AbsBestInd[l]<<" ";
     cout<<" "<<AbsBestEval<<endl; 
     for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
       cout<<" "<<FoldingProtein->EvalOnlyVector(sizeProtein,BestInd); 
       cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }
     if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" Nclusters "<<nclust<<endl;    
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

  
      //if(NPoints>10) IntTree->GenPop(Elit,pop);


      // The most probable  configurations are generated

      int MaxMPC = 10;
     
             
      PropagationJtree  = new JtreeTable(vars,nclust, Cardinalities);   
      PropagationJtree->convertMPM(listclusters); 
      ncMPM = FindMaxConf(Elit,PropagationJtree,selpop, pop, MaxMPC/2); 
      
      cout<<" ncMPM is  "<<ncMPM<<endl;    

      
          PropagationJtree  = new JtreeTable(vars, vars, Cardinalities); 
          PropagationJtree->convert(IntTree); 
          ncTree = FindMaxConf(Elit+ncMPM,PropagationJtree,selpop, pop, MaxMPC/2); 
      
       cout<<" ncTree is  "<<ncTree <<endl;        
        
      
        cout<<"Random MPM: From "<<Elit+ncMPM+ncTree<<" to "<<(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))/2<<endl;
	// cout<<"Random Tree Alone: From "<<Elit+ncMPM+ncTree<<" to "<<(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))<<endl; 
        cout<<"Random Tree: From "<<(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))/2<< " to "<<psize<<endl;
      
      MyMarkovNet->GenPop(Elit+ncMPM+ncTree,(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))/2, pop);  
       //IntTree->GenPop(Elit+ncMPM+ncTree,pop); 
      IntTree->GenPop((Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))/2,pop); 
	 
      //else  MyMarkovNet->GenPop(Elit,pop);    

       MyMarkovNet->Destroy();
   
       
      i++;

  } 
   end_time(); 
 
   if(printvals>0)  cout<<"LastGen : "<<i<<" Best: "<<BestEval<<" Abs: "<<AbsBestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" "<<TotEvaluations<<endl;  //cout<<BestEval<<endl; 
  if(NPoints>10) NPoints = 10;
 
  DeletePopulations(); 
  delete IntTree;
  delete MyMarkovNet; 
  delete AffProp;
  for(i=0;i<nclust;i++) delete[] listclusters[i];
  delete[] listclusters;       
  delete[] allvars; 

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
  
     if(printvals>1)      cout<<"Gen : "<<i<<" Best: "<<BestEval<<" Elit "<<Elit<<" TotEval: "<<TotEvaluations<<" DifPoints: "<<NPoints<<endl; 
 
        
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
          ImproveProtein(pop,0,NPoints,1000,psize);   
          i++;         
          
   }  

 end_time();  

 if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest:  "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<endl;  
  
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
                       case 3: succ = PopulationLocal(Complex,1,Cycles);break;  // Markov Network    
		       case 4: succ = AffEDAMaxConf(Complex,2);  // (Marginal Product Model  learned using affinity propagation)
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



void AllKikuchiApprox()
{
   
  Popul* pop;
  int NPoints,totedges;
  double sumprob,current_score;
  int i,j,a,n_subgraphs,Card; 
  double* OriginalDistrib;
  double e = 2.71828182845904;
  ReadParameters();
  int IntConf[10];

  vars = 8;
  Card = 3;
  psize = (int) pow(3,vars-2); 
  sizeProtein = vars;
     

    cout<<"vars "<<vars<<" psize "<<psize<<endl;

    Cardinalities = new unsigned int[vars];
    OriginalDistrib = new double[psize];

    for(i=0;i<vars;i++) Cardinalities[i] = Card;   //Cardinalities for the variables

    pop = new Popul(psize,vars,psize,Cardinalities); //Space of all vectors
    for(i=0;i<psize;i++)   // All posible  subgraphs are analyzed
     {
       NumConvert(i,vars-2,3,&pop->P[i][2]);
       pop->P[i][0] = 0; pop->P[i][1] = 0;
       //for(j=0;j<8;j++) cout<<pop->P[i][j]<<" ";
       //cout<<endl;
     }
    
    
     for(i=0;i<256;i++)   // All posible  subgraphs are analyzed
     {
         sumprob = 0;
         BinConvert(i,sizeProtein,IntConf); 
         //for(j=0;j<8;j++) cout<<IntConf[j]<<" ";
	 //cout<<endl;
         FoldingProtein = new HPProtein(sizeProtein,IntConf);  
         for(j=0;j<psize;j++)
	   {          
             OriginalDistrib[j] = pow(e,FoldingProtein->EvalOnlyVector(sizeProtein,pop->P[j]));
             sumprob +=  OriginalDistrib[j];
             //cout<<sumprob<<endl;
            } 
         for(j=0;j<psize;j++) 
           {
             OriginalDistrib[j] = OriginalDistrib[j]/sumprob; 
             cout<<OriginalDistrib[j]<<" "; 
           } 
	 cout<<endl;
         delete FoldingProtein; 
     } 
    

  delete pop; 
  delete[] Cardinalities;
  delete[] OriginalDistrib;


}


int main( int argc, char *argv[] )
{
  // ./protein 1 4 2 7 60 5000 15 20 1 1 

  int i,a;
  int prot_inst,modeprotein;
  int T,MaxMixtP,S_alph,Compl; 

  
 if( argc != 11 ) {
    std::cout << "Usage: " <<"cantexp  EDA{0:Markov, 1:Tree  2:Mixture, 4:AffEDA} modeprotein{2,3} prot_inst n psize Trunc max-gen" << std::endl;
    std::cout << "       Please read the README file." << std::endl;
    exit(1);
}

 params = new int[3];    
 //MatrixFileName = "newviewtrees.txt";  
 cantexp = atoi(argv[1]);         // Number of experiments
 ExperimentMode = atoi(argv[2]); // Type of EDA
 modeprotein = atoi(argv[3]);    // Type of regular lattice (d) d-dimensional  
 prot_inst = atoi(argv[4]);       // Protein Instance
 vars =  atoi(argv[5]);           //Number of variables (redundant because depends on instance)
 psize = atoi(argv[6]);          // Population size
 T = atoi(argv[7]);              // Percentage of truncation integer number (1:99)
 Maxgen =  atoi(argv[8]);        // Max number of generations 
 EvaluationMode   =  atoi(argv[9]);    // Type of evaluation function for the proteins:  1) EvalChain
 BestElitism = atoi(argv[10]);         // If there is or not BestElitism, if thereisnot BestElitism, Elitism = 1 by default;

 cout<<"Alg : "<<ExperimentMode<<", prot dim : "<<modeprotein<<", prot inst. : "<<prot_inst<<", n : "<<vars<<", psize : "<<psize<<", Trunc : "<<T<<", max-gen : "<<Maxgen<<", EvalMode : "<<EvaluationMode<<", BestElit. : "<<BestElitism<<endl; 

 Tour = 0;                       // Truncation Selection is used
 func = 0;                       // Index of the function, only for OchoaFun functions
 Ntrees = 2;                     // Number of Trees  for MT-EDA
 Elit = 1;                       // Elitism
 Nsteps = 50;                    // Learning steps of the Mixture Algorithm  
 InitTreeStructure = 1;    // 0 for a random init tree structures, 1 for a Chu&Liu learned init Tree Structure  
 VisibleChoiceVar = 0;     // 0 The Mixture choice var is hidden, 1 & 2 depends on a variable and the  unitation respectively  
 printvals = 2;            // The printvals-1 best values in each generation are printed 
 MaxMixtP = 500;           // Maximum learning parameter mixture 
 S_alph = 0;               // Value alpha for smoothing 
 StopCrit = 1;             // Stop Criteria for Learning of trees alg.  
 Prior = 1;                // Type of prior. 
 Compl=75;                 // Complexities of the trees. 
 Coeftype=2;               // Type of coefficient calculation for Exact Learning. 
 params[0] = 1 ;           //  Params for function evaluation 
 params[1] = 2;  
 params[2] = 10;  
 
 
 //seed =  1200856437; 
 seed = (unsigned) time(NULL);  
 srand(seed); 
 cout<<"seed"<<seed<<endl; 

TypeMixture = 1; // Class of MT-FDA (1-Meila, 2-MutInf)
Mutation = 0 ; // Population based mutation  
CliqMaxLength = 8; // Maximum size of the cliques for Markov  
MaxNumCliq = 300; // Maximum number of cliques for Markov 
OldWaySel = 0; // Selection with sel pop (1) or straight on Sel prob (0) 
LearningType = 5; // Learning for MNFDA (0-Markov, 1-JuntionTree) 
Cycles = 2 ; // Number of cycles for GS in the MNEDA or size for the clique in Markov EDA. 


Trunc = T/double(100);  
Complex  = Compl/double(100);  
Max = auxMax/double(1000);   
MaxMixtProb =MaxMixtP/double(100); 
S_alpha = S_alph/double(100); 

//ReadParameters(); 

Cardinalities  = new unsigned[5000];  

int k,j,u;

 int* IntConf;
 
    
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

  //  unsigned int vector[64] = {0,0,0,2,0,0,0,2,0,0,0,2,0,0,1,2,0,2,2,2,0,2,2,2,0,2,2,2,2,1,2,1,2,0,2,2,2,0,2,0,0,0,2,0,0,0,2,0,0,0,0,0,0,2,1,2,2,0,2,2,2,0,2,2};

 
  //  unsigned int vector60[60] = {0,0,1,1,1,1,2,1,0,2,2,1,1,0,0,0,1,2,0,0,0,0,2,1,0,0,0,0,2,2,2,1,0,2,2,0,0,1,0,2,1,2,2,0,0,2,1,0,1,1,2,2,1,1,0,2,1,2,0,0};
  double eval;
 int ANTrees[5] = {2,4,1,4,3};

 //sizeProtein = vars;
 //AllKikuchiApprox();
 //return 1;


  a = prot_inst;
 // for(int a=20; a<31;a++)
{

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
 

 vars = sizeProtein;


   if(modeprotein == 2)
     {        
       Card = 3;
       FoldingProtein = new HPProtein(sizeProtein,IntConf);
     //FoldingProtein = new HPProtein3D(sizeProtein,IntConf);  
     }
   else if(modeprotein == 3)
     {
      Card = 5;    
      FoldingProtein = new HPProtein3D(sizeProtein,IntConf);  
      Max = 1000;
    }
    else if(modeprotein == 4)
     {
      Card = 3;    
      FoldingProtein = new HPProtein3Diamond(sizeProtein,IntConf);  
      Max = 1000;
    }

  FoldingProtein->create_contact_weights();
  FoldingProtein->SetAlpha(1.0);
  //cout<<"alpha "<< FoldingProtein->alpha;
  for(u=0;u<5000;u++) Cardinalities[u] = Card;  
  
  

succexp =0; nsucc = 0;

// while( psize <= 20000 && succexp<90)
 {
    
        AbsBestInd = new unsigned int [vars];
        AbsBestEval = -1;
        TotEvaluations = 0;       
       	succexp = 0;  meangen = 0; meaneval = 0;  i =0;  nsucc =0; alltime = 0; bestalltime = 0;  
	while (i<cantexp) //&& nsucc<1
        { 
          currentexp = i;	  
	  runOptimizer(ExperimentMode,i);
         i++;
         //PrintStatistics();
        }  
        //cout<<i<<"     "<<cantexp<<endl;     
	PrintStatistics();             
	delete[] AbsBestInd;   
    
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
  FoldingProtein->delete_contact_weights();
  delete FoldingProtein; 
} 
 delete [] params; 
 delete [] Cardinalities; 
 return 1;
}      

