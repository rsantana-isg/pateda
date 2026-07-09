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
#include "NKModel.h"  
#include "Popul.h"  
/* 
#include "EDA.h" 
#include "AbstractTree.h"  
#include "FDA.h"  
#include "MixtureTrees.h" 
#include "RotamerClass.h" 
*/

#define itoa(a,b,c) sprintf(b, "%d", a) 
  
int* params;
int psize;
int vars;
int k;
int NObj;
int Elit;
NK  *nk1,*nk2; 
int* DetailsPF;
unsigned* Cardinalities;
MultiPopul *pop,*selpop; 

void MultiEvalNK(MultiPopul* epop,int nelit, int epsize, int atgen)
{
 double CurrentEval[10];
 int i,start,pos; 

if (atgen==0) start=0;
else start=nelit;
for(pos=start; pos < epsize;  pos ++)  
 {
   CurrentEval[0] = nk1->evalfunc(epop->P[pos]);
   CurrentEval[1] = nk2->evalfunc(epop->P[pos]);
   epop->SetVals(pos,CurrentEval); 
   //cout<<"Pos "<<pos<<" :";
   //for(i=0;i<NObj;i++) cout<<CurrentEval[i]<<" ";
   // cout<<endl; 
   
  }
}



void EvolveInstanceRandom(int nsteps, int typ)
{
  
  int i,j,l,auxk,totake,tochange,pos,oldval,count;
  int best_i, best_j, best_l,flag;
  int BestVal;

   MultiEvalNK(pop,0,psize,0);
   for(int ii=0;ii<psize;ii++) cout<<pop->MultiEvaluations[ii][0]<<" "<<pop->MultiEvaluations[ii][1]<<endl;
   pop->ParetoRankingSelRestricted(selpop,DetailsPF);
   if (typ==-1) BestVal = DetailsPF[1];
   else    BestVal = DetailsPF[0];

   count = 0;
  while(count<nsteps)
    {
      tochange = randomint(vars);
      pos = randomint(k)+1;
      MultiEvalNK(pop,0,psize,0);
      pop->ParetoRankingSelRestricted(selpop,DetailsPF);   
     
       
      oldval = nk1->lattice[tochange][pos];   
      flag = 1;
      while(flag==1)
	{
	 flag = 0;
         totake = randomint(vars);        
         for(auxk=0;auxk<k+1;auxk++)  
           {
             if(totake == nk1->lattice[tochange][auxk]) flag=1;
           }
       }           
      
       nk1->lattice[tochange][pos] = totake;
       MultiEvalNK(pop,0,psize,0);
       pop->ParetoRankingSelRestricted(selpop,DetailsPF);
       cout<<count<<" "<<tochange<<"  "<<pos<< "  "<<oldval<<"  "<<totake<<"  "<<DetailsPF[0]<<" "<<DetailsPF[1]<<"  "<<BestVal<<endl;
       if ((typ==1 && (DetailsPF[0]>=BestVal)) ||  (typ==-1 && (DetailsPF[1]<=BestVal)))
                 {
                   if (typ==1) BestVal = DetailsPF[0];
                   else BestVal = DetailsPF[1];
                   for(int ii=0;ii<psize;ii++) cout<<pop->MultiEvaluations[ii][0]<<" "<<pop->MultiEvaluations[ii][1]<<endl;
                 }
       else nk1->lattice[tochange][pos] = oldval;      
    count++;
  }
  for(i=0;i<psize;i++) cout<<pop->MultiEvaluations[i][0]<<" "<<pop->MultiEvaluations[i][1]<<endl;

}


void EvolveInstance(int nsteps, int typ)
{
  int i,j,l,auxk,totake,tochange,pos,oldval,count;
  int best_i, best_j, best_l;
  int BestVal;

   MultiEvalNK(pop,0,psize,0);
   pop->ParetoRankingSelRestricted(selpop,DetailsPF);
   if (typ==-1) BestVal = DetailsPF[1];
   else    BestVal = DetailsPF[0];


   count = 0;
  while(count<nsteps)
    {
     for(i=0;i<vars;i++)
       for(j=1;j<k+1;j++)
        {
          tochange =  i;   //randomint(n);
          pos = j;         //pos = randomint(k)+1;
          for(l=0;l<vars;l++)
           {
            auxk = 0;
            while(auxk<k+1 && (l != nk1->lattice[tochange][auxk])) auxk++;
            if(auxk==k+1) 
	      {
                oldval = nk1->lattice[tochange][j];   
                nk1->lattice[tochange][j] = l;
                MultiEvalNK(pop,0,psize,0);
                nk1->lattice[tochange][j] = oldval;
                pop->ParetoRankingSelRestricted(selpop,DetailsPF);
                cout<<count<<" "<<i<<"  "<<j<< "  "<<oldval<<"  "<<l<<"  "<<DetailsPF[0]<<" "<<DetailsPF[1]<<"  "<<BestVal<<endl;
                if ((typ==1 && (DetailsPF[0]>BestVal)) ||  (typ==-1 && (DetailsPF[1]<BestVal)))
                 {
                   if (typ==1) BestVal = DetailsPF[0];
                   else BestVal = DetailsPF[1];
                   best_i = tochange;
                   best_j = j;
                   best_l = l;
                 }
            }  
          } 
       }
     nk1->lattice[best_i][best_j] = best_l;
     count++;
    }
}


// ./snps NewConfFileENm10.txt PruebaAlex 1 1 1 100 15 3  1 2 2333
int main(int argc, char *argv[ ])
{
 int i,j,M,seed;  
 unsigned ta;
 ta = (unsigned) time(NULL);  
 FILE *streamPairs, *streamTriples, *streamTags; 

 
 if( argc != 5 ) {
 //if( argc != 15 ) {
    std::cout << "Usage: " <<"./snps ConfFile TagsFilename MAXD MINC MINM psize T Maxgen BestElitism VerboseLevel Seed" << std::endl;
 //   std::cout << "Usage: " <<"./snps MapFilename HapFilename MAXD MINC MINM PairsFilename TriplesFilename TagsFilename(prefix) psize T Maxgen BestElitism VerboseLevel Seed" << std::endl;
    std::cout << "       Please read the README file." << std::endl;
    exit(1);
 }

 
 //ConfFilename = argv[1];
 //TagsFilename = argv[2];
 
 seed = atoi(argv[1]) + ta;
 vars = atoi(argv[2]);
 NObj =   atoi(argv[3]);
 k =   atoi(argv[4]);

 params = new int[3]; 
 DetailsPF = new int[2];
 psize = pow(2,vars);
 Elit = 0;
 


 srand(seed);
  
 Cardinalities  = new unsigned[vars];  
 for(i=0;i<vars;i++) Cardinalities[i] = 2;   
 
 cout<<seed<<" "<<psize<<" "<<vars<<"  "<<k<<"  "<<NObj<<endl; 

 nk1 = new NK(vars,k);
 nk1->RandomInstance();
 nk1->SaveInstance("Prueba1.txt");
 nk2 = new NK(vars,k);
 nk2->RandomInstance();
 nk2->SaveInstance("Prueba2.txt");

 pop = new MultiPopul(psize,vars,Elit,Cardinalities,NObj);  
 selpop = new MultiPopul(psize,vars,Elit,Cardinalities,NObj);
 pop->ProbInit();



 //EvolveInstance(50,-1);

 EvolveInstanceRandom(1000,-1);
 // for(i=0;i<DetailsPF[0];i++) cout<<selpop->MultiEvaluations[i][0]<<" "<<selpop->MultiEvaluations[i][1]<<endl;
 //cout<<endl; 
 //cout<<DetailsPF[0]<<" "<<DetailsPF[1]<<endl;


 //compact_pop = new MultiPopul(psize,vars,Elit,Cardinalities,NObj);  
 

 delete nk1;
 delete nk2;
 delete pop;
 delete selpop;
 delete[] params;
 delete[] Cardinalities;
 delete[] DetailsPF;
  
 return 1;
}       
 
#endif
