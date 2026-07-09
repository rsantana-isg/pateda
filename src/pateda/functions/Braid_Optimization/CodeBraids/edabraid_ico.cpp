#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <math.h> 
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 

#include "auxfunc.h"  
#include "Popul.h"  
//#include "EDA.h" 
#include "AbstractTree.h"  
//#include "FDA.h"  

#include "random.h"
#include "quaternion.h"


using namespace std;
//using namespace quaternion;


FILE *stream;  
FILE *file,*outfile;  	  
 
int Instance;
int Experiment;
int cantexp;  
int ExperimentMode;
int now;  
int vars;  
int auxMax;  
double Max;  
double  Trunc;  
int psize;  
int  Tour;  
int func;  
int Elit;  
int succexp;  
double meangen;   
int Nsteps;  
int InitTreeStructure;  
int VisibleChoiceVar;  
int Maxgen;  
int printvals;   
//unsigned int Card;  
int seed;  
int* params;
int fun;  
int *timevector; 
char filedetails[30]; 
char MatrixFileName[30]; 
int BestElitism;   
int StopCrit; //Stop criteria to stop the MT learning alg. 
int Prior; 
double mComplex; 
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


int TotEvaluations;
int EvaluationMode;
int currentexp;
int length;
long int explength;
int MaxMPC;
int TypeMPC;

// VARIABLES BRAIDS

Quaternion* All_IcoTargets[60];
double LAMBDA = 0.001;
double lambdas[] = {0, 0.01, 0.05, 0.1};
int typelambda;     // Type of LAMBDA values (0:0.0, 1:0.01, 2:0.05, 3:0.1) 
int typesampling;   // Type of sampling (number of variables to  update)
double sampl_threshold=0;
int lambdaCount = 4;
Quaternion* Generators;
unsigned int* auxbraid;
int typetree;  // 0:Tree-UMDA, 1:Tree-Markov, 2:Tree-Tree
int instance;  // Index of the icosahedral instance (between 0 and 59)

// In the original paper by McDonnald and Katzgraber there were two problems
// On with 4 types of generators and another with 10
// In this problem we use the first problem with 4 generators (see below)

#ifndef FAKE_QUATERNIONS
Quaternion InitQuat=(Quaternion){0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
#else
Quaternion InitQuat=(Quaternion){0,0,0,0};
#endif

void _seed(int s)
{
	srand(s);
}

int _nextInt(int max)
{
	return rand()%max;
}

float _next()
{
	return (float)rand()/RAND_MAX;
}

Random r = {&_seed, &_next, &_nextInt};


// The target gate is initialize

#ifndef FAKE_QUATERNIONS
Quaternion Target = (Quaternion){U,Z,Z,Z, Z,U,Z,Z, Z,Z,Z,U, Z,Z,U,Z};
unsigned int Card= 10;
#else
//Quaternion Target = (Quaternion){0,0,0,1};
Quaternion Target = (Quaternion){0,1,0,0};

unsigned int Card=4;
#endif


// The values of generators are initialized
// In the original paper by McDonnald and Katzgraber there were two problems
// On with 4 types of generators and another with 10
// In this problem we use the first problem with 4 generators 

void InitBraids(){   
  int i;
 	
	Generators = (Quaternion*)malloc(sizeof(Quaternion)*16);    
  	#ifndef FAKE_QUATERNIONS

	#else
	
	double tau = (sqrt(5)-1)/2;
	double theta1 = -7*M_PI/10;
	double theta2 = 9*M_PI/10;
      
	Generators[0] = (Quaternion){cos(theta1),sin(theta1),0,0};
	Generators[1] = (Quaternion){tau*cos(theta2),tau*sin(theta2),0,-sqrt(tau)};
        Quaternion_Conjugate(Generators[0],&Generators[2]);
	Quaternion_Conjugate(Generators[1],&Generators[3]);       
    

  for(i=0;i<60;i++) All_IcoTargets[i] = new Quaternion;

  double phi = 4*M_PI/10;
  Quaternion_FromAxisAngle(0.0, 0.0, 1.0, 0.0,All_IcoTargets[0]);
  Quaternion_FromAxisAngle(0.0, 1.0, tau, phi,All_IcoTargets[1]);
  Quaternion_FromAxisAngle(0.0, 1.0, -tau, phi,All_IcoTargets[2]);
  Quaternion_FromAxisAngle(0.0, -1.0, tau, phi,All_IcoTargets[3]);
  Quaternion_FromAxisAngle(0.0, -1.0, -tau, phi,All_IcoTargets[4]);
  Quaternion_FromAxisAngle(tau, 0.0, 1.0, phi,All_IcoTargets[5]);
  Quaternion_FromAxisAngle(-tau, 0.0, 1.0, phi,All_IcoTargets[6]);
  Quaternion_FromAxisAngle(tau, 0.0, -1.0, phi,All_IcoTargets[7]);
  Quaternion_FromAxisAngle(-tau, 0.0, -1.0, phi,All_IcoTargets[8]);
  Quaternion_FromAxisAngle(1.0, tau, 0.0, phi,All_IcoTargets[9]);
  Quaternion_FromAxisAngle(1.0, -tau, 0.0, phi,All_IcoTargets[10]);
  Quaternion_FromAxisAngle(-1.0, tau, 0.0, phi,All_IcoTargets[11]);
  Quaternion_FromAxisAngle(-1.0, -tau, 0.0, phi,All_IcoTargets[12]);
  Quaternion_FromAxisAngle(0.0, 1.0, tau, 2.0 * phi,All_IcoTargets[13]);
  Quaternion_FromAxisAngle(0.0, 1.0, -tau, 2.0 * phi,All_IcoTargets[14]);
  Quaternion_FromAxisAngle(0.0, -1.0, tau, 2.0 * phi,All_IcoTargets[15]);
  Quaternion_FromAxisAngle(0.0, -1.0, -tau, 2.0 * phi,All_IcoTargets[16]);
  Quaternion_FromAxisAngle(tau, 0.0, 1.0, 2.0 * phi,All_IcoTargets[17]);
  Quaternion_FromAxisAngle(-tau, 0.0, 1.0, 2.0 * phi,All_IcoTargets[18]);
  Quaternion_FromAxisAngle(tau, 0.0, -1.0, 2.0 * phi,All_IcoTargets[19]);
  Quaternion_FromAxisAngle(-tau, 0.0, -1.0, 2.0 * phi,All_IcoTargets[20]);
  Quaternion_FromAxisAngle(1.0, tau, 0.0, 2.0 * phi,All_IcoTargets[21]);
  Quaternion_FromAxisAngle(1.0, -tau, 0.0, 2.0 * phi,All_IcoTargets[22]);
  Quaternion_FromAxisAngle(-1.0, tau, 0.0, 2.0 * phi,All_IcoTargets[23]);
  Quaternion_FromAxisAngle(-1.0, -tau, 0.0, 2.0 * phi,All_IcoTargets[24]);

  Quaternion_FromAxisAngle(0.0, 0.0, 1.0, M_PI,All_IcoTargets[25]);
  Quaternion_FromAxisAngle(1.0, 1.0 / tau, tau, M_PI,All_IcoTargets[26]);
  Quaternion_FromAxisAngle(-1.0, 1.0 / tau, tau, M_PI,All_IcoTargets[27]);
  Quaternion_FromAxisAngle(1.0, -1.0 / tau, tau, M_PI,All_IcoTargets[28]);
  Quaternion_FromAxisAngle(-1.0, -1.0 / tau, tau, M_PI,All_IcoTargets[29]);
  Quaternion_FromAxisAngle(1.0, 0.0, 0.0, M_PI,All_IcoTargets[30]);
  Quaternion_FromAxisAngle(tau, 1.0, 1.0 / tau, M_PI,All_IcoTargets[31]);
  Quaternion_FromAxisAngle(tau, -1.0, 1.0 / tau, M_PI,All_IcoTargets[32]);
  Quaternion_FromAxisAngle(tau, 1.0, -1.0 / tau, M_PI,All_IcoTargets[33]);
  Quaternion_FromAxisAngle(tau, -1.0, -1.0 / tau, M_PI,All_IcoTargets[34]);
  Quaternion_FromAxisAngle(0.0, 1.0, 0.0, M_PI,All_IcoTargets[35]);
  Quaternion_FromAxisAngle(1.0 / tau, tau, 1.0, M_PI,All_IcoTargets[36]);
  Quaternion_FromAxisAngle(-1.0 / tau, tau, 1.0, M_PI,All_IcoTargets[37]);
  Quaternion_FromAxisAngle(1.0 / tau, tau, -1.0, M_PI,All_IcoTargets[38]);
  Quaternion_FromAxisAngle(-1.0 / tau, tau, -1.0, M_PI,All_IcoTargets[39]);

  phi = 2.0 * M_PI / 3.0;
  Quaternion_FromAxisAngle(tau, 0.0, 1.0 + 2.0 * tau, phi,All_IcoTargets[40]);
  Quaternion_FromAxisAngle(-tau, 0.0, 1.0 + 2.0 * tau, phi,All_IcoTargets[41]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, tau, 0, phi,All_IcoTargets[42]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, -tau, 0, phi,All_IcoTargets[43]);
  Quaternion_FromAxisAngle(0.0, 1.0 + 2.0 * tau, tau, phi,All_IcoTargets[44]);
  Quaternion_FromAxisAngle(0.0, 1.0 + 2.0 * tau, -tau, phi,All_IcoTargets[45]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, phi,All_IcoTargets[46]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, phi,All_IcoTargets[47]);
  Quaternion_FromAxisAngle(-1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, phi,All_IcoTargets[48]);
  Quaternion_FromAxisAngle(-1.0 - 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, phi,All_IcoTargets[49]);
  Quaternion_FromAxisAngle(tau, 0.0, 1.0 + 2.0 * tau, 2.0 * phi,All_IcoTargets[50]);
  Quaternion_FromAxisAngle(-tau, 0.0, 1.0 + 2.0 * tau, 2.0 * phi,All_IcoTargets[51]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, tau, 0, 2.0 * phi,All_IcoTargets[52]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, -tau, 0, 2.0 * phi,All_IcoTargets[53]);
  Quaternion_FromAxisAngle(0.0, 1.0 + 2.0 * tau, tau, 2.0 * phi,All_IcoTargets[54]);
  Quaternion_FromAxisAngle(0.0, 1.0 + 2.0 * tau, -tau, 2.0 * phi,All_IcoTargets[55]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi,All_IcoTargets[56]);
  Quaternion_FromAxisAngle(1.0 + 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi,All_IcoTargets[57]);
  Quaternion_FromAxisAngle(-1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi,All_IcoTargets[58]);
  Quaternion_FromAxisAngle(-1.0 - 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi,All_IcoTargets[59]);
  
  //Target = All_IcoTargets[instance];
  Quaternion_Copy(&Target,*All_IcoTargets[instance]);
  Quaternion_Print(Target);
#endif
	
	
}


// init_time and end_time are used to measure how long time take the algorithms to completed

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


// Original function proposed by McDonnald and Katzgrabber
// is evaluated for all solutions in the population

void evaloriginal(Popul* epop,int nelit, int epsize, int atgen)
{
  double obj_f,distance;
  int k,j,start,alength;
  Quaternion AuxQuat;
 
  // When there is elitism, we only evaluate solutions from start=nelit,
  // bu when in the first population we evaluate all the solutions

  if (atgen==0) start=0;
  else start=nelit;
  
  // For each individual in the population 
  for(k=start; k < epsize;  k++)  
   { 
    Quaternion_Copy(&AuxQuat,Generators[epop->P[k][0]]); //  The matrix correponding to the first variable
                                                         // in solution k is copied to AuxQuat, where the product
                                                         // of the matrices will be kept   

    // The matrices corresponding to all other variables are multiplied
    for(j=1; j<vars;  j++)   Quaternion_Multiply(AuxQuat, Generators[epop->P[k][j]],&AuxQuat);

    // The distance is computed between the Target and the product of matrices
    distance = Quaternion_Distance(Target, AuxQuat);

    // In the original definitionof the function the length is equal to the number of positions
    alength = vars;
  
    if(distance > 0.0000001 && alength>0)
      obj_f = (1.0-LAMBDA)/(1+distance) + LAMBDA/alength;
    else
      obj_f =  -10000000.0;		//return -INFINITY;  
    
    epop->SetVal(k,obj_f);
    TotEvaluations++;
   }
}


void PrintEvals(unsigned int* thesol)
{
  double obj_f,distance, best_distance;
  int j,k,alength;
  int current, last;
  //unsigned int auxsols[22] =  {3,3,0,0,0,0,3,0,3,0,1,2,2,1,2,3,3,3,3,3,0,3};
  //unsigned int auxsols[20] =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  Quaternion AuxQuat;

  //for(k=0; k < 20;  k++)  thesol[k] = auxsols[k];
      
   current=0;
   last=1;
   auxbraid[current] = thesol[0];
   while(last<vars)
    {
      if(current>-1 && ( (thesol[last]==auxbraid[current]+Card/2)  || (thesol[last]==auxbraid[current]-Card/2) )) current--;                            
      else 
        {
          current++;
          auxbraid[current] = thesol[last];                               
        }
        last++;   
    }
       
   /*
 for(int ll=0;ll<current;ll++)  
     {
       //cout<<thesol[ll];
      cout<<auxbraid[ll];
     }
     cout<<endl;  
   */  
   if(current>=0)
     {  
      best_distance = 1000000000.0;
      Quaternion_Copy(&AuxQuat,Generators[auxbraid[0]]);      
      for(j=1; j<=current;  j++) 
       {      
         Quaternion_Multiply(AuxQuat, Generators[auxbraid[j]],&AuxQuat);     
         distance = Quaternion_Distance(Target, AuxQuat);
         if(distance<best_distance)
          {
            alength = j;
            best_distance=distance;
            //cout<<j<<" "<<alength<<" "<<best_distance<<endl;
          }
        }    
     }
    else 
      {
       alength=0;
       current=0;
      }

   if (func==0)
     {
    if(distance > 0.0000001)
      obj_f = (1.0-LAMBDA)/(1+distance) + LAMBDA/vars;
    else
      obj_f =  -10000000.0;		//return -INFINITY;      
     }
   else
   {
    if(best_distance > 0.0000001 && alength>0)
      obj_f = (1.0-LAMBDA)/(1+best_distance) + LAMBDA/alength;
    else
      obj_f =  -10000000.0;		//return -INFINITY;      
     }
 
   cout<<current<<" "<<alength<<" "<<best_distance<<" "<<obj_f<<" "<<(1.0)/(1+best_distance)<<" "<<log10(best_distance);
  
}


void evalcondensed(Popul* epop,int nelit, int epsize, int atgen)
{
  double obj_f,distance, best_distance;
  int k,j,start,alength;
  int current, last;

 Quaternion AuxQuat;
 if (atgen==0) start=0;
 else start=nelit;
 
 for(k=start; k < epsize;  k++)  
 {  
   current=0;
   last=1;
   auxbraid[current] = pop->P[k][0];
   while(last<vars)
    {
      
      if(current>-1 && ( (pop->P[k][last]==auxbraid[current] + Card/2)  || (pop->P[k][last]==auxbraid[current]-Card/2) )) current--;                              
      else 
        {
          current++;
          auxbraid[current] = pop->P[k][last];                               
        }
       last++;   
    }

   if(current>=0)
     {  
      best_distance = 1000000000.0;
      Quaternion_Copy(&AuxQuat,Generators[auxbraid[0]]);      
      for(j=1; j<=current;  j++) 
       {      
         Quaternion_Multiply(AuxQuat, Generators[auxbraid[j]],&AuxQuat);     
         distance = Quaternion_Distance(Target, AuxQuat);
         if(distance<best_distance)
          {
            alength = j;
            best_distance=distance;
            //cout<<j<<" "<<alength<<" "<<best_distance<<endl;
          }
        }    
     }
    else 
      {
       alength=0;
       current=0;
      }
 
    if(best_distance > 0.0000001 && alength>0)
      obj_f = (1.0-LAMBDA)/(1+best_distance) + LAMBDA/alength;
    else
      obj_f =  -10000000.0;		//return -INFINITY;  
    
   
  if(func==3 || func==7)
      {
	//cout<<current<<" alength "<<alength<<"  "<<epop->P[k][alength-1]<<endl;
	for(j=0; j<=alength;  j++)   epop->P[k][j] = auxbraid[j];       
 	for(j=1; j<alength && alength+j<vars;  j++) epop->P[k][alength+j] = epop->P[k][alength-j+1];
      }
    else  if(func==2  || func==6)
      {
       for(j=0; j<=current;  j++)   epop->P[k][j] = auxbraid[j];
       for(j=1; j<current && current+j<vars;  j++) epop->P[k][current+j] = epop->P[k][current-j+1];
    
      }
 

    //for(j=0; j<vars;  j++)   cout<<epop->P[k][j]<<" ";
    //cout<<endl;
    epop->SetVal(k,obj_f);
    //cout<<k<<" "<<obj_f<<endl;
    TotEvaluations++;
 }
}


void evalgreedy(Popul* epop,int nelit, int epsize, int atgen)
{
  double obj_f, abs_obj_f, distance, best_distance;
  int k,j,start,alength,bestalength,bestpos,bestassig;
  int current, last,improved,thepass;
  int oldval, l,m;

 Quaternion AuxQuat;
 if (atgen==0) start=0;
 else start=nelit;
 
 for(k=start; k < epsize;  k++)  
 {  
   improved=1;
   thepass = 0;
   abs_obj_f = -1000000000.0;
   
   bestalength = 2*vars;
   while(improved==1)  
     {
       improved = 0;
       for(l=0; l < vars;  l++)  
	 { 
	   for(m=0; m < Card, m!= pop->P[k][l];  m++)
	     { 
	       oldval = pop->P[k][l];
	       pop->P[k][l] = m;  
	 
	       current=0;
	       last=1;
	       auxbraid[current] = pop->P[k][0];
	       while(last<vars) 
		 {      
		   if(current>-1 && ( (pop->P[k][last]==auxbraid[current] + Card/2)  || (pop->P[k][last]==auxbraid[current]-Card/2) )) current--;                              
		   else 
		     {
		       current++;
		       auxbraid[current] = pop->P[k][last];                               
		     }
		   last++;   
		 }

	       if(current>=0)
		 {  
		   best_distance = 1000000000.0;
		   Quaternion_Copy(&AuxQuat,Generators[auxbraid[0]]);      
		   for(j=1; j<=current;  j++) 
		     {      
		       Quaternion_Multiply(AuxQuat, Generators[auxbraid[j]],&AuxQuat);     
		       distance = Quaternion_Distance(Target, AuxQuat);
		       if(distance<best_distance)
			 {
			   alength = j;
			   best_distance=distance;
			   //cout<<j<<" "<<alength<<" "<<best_distance<<endl;
			 }
		     }    
		 }
	       else 
		 {
		   alength=0;
		   current=0;
		 }
 
	       if(best_distance > 0.0000001 && alength>0)
		 obj_f = (1.0-LAMBDA)/(1+best_distance) + LAMBDA/alength;
	       else
		 obj_f =  -10000000.0;		//return -INFINITY;  
    
	       // cout<<k<<" "<<alength<<" "<<best_distance<<" "<<obj_f<<endl;
	       //for(j=0; j<vars;  j++)   cout<<epop->P[k][j]<<" ";
	       //cout<<endl;
	       //for(j=0; j<current;  j++)   cout<<auxbraid[j]<<" ";
	       //cout<<endl;
	       // if(best_distance<abs_best_distance || (best_distance==abs_best_distance &&  bestalength>alength))
               if(obj_f>abs_obj_f || (obj_f==abs_obj_f &&  bestalength>alength))
		 {
		   abs_obj_f = obj_f;
		   bestpos = l;
		   bestassig = m;
		   bestalength = alength;
		   improved = 1;
                   //cout<<k<<" "<<thepass<<" "<<bestpos<<" "<<bestassig<<" "<<bestalength<<" "<<abs_best_distance<<" "<<obj_f<<endl;
		 } 
	       pop->P[k][l] = oldval;  
	     }
	 }
       if(improved==1)
	 {
	   epop->P[k][bestpos]= bestassig;
	 }   
       thepass++;
     }
	 //for(j=0; j<vars;  j++)   cout<<epop->P[k][j]<<" ";
	 //cout<<endl;
	 epop->SetVal(k,obj_f);
	 //cout<<k<<" "<<obj_f<<endl;
	 TotEvaluations++;
       }
}


void evalfunction(Popul* epop,int nelit, int epsize, int atgen)
{

  if(func==0) evaloriginal(epop,nelit,epsize,atgen);
  else if(func==1 || func==2 || func==3) evalcondensed(epop,nelit,epsize,atgen);
  else if(func==5  || func==6 || func==7) 
    {
     evalgreedy(epop,nelit,epsize,atgen);
     evalcondensed(epop,nelit,epsize,atgen);
    }
  else  if(func==4) 
    {
      evalgreedy(epop,nelit,epsize,atgen);
      evaloriginal(epop,nelit,epsize,atgen);
    }

}

// Implementation of the different selection methods  
// Tour=0: Truncation selection
//     =1: Tournament selection
//     =2: Proportional selection
//     =3: Boltzmann selection
// Mainly tested for truncation selection 
 
int Selection() 
{ 
   int NPoints=0;  
   if (Tour==0)  
         {  
           pop->TruncSel(selpop,TruncMax); 
           selpop->UniformProb(TruncMax,fvect);
           //selpop->BotzmannDist(1.0,fvect);          
           NPoints = selpop->CompactPopNew(compact_pop,fvect); 	     
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
 

// The position and value of the best solution in the population are found and
// variables BestEval and BestInd are updated


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
 
// The populations are initialized
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
  pop->RandInit();  
  compact_pop = new Popul(psize,vars,Elit,Cardinalities);  
  fvect = new double[psize];
 
 } 
 
// Populations are deleted
inline void DeletePopulations() 
{   
  delete compact_pop; 
  delete pop;  
  delete selpop; 
  if (Tour>0 || (Tour==0 && Elit>TruncMax)) delete elitpop; 
  delete[] fvect; 
} 

// Implementation of the EDAs. Three variants are considered:
//  typetree = 0  : UMDA, variables are independent
//  typetree = 1  : Chain-shaped 1-Markov model: Each variable depends on the previous in the sequence
//  typetree = 2  : Tree-EDA (The probabilistic model learned corresponds to a tree) 


int Intusualinit(double Complexity)  
{  
  int i,fgen;  
  double auxprob,sumprob;     
  IntTreeModel *IntTree;      // The tree is defined only once and updated in every generation
 
  init_time();                // Beginning of the execution,  to measure times of the algorithm
 
  InitPopulations();            // The initial population is randomly initialized 

                                // The tree is constructed using as parameters: the number of variables
                                // the complexity of  the model, the selected population popsize, and 
                                // the cardinalities (maximum number of values) for each variable
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);   


  // The parameters that are used by the EDA are initialized where
  // i is the current generation, auxprob is the probability given by the model to the best solution
  // BestEval is the best solution found to the moment, fgen is the generation where the optimum 
  // was found (if it was found), NPoints is the number of different individuals in the current population
 
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1;  NPoints = 100;
   
  // The stop conditions are: A maximum number of generations, the optimum was found, the number of different
  // individuals in the population is equal or less than 10 (too homogeneous population) 
  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
     evalfunction(pop,Elit,psize,i);     // All solutions in the population are evaluated
     NPoints = Selection();              // The selected population is updated using the Selection method          
     IntTree->rootnode = IntTree->RandomRootNode();   // The root of the tree is randomly selected
     IntTree->CalProbFvect(selpop,fvect,NPoints);     // The univariate and bivariate probabilities are learned

      
     if(typetree==2)   // Tree-EDA model learning
       {
         IntTree->CalMutInf();                       //  Mutual information is computed
         IntTree->MakeTree(IntTree->rootnode);       //   Using Chow-Liu method the structure of the tree is learned
       }
     else if(typetree==1) // 1-Markov-Model
       {
           IntTree->MakeTreeMarkovChain();        // The structure of the sequence is set (not learning needed)
       }
     else if(typetree==0)  // UMDA
       IntTree->MakeTreeIndependentVars();         // The structure (independent variables) is set
      
     FindBestVal();  //  The best solution is the population is found
    
     IntTree->PutPriors(Prior,selpop->psize,1);  //  Priors are used to inject diversity in the population

sumprob = IntTree->SumProb(selpop,NPoints);      //  Sum of the probabilities given by the model to all points in the population
//auxprob = IntTree->Prob(BestInd);          // Probability of the  best solution in the population
     //selpop->Print(0); 
    
      
//  A number of  "printvals" solutions of the selected population are printed 
//  where printvals is a parameter pass to the algorithm     
if(printvals>1) 
   {            
     for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
        for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" ";  
        cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }
     // Descriptors of the EDA behavior are printed in each population
     if(printvals)   cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TreProb:"<<sumprob<<" "<<Elit<<endl;    
   }

    // Best elitism is implemented, where the "Elit" best solutions are passed to the next generation

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

      // If the current population is sufficiently diverse (10 or more different individuals)
      // the new population is generated sampling from the model
      // Two possible sampling methods are used: Normal sampling and Partial sampling
      if(NPoints>10) 
	{
	  if(typesampling==0) IntTree->GenPop(Elit,pop);   
          else // Only a partial number of the variables is generated during sampling
          {
            IntTree->PopulateFromOtherPop(Elit,pop,selpop); 
	    IntTree->GenPopSomeVariables(Elit,pop,sampl_threshold);
          }       
          //pop->Print();
        }     
     i++;
  }  
  end_time(); 
   
  //if(printvals>0)  cout<<"LastGen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" "<<TotEvaluations<<endl;  //cout<<BestEval<<endl;
 
// A summary of the EDA parameters and of the results of the algorithm are printed to the screen 

 cout<<func<<" "<<typelambda<<" "<<typesampling<<" "<<typetree<<" ";
 cout<<Instance<<" "<<Experiment<<" "<<" "<<i<<" "<<BestEval<<" "<<NPoints<<" "<<" ";    
 for(int l=0;l<vars;l++) cout<<BestInd[l]<<" ";           
  PrintEvals(BestInd);
  cout<<endl;
  if(NPoints>10) NPoints = 10;

  DeletePopulations(); 
  delete IntTree; 
  return fgen;  
}  



int main( int argc, char *argv[] )
{
   
  //  EXAMPLE OF HOW TO CALL THE PROGRAM
  //./edabraid 30 1 1 50 500 50 50 1 1000 3 0 0 0 1 59 0


int i, T,u,MaxMixtP,S_alph,Compl; 

  if( argc != 17 ) {
    std::cout << "Usage: " <<"cantexp  EDA{0:Markov, 1:Tree  2:Mixture, 4:AffEDA} modeprotein{2,3} prot_inst n psize Trunc max-gen" << std::endl;
    std::cout << "       Please read the README file." << std::endl;
    exit(1);
}

 params = new int[3];    

 cantexp = atoi(argv[1]);         // Number of experiments
 ExperimentMode = atoi(argv[2]);  // Type of EDA 
 length = atoi(argv[3]);          // Number of bits for  each variable  (We asume length=1)
 vars =  atoi(argv[4]) * length;  //Number of variables (redundant because depends on instance)
 psize = atoi(argv[5]);           // Population size
 T = atoi(argv[6]);               // Percentage of truncation integer number (1:99)
 Maxgen =  atoi(argv[7]);         // Max number of generations 
 BestElitism = atoi(argv[8]);     // If there is or not BestElitism, if thereisnot BestElitism, Elitism = 1 by default;
 Max = atoi(argv[9]);              // Maximum is it is know 
 CliqMaxLength = atoi(argv[10]);
 func  = atoi(argv[11]);         // Type of function (0:original, 1: condensed, 2: condensed:rewired)
 typelambda  = atoi(argv[12]);   // Type of LAMBDA values (0:0.0, 1:0.01, 2:0.05, 3:0.1) 
 typesampling  = atoi(argv[13]); // Type of Lambda (0: Normal, 1: Partial, max n variables,: Partial, max n/2 variables)
 typetree  = atoi(argv[14]);     // 0:Tree-UMDA, 1:Tree-Markov, 2:Tree-Tree
 instance   = atoi(argv[15]);    // index of the icosahedral instance (value between 0 and 59) 
 printvals  = atoi(argv[16]);    // The printvals-1 best values in each generation are printed 
 //Card =  atoi(argv[15]);
 
 LAMBDA = lambdas[typelambda];
 if(typesampling==1) sampl_threshold=0.5;
 else if(typesampling==2) sampl_threshold=0.25;

 Tour = 0;                       // Truncation Selection is used
 Elit = 1;                       // Elitism
 InitTreeStructure = 1;          // 0 for a random init tree structures, 1 for a Chu&Liu learned init Tree Structure  
 Prior = 1;                      // Type of prior. 
 Compl=75;                       // Complexities of the trees. 
 Coeftype=2;                     // Type of coefficient calculation for Exact Learning. 
 params[2] = 10;  
 seed = (unsigned) time(NULL);  
 srand(seed); 
 Trunc = T/double(100);          // Parameter of truncation (from 0 to 1)
 mComplex  = Compl/double(100);  
 Cardinalities  = new unsigned[5000];  


for(u=0;u<5000;u++) Cardinalities[u] = Card;  
 auxbraid = new unsigned int[vars]; // Auxiliary variable for braid evaluation. Made global for efficiency considerations (avoid calls to new)

// The structures needed by the braids are initialized
 InitBraids();    

//
 cout<<"Alg : "<<typetree<<", number codifying bits : "<<length<<", n : "<<vars<<", psize : "<<psize<<", Trunc : "<<T<<", max-gen : "<<Maxgen<<", BestElit. : "<<BestElitism<<", NNeighbors  : "<<CliqMaxLength<<", MaxFun  : "<<Max<<", func : "<<func<<" Card: "<<Card<<endl; 

        succexp =0; nsucc = 0; 
        AbsBestInd = new unsigned int [vars];
        AbsBestEval = -1;
        TotEvaluations = 0;       
       	i =0;   
	while (i<cantexp) //&& nsucc<1
        { 
          currentexp = i;
          Experiment = i;
	  int succ=Intusualinit(mComplex);
          i++;         
        }  

        for(i=0;i<60;i++) delete All_IcoTargets[i];
           	         
	delete[] AbsBestInd;                         
        delete[] auxbraid;
        delete [] params; 
        delete [] Cardinalities; 
        return 0;

}      




