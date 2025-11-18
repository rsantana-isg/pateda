// ------------------------------------------------------------------------ //
// This source file is part of the 'ESA Advanced Concepts Team's			//
// Space Mechanics Toolbox' software.                                       //
//                                                                          //
// The source files are for research use only,                              //
// and are distributed WITHOUT ANY WARRANTY. Use them on your own risk.     //
//                                                                          //
// Copyright (c) 2004-2007 European Space Agency                            //
// ------------------------------------------------------------------------ //
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "mga_dsm.h"


#include <math.h> 
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 

#include "auxfunc.h"  
#include "Popul.h"  
#include "EDA.h" 
#include "AbstractTree.h"  
#include "FDA.h"  
#include "MixtureTrees.h" 
#include "affinity.h" 
#include "JtreeTable.h" 
#include "jtPartition.h" 

using namespace std;




FILE *stream;  
FILE *file,*outfile;  	  
 
double meanlikehood[500]; 
double Sel_Int[11] = {2.326,2.603,1.755,1.554,1.4,1.271,1.159,1.058,0.966,0.88,0.798};  
double SelInt; 

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
 

int TotEvaluations;
int EvaluationMode;
int currentexp;
int length;
long int explength;
 int MaxMPC;
 int TypeMPC;
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
 




/*


// Variables for the trajectory problem

void evalfunction(Popul* epop,int nelit, int epsize, int atgen)
{

#if MGADSM_PROBLEM_TYPE == time2AUs  // SAGAS
	int dim = 3;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 5;
	
	double *Delta_V = new double[dim+1];
	vector<double> t(12);
        t[0] = 7020.49;
	t[1] = 5.34817;
	t[2] = 1;
	t[3] = 0.498915;
	t[4] = 788.763;
	t[5] = 484.349;
	t[6] = 0.4873;
	t[7] = 0.01;
	t[8] = 1.05;
	t[9] = 10.8516;
	t[10] = -1.57191;
	t[11] = -0.685429;
	
	dsm_customobject c_o; // empty in this case
	const double rp = 3950;
	const double e = 0.98;	
	const double Isp = 0.0;
	const double mass = 0.0;	
	const double AUdist = 50.0;
	const double DVtotal = 6.782;
	const double DVonboard = 1.782;

#elif MGADSM_PROBLEM_TYPE == total_DV_rndv  // Cassini with DSM

#if total_DV_rndv_problem == cassini
	int dim = 6;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 2;
	sequence[2] = 2;
	sequence[3] = 3;
	sequence[4] = 5;
	sequence[5] = 6;
	
	double *Delta_V = new double[dim+1];
	vector<double> t(22);
    t[0] = -815.144;
	t[1] = 3;
	t[2] = 0.623166;
	t[3] = 0.444834;
	t[4] = 197.334;
	t[5] = 425.171;
	t[6] = 56.8856;
	t[7] = 578.523;
	t[8] = 2067.98;
	t[9] = 0.01;
	t[10] = 0.470415;
	t[11] = 0.01;
	t[12] = 0.0892135;
	t[13] = 0.9;
	t[14] = 1.05044;
	t[15] = 1.38089;
	t[16] = 1.18824;
	t[17] = 76.5066;
	t[18] = -1.57225;
	t[19] = -2.01799;
	t[20] = -1.52153;
	t[21] = -1.5169;

	// all the rest is empty in this case
	dsm_customobject c_o; 
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;

#endif
#if total_DV_rndv_problem == messenger

	int dim = 5;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 2;
	sequence[3] = 2;
	sequence[4] = 1;

	double *Delta_V = new double[dim+1];
	vector<double> t(18);
    t[0] = 2363.36;
	t[1] = 1.68003;
	t[2] = 0.381885;
	t[3] = 0.512516;
	t[4] = 400;
	t[5] = 173.848;
	t[6] = 224.702;
	t[7] = 211.803;
	t[8] = 0.238464;
	t[9] = 0.265663;
	t[10] = 0.149817;
	t[11] = 0.485908;
	t[12] = 1.34411;
	t[13] = 3.49751;
	t[14] = 1.1;
	t[15] = 1.29892;
	t[16] = 2.49324;
	t[17] = 1.81426;

	// all the rest is empty in this case
	dsm_customobject c_o; 
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;
#endif

#elif MGADSM_PROBLEM_TYPE == rndv

	int dim = 6;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 4;
	sequence[3] = 3;
	sequence[4] = 3;
	sequence[5] = 10;

	double *Delta_V = new double[dim+1];
	vector<double> t(22);
    t[0] = 1524.25;
	t[1] = 3.95107;
	t[2] = 0.738307;
	t[3] = 0.298318;
	t[4] = 365.123;
	t[5] = 728.902;
	t[6] = 256.049;
	t[7] = 730.485;
	t[8] = 1850;
	t[9] = 0.199885;
	t[10] = 0.883382;
	t[11] = 0.194587;
	t[12] = 0.0645205;
	t[13] = 0.493077;
	t[14] = 1.05;
	t[15] = 1.05;
	t[16] = 1.05;
	t[17] = 1.36925;
	t[18] = -1.74441;
	t[19] = 1.85201;
	t[20] = -2.61644;
	t[21] = -1.53468;
	
	dsm_customobject c_o; 
	c_o.keplerian[0] = 3.50294972836275;
	c_o.keplerian[1] = 0.6319356;
	c_o.keplerian[2] =  7.12723;
	c_o.keplerian[3] = 	50.92302;
	c_o.keplerian[4] =  11.36788;
	c_o.keplerian[5] = 0.0;
	c_o.epoch = 52504.23754000012;
	c_o.mu = 0.0;

	// all the rest is empty in this case
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;
#endif

 double obj_f;


 int k,j,eval_local,start;
 if (atgen==0) start=0;
 else start=nelit;

 const double pi = acos(-1.0);
 double minbound[12] = {7000.0, 0.0, 0.0, 0.0, 50.0,   300.0,  0.01,  0.01,  1.05, 8.0,  -1*pi, -1*pi};
 double maxbound[12] = {9100.0, 7.0, 1.0, 1.0, 2000.0, 2000.0, 0.9,   0.9,   7.0,  500.0,   pi,  pi};
 long int locusvalues[22] };
 
 
 //cout<<"ExpLength is "<<explength<<endl;

for(k=start; k < epsize;  k++)  
 {
       
   for (j=0;j<12;j++)
     {
       locusvalues[j] = ConvertNum(length,Card,&epop->P[k][j*length]); 
       //cout<<locusvalues[j]<<" ";
       t[j] =  minbound[j] + (maxbound[j]-minbound[j]) * ((locusvalues[j]+1.0)/explength);
     } 
   
   
   //for (j=0;j<12;j++)    t[j] =  minbound[j] + (maxbound[j]-minbound[j]) * epop->P[k][j]/1024.0;
  
   //cout<<endl;
   //for (j=0;j<12;j++) cout<<t[j]<<" ";
   //cout<<endl;   
       
   MGA_DSM(t, dim, sequence, c_o, rp, e, Isp, mass, AUdist, DVtotal, DVonboard, obj_f, Delta_V);	
   //cout<<setprecision(10)<<endl<<"MGA_DSM objective value = " << obj_f<<endl;
   eval_local = 1;   
     
   epop->SetVal(k,1000000.0 - obj_f); //Minimization is transformed in maximization 
   TotEvaluations += eval_local; 
 }
 delete[] Delta_V;
}

*/




// Variables for the trajectory problem

void evalfunction(Popul* epop,int nelit, int epsize, int atgen)
{
 double obj_f;
 const double pi = acos(-1.0);
 int tract_vars;
 double minbound[22];
 double maxbound[22];
 long int locusvalues[22];

 
 #if MGADSM_PROBLEM_TYPE == time2AUs  // SAGAS
	int dim = 3;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 5;
	
	double *Delta_V = new double[dim+1];
	vector<double> t(12);
        t[0] = 7020.49;
	t[1] = 5.34817;
	t[2] = 1;
	t[3] = 0.498915;
	t[4] = 788.763;
	t[5] = 484.349;
	t[6] = 0.4873;
	t[7] = 0.01;
	t[8] = 1.05;
	t[9] = 10.8516;
	t[10] = -1.57191;
	t[11] = -0.685429;
	
	dsm_customobject c_o; // empty in this case
	const double rp = 3950;
	const double e = 0.98;	
	const double Isp = 0.0;
	const double mass = 0.0;	
	const double AUdist = 50.0;
	const double DVtotal = 6.782;
	const double DVonboard = 1.782;

     
        tract_vars = 12;

        minbound[0] = 7000.0;   maxbound[0] = 9100.0;
        minbound[1] = 0.0;      maxbound[1] = 7.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 50.0;     maxbound[4] = 2000.0;
        minbound[5] = 300.0;    maxbound[5] = 2000.0;
        minbound[6] = 0.01;     maxbound[6] = 0.9;
        minbound[7] = 0.01;     maxbound[7] = 0.9; 
        minbound[8] = 1.05;     maxbound[8] = 7.0;
        minbound[9] = 8.0;      maxbound[9] = 7.0;
        minbound[10] = -1*pi;   maxbound[10] = pi;
        minbound[11] = -1*pi;   maxbound[11] = pi;
  
     

#elif MGADSM_PROBLEM_TYPE == total_DV_rndv  // Cassini with DSM

#if total_DV_rndv_problem == cassini
	int dim = 6;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 2;
	sequence[2] = 2;
	sequence[3] = 3;
	sequence[4] = 5;
	sequence[5] = 6;
	
	double *Delta_V = new double[dim+1];
	vector<double> t(22);
    t[0] = -815.144;
	t[1] = 3;
	t[2] = 0.623166;
	t[3] = 0.444834;
	t[4] = 197.334;
	t[5] = 425.171;
	t[6] = 56.8856;
	t[7] = 578.523;
	t[8] = 2067.98;
	t[9] = 0.01;
	t[10] = 0.470415;
	t[11] = 0.01;
	t[12] = 0.0892135;
	t[13] = 0.9;
	t[14] = 1.05044;
	t[15] = 1.38089;
	t[16] = 1.18824;
	t[17] = 76.5066;
	t[18] = -1.57225;
	t[19] = -2.01799;
	t[20] = -1.52153;
	t[21] = -1.5169;

	// all the rest is empty in this case
	dsm_customobject c_o; 
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;


  tract_vars = 22;

        minbound[0] = -1000.0;   maxbound[0] = 0.0;
        minbound[1] = 3.0;      maxbound[1] = 5.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 100.0;     maxbound[4] = 400.0;
        minbound[5] = 100.0;    maxbound[5] =  500.0;
        minbound[6] = 30.0;     maxbound[6] = 300.0;
        minbound[7] = 400.0;     maxbound[7] = 1600.0; 
        minbound[8] = 800.0;     maxbound[8] = 2200.0;
        minbound[9] = 0.01;     maxbound[9] = 0.9;
        minbound[10] = 0.01;     maxbound[10] = 0.9; 
        minbound[11] = 0.01;     maxbound[11] = 0.9;
        minbound[12] = 0.01;     maxbound[12] = 0.9; 
        minbound[13] = 0.01;     maxbound[13] = 0.9;   
        minbound[14] = 1.05;     maxbound[14] = 6.0; 
        minbound[15] = 1.05;     maxbound[15] = 6.0; 
        minbound[16] = 1.15;     maxbound[16] = 6.5; 
        minbound[17] = 1.7;      maxbound[17] = 291.0;
        minbound[18] = -1*pi;   maxbound[18] = pi;
        minbound[19] = -1*pi;   maxbound[19] = pi;
        minbound[20] = -1*pi;   maxbound[20] = pi;
        minbound[21] = -1*pi;   maxbound[21] = pi;
#endif
#if total_DV_rndv_problem == messenger

	int dim = 5;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 2;
	sequence[3] = 2;
	sequence[4] = 1;

	double *Delta_V = new double[dim+1];
	vector<double> t(18);
    t[0] = 2363.36;
	t[1] = 1.68003;
	t[2] = 0.381885;
	t[3] = 0.512516;
	t[4] = 400;
	t[5] = 173.848;
	t[6] = 224.702;
	t[7] = 211.803;
	t[8] = 0.238464;
	t[9] = 0.265663;
	t[10] = 0.149817;
	t[11] = 0.485908;
	t[12] = 1.34411;
	t[13] = 3.49751;
	t[14] = 1.1;
	t[15] = 1.29892;
	t[16] = 2.49324;
	t[17] = 1.81426;

	// all the rest is empty in this case
	dsm_customobject c_o; 
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;


        tract_vars = 18;

        minbound[0] = 1000.0;   maxbound[0] = 4100.0;
        minbound[1] = 1.0;      maxbound[1] = 5.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 200.0;    maxbound[4] = 400.0;
        minbound[5] = 30.0;    maxbound[5] =  400.0;
        minbound[6] = 30.0;    maxbound[6] =  400.0;
        minbound[7] = 30.0;    maxbound[7] =  400.0;
        minbound[8] = 0.01;     maxbound[8] = 0.99;
        minbound[9] = 0.01;     maxbound[9] = 0.99;
        minbound[10] = 0.01;     maxbound[10] = 0.99;
        minbound[11] = 0.01;     maxbound[11] = 0.99;
        minbound[12] = 1.1;      maxbound[12] = 6.0;
        minbound[13] = 1.1;      maxbound[13] = 6.0;
        minbound[14] = 1.1;      maxbound[14] = 6.0;
        minbound[15] = -1*pi;   maxbound[15] = pi;
        minbound[16] = -1*pi;   maxbound[16] = pi;
        minbound[17] = -1*pi;   maxbound[27] = pi;
    

#endif

#elif MGADSM_PROBLEM_TYPE == rndv

	int dim = 6;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 4;
	sequence[3] = 3;
	sequence[4] = 3;
	sequence[5] = 10;

	double *Delta_V = new double[dim+1];
	vector<double> t(22);
    t[0] = 1524.25;
	t[1] = 3.95107;
	t[2] = 0.738307;
	t[3] = 0.298318;
	t[4] = 365.123;
	t[5] = 728.902;
	t[6] = 256.049;
	t[7] = 730.485;
	t[8] = 1850;
	t[9] = 0.199885;
	t[10] = 0.883382;
	t[11] = 0.194587;
	t[12] = 0.0645205;
	t[13] = 0.493077;
	t[14] = 1.05;
	t[15] = 1.05;
	t[16] = 1.05;
	t[17] = 1.36925;
	t[18] = -1.74441;
	t[19] = 1.85201;
	t[20] = -2.61644;
	t[21] = -1.53468;
	
	dsm_customobject c_o; 
	c_o.keplerian[0] = 3.50294972836275;
	c_o.keplerian[1] = 0.6319356;
	c_o.keplerian[2] =  7.12723;
	c_o.keplerian[3] = 	50.92302;
	c_o.keplerian[4] =  11.36788;
	c_o.keplerian[5] = 0.0;
	c_o.epoch = 52504.23754000012;
	c_o.mu = 0.0;

	// all the rest is empty in this case
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;

       tract_vars = 22;

        minbound[0] = 1460.0;   maxbound[0] = 1825.0;
        minbound[1] = 3.0;      maxbound[1] = 5.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 300.0;    maxbound[4] = 500.0;
        minbound[5] = 150.0;    maxbound[5] = 800.0;
        minbound[6] = 150.0;    maxbound[6] =  800.0;
        minbound[7] = 300.0;    maxbound[7] =  800.0;
        minbound[8] = 700.0;     maxbound[8] = 1850.0;
        minbound[9] = 0.01;     maxbound[9] = 0.9;
         minbound[10] = 0.01;     maxbound[10] = 0.9;
        minbound[11] = 0.01;     maxbound[11] = 0.9;
        minbound[12] = 0.01;      maxbound[12] = 0.9;
        minbound[13] = 0.01;      maxbound[13] = 0.9;
        minbound[14] = 0.01;      maxbound[14] = 0.9;
        minbound[15] = 1.05;      maxbound[15] = 9.0;
        minbound[16] = 1.05;      maxbound[16] = 9.0;
        minbound[17] = 1.05;      maxbound[17] = 9.0;
        minbound[18] = 1.05;      maxbound[18] = 9.0;
        minbound[19] = -1*pi;   maxbound[19] = pi;
        minbound[20] = -1*pi;   maxbound[20] = pi;
        minbound[21] = -1*pi;   maxbound[21] = pi;
    #endif



 int k,j,eval_local,start;
 if (atgen==0) start=0;
 else start=nelit;

cout<<"tract_vars is "<<tract_vars<<"length is "<<length<<endl;

//unsigned int auxsol[66]  = {30, 13, 35, 29, 38, 13, 30, 14, 38, 8, 17, 38, 13, 12, 6, 20, 18, 11, 34, 26, 12, 34, 26,  23, 37, 23, 12, 4, 11, 12, 18, 26, 21, 20, 26, 35, 15, 5, 8, 18, 13, 23, 17, 25, 15, 24, 32, 9, 7, 35, 8, 2, 30, 19, 19, 5, 7, 27, 29, 13, 5, 26, 4, 8, 39, 31}; 
/*
 
for (j=0;j<tract_vars;j++)
     {
       locusvalues[j] = ConvertNum(length,Card,&auxsol[j*length]); 
//cout<<locusvalues[j]<<" ";
       t[j] =  minbound[j] + (maxbound[j]-minbound[j]) * ((locusvalues[j]+1.0)/explength);
       cout<<t[j]<<" ";
     } 

MGA_DSM(t, dim, sequence, c_o, rp, e, Isp, mass, AUdist, DVtotal, DVonboard, obj_f, Delta_V);

cout<<"ABSBEST------------------ "<<obj_f<<endl;

*/
for(k=start; k < epsize;  k++)  
 {
       
   for (j=0;j<tract_vars;j++)
     {
       locusvalues[j] = ConvertNum(length,Card,&epop->P[k][j*length]); 
       //cout<<locusvalues[j]<<" ";
       t[j] =  minbound[j] + (maxbound[j]-minbound[j]) * ((locusvalues[j]+1.0)/explength);
     } 
  
       
   MGA_DSM(t, dim, sequence, c_o, rp, e, Isp, mass, AUdist, DVtotal, DVonboard, obj_f, Delta_V);	
   //cout<<setprecision(10)<<endl<<"MGA_DSM objective value = " << obj_f<<endl;
   eval_local = 1;        
   epop->SetVal(k,1000000.0 - obj_f); //Minimization is transformed in maximization 
   TotEvaluations += eval_local; 
 }
 delete[] Delta_V;
}






// Variables for the trajectory problem

void TrajectLocalOptimizer(Popul* epop,int nelit, int epsize, int atgen)
{
 double obj_f;
 const double pi = acos(-1.0);
 int tract_vars;
 double minbound[22];
 double maxbound[22];
 long int locusvalues[22];

 
 #if MGADSM_PROBLEM_TYPE == time2AUs  // SAGAS
	int dim = 3;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 5;
	
	double *Delta_V = new double[dim+1];
	vector<double> t(12);
        t[0] = 7020.49;
	t[1] = 5.34817;
	t[2] = 1;
	t[3] = 0.498915;
	t[4] = 788.763;
	t[5] = 484.349;
	t[6] = 0.4873;
	t[7] = 0.01;
	t[8] = 1.05;
	t[9] = 10.8516;
	t[10] = -1.57191;
	t[11] = -0.685429;
	
	dsm_customobject c_o; // empty in this case
	const double rp = 3950;
	const double e = 0.98;	
	const double Isp = 0.0;
	const double mass = 0.0;	
	const double AUdist = 50.0;
	const double DVtotal = 6.782;
	const double DVonboard = 1.782;

     
        tract_vars = 12;

        minbound[0] = 7000.0;   maxbound[0] = 9100.0;
        minbound[1] = 0.0;      maxbound[1] = 7.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 50.0;     maxbound[4] = 2000.0;
        minbound[5] = 300.0;    maxbound[5] = 2000.0;
        minbound[6] = 0.01;     maxbound[6] = 0.9;
        minbound[7] = 0.01;     maxbound[7] = 0.9; 
        minbound[8] = 1.05;     maxbound[8] = 7.0;
        minbound[9] = 8.0;      maxbound[9] = 7.0;
        minbound[10] = -1*pi;   maxbound[10] = pi;
        minbound[11] = -1*pi;   maxbound[11] = pi;
  
     

#elif MGADSM_PROBLEM_TYPE == total_DV_rndv  // Cassini with DSM

#if total_DV_rndv_problem == cassini
	int dim = 6;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 2;
	sequence[2] = 2;
	sequence[3] = 3;
	sequence[4] = 5;
	sequence[5] = 6;
	
	double *Delta_V = new double[dim+1];
	vector<double> t(22);
    t[0] = -815.144;
	t[1] = 3;
	t[2] = 0.623166;
	t[3] = 0.444834;
	t[4] = 197.334;
	t[5] = 425.171;
	t[6] = 56.8856;
	t[7] = 578.523;
	t[8] = 2067.98;
	t[9] = 0.01;
	t[10] = 0.470415;
	t[11] = 0.01;
	t[12] = 0.0892135;
	t[13] = 0.9;
	t[14] = 1.05044;
	t[15] = 1.38089;
	t[16] = 1.18824;
	t[17] = 76.5066;
	t[18] = -1.57225;
	t[19] = -2.01799;
	t[20] = -1.52153;
	t[21] = -1.5169;

	// all the rest is empty in this case
	dsm_customobject c_o; 
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;


  tract_vars = 22;

        minbound[0] = -1000.0;   maxbound[0] = 0.0;
        minbound[1] = 3.0;      maxbound[1] = 5.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 100.0;     maxbound[4] = 400.0;
        minbound[5] = 100.0;    maxbound[5] =  500.0;
        minbound[6] = 30.0;     maxbound[6] = 300.0;
        minbound[7] = 400.0;     maxbound[7] = 1600.0; 
        minbound[8] = 800.0;     maxbound[8] = 2200.0;
        minbound[9] = 0.01;     maxbound[9] = 0.9;
        minbound[10] = 0.01;     maxbound[10] = 0.9; 
        minbound[11] = 0.01;     maxbound[11] = 0.9;
        minbound[12] = 0.01;     maxbound[12] = 0.9; 
        minbound[13] = 0.01;     maxbound[13] = 0.9;   
        minbound[14] = 1.05;     maxbound[14] = 6.0; 
        minbound[15] = 1.05;     maxbound[15] = 6.0; 
        minbound[16] = 1.15;     maxbound[16] = 6.5; 
        minbound[17] = 1.7;      maxbound[17] = 291.0;
        minbound[18] = -1*pi;   maxbound[18] = pi;
        minbound[19] = -1*pi;   maxbound[19] = pi;
        minbound[20] = -1*pi;   maxbound[20] = pi;
        minbound[21] = -1*pi;   maxbound[21] = pi;
#endif
#if total_DV_rndv_problem == messenger

	int dim = 5;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 2;
	sequence[3] = 2;
	sequence[4] = 1;

	double *Delta_V = new double[dim+1];
	vector<double> t(18);
    t[0] = 2363.36;
	t[1] = 1.68003;
	t[2] = 0.381885;
	t[3] = 0.512516;
	t[4] = 400;
	t[5] = 173.848;
	t[6] = 224.702;
	t[7] = 211.803;
	t[8] = 0.238464;
	t[9] = 0.265663;
	t[10] = 0.149817;
	t[11] = 0.485908;
	t[12] = 1.34411;
	t[13] = 3.49751;
	t[14] = 1.1;
	t[15] = 1.29892;
	t[16] = 2.49324;
	t[17] = 1.81426;

	// all the rest is empty in this case
	dsm_customobject c_o; 
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;


        tract_vars = 18;

        minbound[0] = 1000.0;   maxbound[0] = 4100.0;
        minbound[1] = 1.0;      maxbound[1] = 5.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 200.0;    maxbound[4] = 400.0;
        minbound[5] = 30.0;    maxbound[5] =  400.0;
        minbound[6] = 30.0;    maxbound[6] =  400.0;
        minbound[7] = 30.0;    maxbound[7] =  400.0;
        minbound[8] = 0.01;     maxbound[8] = 0.99;
        minbound[9] = 0.01;     maxbound[9] = 0.99;
        minbound[10] = 0.01;     maxbound[10] = 0.99;
        minbound[11] = 0.01;     maxbound[11] = 0.99;
        minbound[12] = 1.1;      maxbound[12] = 6.0;
        minbound[13] = 1.1;      maxbound[13] = 6.0;
        minbound[14] = 1.1;      maxbound[14] = 6.0;
        minbound[15] = -1*pi;   maxbound[15] = pi;
        minbound[16] = -1*pi;   maxbound[16] = pi;
        minbound[17] = -1*pi;   maxbound[27] = pi;
    

#endif

#elif MGADSM_PROBLEM_TYPE == rndv

	int dim = 6;
	vector<int> sequence(dim);
	sequence[0] = 3;
	sequence[1] = 3;
	sequence[2] = 4;
	sequence[3] = 3;
	sequence[4] = 3;
	sequence[5] = 10;

	double *Delta_V = new double[dim+1];
	vector<double> t(22);
    t[0] = 1524.25;
	t[1] = 3.95107;
	t[2] = 0.738307;
	t[3] = 0.298318;
	t[4] = 365.123;
	t[5] = 728.902;
	t[6] = 256.049;
	t[7] = 730.485;
	t[8] = 1850;
	t[9] = 0.199885;
	t[10] = 0.883382;
	t[11] = 0.194587;
	t[12] = 0.0645205;
	t[13] = 0.493077;
	t[14] = 1.05;
	t[15] = 1.05;
	t[16] = 1.05;
	t[17] = 1.36925;
	t[18] = -1.74441;
	t[19] = 1.85201;
	t[20] = -2.61644;
	t[21] = -1.53468;
	
	dsm_customobject c_o; 
	c_o.keplerian[0] = 3.50294972836275;
	c_o.keplerian[1] = 0.6319356;
	c_o.keplerian[2] =  7.12723;
	c_o.keplerian[3] = 	50.92302;
	c_o.keplerian[4] =  11.36788;
	c_o.keplerian[5] = 0.0;
	c_o.epoch = 52504.23754000012;
	c_o.mu = 0.0;

	// all the rest is empty in this case
	const double rp = 0;
	const double e = 0;	
	const double Isp = 0.0;
	const double mass = 0.0;
	const double AUdist = 0;
	const double DVtotal = 0;
	const double DVonboard = 0;

       tract_vars = 22;

        minbound[0] = 1460.0;   maxbound[0] = 1825.0;
        minbound[1] = 3.0;      maxbound[1] = 5.0;
        minbound[2] = 0.0;      maxbound[2] = 1.0;
        minbound[3] = 0.0;      maxbound[3] = 1.0;
        minbound[4] = 300.0;    maxbound[4] = 500.0;
        minbound[5] = 150.0;    maxbound[5] = 800.0;
        minbound[6] = 150.0;    maxbound[6] =  800.0;
        minbound[7] = 300.0;    maxbound[7] =  800.0;
        minbound[8] = 700.0;     maxbound[8] = 1850.0;
        minbound[9] = 0.01;     maxbound[9] = 0.9;
         minbound[10] = 0.01;     maxbound[10] = 0.9;
        minbound[11] = 0.01;     maxbound[11] = 0.9;
        minbound[12] = 0.01;      maxbound[12] = 0.9;
        minbound[13] = 0.01;      maxbound[13] = 0.9;
        minbound[14] = 0.01;      maxbound[14] = 0.9;
        minbound[15] = 1.05;      maxbound[15] = 9.0;
        minbound[16] = 1.05;      maxbound[16] = 9.0;
        minbound[17] = 1.05;      maxbound[17] = 9.0;
        minbound[18] = 1.05;      maxbound[18] = 9.0;
        minbound[19] = -1*pi;   maxbound[19] = pi;
        minbound[20] = -1*pi;   maxbound[20] = pi;
        minbound[21] = -1*pi;   maxbound[21] = pi;
    #endif



 int k,j,eval_local,start;
 if (atgen==0) start=0;
 else start=nelit;

cout<<"tract_vars is "<<tract_vars<<"length is "<<length<<endl;

//unsigned int auxsol[66]  = {30, 13, 35, 29, 38, 13, 30, 14, 38, 8, 17, 38, 13, 12, 6, 20, 18, 11, 34, 26, 12, 34, 26,  23, 37, 23, 12, 4, 11, 12, 18, 26, 21, 20, 26, 35, 15, 5, 8, 18, 13, 23, 17, 25, 15, 24, 32, 9, 7, 35, 8, 2, 30, 19, 19, 5, 7, 27, 29, 13, 5, 26, 4, 8, 39, 31}; 
/*
 
for (j=0;j<tract_vars;j++)
     {
       locusvalues[j] = ConvertNum(length,Card,&auxsol[j*length]); 
//cout<<locusvalues[j]<<" ";
       t[j] =  minbound[j] + (maxbound[j]-minbound[j]) * ((locusvalues[j]+1.0)/explength);
       cout<<t[j]<<" ";
     } 

MGA_DSM(t, dim, sequence, c_o, rp, e, Isp, mass, AUdist, DVtotal, DVonboard, obj_f, Delta_V);

cout<<"ABSBEST------------------ "<<obj_f<<endl;

*/

 int nmoves,l,bestvar,bestval,Improve,oldval;
 double oldt,best_obj_f;

for(k=start; k < epsize;  k++)  
 {  

   
  for (j=0;j<tract_vars;j++)
     {
         locusvalues[j] = ConvertNum(length,Card,&epop->P[k][j*length]);    
         t[j] =  minbound[j] + (maxbound[j]-minbound[j]) * ((locusvalues[j]+1.0)/explength);
     } 

    MGA_DSM(t, dim, sequence, c_o, rp, e, Isp, mass, AUdist, DVtotal, DVonboard, best_obj_f, Delta_V);
    TotEvaluations += 1;    

  Improve = 1;
  nmoves = 0;
  while (Improve)
   {
    Improve = 0;
   
    j = randomint(tract_vars);
    //for (j=0;j<tract_vars;j++)
     {
        oldval = epop->P[k][j*length];
        oldt = t[j];
        for (l=0;l<Card;l++)
	 { 
	   if(l != oldval)
           { 
             epop->P[k][j*length] = l;
	     locusvalues[j] = ConvertNum(length,Card,&epop->P[k][j*length]);    
             t[j] =  minbound[j] + (maxbound[j]-minbound[j]) * ((locusvalues[j]+1.0)/explength);
             MGA_DSM(t, dim, sequence, c_o, rp, e, Isp, mass, AUdist, DVtotal, DVonboard, obj_f, Delta_V);
             if (obj_f < best_obj_f)
	     {
               Improve = 1;
               bestvar = j;
               bestval = l; 
               best_obj_f = obj_f;
             }
	     //   cout<<k<<" "<<j<<" "<<l<<" "<<setprecision(10)<<" MGA_DSM objective value = " << obj_f<<endl;
           TotEvaluations += 1;        
         }
        }
	epop->P[k][j*length] = oldval;
        t[j] = oldt;
     }
    if(Improve==1)     
      {
	epop->P[k][bestvar*length] = bestval;
        locusvalues[bestvar] = ConvertNum(length,Card,&epop->P[k][bestvar*length]); 
        t[bestvar] =  minbound[bestvar] + (maxbound[bestvar]-minbound[bestvar]) * ((locusvalues[bestvar]+1.0)/explength);
        //cout<<k<<" TotEvals "<<TotEvaluations<<" "<<setprecision(10)<<" MGA_DSM objective value = " << 1000000.0 - best_obj_f<<endl;
      }
    nmoves++;
   }


   epop->SetVal(k,1000000.0 - best_obj_f); //Minimization is transformed in maximization 
   
 }
 delete[] Delta_V;
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
  pop->RandInit();  
  compact_pop = new Popul(psize,vars,Elit,Cardinalities);  
  fvect = new double[psize];
 
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
 

  i=0;  fgen = -1;  auxprob =0; BestEval  = Max -1; NPoints=TruncMax; 

  Popul* aux_pop;
  aux_pop = new Popul(psize,vars,Elit,Cardinalities);  
 
 NPoints = psize;
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 1;
 TotEvaluations = 0; 


  while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
        evalfunction(pop,Elit,psize,i);       
        NPoints = Selection(); 
        MyMarkovNet->SetNPoints(selpop->psize,NPoints,fvect); 
        MyMarkovNet->SetPop(selpop); 
   
  FindBestVal(); 

  if(printvals>1) 
    {     
         cout<<"Best : ";
     for(int l=0;l<vars;l++) cout<<BestInd[l]<<" ";
     printf("%10f \n", BestEval); 
  
     if(printvals>0)     cout<<"Gen : "<<i<<" Best: "<<BestEval<<" Elit "<<Elit<<" TotEval: "<<TotEvaluations<<" DifPoints: "<<NPoints<<" sizecliq "<<sizecliq<<" "<<OldBest<<" "<<BestEval<<" "<<gap<<endl; 
    }

     MyMarkovNet->UpdateModelProtein(typemodel,sizecliq); 
     auxprob = MyMarkovNet->Prob(BestInd);  
     
           
     //  if(printvals>1)     cout<<"Gen : "<<i<<" Best: "<<BestEval<<" Elit "<<Elit<<" TotEval: "<<TotEvaluations<<" DifPoints: "<<NPoints<<" sizecliq "<<sizecliq<<endl; 
         


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
     
     if(NPoints>10)  MyMarkovNet->GenPop(Elit,pop);       
  
       
           i++;         
          MyMarkovNet->Destroy();    
     
  }  

  if(NPoints>10) NPoints = 10;
   end_time();  

 if(printvals>0)  cout<<"Gen : "<<i<<" Best: "<<BestEval<<" ProbBest: "<<auxprob<<" DifUhyPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" Best Solution:  ";  
  
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
   
     //pop->Print();
     evalfunction(pop,Elit,psize,i);     
     NPoints = Selection(); 
     
   
       
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
    
     i++;
  }  
  
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
     
  
   //pop->Print(); 

   evalfunction(pop,Elit,psize,i); 
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
                
      	  }   

       MixtureInt->RemoveTrees(); 
       MixtureInt->RemoveProbabilities();
        
      i++;  

 
  }  


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
  //MyMarkovNet->SetProtein(FoldingProtein); 
 
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

 TotEvaluations = 0;
 while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
  
     evalfunction(pop,Elit,psize,i); 
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
     MyMarkovNet->UpdateModelProteinMPM(2,nclust,listclusters); //Non overlapping sets for the factorization
     FindBestVal();   
     auxprob = MyMarkovNet->Prob(BestInd);  
           
       
   if(printvals>1) 
    {     
     
     for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
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
 

  CliqMaxLength = 3; //(int) (log(selpop->psize)/log(Cardinalities[0])) ;
 
  MaxNumCliq = vars;
  Kmpc = psize-1; 
  ncMPM = 0; ncTree = 0;
 
  LearningType=5; // A Marginal Product  Model 
  MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
  //MyMarkovNet->SetProtein(FoldingProtein); 
 
  IntTree = new IntTreeModel(vars,Complexity,selpop->psize,Cardinalities);  //Needed for learning mutual information
  
  AffPropagation* AffProp; 
  AffProp = new AffPropagation(vars);
  
  listclusters = new  unsigned long* [vars];
  allvars = new  unsigned long [vars];
  for(i=0;i<vars;i++) allvars[i] = i;
  
 
  lam = 0.5; maxits = 1000; convits = 50; deph = 30; SimThreshold = 0.00001;
  i=0; auxprob =0; BestEval  = Max -1; fgen = -1; NPoints = 100;  

 cout<<CliqMaxLength<<" "<<MaxNumCliq<<" "<<Complexity<<" "<<selpop->psize<<" "<<BestEval<<"  "<<NPoints<<endl; 
 OldBest = 0;
 AbsBestEval = 0;
 gap  = 0;
 TotEvaluations = 0;
 while (i<Maxgen && BestEval<Max && NPoints>10)  
  {  
   
    // pop->Print();
    TrajectLocalOptimizer(pop,Elit,psize,i);  //Local optimizer
     //evalfunction(pop,Elit,psize,i);  
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
 
     MyMarkovNet->UpdateModelProteinMPM(2,nclust,listclusters); //Non overlapping sets for the factorization
     FindBestVal();   
     IntTree->PutPriors(Prior,selpop->psize,1);
     auxprob = MyMarkovNet->Prob(BestInd);  
           

       
   if(printvals>1) 
    {     
         for(int ll=0;ll<printvals-1;ll++)// NPoints 
      { 
       cout<<"Best :";
       for(int l=0;l<vars;l++) cout<<selpop->P[ll][l]<<" "; 
       cout<<" "<<selpop->Evaluations[ll]<<endl; 
      }
    }
	
     if(printvals)
      {
        cout<<"Gen : "<<i<<" Best: ";
        printf("%10f ", BestEval);
	cout<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" Nclusters "<<nclust<<" TotEval: "<<TotEvaluations<<endl;    
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

      if(TypeMPC == 0 && MaxMPC>0)
	{
          PropagationJtree  = new JtreeTable(vars,nclust, Cardinalities);   
          PropagationJtree->convertMPM(listclusters); 
           ncMPM = FindMaxConf(Elit,PropagationJtree,selpop, pop, MaxMPC); 
        }     
      else if(TypeMPC == 1 && MaxMPC>0) 
	{          
          PropagationJtree  = new JtreeTable(vars, vars, Cardinalities); 
          PropagationJtree->convert(IntTree); 
          ncTree = FindMaxConf(Elit+ncMPM,PropagationJtree,selpop, pop, MaxMPC); 
        }
     
       cout<<" ncMPM is  "<<ncMPM<<endl;   
       cout<<" ncTree is  "<<ncTree <<endl;        
        
        if(TypeMPC == 0)
	{
          cout<<"Random MPM: From "<<Elit+ncMPM+ncTree<<" to "<<(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))<<endl;
	  MyMarkovNet->GenPop(Elit+ncMPM+ncTree,(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree)), pop);  
	}
        if(TypeMPC == 1)
	{
          cout<<"Random Tree Alone: From "<<Elit+ncMPM+ncTree<<" to "<<(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))<<endl;
          IntTree->GenPop(Elit+ncMPM+ncTree, pop); 
        }
 
        //cout<<"Random Tree: From "<<(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))/2<< " to "<<psize<<endl;
      	//MyMarkovNet->GenPop(Elit+ncMPM+ncTree,(Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree)), pop);  
        //IntTree->GenPop(Elit+ncMPM+ncTree,pop); 
        //IntTree->GenPop((Elit+ncMPM+ncTree) + (psize-(Elit+ncMPM+ncTree))/2,pop); 
	//else  MyMarkovNet->GenPop(Elit,pop);    

       MyMarkovNet->Destroy();
   
       
      i++;

  } 
   end_time(); 
 
   if(printvals>0)
    {
      cout<<"LastGen : "<<i<<" Best: ";
      printf("%10f ", BestEval);
      cout<<" ProbBest: "<<auxprob<<" DifPoints: "<<NPoints<<" TotEval: "<<TotEvaluations<<" time "<<auxtime<<" "<<TotEvaluations<<endl;  
    }

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
                       case 4: succ = AffEDAMaxConf(Complex,2);  // (Marginal Product Model  learned using affinity propagation)
                     } 

  

    if (succ>-1)  
   { 
       succexp++; 
       meangen += succ;    
       bestalltime +=auxtime;      
        
   } 
   else nsucc++;
   alltime += auxtime;  
   meaneval += BestEval; 
   meanlikehood[nrun] = BestEval;  
} 



int main( int argc, char *argv[] )
{
  // ./trajectory 1 4 3 22 20000 15 250 1 0 0  

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
 
 length = atoi(argv[3]);    // Number of bits for  each variable   
 vars =  atoi(argv[4]) * length;           //Number of variables (redundant because depends on instance)
 psize = atoi(argv[5]);          // Population size
 T = atoi(argv[6]);              // Percentage of truncation integer number (1:99)
 Maxgen =  atoi(argv[7]);        // Max number of generations 
 BestElitism = atoi(argv[8]);         // If there is or not BestElitism, if thereisnot BestElitism, Elitism = 1 by default;
 MaxMPC = atoi(argv[9]);
 TypeMPC = atoi(argv[10]);

 cout<<"Alg : "<<ExperimentMode<<", number codifying bits : "<<length<<", n : "<<vars<<", psize : "<<psize<<", Trunc : "<<T<<", max-gen : "<<Maxgen<<", BestElit. : "<<BestElitism<<", MaxMPC  : "<<MaxMPC<<", TypeMPC  : "<<TypeMPC<<endl; 

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
Cycles = 0 ; // Number of cycles for GS in the MNEDA or size for the clique in Markov EDA. 


Trunc = T/double(100);  
Complex  = Compl/double(100);  
Max = 1000000.0;   
MaxMixtProb =MaxMixtP/double(100); 
S_alpha = S_alph/double(100); 

//ReadParameters(); 

Cardinalities  = new unsigned[5000];  
int k,j,u;
double eval;


Card = 40;
for(u=0;u<5000;u++) Cardinalities[u] = Card;  
 
 explength = 1.0;
 for(i=0;i<length;i++) explength*=Card;  
  

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
 delete [] params; 
 delete [] Cardinalities; 
 return 0;

}      




