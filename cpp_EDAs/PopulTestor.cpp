#include "auxfunc.h"
#include "Ochoafun.h"
#include "Popul.h"

//Funciones
// order 5

extern int* params;
double cuban1_3[]={0.595,0.2,0.595,0.1,1,0.05,0.09,0.15};
double onemax3[] ={0,1,1,2,1,2,2,3};
double dmh3[]    ={28,26,22,0,14,0,0,30};
double goldberg[]={0.9,0.8,0.8,0.0,0.8,0.0,0.0,1.0};
double mia[]={1,0,1,0,0,1,0,1};

// order 5
double cubanI[] = {2.38,0,0,0,0,0.8,0,0,0,0,2.38,0,0,0,0,0.4,4,0,0,0,0,0.2,0,0,0,0,0.6,0,0,0,0,0.36};
double cubanII[]={0,0,1,0,1,0,2,0,1,0,2,0,2,0,3,0,1,0,2,1,2,1,3,2,2,1,3,2,3,2,4,3};


Popul::Popul(int size, int cvars,int Elit)
{
 int i;
 psize = size;
 vars = cvars;
 elit = Elit;
 P =  new unsigned*[psize];
 dim = new unsigned[vars];
 for(i=0; i < psize; i++) P[i] = new unsigned[cvars];
 for(i=0; i < vars; i++) dim[i] = 2;
 index = new int[psize];
 Evaluations = new double[psize];
}

Popul::Popul(int size, int cvars,int Elit, unsigned* dims)
{
 int i;
 psize = size;
 vars = cvars;
 elit = Elit;
 P =  new unsigned*[psize];
 dim = new unsigned[vars];
 for(i=0; i < psize; i++) P[i] = new unsigned[cvars];
 for(i=0; i < vars; i++) dim[i] = dims[i];
 index = new int[psize];
 Evaluations = new double[psize];
}

void Popul::RandInit()
{
  int i,j;
  
 
 for(i=0;i<psize;i++)
	 for(j=0;j<vars;j++) 
		P[i][j] = randomint(dim[j]);
	 
}

void Popul::Repair(int min,int max, double* univprob)
{
  int i,j, auxunit,auxindex;
   
  auxunit = 0;
 for(i=0;i<psize;i++)
 {
	 auxunit = 0;
	 for(j=0;j<vars;j++) auxunit += P[i][j];
	 while (auxunit>max) 
	 {
        auxindex = randomint(vars);
		if(P[i][auxindex]==1 && univprob[auxindex] !=1)
		{ 
		 auxunit--;
		 P[i][auxindex] = 0;
        }
	 }
   
	while (auxunit<min) 
	 {
		auxindex = randomint(vars);
		if(P[i][auxindex]==0 && univprob[auxindex] !=0)
		{ 
		 auxunit++;
		 P[i][auxindex] = 1;
        }
	 }

 }
}



void Popul::Repair(int min,int max)
{
  int i,j, auxunit,auxindex;
   
  auxunit = 0;
 for(i=0;i<psize;i++)
 {
	 auxunit = 0;
	 for(j=0;j<vars;j++) auxunit += P[i][j];
	 while (auxunit>max) 
	 {
        auxindex = randomint(vars);
		if(P[i][auxindex]==1)
		{ 
		 auxunit--;
		 P[i][auxindex] = 0;
        }
	 }
   
	while (auxunit<min) 
	 {
		auxindex = randomint(vars);
		if(P[i][auxindex]==0)
		{ 
		 auxunit++;
		 P[i][auxindex] = 1;
        }
	 }

 }
}

void Popul::SetGenePoolSize(int size)
{
  genepoollimit= size;  
}
void Popul::ProbInit()
{
  int i,j;

 for(i=0;i<psize;i++)
	 for(j=0;j<vars;j++)  BinConvert(i, vars, P[i]); 

 }


void Popul::TournSel(Popul *Opop, int tour)
{
 int j,k,pick,auxindex;
 float auxval;
 Tour  = tour;

 for(j=0;j<elit;j++) 
 { 
   for(k=j+1;k<psize;k++) 
  {
   if (Evaluations[index[k]]>Evaluations[index[j]]) 
	   {
		   auxindex = index[j];
		   index[j] = index[k];
		   index[k] =auxindex;
	   }
   }
    Opop->AssignChrom(j,this,index[j]);
	Opop->Evaluations[j] = Evaluations[index[j]];
 }

  for(j=elit;j<psize;j++) 
	{	
	 auxval = 0;
	 for(k=1;k<=Tour;k++) 
	 {
	   pick= randomint(psize);
	   if (Evaluations[index[pick]]>auxval) 
	   {
		   auxval =  Evaluations[index[pick]];
		   auxindex = pick;
	   }
     }
    Opop->AssignChrom(j,this,index[auxindex]);
  }
	
}


void Popul::SetElit(int To, Popul *Opop)
{
 int j;

 for(j=0;j<To;j++) 
 { 
    Opop->AssignChrom(j,this,index[j]);
	Opop->Evaluations[j] = Evaluations[index[j]];
 }
}


void Popul::TruncSel(Popul *Opop, int howmany)
{
 int j,k,auxindex;
 
 for(j=0;j<howmany;j++) 
 { 
   for(k=j+1;k<psize;k++) 
  {
   if (Evaluations[index[k]]>Evaluations[index[j]]) 
	   {
		   auxindex = index[j];
		   index[j] = index[k];
		   index[k] =auxindex;
	   }
   }
    Opop->AssignChrom(j,this,index[j]);
	Opop->Evaluations[j] = Evaluations[index[j]];
//	printf("%f \n ",Opop->Evaluations[j]);
	Opop->index[j] = j;
 }

  
}


void Popul::AssignChrom(int currpos,Popul* Opop,int otherpos)
{
 int j;
 for(j=0;j<vars;j++) P[currpos][j] = Opop->P[otherpos][j];
}

void Popul::InitIndex()
{
 int j;
 for(j=0;j<psize;j++) index[j] = j;
}

 void Popul::Evaluate(int poschrom, int func)
 {
   int i;
   float faux = 0;
   double *fun;

    switch( func  )
    { 
	  case 1: fun = cuban1_3; break;
	  case 2: fun = onemax3;  break;
	  case 3: fun = dmh3;  break;
	  case 4: fun = goldberg; break;
	  case 5: fun = cubanI; break;
	  case 6: fun = cubanII; break;
	}
   for(i=0; i<vars;i=+2) 
	    faux+=fun[4*P[poschrom][i]+2*P[poschrom][i+1]+P[poschrom][i+2]];
   Evaluations[poschrom] = faux;
   
     
 }


unsigned* Popul::Ind(int row)
{
	return P[row];
}

 void Popul::SetVal(int pos, double val)
 {
    Evaluations[pos] = val;
 }

 void Popul::EvaluateAll(int func)
 {
   int i,j;
   float faux;
   //double *fun;
/*
    switch( func  )
    { 
	  case 1: fun = cuban1_3; break;
	  case 2: fun = onemax3;  break;
	  case 3: fun = dmh3;  break;
	  case 4: fun = goldberg; break;
	  case 5: fun = cubanI; break;
	  case 6: fun = cubanII; break;
      case 7: fun = mia; break;
	}
*/
	for (j=0;j<psize;j++)
	{
     //faux = 0;
     //for(i=0; i<vars;i++)   faux+=P[j][i];
     //Evaluations[j] = faux;
	 // Para testores Evaluations[j] = evalua(j,this, func,params);//Evaluations[j] = faux; 

    }
	
   
 }
 
double Popul::Fitness(int individual) 
{
	return Evaluations[index[individual]];
}
 


Popul::~Popul()
{
 int i;
 
 for(i=0; i < psize; i++) delete[] P[i];
 delete[] P;
 delete[] index;
 delete[] Evaluations;
 delete[] dim;
}
