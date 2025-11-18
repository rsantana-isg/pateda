#include <math.h> 
#include <iostream> 
#include <fstream> 
#include "auxfunc.h" 
#include "AllFunctions.h" 
#include "Popul.h"  
 
using namespace std;

extern FILE* outfile; 
 
//Funciones 
// order 5 
 
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
 for(i=0; i < psize; i++) P[i] = new unsigned [cvars]; 
 for(i=0; i < vars; i++) dim[i] = 2; 
 index = new unsigned int[psize]; 
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
 index = new unsigned int[psize]; //Initiation for the CopyPop
 for(i=0; i < psize; i++)
  {
    P[i] = new unsigned[cvars];
    index[i] = psize+1;
  }
 Evaluations = new double[psize];
 for(i=0; i < vars; i++) dim[i] = dims[i]; 
} 
 
void Popul::RandInit() 
{ 
  int i,j; 
   
 
 for(i=0;i<psize;i++) 
   { 	
         for(j=0;j<vars;j++)  
	   {
	     P[i][j] = randomint(dim[j]); 
     	       
           }

   }   
  
} 
 
void Popul::RandInitIndiv(int pos) 
{ 
  int j; 
     
	 for(j=0;j<vars;j++)  
	 	 P[pos][j] = randomint(dim[j]); 
     	  
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
 
void Popul::Print() 
{ 
  int i,j; 
 
 for(i=0;i<psize;i++) 
   { 
    for(j=0;j<vars;j++)  printf("%d ",P[i][j]);      
    printf("  %e ",Evaluations[i]);
    printf("\n "); 
   } 
    
 } 

void Popul::Print(int from, int to) 
{ 
  int i,j; 
 
 for(i=from;i<to;i++) 
   { 
    for(j=0;j<vars;j++)  printf("%d ",P[i][j]);      
    printf("  %e ",Evaluations[i]);
    printf("\n "); 
   } 
    
 } 
void Popul::Print(int i) 
{ 
  int j; 
   for(j=0;j<vars;j++)  printf("%d ",P[i][j]);       
    printf("\n "); 
  
 } 

double Popul::FindBestVal()
{ int i;
 double best;
 best = Evaluations[0];
 for(i=0;i<psize;i++) 
 { 
    if(Evaluations[i]>best) best = Evaluations[i];
 }
 return best;
}

int Popul::FindBestIndPos()
{ int i,indpos;
 double best;
 best = Evaluations[0];
 indpos = 0;
 for(i=0;i<psize;i++) 
 { 
    if(Evaluations[i]>best) 
       {
         best = Evaluations[i];
         indpos = i;
       }
 }
 return indpos;
}

void Popul::TournSel(Popul *Opop, int tour) 
{ 
 int j,k,pick,auxindex; 
 float auxval; 
 Tour  = tour; 
 InitIndex();
 elit = 1; 
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
	Opop->index[j] = j; 
 } 
 
  for(j=elit;j<Opop->psize;j++)  
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
	Opop->Evaluations[j] = Evaluations[index[j]]; 
	Opop->index[j] = j; 
  } 
	 
} 


void Popul::CopyPop(Popul *Opop) 
{ 
 int i,j; 
 for(i=0;i<Opop->psize;i++) 
  { 
   for(j=0;j<vars;j++) 
   { 
    P[i][j] = Opop->P[i][j]; 
   } 
   Evaluations[i] =  Opop->Evaluations[i];
  }
}
 
int Popul::CompactPop(Popul *Opop, double* pvect ) 
{ 
 int i,j,k,first,next,last; 
 int  auxtest; 
 double sum;

 sum =0;
  
 for(i=0;i<psize;i++) 
  { 
   pvect[i] = 1; 
   for(j=0;j<vars;j++) 
   { 
    Opop->P[i][j] = P[i][j]; 
    Opop->Evaluations[i] = Evaluations[i];
   } 
  } 
 
 //Opop->Print(); 
 first = 0; 
 last = psize-1; 
  
 
 while(first<=last) 
   { 
     //pvect[first] = 1; 
     next = first+1; 
     while (next <= last) 
      { 
       i=0; 
       while(i<vars && Opop->P[first][i] == Opop->P[next][i]) i++; 
       if(i==vars) 
        { 
	  // cout<<"first"<<first<<" last "<<last<<" next "<<next<<"pvect[first] "<<pvect[first]<<endl;       
         Opop->Evaluations[next] = Opop->Evaluations[last]; 
         for(k=0;k<vars;k++) Opop->P[next][k] = Opop->P[last][k]; 
         pvect[first]++; 
         last = last-1; 
        } 
       else   next++; 
      }   
     //cout<<endl<<"first "<<first<<" last "<<last<<endl; 
     //Opop->Print(); 
     first++; 
      
    } 
 cout<<"psize is"<<psize; 
 
 
 auxtest = 0; 
 
 for(i=0;i<=last;i++) 
  { 
    auxtest += pvect[i]; 
    pvect[i] /= psize;  
    sum += pvect[i];
   cout<<i<<"  "<<pvect[i]<<" sum = "<<sum<<" Eval = "<<Opop->Evaluations[i]<<endl; 
    //cout<<pvect[i]<<" "; 
  } 
 
 if (auxtest != psize)  
        cout<<"ERROR WITH THE PROB VECTOR  "<<auxtest<<endl; 
 
 return last+1; 
  
} 
 

int Popul::CompactPopNew(Popul *Opop, double* pvect ) 
{ 
 int i,j,k,first,next,last; 
 double sum;

 sum = 0;
 for(i=0;i<psize;i++) 
 {
    for(j=0;j<vars;j++) 
           Opop->P[i][j] = P[i][j]; 
    Opop->Evaluations[i] = Evaluations[i];
    Opop->index[i] = i; 
 }

   
 first = 0;
 last = psize-1; 
 
 while(first<=last) 
   { 
     next = first+1; 
     while (next <= last) 
      { 
       i=0; 
       while(i<vars && Opop->P[first][i] == Opop->P[next][i]) i++; 
       if(i==vars) 
        { 
	 for(k=0;k<vars;k++) Opop->P[next][k] = Opop->P[last][k]; 
         pvect[first] += pvect[next];
         pvect[next] = pvect[last] ;
         Opop->Evaluations[next] = Opop->Evaluations[last];
         last = last-1; 
        } 
       else   next++; 
      }   
     first++;   
    } 
/*
 for(i=0;i<=last;i++)
  { 
      sum += pvect[i];
      cout<<i<<"  "<<pvect[i]<<" sum = "<<sum<<" Eval = "<<Opop->Evaluations[i]<<endl; 
  }
*/
 return last+1; 
  
} 

  
void Popul::BotzmannDist(double temp, double* pvect) 
{ 
  int i; 
  double Z,base; 
 
  Z=0;
  base = 2.7182818; 
  //base /=2.0; 
  //cout<<beta<<" beta "; 
   
 for(i=0;i<psize;i++)  
   {  
     pvect[i] = pow(base, temp*Evaluations[i]); 
     Z += pvect[i];  
     //cout<<pvect[i]<<" "; 
   } 
  for(i=0;i<psize;i++) 
    {
       pvect[i]/=Z; 
       // cout<<pvect[i]<<" ";
    }
} 
 
void Popul::ProporDist(double* pvect) 
{ 
  int i; 
  double Z,sum; 
 
  Z=0;  
  sum = 0;  
 for(i=0;i<psize;i++)  
   {  
     pvect[i] = Evaluations[i]; 
     Z += pvect[i];  
     //cout<<pvect[i]<<" "; 
   } 
  if(Z!=0) 
    for(i=0;i<psize;i++)
     {
       pvect[i]/=Z; 
       //sum += pvect[i];
       //cout<<i<<"  "<<pvect[i]<<" sum = "<<sum<<endl; 
     }
  else
   for(i=0;i<psize;i++) pvect[i]=1.0/psize;
} 
 
void Popul::UniformProb(int NPoints , double* pvect)
{ 
  int i;
  for(i=0;i<NPoints;i++)  pvect[i]= (1.0)/NPoints;
}

void Popul::SUSSel(int vectpoints,Popul* origpop,double* pvect)
{
    int i,indsel;
    double prob,tot,frac;
    i=0;
    prob = myrand();
    indsel = 0;
    tot = pvect[0];
    frac = (1.0/psize);

	while(prob>tot && i<vectpoints)
         {
           i++; 
           tot += pvect[i];
         }

        while(indsel<psize) 
	{
	    while(prob<=tot && indsel<psize)
            {
             AssignChrom(indsel,origpop,i);
             Evaluations[indsel] = origpop->Evaluations[i]; 
	     prob += frac; 
             indsel++;
            }
	    i++;
            if(i==vectpoints) i=0;
            tot += pvect[i];
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
 InitIndex();
 for(j=0;j<howmany;j++)  
 {  
   for(k=j+1;k<psize;k++)  
  { 
   if (Evaluations[index[k]]>=Evaluations[index[j]])  
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
 

void Popul::OrderPop() 
{ 
 int j,k,auxindex; 
 InitIndex();
 for(j=0;j<psize;j++)  
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
 } 
}


void Popul::Merge2Pops(Popul *pop1, int P1size, Popul *pop2, int P2size ) 
{ 
 int i,j,k,auxindex;
 j = 0; k = 0; i = 0;

 pop1->OrderPop();
 pop2->OrderPop();

while (i<psize && j< P1size && k< P2size) 
{
  while(i<psize &&  j< P1size &&  pop1->Evaluations[pop1->index[j]] >= pop2->Evaluations[pop2->index[k]])
   {
     this->AssignChrom(i,pop1,pop1->index[j]); 
     this->Evaluations[i] = pop1->Evaluations[pop1->index[j]]; 
     j++;
     i++;
   }
  while(i<psize && k< P2size && pop2->Evaluations[pop2->index[k]] >= pop1->Evaluations[pop1->index[j]])
   {
     this->AssignChrom(i,pop2,pop2->index[k]); 
     this->Evaluations[i] = pop2->Evaluations[pop2->index[k]]; 
     k++;
     i++;
   }
}
 while(i<psize && j< P1size)
   {
     this->AssignChrom(i,pop1,pop1->index[j]); 
     this->Evaluations[i] = pop1->Evaluations[pop1->index[j]]; 
     j++;
     i++;
   }

 while(i<psize && k< P2size)  
 {
     this->AssignChrom(i,pop2,pop2->index[k]); 
     this->Evaluations[i] = pop2->Evaluations[pop2->index[k]]; 
     k++;
     i++;
  }
} 

 
void Popul::AssignChrom(int currpos,Popul* Opop,int otherpos) 
{ 
 int j; 
 for(j=0;j<vars;j++)  
 { 
	 P[currpos][j] = Opop->P[otherpos][j]; 
	// index[currpos] = currpos; 
 } 
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

double Popul::GetVal(int pos) 
 { 
    return Evaluations[pos]; 
 } 
 
 void Popul::EvaluateAll(int func) 
 { 
   int j;   
//   int i; 
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
   meaneval = 0; 
	for (j=0;j<psize;j++) 
	{ 
     //faux = 0; 
     //for(i=0; i<vars;i++)   faux+=P[j][i]; 
     //Evaluations[j] = faux; 
	 Evaluations[j] = evalua(j,this, func,params); 
	 meaneval += Evaluations[j]; 
	} 
	meaneval /= psize; 
	    
 } 
  

void Popul::Merge(int elit, int toursize, Popul* other)
{
  int i,j, bestind;
  unsigned int* auxint;
  double auxval;

for(i=0; i < psize; i++)  index[i] = psize+1;
  for(i=elit; i < psize; i++)
   {
     bestind = other->FindBestClosestChrom(elit,toursize,P[i],Evaluations[i]);
     
     if(bestind > -1)
       {
         for(j=0; j < vars; j++) other->P[bestind][j] = P[i][j];
         other->Evaluations[bestind] = Evaluations[i];      
       }

   }

   for(i=0; i < elit; i++)
     {
      for(j=0; j < vars; j++) other->P[i][j] = P[i][j];
      other->Evaluations[i] = Evaluations[i];
     }
} 

int Popul::FindBestClosestChrom(int elit, int toursize, unsigned int* vect, double value)
{
  int i,a,j,eqcomp,maxncomp,bestcomp;
  
  maxncomp =-1;
  bestcomp = -1;
  
  //cout<<"initeval "<<value<<endl;
  // for(j=0; j < vars; j++) cout<<vect[j]<<" ";
  // cout<<endl;

  for(i=0; i < toursize; i++)
   {
    a =  randomint(psize);
    eqcomp = 0;
    for(j=0; j < vars; j++) eqcomp +=(vect[j]==P[a][j]);
   
   
    if ((eqcomp !=vars) && (eqcomp>maxncomp || (eqcomp==maxncomp &&Evaluations[a] < value )) )
      {
	//  cout<<"i "<<i<<" a "<<a<<" eval "<<Evaluations[a]<<endl;
	//for(j=0; j < vars; j++) cout<<P[a][j]<<" ";
	//cout<<endl;
       bestcomp = a;
       maxncomp = eqcomp;
      }    
   }
  //cout<< " a "<<a<<" value  "<<value<<" maxncomp    "<<maxncomp<<endl;

    if((bestcomp >-1) && (Evaluations[bestcomp]<=value)) return bestcomp;
    return -1;
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










