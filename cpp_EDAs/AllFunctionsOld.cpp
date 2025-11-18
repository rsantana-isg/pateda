#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>  
#include <memory.h> 
#include <iostream.h> 
#include <fstream.h> 
#include "AllFunctions.h" 
 
 
  
int *Funpars;  
Popul *p;  
  
 
extern void i_user_defined(int); 
extern void e_user_defined(int *, double, int); 
extern int s_user_defined(double, double, int,  Popul *); 
extern double user_defined_function(int *, int ); 
 
//******************************************************** 
  
//-------------------------------------------------------- 
void  default_end(int *buff, double Fv, int n) 
{ 
  printf("\n\nBest:\n"); 
  for(int i=0; i < n; i++) printf("%d", buff[i]); 
  printf("\nFvalue: %f\n", Fv); 
} 
 
//-------------------------------------------------------- 
 
// ================================================================================ 
// 
// name:          trap5 
// 
// function:      computes the value of trap-5 function 
// 
// parameters:    x............a binary string 
//                n............the length of a string 
// 
// returns:       (float) the value of a trap5 function 
// 
// ================================================================================ 
 
double trap5(int *x, int n) 
{ 
  int s; 
  int u; 
  register int i; 
 
  s = 0; 
 
  for (i=0; i<n; ) 
    { 
      u  = x[i++]; 
      u += x[i++]; 
      u += x[i++]; 
      u += x[i++]; 
      u += x[i++]; 
       
      if (u<5) 
	s += 4-u; 
      else 
	s += 5; 
    }; 
 
   return (double) s; 
} 
 
int s_trap5(double BestFv, double AvgFv, int StopGen,Popul *p) 
{ 
  if( BestFv == p->vars ) return 1; 
  return 0; 
} 
 
 
//-------------------------------------------------------- 
// Funcion FourPeaks 
double T_FourPeaks; // parameter of the function 
double FourPeaks(int *buff,int n) 
{ 
  int *p; 
  int head_1, tail_0, min, max; 
   
  T_FourPeaks = Funpars[0]; 
 
  p=buff; 
  head_1=0; 
  while(*p) 
    { 
      head_1 ++; 
      p++; 
    } 
  p=buff+n-1; 
  tail_0=0; 
  while(*p==0) 
    { 
      tail_0++; 
      p--; 
    } 
  if(head_1>tail_0) 
    { 
      min=tail_0; max=head_1; 
    } 
  else 
    { 
      min=head_1; max=tail_0; 
    } 
  if(min>T_FourPeaks) 
    return (100+max); 
  else 
    return (max); 
} 
 
//------------------------------------------------------ 
// Funcion SixPeaks 
double T_SixPeaks; // parameter of the function 
double SixPeaks(int *buff,int n) 
{ 
 
  int head_1, head_0, tail_0, tail_1, min1, max1, min2, max2, i, max; 
  double res; 
  //T_SixPeaks = Funpars[0]; 
  T_SixPeaks = n/2 - 1; 
  
  head_1=0; 
  i = 0; 
  while(i<n  && buff[i]==1) 
    { 
      head_1 ++; 
      i++; 
    } 
 
  head_0=0; 
  i = 0; 
 while(i<n  && buff[i]==0) 
    { 
      head_0 ++; 
      i++; 
    } 
 
  tail_0=0; 
  i = n-1; 
 while(i>=0 && buff[i]==0) 
    { 
      tail_0 ++; 
      i--; 
    } 
 
  tail_1=0; 
 
  i = n-1; 
 while(i>=0 && buff[i]==1 ) 
    { 
      tail_1 ++; 
      i--; 
    } 
 
  max1 = (head_0>head_1)?head_0:head_1; 
  max2 = (tail_0>tail_1)?tail_0:tail_1; 
  max = (max1>max2)?max1:max2; 
 
  if( ((tail_0 > T_SixPeaks) && (head_1 > T_SixPeaks))   || ((tail_1 > T_SixPeaks) && (head_0 > T_SixPeaks)) ) res = (n+max); 
  else    res = (max); 
 
  if (res>54) { 
 for(i=0; i < n; i++) cout<<buff[i]<<" "; 
  cout<<endl; 
  } 
  return res; 
 
} 
//------------------------------------------------------ 
int *grid; 
double Checkerboard(int *buff, int n) 
{ 
  int i, j, sum, temp; 
   
  sum = 0;  
   
  int N = (int) Funpars[0];  
   
 
  for(i=0; i < N; i++) 
    for(j=0; j < N; j++) 
      grid[N*i+j] = *(buff++); 
  for(i=1;i<(N-1);i++) 
    for(j=1;j<(N-1);j++) 
      { 
	temp=grid[N*i+(j-1)]+grid[N*(i-1)+j]+grid[N*i+j+1]+grid[N*(i+1)+j]; 
	if(grid[N*i+j]) temp = 4 - temp; 
	sum+=temp; 
      }  
 
  return( (double) sum); 
} 
 
//------------------------------------------------------ 
 
double IntCheckerboard(int *buff, int n) 
{ 
  int i, j, sum, temp; 
   
  sum = 0;  
   
  int N = (int) Funpars[0];  
   
   
  for(i=1;i<(N-1);i++) 
    for(j=1;j<(N-1);j++) 
      { 
	  temp=(buff[N*i+(j-1)]==buff[N*i+j]); 
          temp += (buff[N*(i-1)+j]==buff[N*i+j]); 
          temp += (buff[N*i+j+1]==buff[N*i+j]); 
          temp += (buff[N*(i+1)+j]==buff[N*i+j]); 
	  sum+= (4-temp); 
      }  
 
  return( (double) sum); 
} 
 
 
 
int s_Checkerboard(double BestFv, double AvgFv, int StopGen,Popul *p) 
{ 
  int N = (int)  Funpars[0]; 
  if(BestFv == (double)(N-2)*(N-2)*4) return 1; 
  return 0; 
} 
 
void i_Checkerboard(int n) 
{ 
  grid = new int[ Funpars[0]* Funpars[1]]; 
} 
 
void e_Checkerboard(int *buff, double Fv, int n) 
{ 
/*  int par1 = int(Funpars[0]); 
  printf("\nBest:\n"); 
  for(int i=0; i < par1; i++) 
    { 
      for(int j=0; j < par1; j++) 
	printf("%d", *buff++); 
      printf("\n"); 
    } 
  printf("Fvalue: %f\n",Fv);*/ 
  delete[] grid;  
   
} 
 
//------------------------------------------------------ 
 
 
 
//.............................. 
double EqualProducts(int *buff, int n) 
{ 
  int i; 
  double p1,p2; 
  p1 = 1.0; 
  p2 = 1.0;  
  for(i=0; i < n; i++) 
    if(buff[i]) 
      p1*=EqualProductsNumbers[i]; 
    else 
      p2*=EqualProductsNumbers[i]; 
  //cout<<-1.0*fabs(p1-p2)<<endl; 
  return(-1.0*fabs(p1-p2)); 
} 
 
//----------------------------------------------------- 
 
double OneMax(int *buff, int n) 
{ 
  int i, sum; 
  sum = 0; 
  for(i = 0; i < n; i++) sum += buff[i]; 
  return sum; 
} 
 
int s_OneMax(double BestFv, double AvgFv, int StopGen,Popul *p) 
{ 
  if( BestFv == p->vars ) return 1; 
  return 0; 
} 
 
double NOneMax(int *buff, int n) 
{ 
  int i, sum; 
  sum = 0; 
  for(i = 0; i < n; i++) sum += buff[i]; 
  return (n-sum); 
} 
 
int s_NOneMax(double BestFv, double AvgFv, int StopGen,Popul *p) 
{ 
  if( BestFv ==  p->vars ) return 1; 
  return 0; 
} 
 
 
//----------------------------------------------------- 
int K_BigJump = 1, M_BigJump = 3; 
 
double BigJump(int *buff, int n) 
{ int Nvars = p->vars; 
 
  K_BigJump = int(Funpars[0]); 
  M_BigJump = int(Funpars[1]); 
 
  double r = OneMax(buff, n); 
  if( r == Nvars) return(K_BigJump * Nvars); 
  if( r > (Nvars - M_BigJump) ) return 0; 
  return r; 
} 
 
int s_BigJump(double BestFv, double AvgFv, int StopGen, Popul *p ) 
{ int Nvars = p->vars; 
  if( BestFv == Nvars * K_BigJump ) return 1;   
  return 0; 
} 






void  OrderErrorCodes(int *buff, int n, int  worldlength, int numberwords) 
{ 
 
   int i,j,k,pos_i,pos_j,mindist,auxint,lessthan;
   int *orderedcodes,*auxchrom ;

  orderedcodes = new int[numberwords];
  for(i=0;i<numberwords;i++) orderedcodes[i] = i;
  
 for(i=0;i<numberwords-1;i++) 
   {
     for(j=i+1;j<numberwords;j++) 
     {
      pos_i = orderedcodes[i]*worldlength;
      pos_j = orderedcodes[j]*worldlength;
      k = 0;
      while(k<worldlength && (buff[pos_i+k] == buff[pos_j+k])) k++;
    
      if(k<worldlength && (buff[pos_i+k] > buff[pos_j+k])) 
	{ auxint = orderedcodes[i];
	  orderedcodes[i] = orderedcodes[j];
          orderedcodes[j] = auxint;
        }
      }
   } 
 /*
for(i=0;i<numberwords;i++)
    {
     for(k=0;k<worldlength;k++)   cout<<buff[i*worldlength+k]<<" ";
     cout<<endl;
    }
 */
  auxchrom = new int[n];
  for(i=0;i<numberwords;i++)
    {
     for(k=0;k<worldlength;k++)  
        auxchrom[i*worldlength+k] = buff[orderedcodes[i]*worldlength+k];
     //cout<<orderedcodes[i]<<" ";
    }
  //cout<<endl;

  for(i=0;i<n;i++) buff[i] =auxchrom[i];
 

  delete[] auxchrom;
  delete[] orderedcodes;
  
} 



 //----------------------------------------------------- 
double ErrorCodes(int *buff, int n) 
 { 
  int  worldlength, numberwords,i,j,k,pos_i,pos_j,dist_ij,mindist;
  double fitness;
  worldlength  = int(Funpars[0]); 
  numberwords = int(Funpars[1]);
  fitness = 0; 
  mindist = worldlength;
  for(i=0;i<numberwords-1;i++) 
   {
     pos_i = i*worldlength;
     for(j=i+1;j<numberwords;j++) 
     {
      pos_j = j*worldlength;
      dist_ij = 0;
      for(k=0;k<worldlength;k++) dist_ij  += (buff[pos_i+k] != buff[pos_j+k]);
       if(dist_ij>0) 
       {
        fitness += 1.0/(dist_ij*dist_ij);
        if (dist_ij < mindist) mindist = dist_ij;
       }  
     }
   } 
  // OrderErrorCodes(buff,n,worldlength,numberwords); 
  // cout<<fitness<<endl;
  /* for(i=0;i<numberwords;i++)
    {
     for(k=0;k<worldlength;k++)   cout<<buff[i*worldlength+k]<<" ";
     cout<<endl;
     }*/
 if (fitness>0) fitness = (1.0/fitness) + (mindist/12.0 - (mindist*mindist)/4.0 + (mindist*mindist*mindist)/6.0);
  return fitness*1000;
  //return (mindist);
 }





 
//------------------------------------------------------------ 
int Kdec = 4; 
double decepK(int *buff) 
{   
  double r = OneMax(buff, Kdec); 
  if( r == Kdec )  return( Kdec ); 
  else  return(Kdec - 1 - r); 
      /* 
    if(Kdec == 3) return 1; else return( Kdec ); 
  else 
    if(Kdec == 3){  
	if(r==0) return(0.9); 
	if(r==1) return(0.8); 
	return(0); 
    } else  return(Kdec - 1 - r); 
      */ 
} 
 
double KDeceptive(int *buff, int n){ 
double sum = 0; 
Kdec = int(Funpars[0]); 
 for(int i = 0; i < n; i += Kdec ) 
  sum += decepK(buff + i); 
return sum; 
} 
 
int s_KDeceptive(double BestFv, double AvgFv, int StopGen, Popul *p) 
{ 
 
  if(Kdec == 3) { 
      if( BestFv == p->vars/3.0 ) return 1; 
  }else{ 
      if( BestFv == p->vars ) return 1; 
  } 
  return 0; 
} 
 








//------------------------------------------------------------ 
// Generalized Deceptive function for integer values 

double GenKDecep(int *buff, int n)
{
double sum;
int i,j,a,s,Card;


Kdec = int(Funpars[0]); 
Card = int(Funpars[1]); 

sum = 0;
a= 0;
for(i=0; i < n/Kdec; i ++) 
{   
	s = 0;
	for(j=0; j < Kdec; j ++)  s += buff[a++];
	if( (Card-1)*Kdec == s) sum += (Card-1)*Kdec;
        else if(s==0) sum += ((Card-1)*Kdec - 1);
	else sum += ((Card-1)*Kdec - s - 1);
}
return sum;
}



//------------------------------------------------------------ 
int f1[] = {0,0,0,0,1}; 
int f2[] = {0,0,0,1,1}; 
int f3[] = {0,0,1,1,1}; 
int f4[] = {1,1,1,1,1}; 
int f5[] = {0,0,0,0,0}; 
 
double F5Muhl(int* buff) 
{ 
  if ( memcmp(buff, f1, 5*sizeof(int)) == 0 ) 
       return 3.0; 
  else 
    if ( memcmp(buff, f2, 5*sizeof(int)) == 0 ) 
      return 2.0; 
  else 
    if ( memcmp(buff, f3, 5*sizeof(int)) == 0 ) 
      return 1.0; 
  else 
    if ( memcmp(buff, f4, 5*sizeof(int)) == 0 ) 
      return 3.5; 
  else 
    if ( memcmp(buff, f5, 5*sizeof(int)) == 0 ) 
      return 4.0; 
  else 
    return 0; 
} 
 
double Fc2(int* buff, int n) 
{ 
  int m = n/5; 
  double sum = 0; 
  for (int i=0; i < m; i++) 
    sum += F5Muhl(&buff[5*i]); 
  return sum; 
} 
/* 
#define SIZE2 20 
int Fc2Optimo[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; 
int s_Fc2(double BestFv, double AvgFv, int StopGen,pop *Pop) 
{ 
  if( memcmp(Pop->indv(Pop->who()), Fc2Optimo, SIZE2*sizeof(int)) == 0 )  
    return 1;   
  return 0; 
} 
*/ 
//--------------------------------------------------------- 
 
double g(int* buff) 
{ 
  if ( int(OneMax(buff, 5)) % 2 ) 
    return 1; 
  return 0; 
}  
 
double F5Multimodal(int* buff) 
{ 
  return OneMax(buff, 5) + 2*g(buff); 
} 
 
double Fc3(int* buff, int n) 
{ 
  int m = n/5; 
  double sum = 0; 
  for (int i=0; i < m; i++) 
    sum += F5Multimodal(&buff[i*5]); 
  return sum; 
} 
/* 
#define SIZE3 20 
int Fc3Optimo[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}; 
int s_Fc3(double BestFv, double AvgFv, int StopGen,pop *Pop) 
{ 
  if( memcmp(Pop->indv(Pop->who()), Fc3Optimo, SIZE3*sizeof(int)) == 0 )  
      return 1;   
  return 0; 
} 
*/ 
//----------------------------------------------------------- 
double F3Cuban1[] = {0.595, 0.2, 0.595, 0.1, 1.0, 0.05, 0.09, 0.15}; 
 
double F5Cuban1( int* buff ) 
{ 
  if ( (buff[1] == buff[3]) && (buff[2] == buff[4]) ) 
    return 4*F3Cuban1[ 4*buff[0] + 2*buff[1] + buff[2] ]; 
  else 
    return 0; 
} 
 
double F5Cuban2(int* buff) 
{ 
  if ( buff[4] == 0 ) 
    return OneMax(buff, 5); 
  else 
    if ( (buff[0] == 0) ) 
      return 0; 
    else 
      return OneMax(buff, 5) - 2; 
} 
 
// Pongo el mismo nombre que tiene Larranaga. 
 
//----------------------------------------------------------- 
double Fc4(int* buff, int n) 
{ 
  int m = (n - 1)/4; 
  double sum = 0; 
  for (int i=0; i < m; i++) 
    sum +=  F5Cuban1(&buff[4*i]); 
  return sum; 
} 
/* 
#define SIZE4 21 
int Fc4Optimo[SIZE4] = {1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0}; 
 
int s_Fc4(double BestFv, double AvgFv, int StopGen,pop *Pop) 
{ 
  if( memcmp(Pop->indv(Pop->who()), Fc4Optimo, SIZE4*sizeof(int)) == 0 )  
      return 1;   
  return 0; 
} 
*/ 
//------------------------------------------------------------ 
double Fc5(int* buff, int n) 
{ 
  int m = (n - 5)/8; 
  double sum = F5Cuban1(buff); 
  for (int i=0; i < m; i++) 
    sum += F5Cuban2(&buff[4*(2*i+1)]) + F5Cuban1(&buff[4*(2*i+2)]); 
  return sum; 
}  
 
#define SIZE5 21 
int Fc5Optimo[SIZE5] = {1,0,0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0,0,0,0}; 
 
int s_Fc5(double BestFv, double AvgFv, int StopGen,Popul *p) 
{int n=p->vars; 
/*  if( memcmp(Pop->indv(Pop->who()), Fc5Optimo, SIZE5*sizeof(int)) == 0 )  
      return 1;   
  return 0; 
*/ 
 double opitmo; 
switch(n){ 
    case 21: opitmo = 14.8; break; 
    case 37: opitmo = 25.6; break; 
    case 69: opitmo = 47.2; break; 
    case 101: opitmo = 68.8; break; 
    case 133: opitmo = 90.4; break; 
    case 205: opitmo = 138.98;break; 
    default: 
	printf("eeror en FC5\n"); 
} 
if(opitmo==BestFv) return 1;   
  return 0; 
} 
 
 
//---------------------------------------------------- 
double decep3[]={0.9, 0.8, 0.8, 0, 0.8, 0, 0, 1}; 
double Decep3(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n; i += 3) 
       sum += decep3[ buff[i] + 2 * buff[i + 1] + 4 * buff[i + 2] ]; 
  //sum += decep3[ (1-buff[i]) + 2 * (1-buff[i + 1]) + 4 * (1-buff[i + 2]) ]; 
  return sum; 
} 
 //---------------------------------------------------- 
double decepMart[]={0.553827, 0.049179, 0.856078, 1.09925, 0.980221, -0.298355, 0.370961, -0.192739}; 
double DecepMarta3(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n; i += 3) 
    sum += decepMart[ 4*buff[i] + 2 * buff[i + 1] +  buff[i + 2] ]; 
  return sum; 
} 
//double decepMart3[]={0.9, 1.1, 1.1, 0.7, 1.1, 0.7, 0.7, 0.7}; 
double decepMart3[]={1.5, 0.9,0.9,0.9,0.9,0.9,0.9, 1.5}; 
double DecepMarta3New(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n; i += 3) 
    sum += decepMart3[ 4*buff[i] + 2 * buff[i + 1] +  buff[i + 2] ]; 
  return sum; 
} 

double FirstPolytree5[]={-1.141,1.334,-5.353,-1.700,0.063,-0.815,-0.952,-0.652,0.753,1.723,-4.964,-1.311,1.454,0.576,0.439,0.739,-3.527,1.051,-7.738,-4.085,1.002,0.124,-0.013,0.286,-6.664,-4.189,-10.876,-7.223,-1.133,-2.011,-2.148,-1.849}; 
 
double FirstPolytree5Ochoa(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n; i += 5) 
    sum += FirstPolytree5[ buff[i] + 2 * buff[i + 1] +  4*buff[i + 2]+ 8 * buff[i + 3] +  16*buff[i + 4]  ]; 
  return sum; 
} 
    
double FirstPolytree3[]={1.042,-0.736,0.357,-1.421,-0.083,0.092,-0.768,-0.592}; 
 
double FirstPolytree3Ochoa(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n; i += 3) 
    sum += FirstPolytree3[ buff[i] + 2 * buff[i + 1] +  4*buff[i + 2]]; 
  return sum; 
} 

 double FirstPolytree3OchoaSolap(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n-2; i += 2) 
  {
      //   cout<<i<<" "<<i+1<<" "<<i+2<<endl;
    sum += FirstPolytree3[ buff[i] + 2 * buff[i + 1] +  4*buff[i + 2]]; 
  }
  return sum; 
} 


// Roberto (Esta funcion toma valores que dependen  
// principalmente de la primera variable  
  
//---------------------------------------------------- 
double decep3_Mh[]={2, 1, 1, 0, 1, 0, 0, 3}; 
double Decep3_Mh(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n; i += 3) 
    sum += decep3_Mh[ buff[i] + 2 * buff[i + 1] + 4 * buff[i + 2] ]; 
  return sum; 
} 
  
 
// Roberto (Esta funcion toma valores que dependen  
// principalmente de la primera variable  
  
double TwoPeaksDecep3(int *buff, int n)  
{  
  double sum = 0;  
  if (buff[0]==0)  
  {  
   for(int i = 1; i < n; i += 3)  
    sum += decep3[ buff[i] + 2 * buff[i + 1] + 4 * buff[i + 2] ];  
  }  
  else   
  {  
   for(int i = 1; i < n; i += 3)  
    //sum += decep3[ (1-buff[i]) + 2 * (1-buff[i + 1]) + 4 * (1-buff[i + 2]) ];  
    sum += 1 - decep3[ (buff[i]) + 2 * (buff[i + 1]) + 4 * (buff[i + 2]) ];  
  }  
return sum;  
} 
int s_Decep3(double BestFv, double AvgFv, int StopGen,Popul *p){ 
  /*int flag = 1; 
    for(int i=0; i < Nvars; i++) 
    { 
    if( Pop->freq(i, 1) < 0.95 ) flag = 0; 
    printf("%f ",Pop->freq(i,1)); 
    } 
    printf("\n"); 
    if( flag ) return 1; 
  */ 
 
   if( BestFv == (p->vars/3.0) ) return 1; 
   return 0; 
} 
//----------------------------------------------------- 
// IsoTorus 
int m; 
double IsoT1(int *buff){ 
  double r = OneMax(buff, 5); 
 
  if( r ==  0) return m; 
  if( r ==  5) return(m-1); 
  return 0; 
} 
double IsoT2(int *buff){ 
  double r = OneMax(buff, 5); 
 
  if( r ==  5) return  (m*m); 
  return 0; 
} 
 
int s_IsoTorus(double BestFv, double AvgFv, int StopGen,Popul *p){ 
   if( BestFv == (m*m*m-m+1) ) return 1; 
   return 0; 
} 
 
double IsoTorus(int *buff, int n){ 
m = int(sqrt(n)); 
 
int cinco[5], xup, xdown, xleft, xright; 
cinco[0] =  buff[n-m]; 
cinco[1] =  buff[m-1]; 
cinco[2] =  buff[0]; 
cinco[3] =  buff[1]; 
cinco[4] =  buff[m]; 
double sum = IsoT2(cinco); 
 
for(int i = 0; i < m; i++) 
    for(int j = 0; j < m; j++){ 
       if(!(i+j)) continue; 
       xup   = i - 1; 
       xdown = i + 1; 
       xleft = j - 1; 
       xright= j + 1; 
       if(i==0)     xup    = m-1; 
       if(i==(m-1)) xdown  = 0;  
       if(j==0)     xleft  = m-1; 
       if(j==(m-1)) xright = 0;  
       cinco[0] = buff[xup*m+j]; 
       cinco[1] = buff[i*m+xleft]; 
       cinco[2] = buff[i*m+j]; 
       cinco[3] = buff[i*m+xright]; 
       cinco[4] = buff[xdown*m+j]; 
       sum += IsoT1(cinco); 
    } 
return sum; 
} 
 
double PartialCheckerBoard(int *buff, int n){ 
m = int(sqrt(n)); 
double temp,sum; 
int xup, xdown, xleft, xright; 
 sum = 0; 
for(int i = 0; i < m; i++) 
    for(int j = 0; j < m; j++){ 
       xup   = i - 1; 
       xdown = i + 1; 
       xleft = j - 1; 
       xright= j + 1; 
       if(i==0)     xup    = m-1; 
       if(i==(m-1)) xdown  = 0;  
       if(j==0)     xleft  = m-1; 
       if(j==(m-1)) xright = 0;  
       temp=buff[xup*m+j]+buff[i*m+xleft]+buff[i*m+xright]+buff[xdown*m+j]; 
	if(buff[i*m+j]) 
          { 
            if (temp == 0)  temp = 4; 
             
          } 
        else 
          { 
            if (temp !=4) temp = 4 - temp; 
          } 
	sum+=temp; 
    } 
return sum; 
} 
 
 
double FullCheckerBoard(int *buff, int n){ 
m = int(sqrt(n)); 
double temp,sum; 
int xup, xdown, xleft, xright; 
 sum = 0; 
for(int i = 0; i < m; i++) 
    for(int j = 0; j < m; j++){ 
       xup   = i - 1; 
       xdown = i + 1; 
       xleft = j - 1; 
       xright= j + 1; 
       if(i==0)     xup    = m-1; 
       if(i==(m-1)) xdown  = 0;  
       if(j==0)     xleft  = m-1; 
       if(j==(m-1)) xright = 0;  
       temp=buff[xup*m+j]+buff[i*m+xleft]+buff[i*m+xright]+buff[xdown*m+j]; 
	if(buff[i*m+j]) temp = 4 - temp; 
	sum+=temp; 
    } 
return sum; 
} 
 
 
double IntFullCheckerBoard(int *buff, int n){ 
m = int(sqrt(n)); 
double temp,sum; 
int xup, xdown, xleft, xright; 
 sum = 0; 
for(int i = 0; i < m; i++) 
    for(int j = 0; j < m; j++){ 
       xup   = i - 1; 
       xdown = i + 1; 
       xleft = j - 1; 
       xright= j + 1; 
       if(i==0)     xup    = m-1; 
       if(i==(m-1)) xdown  = 0;  
       if(j==0)     xleft  = m-1; 
       if(j==(m-1)) xright = 0;  
       temp=(buff[xup*m+j]==buff[i*m+j]) +(buff[i*m+xleft]==buff[i*m+j])+(buff[i*m+xright]==buff[i*m+j])+(buff[xdown*m+j]==buff[i*m+j]); 
	sum+=temp; 
    } 
return sum; 
} 
 
 
 
 
 
 
 
 
//----------------------------------------------------- 
// IsoChain 
int l; 
double Iso1(int *buff){ 
  double r = OneMax(buff, 3); 
 
  if( r ==  0) return l; 
  if( r ==  3) return(l-1); 
  return 0; 
} 
double Iso2(int *buff){ 
  double r = OneMax(buff, 3); 
 
  if( r ==  3) return(l); 
  return 0; 
} 
 
 
double IsoChain(int *buff, int n){ 
double sum = 0; 
l = int(Funpars[0]); 
 
int s = 0; 
for(int i = 0; i < (l-1); i++,s+=2) 
  sum += Iso1(buff + s); 
sum += Iso2(buff + s);   
 
return sum; 
} 
  
 
int s_IsoChain(double BestFv, double AvgFv, int StopGen,Popul *p){ 
   if( BestFv == (l*(l-1)+1) ) return 1; 
   return 0; 
} 
//------------------------------------------------  
  
double S_Peak(int *buff, int n){  
double sum = 0;  
int m,k,i;  
  
k = int(Funpars[0]);  
m = int(Funpars[1]);  
  
sum = OneMax(buff,n);  
int s = 1;  
for(i = 0; i < m; i++)  
  s *=  (1-buff[i]);  
for(i = m; i < n; i++)  
  s  *=  buff[i];  
return sum+s*k*(m+1);  
}   
 
 
 
//----------------------------------------------------- 
double posFun(int *buff, int n) 
{ 
  int i, sum; 
  sum=0; 
  for(i = 0; i < n; i++) sum += (buff[i]==1)*(i+1); 
  return sum; 
} 
 
 
//----------------------------------------------------- 
double IsoPeak(int *buff, int n) 
{ 
  int i, sum, l; 
  l = n-1; 
  sum = ((buff[0]+buff[1])==2)*(l); 
   
  for(i = 1; i < l; i+=1) sum +=  ((buff[i]==buff[i+1]))*(l-buff[i]); 
  return sum; 
} 
 
 
 
int s_PosFun(double BestFv, double AvgFv, int StopGen,Popul *p) 
{ 
  if(BestFv == p->vars*(p->vars+1)/2.0) return 1; 
  return 0; 
} 
 
//----------------------------------------------------- 
double posFun_to_2(int *buff, int n) 
{ 
  int i, sum; 
  sum = 0; 
  for(i = 0; i < n; i++) sum += (buff[i])? (i+1)*(i+1): 0; 
  return sum; 
} 
 
int s_PosFun_to_2(double BestFv, double AvgFv, int StopGen,Popul *p) 
{ 
  // OJO  if(BestFv == Nvars*(Nvars+1)/2.0) return 1; 
  return 0; 
} 
 
 
//---------------------------------------------------------- 
 
//Function block is a symmetric function with n optima 
 
double BlockFunc(int* buff, int n) 
{ 
  int i,tot,aux,min,pos,Block; 
  double resp; 
   
  
  Block  = Funpars[0];   
  resp = 0;    
  i=1; aux = buff[0]; pos=0; tot = buff[0]; 
  min = (n-Block>=Block)?n:0; 
 
  while( i<n ) 
    { 
     if(buff[i]==1 && buff[i]==buff[i-1]) aux++; 
     else if(buff[i]==1)  
      { 
        aux=1;   
        pos=i; 
      } 
     else if(buff[i]==0 && buff[i-1]==1)  
      { 
        if (( abs(Block-aux)<abs(Block-min)) && !(pos==0 && buff[n-1]==1)) 
        min = aux; 
      } 
     tot += buff[i]; 
     i++; 
    } 
 
 
if (buff[n-1]==1) 
  { 
    if(buff[0]==1 ) // This considers the case of the ring 
      { 
       i=0; 
       while(i<n && buff[i]==1 ) i++; 
       if (i<n) aux+= i;         
      } 
    if (( abs(Block-aux)<abs(Block-min)) && !(pos==0 && buff[n-1]==1)) min =aux; 
  } 
  
  
  if (tot==0) return resp; 
  resp += (2*n -abs(Block-min) -abs(tot-min)); 
  //cout<<"Block= "<<Block<<" Min= "<<min<<"  Tot="<<tot<<" Resp= "<<resp<<endl; 
  return resp; 
 
} 
 
 
//------------------------------------------------------------ 
 
 
double 
symmetryfunc(int* buff, int n) 
{ 
 int u; 
  u = int(OneMax(buff,n)); 
  if(2*u < n) 
    u= n-u; 
  return u; 
} 
 
 
 
//------------------------------------------------------------ 
 
 
double 
fhtrap1(int* buff, int n) 
{ 
  int j,k,l,auxn,nv; 
  int auxbuff[729]; 
  int powers[8] = {1,3,9,27,81,243,729};   
  double sum; 
 
     
  auxn = n; 
  sum = 0; 
  j= 1; 
  while(auxn>=3) 
   { 
     l = 0; 
     //cout<<auxn<<"  "<<sum<<endl; 
     if (auxn == n) 
       { 
          for(k=0;k<n;k+=3) 
            {  
              nv =(buff[k] + buff[k+1] + buff[k+2]); 
	      sum += (nv==3 || nv==0)*powers[j]; 
              if (nv==3) auxbuff[l++] = 1; 
              else if(nv==0) auxbuff[l++] = 0; 
              else auxbuff[l++] = -10; 
            }  
	  auxn = n/3;      
       } 
     else if(auxn==3) 
       { 
         k=0; 
         nv =(auxbuff[k] + auxbuff[k+1] + auxbuff[k+2]); 
	 if(nv==3) sum += powers[j]; 
         else if(nv==0) sum += (powers[j]); 
         auxn /= 3;          
       } 
      else  
       { 
          for(k=0;k<auxn;k+=3) 
            {  
              nv =(auxbuff[k] + auxbuff[k+1] + auxbuff[k+2]); 
              sum += (nv==3 || nv==0)*powers[j]; 
              if (nv==3) auxbuff[l++] = 1; 
              else if(nv==0) auxbuff[l++] = 0; 
              else auxbuff[l++] = -10;	      
            }    
           auxn /= 3;          
       } 
     j ++;     
   }  
  
  return sum; 
} 
 
 
 
 
//------------------------------------------------------------ 
 
 
 
double 
HIFF(int* buff, int n) 
{ 
  int j,k,l,auxn,nv; 
  int auxbuff[1024]; 
  int powers[15] = {1,2,4,8,16,32,64,128,256,512,1024};   
  double sum; 
 
     
  auxn = n; 
  sum = n; 
  j= 1; 
  while(auxn>=2) 
   { 
     l = 0; 
     //cout<<auxn<<"  "<<sum<<endl; 
     if (auxn == n) 
       { 
          for(k=0;k<n;k+=2) 
            {  
              nv =(buff[k] + buff[k+1]); 
              sum += (nv==2 || nv==0)*powers[j]; 
              if (nv==2) auxbuff[l++] = 1; 
              else if(nv==0) auxbuff[l++] = 0; 
              else auxbuff[l++] = -10; 
            }  
	  auxn = n/2;      
       } 
     else if(auxn==2) 
       { 
         k=0; 
         nv =(auxbuff[k] + auxbuff[k+1]); 
	 if(nv==2) sum += powers[j]; 
         else if(nv==0) sum += (powers[j]); 
         auxn /= 2;          
       } 
      else  
       { 
          for(k=0;k<auxn;k+=2) 
            {  
              nv =(auxbuff[k] + auxbuff[k+1]); 
              sum += (nv==2 || nv==0)*powers[j]; 
              if (nv==2) auxbuff[l++] = 1; 
              else if(nv==0) auxbuff[l++] = 0; 
              else auxbuff[l++] = -10;	      
            }    
           auxn /= 2;          
       } 
     j ++;     
   }  
  
  return sum; 
} 
 
//------------------------------------------------------------ 
 
 
double Automata(int* buff, int n) 
{    int i,j,state; 
     double value;  
     int chain[200] = {0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1};  
     int allstates[200]; 
     int allpos[200]; 
     int chainlength = 64;  
     int orderstates; 
     int* auxstring; 
     unsigned int vect[4]; 
     int auxint=0; 
     int l; 
 
     l=params[0]; 
     chainlength = (n/2)*(l+1); 
    for(i=0;i<n;i+=4)  
    { 
 
	BinConvert(buff[i],l,vect); 
        for(j=0;j<l;j++) chain[auxint++] = vect[j];  
        chain[auxint++] = buff[i+1]; 
	BinConvert(buff[i+2],l,vect); 
        for(j=0;j<l;j++) chain[auxint++] = vect[j];   
        chain[auxint++] = buff[i+3];     
    } 
  
     auxstring = new int[n]; 
 
     for(i=0;i<200;i++)  allstates[i] = -1; 
 
	//Number of digits that were predicted by the automata in the correct way 
       	i=0; 
        state = 0;  
	value = 0;  
        orderstates = 0;  
 
	while(i<chainlength-1) 
	{  
           if (allstates[state]==-1) 
             { 
	      allpos[orderstates] = state; 
              allstates[state] =  orderstates; 
              orderstates++; 
             } 
           value += (chain[i+1] == buff[4*state+1+2*chain[i]]);  
           state = buff[4*state+2*chain[i]];  
           //cout<<"Val: "<<value<<" State:"<<state<<endl; 
	   i++;  
	}   
 
	if((orderstates)*4 < n)  
	{ 
         delete [] auxstring; 
         return (orderstates)*4; 
        } 
 
       for(i=0;i<orderstates*4;i+=4)  
       { 
           int p=4*allpos[int((i+1)/4)]; 
           //cout<<"p="<<p<<endl; 
	   auxstring[i] =   allstates[buff[p]]; 
           auxstring[i+1] = buff[p+1]; 
           auxstring[i+2] = allstates[buff[p+2]]; 
           auxstring[i+3] = buff[p+3]; 
           //for(int j=0;j<i+4;j++)  cout<<   auxstring[j]<<" "; 
           //cout<<endl;   
       }   
 
       for(i=0;i<n;i++) buff[i] = auxstring[i]; 
       delete [] auxstring; 
       //cout<<"order "<<orderstates<<" Val: "<<value<<endl; 
 
       return value; 
 
 
} 
 
 
//------------------------------------------------------------ 
 
 
double 
Barsfunc(int* buff, int n) 
{ 
 int i,j, tot0,tot1,sum0,sum1; 
 tot0 = 0; 
 tot1 = 0; 
 
 
 m = int(sqrt(n)); 
 for (i=0; i<m; i++) 
   {  
    sum0 = 0; 
    sum1 = 0; 
    for (j=0; j<m; j++) 
      { 
       if(buff[i+j*m]==1) sum1++; 
       else sum1=0; 
       tot1 += sum1; 
       if(buff[i*m+j]==0) sum0++; 
       else sum0=0; 
       tot0 += sum1; 
      } 
   } 
  if (tot0>tot1) return tot0; 
  return tot1; 
} 
 
//------------------------------------------------------------ 
 
 
double 
DecBarsfunc(int* buff, int n) 
{ 
 int i,j, tot0,tot1,sum0,sum1; 
 tot0 = 0; 
 tot1 = 0; 
 
 m = int(sqrt(n-1)); 
 for (i=1; i<m; i+=3) 
   {  
    for (j=0; j<m; j++) 
      { 
       sum0 =  buff[i+j*m-1]+ buff[i+j*m]+buff[i+j*m+1]; 
       if (sum0==3) tot0+=3; 
       else if (sum0==0) tot0+=2; 
       else if (sum0==1) tot0+=1; 
       sum1 =  (1-buff[(i-1)*m+j])+ (1-buff[i*m+j])+(1-buff[(i+1)*m+j]); 
       if (sum1==3) tot1+=3; 
       else if (sum1==0) tot1+=2; 
       else if (sum1==1) tot1+=1; 
      } 
   } 
  
  if (buff[n-1]==1) return tot0; 
  return tot1; 
} 
//------------------------------------------------------------ 
 
 
double 
GDecBarsfunc(int* buff, int n) 
{ 
 int i,j,sum0,sum1; 
 double  tot0,tot1; 
 tot0 = 0; 
 tot1 = 0; 
 
 m = int(sqrt(n-1)); 
 for (i=1; i<m; i+=3) 
   {  
    for (j=0; j<m; j++) 
      { 
       sum0 =  buff[i+j*m-1]+ buff[i+j*m]+buff[i+j*m+1]; 
       if (sum0==3) tot0+=1; 
       else if (sum0==0) tot0+=0.9; 
       else if (sum0==1) tot0+=0.8; 
       //sum1 =  (1-buff[(i-1)*m+j])+ (1-buff[i*m+j])+(1-buff[(i+1)*m+j]); 
       sum1 =  (buff[(i-1)*m+j])+ (buff[i*m+j])+(buff[(i+1)*m+j]); 
       if (sum1==3) tot1+=1; 
       else if (sum1==0) tot1+=0.9; 
       else if (sum1==1) tot1+=0.8; 
      } 
   } 
  
  if (buff[n-1]==1) return tot0; 
  return tot1; 
} 
 
 
//------------------------------------------------------------ 
 
 
double 
Lector(int* buff, int n) 
{ 
 int i,ind,sum; 
 sum = 0; 
  
 //ind = 8*buff[16]+4*buff[17]+2*buff[18]+buff[19]; 
 ind =16*buff[32]+8*buff[33]+4*buff[34]+2*buff[35]+buff[36]; 
 for (i=0; i<32; i++) sum +=buff[i]; 
 
 
 if (buff[n-1]==buff[n-2]) return (32-sum-1+2*buff[ind]); 
  return (sum-2*buff[ind]+1); 
} 
 
double 
PathFunction(int* buff, int n) 
{ 
  int value=0; 
  int i; 
 for(int i=0;i<n;i++)     //value += genes[i]; 
               { 
                if (buff[i]==0) {return value;} 
                else {value += 1;}; 
                } 
 return value; 
} 
 
 
 
//------------------------------------------------------------ 
 
double 
Uniform(int* buff, int n) 
{ return 1.0; 
} 
double 
Multiplexor(int* buff, int n) 
{ 
 int i,ind,sum,aux; 
 sum = 0; 
  
 //ind = 8*buff[16]+4*buff[17]+2*buff[18]+buff[19]; 
 ind =16*buff[32]+8*buff[33]+4*buff[34]+2*buff[35]+buff[36]; 
 for (i=0; i<32; i++) sum +=buff[i]; 
 
 aux = buff[n-3]+buff[n-2]+buff[n-1]; 
 if (aux==3 || aux==1 ) return (32-sum-1+2*buff[ind]); 
  return (sum-2*buff[ind]+1); 
} 
//------------------------------------------------------------ 
 
// Funcion de Mh para la deceptive 5 
double Fharddec5[] = {0.9, 0.85, 0.8, 0.75, 0, 1}; 
double harddecp5(int *x, int n) 
{ 
  double s; 
  int u; 
   
  register int i; 
 
  s = 0; 
 
  for (i=0; i<n; ) 
    { 
      u  = x[i++]; 
      u += x[i++]; 
      u += x[i++]; 
      u += x[i++]; 
      u += x[i++]; 
 
      s +=  Fharddec5[u]; 
     }; 
 
   return s; 
} 
 
//---------------------------------------------------- 
double decepV[]={0.2, 1.4, 1.6, 2.9, 3, 1, 0.8, 0.6}; 
double DecepVenturini(int *buff, int n) 
{ 
  double sum = 0; 
  for(int i = 0; i < n; i += 3) 
        sum += decepV[ (buff[i+2] + 2 * buff[i + 1] + 4 * buff[i]) ]; 
  //sum += decepV[ (1-buff[i]) + 2 * (1-buff[i + 1]) + 4 * (1-buff[i + 2]) ]; 
  return sum; 
} 
 
 
 
 
int darValor(int a, int b) 
{ 
 if (a==0 && b==0) return 0; 
 if (a==0 && b==1) return 1; 
 if (a==1 && b==0) return 2; 
 if (a==1 && b==1) return 3; 
} 
 
 
 
 
int msuma(int *genes, int N) 
 
{ 
 
int sum=0; 
 
for(int j=0; j<N; j++) 
 
        { 
 
        if (genes[j]==1) { sum--; }; 
 
        } 
 
return sum; 
 
} 
 
 
 
bool esLongPath(int *genes,int IND_SIZE ) 
 
{  
 
bool FINAL=true; 
bool ESTADO=true; 
bool CEROS; 
bool CERO; 
bool UNO; 
bool TRES; 
int izq; 
int der; 
int valor; 
 
if (genes[IND_SIZE-1]==0) 
        { 
        CERO=true; 
        UNO=false; 
        TRES=true; 
        CEROS=true; 
        } 
 
else 
        { 
        CERO=true; 
        UNO=true; 
        TRES=true; 
        CEROS=false; 
        }; 
 
 
for(int k=IND_SIZE-2; k>=0; k=k-2) 
        { 
        der=genes[k]; 
        izq=genes[k-1]; 
        valor=darValor(izq,der); 
        if (ESTADO==false) { return false; } 
        else { ESTADO=false; }; 
 
        if (valor==2) return false; 
        if (valor==0 && CERO==true) 
               { 
                CERO=true; 
                UNO=false; 
                TRES=true; 
                ESTADO=true; 
                continue; 
                }; 
 
        if (valor==3 && TRES==true && CEROS==true) 
                { 
                CERO=true; 
                UNO=true; 
                TRES=true; 
                CEROS=false; 
                ESTADO=true; 
                continue; 
                }; 
 
        if (valor==3 && TRES==true && CEROS==false) 
                { 
                CERO=true; 
                UNO=false; 
                TRES=true; 
                ESTADO=true;             
                continue; 
                } 
 
        if (valor==1 && UNO==true && FINAL==true) 
                { 
                CERO=true; 
                UNO=false; 
                TRES=true; 
                CEROS=false; 
                ESTADO=true; 
                FINAL=false; 
                continue; 
                } 
        }; 
return ESTADO; 
} 
 
 
 
 
double valorP(int *genes, int N) 
 
{ 
 
double valorN=ldexpl(3,(N-1)/2)-1; 
double P; 
double ALTO; 
double BAJO; 
double M_BAJO=0; 
double M_ALTO=valorN-1; 
bool ESTADO=true; 
int valor; 
int izq; 
int der; 
int contador=-2; 
valorN=(valorN*2)+1; 
 
 
 
for(N; N>1; N=N-2) 
        { 
        valorN=(valorN-1)/2; 
        contador=contador+2; 
        der=genes[contador+1]; 
        izq=genes[contador]; 
        valor=darValor(izq,der); 
 
        if (ESTADO==true) 
                { 
                if (valor==0) 
                        { 
                        BAJO=0; 
                        ALTO=(valorN-3)/2; 
                        M_ALTO=M_BAJO + (ALTO - BAJO); 
                        continue; 
                        }; 
                if (valor==3) 
                        { 
                        ALTO=valorN-1; 
                        BAJO=(valorN+1)/2; 
                        M_BAJO=M_ALTO - (ALTO - BAJO); 
                        ESTADO=false; 
                        continue; 
                        }; 
                } 
 
        if (ESTADO==false) 
                { 
                if (valor==3) 
                        { 
                        BAJO=0; 
                        ALTO=(valorN-3)/2; 
                        M_ALTO=M_BAJO + (ALTO - BAJO); 
                        ESTADO=true; 
                        continue; 
                        }; 
                if (valor==0) 
                        { 
                        ALTO=valorN-1; 
                        BAJO=(valorN+1)/2; 
                        M_BAJO=M_ALTO - (ALTO - BAJO); 
                        continue; 
                        }; 
                } 
 
        if (valor==1) 
               { 
                P=M_BAJO + ldexpl(3,(N-3)/2)-1; 
                return P; 
                } 
        } 
 
der=genes[contador+2]; 
valor=darValor(der,der); 
 
if (ESTADO==true) 
        { 
        if (valor==0) 
                { 
                P=M_BAJO; 
                return P; 
                } 
        else 
                { 
                P=M_ALTO; 
                return P; 
                } 
        } 
if (ESTADO==false) 
        { 
        if (valor==3) 
 
                { 
                P=M_BAJO; 
               return P; 
                } 
        else 
                { 
                P=M_ALTO; 
               return P; 
               } 
        } 
return P; 
} 
 
 
double LongPath(int *buff, int n) 
{ 
  double val; 
 
 val = 0; 
if (esLongPath(buff,n)) 
        { 
        val = valorP(buff,n); 
        } 
else 
        { 
        val = msuma(buff,n); 
        }; 
 
//val  = val - ldexpl(3,(n-1)/2)-2;    
//cout<<val<<endl; 
  val -= (ldexpl(3,(n-1)/2)-2); 
  //cout<<val<<endl; 
return  val; 
}  
 
 
 
 
//------------------------------------------------------ 
void InitProblem(int Fun, int n) 
{ 
  switch(Fun) 
    { 
    case -1: i_user_defined(n); return; 
    case  3: i_Checkerboard(n); return; 
   
    } 
} 
//---------------------------------------- 
void EndProblem(int Fun, int *buff, double Fv, int n) 
{ 
  switch(Fun) 
    { 
    case -1: e_user_defined(buff,Fv,n); return; 
    case  3: e_Checkerboard(buff,Fv,n); return; 
   
    } 
  e_user_defined(buff,Fv,n); return;//default_end(buff, Fv, n); 
} 
 
 
double eval(int Fun, int *buff,int n){ 
  SetParam(params);
  switch(Fun){ 
  case -1: return user_defined_function(buff,n); 
  case 0: return FourPeaks(buff,n); 
  case 1: return SixPeaks(buff,n); 
  //case 2: ni(); //return ContinuosPeaks(buff,n); 
  case 3: return Checkerboard(buff,n); 
  case 4: return EqualProducts(buff,n); 
  case 5: return OneMax(buff,n); 
  case 6: return posFun(buff,n); 
  case 7: return posFun_to_2(buff,n); 
  case 8: return Decep3(buff,n); 
  case 9: return Fc4(buff, n); 
  case 10: return Fc5(buff, n); 
  case 11: return Fc2(buff, n); 
  case 12: return Fc3(buff, n); 
  case 13: return trap5(buff, n); 
  case 14: return BigJump(buff, n); 
  case 15: return KDeceptive(buff, n); 
  case 16: return IsoChain(buff,n); 
//  case 17: return IsoTree(buff,n); 
//  case 18: return IsoCirc(buff,n); 
  case 19: return IsoTorus(buff,n); 
  case 20: return NOneMax(buff,n);  
  case 21: return TwoPeaksDecep3(buff,n); // Deceptive de dos picos Roberto  
  case 22: return S_Peak(buff,n); 
  case 23: return Decep3_Mh(buff,n); 
  case 24: return harddecp5(buff,n); 
  case 25: return IsoPeak(buff,n); 
  case 26: return BlockFunc(buff,n); 
  case 27: return symmetryfunc(buff,n); 
  case 28: return Barsfunc(buff,n); 
  case 29: return DecBarsfunc(buff,n); 
  case 30: return GDecBarsfunc(buff,n);  
  case 31: return Lector(buff,n); 
  case 32: return Multiplexor(buff,n);  
  case 33: return Uniform(buff,n); 
  case 34: return DecepMarta3(buff,n); 
  case 35: return FullCheckerBoard(buff,n); 
  case 36: return PartialCheckerBoard(buff,n); 
  case 37: return DecepMarta3New(buff,n); 
  case 38: return DecepVenturini(buff,n); 
  case 39: return Automata(buff,n); 
  case 40: return IntCheckerboard(buff,n); 
  case 41: return IntFullCheckerBoard(buff,n); 
  case 42: return fhtrap1(buff,n); 
  case 43: return HIFF(buff,n); 
  case 44: return PathFunction(buff,n); 
  case 45: return LongPath(buff,n); 
  case 46: return FirstPolytree5Ochoa(buff,n); 
  case 47: return FirstPolytree3Ochoa(buff,n); 
  case 48: return FirstPolytree3OchoaSolap(buff,n); 
  case 49: return ErrorCodes(buff,n); 
  case 50: return GenKDecep(buff,n);
 } 
 
printf("function does not exit\n"); 
return -1; 
} 
 
int StopConditions(int Fun, double BestFv, double AvgFv, 
	 int StopGen, Popul *Pop) 
{ 
  int hit; 
  hit = 0; 
  switch(Fun) 
    { 
    case -1: hit=s_user_defined(BestFv,AvgFv,StopGen, Pop); break; 
      //  case 0:  hit=s_FourPeaks(BestFv,AvgFv,StopGen, Pop); break; 
      //  case 1:  hit=s_SixPeaks(BestFv,AvgFv,StopGen, Pop); break; 
      //  case 2:  hit=s_ContinuosPeaks(BestFv,AvgFv,StopGen, Pop); break; 
    case 3:  hit=s_Checkerboard(BestFv,AvgFv,StopGen, Pop); break; 
      //  case 4:  hit=s_EqualProducts(BestFv,AvgFv,StopGen, Pop); break; 
    case 5:  hit=s_OneMax(BestFv,AvgFv,StopGen, Pop); break; 
    case 6:  hit=s_PosFun(BestFv,AvgFv,StopGen, Pop); break; 
    case 7:  hit=s_PosFun_to_2(BestFv,AvgFv,StopGen, Pop); break; 
    case 8:  hit=s_Decep3(BestFv,AvgFv,StopGen, Pop); break;     
 //   case 9:  hit = s_Fc4(BestFv, AvgFv, StopGen, Pop); break; 
    case 10: hit = s_Fc5(BestFv, AvgFv, StopGen, Pop); break; 
//    case 11: hit = s_Fc2(BestFv, AvgFv, StopGen, Pop); break; 
//    case 12: hit = s_Fc3(BestFv, AvgFv, StopGen, Pop); break; 
    case 13: hit = s_trap5(BestFv,AvgFv,StopGen, Pop); break; 
    case 14: hit = s_BigJump(BestFv,AvgFv,StopGen, Pop); break; 
    case 15: hit = s_KDeceptive(BestFv,AvgFv,StopGen, Pop); break; 
    case 16: hit = s_IsoChain(BestFv,AvgFv,StopGen, Pop); break; 
 //   case 17: hit = s_IsoTree(BestFv,AvgFv,StopGen, Pop); break; 
//    case 18: hit = s_IsoCirc(BestFv,AvgFv,StopGen, Pop); break; 
    case 19: hit = s_IsoTorus(BestFv,AvgFv,StopGen, Pop); break; 
    case 20:  hit=s_NOneMax(BestFv,AvgFv,StopGen, Pop); break; 
    } 
/*  if( hit ) 
    { 
      fprintf(flog, "Hit! at gen: %d -- BestFv: %f -- Avg selected set: %f\n", StopGen, BestFv, AvgFv); 
     GenStop += StopGen; 
      return 1; 
    }  
  else 
    { 
      fprintf(flog, "gen: %d -- BestFv: %f -- Avg selected set: %f\n", 
StopGen, BestFv, AvgFv);       return 0; 
    } 
*/  
return hit; 
} 
  
void SetParam(int* params) 
{ 
  Funpars = params; 
} 
  // Evaluation function  
double evalua(int i,Popul* p1, int fun, int* params){  
p = p1;  
int m = p->psize;  
int n = p->vars;  
  
int j;  
double fv;  
  
//int *buff = new int[n];  
Funpars = params;  
//for(j=0; j < n; j++)  buff[j] = p->P[i][j];  
  InitProblem(fun,n);  
  // fv = eval(fun,buff,n);  
  fv = eval(fun,(int*)p->P[i],n); 
  //EndProblem(fun, buff, fv,n);  
EndProblem(fun,(int*) p->P[i], fv,n);  
//delete[] buff;  
return fv;  
}  
 
//------------------------------------------------------------- 
 
 
 
 
 
 
 
 
 
 
