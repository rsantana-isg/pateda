#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pada_globals.h"
#include "pada.h"
#include "fun.h"


extern void i_user_defined(int);
extern void e_user_defined(int *, double, int);
extern int s_user_defined(double, double, int,  pop *);
extern double user_defined_function(int *, int );

//********************************************************
pada_params *W;
// Evaluation function
void evalua(pop &pPop, double *fv, pada_params *Wp){
int m = pPop.Getsize();
int n = pPop.Nvars();
int i, j;
int buff[n];

W = Wp;

for(i=0; i < m; i++){
  for(j=0; j < n; j++)
    buff[j] = pPop(i,j);
  fv[i] = eval(W->pada.Fun.No,buff,n);
}
}
//-------------------------------------------------------
void ni(){
printf("Function not implemented yet\n");
exit(0);
}
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

int s_trap5(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  if( BestFv == W->pada.Nvars ) return 1;
  return 0;
}


//--------------------------------------------------------
// Funcion FourPeaks
double T_FourPeaks; // parameter of the function
double FourPeaks(int *buff,int n)
{
  int *p;
  int head_1, tail_0, min, max;
  
  T_FourPeaks = W->pada.Fun.pars[0];

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
  int *p;
  int head_1, head_0, tail_0, tail_1, min1, max1, min2, max2, min, max;

  T_SixPeaks = W->pada.Fun.pars[0];
  
  p=buff;
  head_1=0;
  while(*p)
    {
      head_1 ++;
      p++;
    }
  p=buff;
  head_0=0;
  while(*p==0)
    {
      head_0 ++;
      p++;
    }

  p=buff+n-1;
  tail_0=0;
  while(*p==0)
    {
      tail_0++;
      p--;
    }

  p=buff+n-1;
  tail_1=0;
  while(*p)
    {
      tail_1++;
      p--;
    }

  if(head_1>tail_0)
    {
      min1=tail_0; max1=head_1;
    }
  else
    {
      min1=head_1; max1=tail_0;
    }

  if(head_0>tail_1)
    {
      min2=tail_1; max2=head_0;
    }
  else
    {
      min2=head_0; max2=tail_1;
    }
  min = (min1>min2)?min2:min1;
  max = (max1>max2)?max1:max2;
  
  if(min>T_SixPeaks)  // OJO no clara la definicion
    return (100+max);
  else
    return (max);
}
//------------------------------------------------------
int *grid;
double Checkerboard(int *buff, int n)
{
  int i, j, sum, temp;
  sum = 0;

  int N = int(W->pada.Fun.pars[0]);

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

int s_Checkerboard(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  int N = (int)  W->pada.Fun.pars[0];
  if(BestFv == (double)(N-2)*(N-2)*4) return 1;
  return 0;
}

void i_Checkerboard(int n)
{
  grid = new int[ W->pada.Fun.pars[0]* W->pada.Fun.pars[1]];
}

void e_Checkerboard(int *buff, double Fv, int n)
{
  int par1 = int(W->pada.Fun.pars[0]);
  printf("\nBest:\n");
  for(int i=0; i < par1; i++)
    {
      for(int j=0; j < par1; j++)
	printf("%d", *buff++);
      printf("\n");
    }
  printf("Fvalue: %f\n",Fv);
  delete grid;
}

//------------------------------------------------------
float *EqualProductsNumbers;

void i_EqualProductsNumbers(int n)
{
  EqualProductsNumbers = new float[n];
}
//..............................
void e_EqualProductsNumbers(int *buff, double Fv, int n)
{
  delete EqualProductsNumbers;
}
//..............................
double EqualProducts(int *buff, int n)
{
  int i;
  double p1,p2;
  p1 = 1.0;
  p2 = 1.0; 
  for(i=0; i < n; i++)
    if(buff[i])
      p1*=EqualProductsNumbers[i]*5;
    else
      p2*=EqualProductsNumbers[i]*5;
  return(fabs(p1-p2));
}

//-----------------------------------------------------

double OneMax(int *buff, int n)
{
  int i, sum;
  sum = 0;
  for(i = 0; i < n; i++) sum += buff[i];
  return sum;
}

int s_OneMax(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  if( BestFv == W->pada.Nvars ) return 1;
  return 0;
}

double NOneMax(int *buff, int n)
{
  int i, sum;
  sum = 0;
  for(i = 0; i < n; i++) sum += buff[i];
  return (n-sum);
}

int s_NOneMax(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  if( BestFv ==  W->pada.Nvars ) return 1;
  return 0;
}


//-----------------------------------------------------
int K_BigJump = 1, M_BigJump = 3;

double BigJump(int *buff, int n)
{ int Nvars = W->pada.Nvars;

  K_BigJump = int(W->pada.Fun.pars[0]);
  M_BigJump = int(W->pada.Fun.pars[1]);

  double r = OneMax(buff, n);
  if( r == Nvars) return(K_BigJump * Nvars);
  if( r > (Nvars - M_BigJump) ) return 0;
  return r;
}

int s_BigJump(double BestFv, double AvgFv, int StopGen, pop *Pop)
{ int Nvars = W->pada.Nvars;
  if( BestFv == Nvars * K_BigJump ) return 1;  
  return 0;
}


//------------------------------------------------------------
int Kdec = 6;
double decepK(int *buff)
{  
  double r = OneMax(buff, Kdec);
  if( r == Kdec ) 
    if(Kdec == 3) return 1; else return( Kdec );
  else
    if(Kdec == 3){ 
	if(r==0) return(0.9);
	if(r==1) return(0.8);
	return(0);
    } else  return(Kdec - 1 - r);
}

double KDeceptive(int *buff, int n){
double sum = 0;
Kdec = int(W->pada.Fun.pars[0]);
for(int i = 0; i < n; i += Kdec )
  sum += decepK(buff + i);
return sum;
}

int s_KDeceptive(double BestFv, double AvgFv, int StopGen, pop *Pop)
{

  if(Kdec == 3) {
      if( BestFv == W->pada.Nvars/3.0 ) return 1;
  }else{
      if( BestFv == W->pada.Nvars ) return 1;
  }
  return 0;
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

#define SIZE2 20
int Fc2Optimo[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
int s_Fc2(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  if( memcmp(Pop->indv(Pop->who()), Fc2Optimo, SIZE2*sizeof(int)) == 0 ) 
    return 1;  
  return 0;
}

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

#define SIZE3 20
int Fc3Optimo[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
int s_Fc3(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  if( memcmp(Pop->indv(Pop->who()), Fc3Optimo, SIZE3*sizeof(int)) == 0 ) 
      return 1;  
  return 0;
}

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

#define SIZE4 21
int Fc4Optimo[SIZE4] = {1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0};

int s_Fc4(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  if( memcmp(Pop->indv(Pop->who()), Fc4Optimo, SIZE4*sizeof(int)) == 0 ) 
      return 1;  
  return 0;
}

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

int s_Fc5(double BestFv, double AvgFv, int StopGen,pop *Pop)
{int n=Pop->Nvars();
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
  return sum;
}

int s_Decep3(double BestFv, double AvgFv, int StopGen,pop *Pop){
  /*int flag = 1;
    for(int i=0; i < Nvars; i++)
    {
    if( Pop->freq(i, 1) < 0.95 ) flag = 0;
    printf("%f ",Pop->freq(i,1));
    }
    printf("\n");
    if( flag ) return 1;
  */

   if( BestFv == (W->pada.Nvars/3.0) ) return 1;
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

  if( r ==  5) return(m*m);
  return 0;
}

int s_IsoTorus(double BestFv, double AvgFv, int StopGen,pop *Pop){
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
l = int(W->pada.Fun.pars[0]);

int s = 0;
for(int i = 0; i < (l-1); i++,s+=2)
  sum += Iso1(buff + s);
sum += Iso2(buff + s);  

return sum;
}

int s_IsoChain(double BestFv, double AvgFv, int StopGen,pop *Pop){
   if( BestFv == (l*(l-1)+1) ) return 1;
   return 0;
}



//-----------------------------------------------------
double posFun(int *buff, int n)
{
  int i, sum;
  sum=0;
  for(i = 0; i < n; i++) sum += (buff[i]) ? (i+1): 0;
  return sum;
}

int s_PosFun(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  if(BestFv == W->pada.Nvars*(W->pada.Nvars+1)/2.0) return 1;
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

int s_PosFun_to_2(double BestFv, double AvgFv, int StopGen,pop *Pop)
{
  // OJO  if(BestFv == Nvars*(Nvars+1)/2.0) return 1;
  return 0;
}

//------------------------------------------------------
void InitProblem(int Fun, int n)
{
  switch(Fun)
    {
    case -1: i_user_defined(n); return;
    case  3: i_Checkerboard(n); return;
    case  4: i_EqualProductsNumbers(n); return;
    }
}
//----------------------------------------
void EndProblem(int Fun, int *buff, double Fv, int n)
{
  switch(Fun)
    {
    case -1: e_user_defined(buff,Fv,n); return;
    case  3: e_Checkerboard(buff,Fv,n); return;
    case  4: e_EqualProductsNumbers(buff,Fv,n); break;
    }
  default_end(buff, Fv, n);
}
//----------------------------------------

void funname(int Fun, char *dest){
  switch(Fun){
  case -1: strcpy(dest, "user_defined_function"); break;
  case 0: strcpy(dest, "FourPeaks"); break;
  case 1: strcpy(dest, "SixPeaks"); break;
  case 2: strcpy(dest, "No imple"); break; // continuos peaks
  case 3: strcpy(dest, "Checkerboard"); break;
  case 4: strcpy(dest, "EqualProducts"); break;
  case 5: strcpy(dest, "OneMax"); break;
  case 6: strcpy(dest, "posFun"); break;
  case 7: strcpy(dest, "posFun_to_2"); break;
  case 8: strcpy(dest, "Decep3Goldb"); break;
  case 9: strcpy(dest, "Fcuban4"); break;
  case 10: strcpy(dest, "Fcuban5"); break;
  case 11: strcpy(dest, "Fc2"); break;
  case 12: strcpy(dest, "Fc3"); break;
  case 13: strcpy(dest, "trap5"); break;
  case 14: strcpy(dest, "BigJump"); break;
  case 15: strcpy(dest, "KDeceptive (K en p0)"); break;
  case 16: strcpy(dest, "IsoChain"); break;
  case 17: strcpy(dest, "IsoTree"); break;
  case 18: strcpy(dest, "IsoCirc"); break;
  case 19: strcpy(dest, "IsoTorus"); break;
  case 20: strcpy(dest, "NOneMax"); break;
 }
}

double eval(int Fun, int *buff,int n){
  switch(Fun){
  case -1: return user_defined_function(buff,n);
  case 0: return FourPeaks(buff,n);
  case 1: return SixPeaks(buff,n);
  case 2: ni(); //return ContinuosPeaks(buff,n);
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
 }
printf("function does not exit\n");
return -1;
}

int StopConditions(int Fun, double BestFv, double AvgFv,
	 int StopGen, pop *Pop, dist1 *D1, dist2 *D2, dist3 *D3)
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
    case 9:  hit = s_Fc4(BestFv, AvgFv, StopGen, Pop); break;
    case 10: hit = s_Fc5(BestFv, AvgFv, StopGen, Pop); break;
    case 11: hit = s_Fc2(BestFv, AvgFv, StopGen, Pop); break;
    case 12: hit = s_Fc3(BestFv, AvgFv, StopGen, Pop); break;
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

//-------------------------------------------------------------










