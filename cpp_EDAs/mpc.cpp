#include "mpc.h" 
#include <assert.h> 
 
void  
mpc0::mpPop(jtree& JT, pop &Pop){ 
  int nvars = Pop.Nvars(); 
  int* vector= new int[ nvars ]; 
  int NoConfig = 1 << nvars; 
  pop AllConfig( NoConfig, nvars, 1); 
  double *fv = AllConfig.getfvPtr(); 
  int MPConf = Pop.Getsize();       // numero de configuraciones a buscar 
  for(int i=0; i < NoConfig; i++){ 
	getCase(i, vector); 
	fv[i] = JT.ReturnProbOfConfiguration(vector); 
	for(int j=0; j < nvars; j++) 
	  AllConfig.set(i, j, vector[j]); 
  } 
  AllConfig.ordena(); 
  AllConfig.best(MPConf, &Pop); 
  delete []vector; 
} 
 
mpc0::mpc0(int* aVar, int* aCard, int aNumVars, int _SampleSize): 
                                    marg(aVar, aCard, aNumVars){ 
if( aNumVars > 21 ){ 
	printf("muy grande el MPC\n"); 
        exit(0); 
}				 
SampleSize = _SampleSize; 
int __C_size = this -> getC_size(); 
indices = new int[__C_size]; 
for(int i=0; i < __C_size; i++) 
  indices[i] = i; 
}; 
 
mpc0::~mpc0(){ 
delete indices; 
} 
 
//-------------------------------------------------------------- 
// find the less probable vector in Pop 
// given the K most probable vectors in 
// K_MPC. Returns negative if there is 
// overflow. In __OverFlow leaves the number of 
// overflow 
 
int LessProbableInPop(pop &Pop, pop &K_MPC, int &__OverFlow ){ 
int LessP = 0, ConfNo; 
 
__OverFlow = 0; 
for(int s=0; s < Pop.Getsize(); s++){ 
  ConfNo = K_MPC.in(Pop.indv(s)); 
  if( !ConfNo ) __OverFlow ++; 
  if( ConfNo > LessP  ) LessP = ConfNo; 
} 
 
if( __OverFlow ) 
  return ( -1 * LessP ); 
else 
  return ( LessP ); 
 
} 
