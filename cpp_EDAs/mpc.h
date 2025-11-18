#include "marg.h" 
#include "jtree.h" 
 
class mpc0 : public marg{ 
int *indices; 
int SampleSize; 
public: 
  mpc0(int* aVar, int* aCard, int aNumVars, int _SampleSize); 
  ~mpc0(); 
  void ordena(); 
  void mpPop(jtree& JT, pop &Pop); 
}; 
 
// find the less probable vector in Pop 
// given the K most probable vectors in 
// K_MPC. Returns negative if there is 
// overflow. In __OverFlow leaves the number of 
// overflow 
 
int LessProbableInPop(pop &Pop, pop &K_MPC, int &__OverFlow ); 
 
