#ifndef __MPCJTCLIQUE__

#define __MPCJTCLIQUE__

#include <string.h>

#include <stdio.h>

#include "mpcmarg.h"
  


class mpcjtclique{

  public:

  int size;

  int *vars;

  int *cards;

  mpcmarg*  TableCount[1];

 
  mpcjtclique(int *vars, int size); // not overlapped clique

  mpcjtclique(int *vars, int size, int *cards); // not overlapped clique

  mpcjtclique( mpcjtclique *other);

  ~mpcjtclique();

  void compute(Popul* Pop);

  //Funciones incluidas



  int contain(int var);

  inline int* getvars(void) {return vars;};

  inline int* getcards(void) {return cards;};

  inline int getvar(int var) { return vars[var];};

  inline int getsize() { return size;};

  inline mpcmarg* getmarg(void) { return *TableCount;};

  void setmarg(mpcmarg* newmarg);
 
  void SetCliqTo( int *x, int* y );

  void ResetCliqTo( int *x, int* y );

  //mpcjtclique::operator=(jtclique other);



  void generate(int *vector);

	 void print();

  void shortPrint();

  double evaluate(int*);



};

#endif






