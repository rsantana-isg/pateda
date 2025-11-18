#ifndef __JTREETABLE__

#define __JTREETABLE__

#include "mpcjtree.h"

#include "AbstractTree.h"


///////////////// Clase JtreeTable



class JtreeTable{


 public:

  mpcjtclique** cliqs;  // OJO prueba

  mpcjtclique**   solaps;

 int*       PrevCliq;

 int*       PrevSolap;

 int       *ordering;

 unsigned int       *cards;

 int       __cliq;

 int       __solp;

 int       NoVars;

 int       NoCliques;

 int       *BestConf;

 double    HighestProb;

 int       Marca;



 JtreeTable(int _nvars, int _NoCliques);

 JtreeTable(int _nvars, int _NoCliques, unsigned int* _cards);

 JtreeTable(JtreeTable* other,int _Marca);

 ~JtreeTable();

 //void add(c C);

 //void add(c C, c O);

 void add(int *vars, int size);

 void add(int *vars, int size, int *Over, int over_size);

 int  WhereIsStore(int* solp, int size);

 //void MakeJt( cl &O );

 void PassingFluxesBottomUp();

 void PassingFluxesTopDown();

 mpcjtclique* FindSon(int Father, int* pos);

 void SetCliqTo(int i, int* x, int* y);

 void ResetCliqTo(int i, int* x, int* y);

 void  Compute (Popul* Pop);

 inline int getNoCliques()  { return NoCliques;};

 inline int getcliqc()  { return __cliq;};

 inline int getsolpc()  { return __solp;};

 inline int getNoVars()  { return NoVars;};

 inline mpcjtclique* getSolap(int i)  { return solaps[i];};

 inline mpcjtclique* getCliq(int i)  { return cliqs[i];};

 inline int getPrevCliq(int i)  { return PrevCliq[i];};

 inline int getPrevSolap(int i)   { return PrevSolap[i];};

 inline int  getordering(int i)  { return ordering[i];};

 inline unsigned int getcard(int i) { return cards[i];}
 
 void FindBestConf();

 inline int* GetBestConf(){return BestConf;};

 inline float GetMaxProb(){ return HighestProb;};

 inline int  GetMarca(){ return Marca;};

 inline void  SetMarca(int _Marca){ Marca = _Marca;};

 double evaluate(int* vector);

 //int FindMaxConf (Popul*,Popul*,int); 

 void convert(IntTreeModel*);

 void convertMPM(unsigned long**);
};

#endif

