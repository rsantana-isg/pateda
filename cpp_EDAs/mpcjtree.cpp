//#include "mpcmarg.h" 
#include <iostream> 
#include <fstream>  
#include "mpcjtree.h" 
 
 
 
void mpcjtclique::SetCliqTo( int *x, int* y ) { 
 
 int CantToSet = 0; 
 
 int *Nvars = new int [size]; 
 
 int *Nvalues = new int [size]; 
 
 int i; 
 
 for ( i=0; i<size; i++){ 
 
  if (y[vars[i]] == -1){ 
 
	  Nvars[CantToSet] = i; 
 
	  Nvalues[CantToSet++] = x[vars[i]]; 
 
	  y[vars[i]] = -1; 
 
  } 
 
 } 
 
  TableCount[0]->SetInMarg(Nvars,Nvalues,CantToSet); 
 
  delete[] Nvars; 
 
  delete[] Nvalues; 
 
 
} 
 
double mpcjtclique::evaluate(int *vector){ 
 int *Case =new int[size]; 
 for(int i=0;i<size;i++)  Case[i] = vector[vars[i]]; 
  double resp = TableCount[0]->get(Case); 
  delete Case; 
 return resp; 
} 
 
 
void mpcjtclique::ResetCliqTo( int *x, int* y ) { 
 
 int CantToSet = 0; 
 
 int *Nvars = new int [size]; 
 
 int *Nvalues = new int [size]; 
 
 int i; 
 
 for ( i=0; i<size; i++){  
  
 if (y[vars[i]] == -1){ 
 
          Nvars[CantToSet] = i; // Se pasa el orden de la variable 
 
	  Nvalues[CantToSet++] = x[vars[i]]; 
 
	  y[vars[i]] = -1; 
 
  } 
 } 
 
  TableCount[0]->ResetInMarg(Nvars,Nvalues,CantToSet); 
 
  delete[] Nvars; 
 
  delete[] Nvalues; 
 
 
} 
 
 
void mpcjtclique::setmarg(mpcmarg* newmarg){ 
 
 
  TableCount[0] = newmarg; 
 
 } 
 
 
 
int mpcjtclique::contain(int var) { 
 
  int i=0; 
 
  while( (i<size) && (vars[i] != var)) i++; 
 
  return(i<size); 
 
} 
 
 
 
 mpcjtclique::mpcjtclique(mpcjtclique *other){ 
 
  size =  other->getsize(); 
 
  vars = new int[size]; 
 
  cards = new int[size]; 
 
  memcpy(vars,other->getvars(), size * sizeof(int) ); 
 
  memcpy(cards,other->getcards(), size * sizeof(int) ); 
 
  // Ya aqui el contenido de las variables esta actualizado 
 
  TableCount[0] = new mpcmarg(other->getmarg()); 
 
} 
 
 
 
 // ************************************************ 
 
// Los que siguen ya estaban definidos 
 
 // ************************************************ 
 
 
 
 
 
mpcjtclique::mpcjtclique(int *_vars, int _size){ 
  // Esta funcion crea el clique pero posteriormente es necesario 
  // asignarle un marginal ya inicializado (setmarg) 
  // Se puede utilizar en su lugar el constructor que incluye los cardinales 
 
 
  cards      = (int   *)0; 
 
  size = _size; 
 
  vars = new int[size]; 
 
  memcpy(vars, _vars, size * sizeof(int));                                                                  
  TableCount[0] = (mpcmarg*) 0; 
 
} 
 
mpcjtclique::mpcjtclique(int *_vars, int _size, int *_cards){ 
 
 
  size = _size; 
 
  vars = new int[size]; 
 
  memcpy(vars, _vars, size * sizeof(int) ); 
   
  cards = new int[size]; 
 
  memcpy(cards, _cards, size * sizeof(int) ); 
 
  TableCount[0] = new mpcmarg(_vars,cards,size); 
} 
 
//------------------------------------------------------------------ 
 
mpcjtclique::~mpcjtclique(){ 
 
 
	if( vars  ) delete[] vars; 
 
	if( cards ) delete[] cards; 
 
        if (TableCount[0]) 
         {
	    delete TableCount[0]; 
         }
}  
 
//----------------------------------------------------------------- 
 
void 
 
mpcjtclique::compute(Popul* Pop){ 
 
   TableCount[0]->compute(Pop); 
  
} 
 
 
 
 
 
 
 
 
 
 
 
