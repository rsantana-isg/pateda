#include <stdlib.h> 
#include <iostream> 
#include <fstream>  
#include <string.h> 
 
#include "mpcmarg.h" 
 
 
 
// Constructor de copia para la clase 
 
mpcmarg::mpcmarg (mpcmarg* other){   
 
  int i ; 
 
  NumVars = other->getNumVars(); 
 
  Vars = new int[NumVars]; 
 
  Cardinality = new int[NumVars]; 
 
  memcpy( Cardinality, other->getCards(), sizeof(int)*NumVars ); 
 
  memcpy( Vars, other->getVars(), sizeof(int)*NumVars ); 
 
 
 
  C_Size = 1; 
 
  for (i=0; i < NumVars; i++) 
 
	 C_Size *= Cardinality[i]; 
 
 
  C_Counts = new double [C_Size]; 
 
  memset(C_Counts, 0, sizeof(double)*C_Size ); 
 
  //for (i=0; i < C_Counts; i++) 
 
  Size = other->getSize(); 
 
  for (i=0; i<C_Size;i++) set(i,other->get(i)); 
 
 
 
 } 
 
 
// Crea una tabla de frecuencias a partir de un vector de reales 
 
 void mpcmarg::setmarg(double *vector){ 
 
  memcpy(C_Counts,vector,sizeof(double)*C_Size); 
 
 } 
 
// Llena una entrada de la tabla de frecuencias. 
 
void  mpcmarg::set( int Index, double value ){ 
 
	 C_Counts[Index] = value; 
 
} 
 
 
 
int mpcmarg::findpos(int var){ 
 
 int i=0; 
 
 while( (i< NumVars) && (Vars[i]) !=var) i++; 
 
 return i; // Si i==size hay un error 
 
} 
 
 
 
 
 
mpcmarg*  mpcmarg::operator*(mpcmarg othermarg){ 
 
 int i,j; 
 
 mpcmarg *resp, *auxsolap, *auxcliq; 
 
 if(this->getC_size() >= othermarg.getC_size() ){ 
 
	auxcliq=this; 
 
	auxsolap = &othermarg; 
 
  } 
 
 else { 
 
	auxcliq = &othermarg; 
 
	auxsolap = this; 
 
  } 
 
 
 
  resp = new mpcmarg(auxcliq->getVars(),auxcliq->getCards(),auxcliq->getNumVars()); 
 
  int* Case = new int[auxcliq->getNumVars()]; 
 
  int* overcase = new int[auxsolap->getNumVars()]; 
 
  int* auxvar = new int[auxsolap->getNumVars()]; 
 
 
 
  for(i=0;i<auxsolap->getNumVars();i++) 
 
	auxvar[i] = auxcliq->findpos(auxsolap->getvar(i)); 
 
 
 
  for(i=0;i<auxcliq->getC_size();i++){ 
 
	  auxcliq->getCase( i, Case ); 
 
	  for( j=0;j<auxsolap->getNumVars();j++) overcase[j] = Case[auxvar[j]]; 
 
	  resp->set( i,auxcliq->get(i)*auxsolap->get(overcase) ); 
 
	 } 
 
 delete []Case; 
 
 delete []overcase; 
 
 delete []auxvar; 
 
 return resp; 
 
} 
 
 
 
 
 
mpcmarg* mpcmarg::getMaxOver( mpcmarg* auxcliq, mpcmarg* auxsolap ){ 
 
 
 
  mpcmarg* resp= new mpcmarg(auxsolap->getVars(),auxsolap->getCards(),auxsolap->getNumVars()); 
 
 
 
  int* Case = new int[auxcliq->getNumVars()]; 
 
  int* overcase = new int[auxsolap->getNumVars()]; 
 
  int* auxvar = new int[auxsolap->getNumVars()]; 
 
  int i; 
 
 
 
  for(i=0;i<auxsolap->getNumVars();i++) 
 
	auxvar[i] = auxcliq->findpos(auxsolap->getvar(i)); 
 
 
 
  for(i=0;i<auxcliq->getC_size();i++){ 
 
	  auxcliq->getCase( i, Case ); 
 
	  for(int j=0;j<auxsolap->getNumVars();j++) overcase[j] = Case[auxvar[j]]; 
 
	  double currmax = auxcliq->get(i); 
 
	  if(currmax>resp->get(overcase)) resp->set(overcase,currmax); 
 
  } 
 
 delete []auxvar; 
 
 delete []Case; 
 
 delete []overcase; 
 
 
 
 return resp; 
 
} 
 
 
 
mpcmarg*   mpcmarg::operator/(mpcmarg othermarg){ 
 
// Se divide this entre othermarg 
 
// Ambos marginales tienen la misma dimension 
 
 int i; 
 
 mpcmarg *resp; 
 
 
 
  resp = new mpcmarg(this->getVars(),this->getCards(),this->getNumVars()); 
 
 
 
  for(i=0;i<this->getC_size();i++){ 
 
	  if (othermarg.get(i) > 0 ) resp->set(i,this->get(i)/othermarg.get(i)); 
 
	  else resp->set(i,this->get(i)); // Que se debe hacer cuando hay indefinicion ? 
 
	 } 
 
 return resp; 
 
 } 
 
 
 
void mpcmarg::SetInMarg( int* vars, int* values, int cantvar){ 
 
 int* Case = new int[NumVars]; 
 
 int i,j; 
 
 for(i=0; i< C_Size; i++){ 
 
  getCase(i,Case); 
 
  j = 0; 
 
  while ( (j<cantvar) && (Case[vars[j]] ==  values[j]) ) j++; 
 
  if (j != cantvar) set(i,0); 
 
 } 
 
  delete[] Case; 
 
} 
 
 
 
void mpcmarg::ResetInMarg( int* vars, int* values, int cantvar){ 
 
 int* Case = new int[NumVars]; 
 
 int i,j; 
 
 for(i=0; i< C_Size; i++){ 
 
  getCase(i,Case); 
 
  j = 0; 
 
  while ( (j<cantvar) && (Case[vars[j]] ==  values[j]) ) j++; 
 
  if (j == cantvar) set(i,0); 
 
 } 
 
 delete[] Case; 
 
} 
 
 
 
void mpcmarg::FindBestConf(int *BestConf){ 
 
 // En BestConf se va actualizando el valor de las variables 
 
 // de la mejor configuracion 
 
 int*  Case = new int [NumVars]; 
 
 int*  PosVars = new int [NumVars]; 
 
 int alreadyset = 0; 
 
 int i,j,maxindex; 
 
 double max; 
  // El valor -1 indica que no se actualizado 
 
 //e cout<<"The number of Vars is "<<NumVars<<endl;
  for( i=0; i< NumVars; i++) { if ( BestConf[Vars[i]] != -1) PosVars[alreadyset++] = i; } 
 
	// Una solucion alternativa a la que sigue no seria generar 
 
	// las soluciones fatibles y escoger el maximo entre ellos. 
 
	max = 0; 	maxindex = 0; 	i = 0; 
 
	while ( i < C_Size) { 
 
	 getCase(i,Case); 
 
	 j = 0; 
 
	 while ((j<alreadyset) && (Case[PosVars[j]] == BestConf[Vars[PosVars[j]]])) j++; 
 
	 if ( (j == alreadyset) && ( get(i)> max) ) { 
 
	  max = get(i); 
 
	  maxindex = i; 
 
	 } 
 
	 i++; 
 
	} 
 
	 getCase(maxindex,Case); 
 
	 for( i=0; i<NumVars; i++) 
           {
              BestConf[Vars[i]]= Case[i];
              //cout<<Case[i]<<" ";
           } 
         //cout<<endl;
	 // Hay algo de redundancia pero se justifica por el ahorro de tiempo 
 
	 delete[] Case; 
 
    delete[] PosVars; 
 
  } 
 
 
 
 mpcmarg& mpcmarg::operator=(mpcmarg& other){ 
 
  int i ; 
 
  mpcmarg* resp = new mpcmarg(other.getVars(),other.getCards(),other.getNumVars()); 
 
  Size = other.getSize(); 
 
  for (i=0; i<C_Size;i++) set(i,other.get(i)); 
 
  return *resp; 
 
 } 
 
 
 
mpcmarg::mpcmarg( int* aVar, int* aCard, int aNumVars, Popul* aPop ){ 
 
  NumVars = aNumVars; 
 
  Vars = new int[NumVars]; 
 
  Cardinality = new int[NumVars]; 
 
 
 
  memcpy( Cardinality, aCard, sizeof(int)*NumVars ); 
 
  memcpy( Vars, aVar, sizeof(int)*NumVars ); 
 
 
 
  C_Size = 1; 
 
  for (int i=0; i < NumVars; i++) 
 
	C_Size *= Cardinality[i]; 
 
 
 
  C_Counts = new double[C_Size];    // OJO Cambiado a double 
 
  memset( C_Counts, 0, sizeof(double)*C_Size );     // OJO Cambiado a double 
 
 
 
  compute( aPop ); 

} 
 
 
 
 
void mpcmarg::compute(Popul* aPop){ 
 
  int* Casos; 
 
  Size = aPop->psize; 
 
 
 
  Casos = new int[NumVars]; 
 
 
 
  for(int j = 0; j < Size; j++){ 
 
    for (int i = 0; i < NumVars; i++) 
 
      Casos[i] = aPop->P[j][Vars[i]]; 
 
    C_Counts[index( Casos )]++; 
 
  } 
 
  //print(); 
 
  delete[] Casos; 
 
} 
 
 
void mpcmarg::compute(Popul* aPop, int *CondVars, int *values, int CondSize){ 
 
  int* Casos; 
 
  int __Size = aPop->psize; 
 
 
 
  Size = 0; 
 
  Casos = new int[NumVars]; 
 
 
 
  int flag; 
 
  for(int j = 0; j < __Size; j++){ 
 
	 // checking conditions 
 
	 flag = 0; 
 
	 for(int k = 0; k < CondSize; k++) 
 
		if( (    aPop->P[j][CondVars[k]]) != values[k] ){ flag = 1; break;} 
 
	 if( flag ) continue; 
 
	 Size++; 
 
	 for (int i = 0; i < NumVars; i++) 
 
		Casos[i] = aPop->P[j][Vars[i]]; 
 
	 C_Counts[index( Casos )]++; 
 
  } 
 
 
 
  delete[] Casos; 
 
} 
 
 
 
 
 
 
 
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
// A partir de aqui las clases tal y como estaban definidas 
 
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
 
 
 
 
 
mpcmarg::mpcmarg( int* aVar, int* aCard, int aNumVars ){ 
 
  NumVars = aNumVars; 
 
  Vars = new int[NumVars]; 
 
  Cardinality = new int[NumVars]; 
 
 
 
  memcpy( Cardinality, aCard, sizeof(int)*NumVars ); 
 
  memcpy( Vars, aVar, sizeof(int)*NumVars ); 
 
 
 
  C_Size = 1; 
 
  for (int i=0; i < NumVars; i++) 
 
	 C_Size *= Cardinality[i]; 
 
 
 
  C_Counts = new double [C_Size]; 

  for (int i=0; i < C_Size; i++) C_Counts[i] = 0.0;
 
  //memset( C_Counts, 0.0, sizeof(double)*C_Size ); 
 
} 
 
 
 
/* 
mpcmarg::mpcmarg( int* aVar, int* aCard, int aNumVars, pop* aPop ){ 
 
  NumVars = aNumVars; 
 
  Vars = new int[NumVars]; 
 
  Cardinality = new int[NumVars]; 
 
 
 
  memcpy( Cardinality, aCard, sizeof(int)*NumVars ); 
 
  memcpy( Vars, aVar, sizeof(int)*NumVars ); 
 
 
 
  C_Size = 1; 
 
  for (int i=0; i < NumVars; i++) 
 
	C_Size *= Cardinality[i]; 
 
 
 
  C_Counts = new int[C_Size];    // OJO Cambiar a double 
 
  memset( C_Counts, 0, sizeof(int)*C_Size );     // OJO Cambiar a double 
 
 
 
  compute( aPop ); 
 
} 
 */ 
 
 
 
void 
 
mpcmarg::clear(){ 
 
  memset(C_Counts, 0, sizeof(double)*C_Size ); 
 
} 
 
 
 
mpcmarg::~mpcmarg(){ 
 
  delete[] Vars; 
 
  delete[] Cardinality; 
 
  delete[] C_Counts; 
 
} 
 
 
 
int mpcmarg::index( int* Casos ){ 
 
  int tmp1, tmp2; 
 
 
 
  tmp2 = 0; 
 
  for (int i = NumVars-1; i > -1; i--){ 
 
	 tmp1 = 1; 
 
	 for (int j = NumVars-1; j > i; j--) 
 
		tmp1 *= Cardinality[j]; 
 
	 tmp1 *= Casos[i]; 
 
	 tmp2 += tmp1; 
 
  } 
 
  return tmp2; 
 
} 
 
 
 
void mpcmarg::set( int* Casos, double Cell_Count ){ 
 
  C_Counts[index( Casos )] = Cell_Count; 
 
} 
 
 
 
double mpcmarg::get( int* Casos ){ 
 
  //cout<<"index "<<index( Casos )<<endl;
  return C_Counts[index( Casos )]; 
 
}   
 
/* 
 
void mpcmarg::compute(pop* aPop){ 
 
  int* Casos; 
 
  Size = aPop->Getsize(); 
 
 
 
  Casos = newvoid mpcmarg::compute(pop* aPop){ 
 
  int* Casos; 
 
  Size = aPop->Getsize(); 
 
 
 
  Casos = new int[NumVars]; 
 
 
 
  for(int j = 0; j < Size; j++){ 
 
    for (int i = 0; i < NumVars; i++) 
 
      Casos[i] = aPop->get(j, Vars[i]); 
 
    C_Counts[index( Casos )]++; 
 
  } 
 
 
 
  delete[] Casos; 
 
} 
 int[NumVars]; 
 
 
 
  for(int j = 0; j < Size; j++){ 
 
    for (int i = 0; i < NumVars; i++) 
 
      Casos[i] = aPop->get(j, Vars[i]); 
 
    C_Counts[index( Casos )]++; 
 
  } 
 
 
 
  delete[] Casos; 
 
} 
 
*/ 
 
//-------------------------------------------- 
 
/* 
 
void mpcmarg::compute(pop* aPop, int *CondVars, int *values, int CondSize){ 
 
  int* Casos; 
 
  int __Size = aPop->Getsize(); 
 
 
 
  Size = 0; 
 
  Casos = new int[NumVars]; 
 
 
 
  int flag; 
 
  for(int j = 0; j < __Size; j++){ 
 
	 // checking conditions 
 
	 flag = 0; 
 
	 for(int k = 0; k < CondSize; k++) 
 
		if( (aPop -> get(j, CondVars[k])) != values[k] ){ flag = 1; break;} 
 
	 if( flag ) continue; 
 
	 Size++; 
 
	 for (int i = 0; i < NumVars; i++) 
 
		Casos[i] = aPop->get(j, Vars[i]); 
 
	 C_Counts[index( Casos )]++; 
 
  } 
 
 
 
  delete[] Casos; 
 
} 
 
*/ 
 
//---------------------------------------- 
 
 
 
void mpcmarg::print(void){ 
 
  int i; 
 
 
 
  printf("VARS:"); 
 
  printf("\n"); 
 
  for (i = 0; i<NumVars; i++) 
 
	 printf(" <%d> ", Vars[i]); 
 
  printf("\n"); 
 
 
 
  for (i = 0; i < C_Size; i++) 
 
	 printf(" <%f> ", C_Counts[i]); 
 
  printf("\n"); 
 
} 
 
 
 
double mpcmarg::get( int Index ){ // get cambia de int a double 
 
	return C_Counts[Index]; 
 
} 
 
 
 
 
 
void mpcmarg::getCase( int I, int* caso ) 
 
{ 
 
	int I1; 
 
	int tmp; 
 
	I1 = I; 
 
 
 
	for (int i=NumVars-1; i > -1; i--){ 
 
		caso[i] = I1 % Cardinality[i]; 
 
		if ( !i ) 
 
			continue; 
 
		tmp = 1; 
 
		for (int j=NumVars-1; j > i; j--) 
 
			tmp *= Cardinality[j]; 
 
		I -= caso[i]*tmp; 
 
		I1 = I/(tmp*Cardinality[i]); 
 
	} 
 
} 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
