#ifndef __MPCMARG__ 
 
#define __MPCMARG__ 
 
 
#include "Popul.h" 
 
///////////////// Cambios realizados a la clase marginal 
 
 
 
//......Clase mpcmarg, marginal generalizado................. 
 
class mpcmarg{ 
 
	 int* Vars;       // Variables del marginal 
 
	 int* Cardinality;  // Su respectiva cardinalidad      
 
	 double* C_Counts; // cell counts // Es necesario que cellcounts sea real 
 
	 int  Size;     // sample's size 
 
	 int  NumVars;  // Variables count 
 
	 int  C_Size;   // Tamanho de la tabla de frecuencias 
 
 
 
  public: 
 
	 mpcmarg( int* aVar, int* aCard, int aNumVars ); 
 
	 mpcmarg (mpcmarg *other); 
 
	 mpcmarg( int* aVar, int* aCard, int aNumVars, Popul* aPop ); 
 
	 ~mpcmarg(); 
 
	 void set( int* casos, double cell_counts );  // Asigna valor de frecuencia a la entrada 
                                                      // de la tabla corresp. con casos. 
	 double get( int* casos );                   // Devuelve valor de frecuencia de la tabla  
 
	 double get( int Index );                     // Idem ant. pero utilz directamente el Ind. 
 
	 inline int getC_size( void ){ return C_Size;} 
 
	 inline int getS_size( void ){ return Size;} 
 
	 void getCase( int I, int* caso );      
 
	 void compute( Popul* aPop );                 // A partir de una poblacion actualiza la tabla 
 
	 void compute( Popul* aPop, int *CondVars, int *values, int CondSize); 
 
	 int index( int* Casos ); 
 
	 void clear(); 
 
	 void print( void ); 
 
 mpcmarg*      operator*(mpcmarg othermarg);  // Multiplica 2 marginales de acuerdo al paso de  
                                              // de flujos definido en el algoritmo de propagacion 
                                              // de Nilssen 
 
 mpcmarg*      operator/(mpcmarg othermarg);  //   Idem para la division 
 
 mpcmarg&      operator=(mpcmarg& other);     // Constructor de copia. 
 
 int        findpos(int var);                 
 
 inline int getNumVars( void ){ return NumVars;} 
 
 void       set( int Index, double value ); // Llena una entrada de la tabla 
 
 mpcmarg*    getMaxOver( mpcmarg* auxcliq, mpcmarg* auxsolap ); //Max del marg dado valor de solap. 
 
 inline int* getVars() {return Vars;}; 
 
 inline int  getvar(int i) {return Vars[i];}; 
 
 inline int* getCards(){return Cardinality;}; 
 
 inline int  getSize() {return Size;}; 
 
 void SetInMarg( int* vars, int* values, int cantvar); // Adiciona evidencia a la tabla de frecuen. 
                                                       // para las var vars (Ver alg. Nillssen) 
 
 void ResetInMarg( int* vars, int* values, int cantvar); // Idem ant. otro tipo de evidencia 
 
 void FindBestConf(int *BestConf);    // Halla la configuracion mas probable para el marg.     
 
 void setmarg(double *vector);      // Actualiza los valores de la tabla a partir de un vector real 
 
 
 
}; 
 
#endif 
 
 
 
 
 
 
 
 
 
 
 
 
