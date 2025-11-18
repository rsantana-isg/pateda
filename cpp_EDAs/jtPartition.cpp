#include "jtPartition.h" 
#include <iostream> 
#include <fstream> 

//#include "jtPart~1.h" 
/*  Cycle implementa el procedimiento basico por el cual se generan las configuraciones 
    mas probables. Se cuenta con una lista en la cual cada una de las configuraciones 
    esta ordenada de acuerdo a su valor de probabilidad. El procedimiento comienza 
    conteniendo la configuracion de mayor probabilidad. Mientras no se cumplan las  
    condiciones de parada el algoritmo escoge la mejor configuracion existente en la 
    lista y que aun no ha sido visitada y partir de ella crea un conjunto de configuraciones 
    descendientes que son a su vez incorporadas a la lista si su probabilidad no es cero */     
 
 
void JtPartition::Cycle() { 
 
 
  
    int i,j; 
 
    int *index, *best; 
 
    int Marca; 
  
    double aux;

    index = new int[NumVars]; 
  
     
 
 while ( (first<MaxKConf) && (firstindex != -1))    // El procedimiento termina cuando  
                                                    // la proxima configuracion no visitada es la k                                                    // esima o cuando no existe ninguna otra conf.                                                     // diferente de cero.   
  { 
 
         JtreeTable* BestTree= GetBestTree();  // Mejor conf. no visitada. 
	 
         best = BestTree->GetBestConf();         
         
         double aux = BestTree->GetMaxProb(); 
  
	 //cout<<first<< " Prob  "<<aux<<endl;
    
	 Marca = BestTree->GetMarca();       
 
	 for (i=0;i<NumVars;i++) index[i] = -1; 
 
 
	 for (i=Marca;i<NumCliq;i++){ 
 
	        JtreeTable* newtablecliq = new JtreeTable(BestTree,i); // Se crean los descendientes 
 
		for (j=0;j<=i;j++) {    // i desempenha el papel de la marca 
 
		 if (j<i) newtablecliq->SetCliqTo(newtablecliq->ordering[j],best,index); 
 
		 else     newtablecliq->ResetCliqTo(newtablecliq->ordering[j],best,index); 
 
		} 
 
	 newtablecliq->PassingFluxesTopDown();    // Se realiza la propagacion de los flujos 
 
	 newtablecliq->PassingFluxesBottomUp(); 
 
         newtablecliq->FindBestConf();           // Se calcula la forma y prob de la configuracion 
          
         aux = newtablecliq->GetMaxProb(); 
  
	 if (aux > -10000000.0)  Add(newtablecliq,aux);
         else delete newtablecliq; 
	 //if (aux > 0)  Add(newtablecliq,aux);   // De ser positiva se anhade a la lista 
         // else delete newtablecliq;  
	} 
 
 firstindex = ordering[firstindex]; 
 first++; 
 
 
 } 
  
 delete[] index; 
} 
 
 
 
JtPartition::JtPartition(int MaxConf, int CantVar, int CantCliq){ 
 
 MaxKConf = MaxConf; 
 
 NumVars = CantVar; 
 
 NumCliq  = CantCliq; 
 
 KbestTrees  = new JtreeTable*[MaxKConf]; 
 
 ordering = new int [MaxKConf]; 
 
 CantConf = 0; 
 
 BestTree = 0; 
 
 first = -1; 
  
 last = -1; 
 
 
} 
 
/* La funcion Add se encarga de adicionar a la lista cada nueva configuracion ( arbol) 
   garantizando: 
  
   1) Que la lista permanece ordenada 
   2) El numero de elementos de la lista nunca excede las  k configuraciones. 
 
   first es la primera configuracion que todavia no ha sido completamente visitada 
   se inicializa su valor en cero. 
   firstindex es la posicion ( indice en la lista ) de la configuracion numero first. 
   ordering es un arreglo reflejando el orden de las configuraciones en la lista. 
    
*/   
 
void JtPartition::Add(JtreeTable* tree, float val) { 
 
 int i,F; 
 
  if ( first == -1){ 
 
	  first = last = 0; 
 
	  ordering[0] = -1; 
 
          firstindex = 0; 
 
 	  KbestTrees[0]= tree;  
 
	  return; 
 
	 } 
 
  if (CantConf == MaxKConf-1){ 
 
    if (val <= KbestTrees[last]->GetMaxProb()) delete tree; // Si es menor o igual que la peor no se incluye 
 
	else { 
 
	 delete KbestTrees[last]; 
 
	 F = FindFather(firstindex,last); 
      
         if (last != CantConf) { 
         
	   int AuxF = FindFather(0,CantConf); // MaxKConf 
 
           ordering[F] = -1; 
             
           KbestTrees[last] = KbestTrees[CantConf]; 
 
           KbestTrees[CantConf] = (JtreeTable *)0; 
 
           ordering[last] = ordering[CantConf]; 
          
           if (firstindex==CantConf) firstindex = last; // Porque CantConf ya no existe 
         
           ordering[AuxF] = last; 
 
         } 
          
         last = (F==CantConf) ? last : F; 
             
         ordering[last] = -1; 
 
         CantConf--; 
 
	 Add(tree,val ); 
 
	} 
         
        return;  
  }  
 
  	i = firstindex; 
  
	while ((ordering[i] != -1) && ( KbestTrees[i]->GetMaxProb()>=val)) { 
 
	        F = i;             //Se busca el antecesor de i 
 
		i = ordering[i]; 
 
	 } 
 
  if (ordering[i] == -1){ 
         
         KbestTrees[++CantConf] = tree; 
         
	 if (val > KbestTrees[i]->GetMaxProb()) { // El nuevo valor es > que el ultimo  
               
             ordering[F] = CantConf; 
        
             ordering[CantConf] = i; 
              
             last = i;  
         }else 
	 { 
           ordering[i] = CantConf; // El nuevo valor es < que el ultimo 
 
	   ordering[CantConf] = -1; 
 
	   last = CantConf; 
         } 
 
	}else { 
 
         KbestTrees[++CantConf] = tree; 
 
	 ordering[CantConf] = i; 
 
         ordering[F] = CantConf;  // Se inserta el valor en la lista    
 
       } 
 
  return; 
 
} 
 
// SetPop llena una Poblacion con los howmany mejores configuraciones de la lista 
 
void JtPartition::SetPop(int Elit, Popul* aPop, int howmany){ 
  // Se debe garantizar que howmany sea <= que el tamanho de poblacion 
  // y la que  cantidad de configuraciones disponibles 
 
int i = 0; 
int cant = Elit; 
int *best; 
 
while (  (i != -1) && (cant<howmany+Elit)) {   
    best = KbestTrees[i]->GetBestConf(); 
    for (int j=0;j<NumVars;j++) 
       {
         aPop->P[cant][j] = best[j]; 
         
	}
    
    aPop->Evaluations[cant] = KbestTrees[i]->GetMaxProb(); 
    i = ordering[i] ; 
    cant++; 
 } 
 
} 
 
 
 
// FindFather encuentra el nodo anterior a son en la lista buscando  
// a partir de from 
 
int JtPartition::FindFather(int from, int son) { 
 
 int i = from; 
 
 while (ordering[i] != son) i = ordering[i]; 
 
 return i; 
 
} 
 
 
 
JtPartition::~JtPartition(){ 
 
int i=0; 
 
while (i != -1) 
   
 { 
   
    delete KbestTrees[i]; 
     i = ordering[i] ; 
  } 
 
 
  delete[] ordering; 
 
  delete[] KbestTrees; 
 
} 
 
 
 
 
 
 
 
 
 
 
 
