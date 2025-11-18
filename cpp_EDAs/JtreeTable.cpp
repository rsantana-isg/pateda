#include <iostream> 
#include <fstream> 
#include "JtreeTable.h"


using namespace std;

JtreeTable::JtreeTable(int _nvars, int _NoCliques, unsigned int* _cards){


NoVars = _nvars;

Marca = 0;

NoCliques = _NoCliques;

cliqs =  new mpcjtclique*[NoCliques];

solaps = new mpcjtclique*[NoCliques-1];

 cards = new unsigned int[NoVars];

 memcpy(cards, _cards, NoVars * sizeof(unsigned int) );

PrevCliq = new int[NoCliques];

PrevSolap = new int[NoCliques];

__cliq  = 0;

__solp = 0;



ordering = new int[NoCliques];

BestConf = new int[NoVars];

for(int i=0; i < NoCliques; i++) ordering[i]=i;

}


JtreeTable::JtreeTable(int _nvars, int _NoCliques){

NoVars = _nvars;

Marca = 0;

NoCliques = _NoCliques;

cliqs =  new mpcjtclique*[NoCliques];

solaps = new mpcjtclique*[NoCliques-1];

PrevCliq =new int[NoCliques];

PrevSolap = new int[NoCliques];

__cliq  = 0;

__solp = 0;


cards = new unsigned int[NoVars];

ordering = new int[NoCliques];

BestConf = new int[NoVars];

for(int i=0; i < NoCliques; i++) ordering[i]=i;

}


JtreeTable::~JtreeTable(){

int i;

  for(i = 0; i < NoCliques; i++) 
   {
    if( cliqs[i] )  delete cliqs[i];
    else cout<<"Not destroyed "<<i<<endl;
   }
  

  for(i = 0; i < __solp; i++) if( solaps[i] )  delete solaps[i];


  delete[] cliqs;


  delete[] solaps;

  delete[] ordering;

  delete[] BestConf;

  delete[] PrevSolap;

  delete[] PrevCliq;

  delete[] cards;
}

void JtreeTable::Compute(Popul* Pop){

int i;

int a = Pop->psize;

  for(i = 0; i < NoCliques; i++) cliqs[i]->compute(Pop);

  for(i = 0; i < __solp; i++) solaps[i]->compute(Pop); 

}


int

JtreeTable::WhereIsStore(int* solp, int size){

 int i,j;

  for(i=0;i<__cliq;i++){

	 j=0;

	 while ( (j<size) && (cliqs[i]->contain(solp[j])) )j++;

    if (j==size) return i;

  }

  return -1; // Si el algoritmo llega a este punto

				 // debe existir un error (una variable de solp que no

				 // esta en ningun clique

 }

/*

void  // Aqui se asume que el mismo clique de solapamiento

      // no se entrara dos veces, OJO: Es mejor verificar esto

JtreeTable::add(c Cliq, c Over){

  int i;
  
  int* auxcard = new int[Cliq.size];
 
  for(i=0;i<Cliq.size;i++) auxcard[i] = cards[Cliq.vars[i]];

  cliqs[__cliq] = new mpcjtclique(Cliq.vars, Cliq.size, auxcard);

  int* aux = new int[Over.size];

  for(i=0;i<Over.size;i++) aux[i] = Cliq.vars[Over.vars[i]];
 
  int* auxcard1 = new int[Over.size];
 
  for(i=0;i<Cliq.size;i++) auxcard1[i] = cards[aux[i]];

  solaps[__solp] = new mpcjtclique(aux, Over.size, auxcard1);

  PrevCliq[__cliq] = WhereIsStore(aux,Over.size);

  PrevSolap[__cliq++] = __solp++;

  delete[] aux;
  
  delete[]  auxcard;

  delete[] auxcard1;
}


void

JtreeTable::add(c Cliq){

  int* auxcard = new int[Cliq.size];
 
  for(int i=0;i<Cliq.size;i++) auxcard[i] = cards[Cliq.vars[i]];

  cliqs[__cliq] = new mpcjtclique(Cliq.vars, Cliq.size, auxcard);

  PrevSolap[__cliq] = -1; // NO hay solapamiento

  PrevCliq[__cliq++] = -1; // Al no tener solapamiento este clique no tiene padre

  delete[] auxcard;
}
*/


double JtreeTable::evaluate(int* vector){
double prob = 0;
double aux, auxval;
 
//for(int i=0;i< NoVars;i++) cout<<vector[i]<<" ";
//cout<<endl;

  for(int i=0;i< NoCliques;i++){
   auxval = cliqs[ordering[i]]->evaluate(vector);
   //cout<<i<<" "<<auxval<<" "<<prob<<endl;
   if ( auxval == 0)
      {
       return -11000000.0; // Retorna 0 cuando uno de los solp tiene ese valor
       cout<<"Some body is out"<<endl;
       }
   prob+=log(auxval);
   
   if (PrevCliq[ordering[i]] != -1){
    aux = solaps[PrevSolap[ordering[i]]]->evaluate(vector);
    if ( aux == 0) return -11000000.0; // Retorna 0 cuando uno de los solp tiene ese valor
    else prob-=log(aux);
    //cout<<i<<" "<<ordering[i]<<" "<<log(auxval)<<" "<<log(aux)<<" "<<prob<<endl;
   }
   //else cout<<i<<" "<<ordering[i]<<" "<<log(auxval)<<" "<<prob<<endl;
  }
  return prob;
}


/*

double JtreeTable::evaluate(int* vector){
double prob = 1;
double aux, auxval;
 
for(int i=0;i< NoVars;i++) cout<<vector[i]<<" ";
cout<<endl;
  for(int i=0;i< NoCliques;i++){
   auxval = cliqs[ordering[i]]->evaluate(vector);
   prob*=auxval;
   if (PrevCliq[ordering[i]] != -1){
    aux = solaps[PrevSolap[ordering[i]]]->evaluate(vector);
    if ( aux == 0) return 0; // Retorna 0 cuando uno de los solp tiene ese valor
    else prob/=aux;
    cout<<i<<" "<<ordering[i]<<" "<<auxval<<" "<<aux<<endl;
   }
   else cout<<i<<" "<<ordering[i]<<" "<<auxval<<endl;
  }
  return prob;
}

/*
void

JtreeTable::MakeJt( cl &O ){

  int i;

  int CantCliques = O.length();

  int auxvector[nvars];



  // mark as NON-visited all the variables

  memset(auxvector, 0, nvars * sizeof(int));



  // -------------------------------------------------------

  // -------- Insertion of the root of the JT --------------

  // ...............  Selection of the root  ...............

  int r = 0, root = RgG.rndN( CantCliques ) + 1;



root = 1;



  for (Pix pi = O.first(); pi != 0; O.next(pi)){

    if(r++ < root ) continue;

	 c *C = O.clique(O(pi));

    add( *C );

	 C -> visit( auxvector ); // mark as visited the root's nodes

    O.prev(pi);

    O.del_after(pi); // remove from O the root

	 break;

  }

  // -------------------------------------------------------

  //

  int msolap, csolap;

  Pix maxPix;

  while( !O.empty() ){

		msolap = 0;

      // find the next clique with maximum overlapping with the visited vars

		for (Pix pi = O.first(); pi != 0; O.next(pi)){

	c *C = O.clique(O(pi));

	csolap = 0;

	for(i=0; i< C -> size; i++){

		csolap += auxvector[ C -> vars[i] ];

		if(csolap > msolap){

	      msolap = csolap; 

			maxPix = pi;

	   }

        }

		}

      // if not overlapping found, include the first clique in the JT

		// and remove it from O

      if(!msolap){ 

        c *C = O.clique( O.remove_front() );

		  add( *C );

		  C -> visit( auxvector ); // mark as visited the root's vars

		}else{ // overlapping case

        c *C = O.clique( O(maxPix) );

		  int solp = 0;

	int WhichSolp[ C -> size ];

        for(int k = 0; k < C -> size; k++)

	  // Note that the overlapping part contains the relative

	  // position of the variables in the clique

	  if( auxvector[ C -> vars[k] ] )

			WhichSolp[ solp++ ] = k;

		  add(C -> vars, C -> size, WhichSolp, solp);

		  C -> visit( auxvector ); // mark as visited the vars of maxPix

		  O.prev( maxPix );

		  O.del_after( maxPix ); // remove from O the root

		}

  }

}

*/
void JtreeTable::PassingFluxesTopDown() {

 mpcmarg *current, *auxmarg, *lambda;

 mpcjtclique *cliq1, *cliq2;

 int i,curindex;

 // Se asume que ordering contiene el ordenamiento apropiado de los cliques

  curindex =0;

  while (curindex<NoCliques-1){
  
  int a =ordering[0];
  current = (cliqs[ordering[curindex]])->getmarg();

  i = curindex+1;

  while (i<NoCliques){

	cliq1 = FindSon(curindex,&i); //Esta funcion actualiza el valor de i

	if (i<NoCliques){

	 cliq2 = solaps[PrevSolap [ordering[i]]];

	 auxmarg = auxmarg->getMaxOver(current,cliq2->getmarg());

	 lambda =(*auxmarg)/(*cliq2->getmarg());

	 cliq1->setmarg( *(lambda) * (*cliq1->getmarg() ));

	 cliq2->setmarg(auxmarg);

	 delete lambda;

     
	}

	i=i+1;

  }

  curindex++;

 }

}



void JtreeTable::PassingFluxesBottomUp() {

 mpcmarg *current, *auxmarg, *lambda;

 mpcjtclique *cliq1, *cliq2;

 int curindex;



  curindex = NoCliques-1;

 while (curindex > 0){
      
        if (PrevCliq[ordering[curindex]] != -1) { // No es una componente independiente  	
          current = cliqs[ordering[curindex]]->getmarg();

	  cliq1 = cliqs[PrevCliq[ordering[curindex]]];

	  cliq2 = solaps[PrevSolap[ordering[curindex]]];

	  auxmarg = auxmarg->getMaxOver(current,cliq2->getmarg());

	  lambda =(*auxmarg)/(*cliq2->getmarg());

	  cliq1->setmarg( *(lambda) * (*cliq1->getmarg()) );

	  cliq2->setmarg(auxmarg);

	  delete lambda;
         
        } 

	curindex--;

 }

}





mpcjtclique* JtreeTable::FindSon(int Father, int* pos){

 while( (*pos<NoCliques)&& (PrevCliq[ordering[*pos]]!= Father) ) (*pos)++;

 if ( *pos == NoCliques) return (mpcjtclique*)0;  //Error

 return cliqs[ordering[*pos]];

}



void JtreeTable::FindBestConf(){

 int i;

 //cout<<NoVars<<" "<<NoCliques<<endl;
 for(i=0; i<NoVars;i++) BestConf[i] = -1;

 for(i=0; i<NoCliques;i++) {

   //cout<<i<<" "<<ordering[i]<<endl;
  mpcmarg* current = cliqs[ordering[i]]->getmarg();

  current->FindBestConf(BestConf);
 
 }
  HighestProb = evaluate(BestConf);

}



 void JtreeTable::SetCliqTo(int i, int* x, int* y){

  cliqs[i]->SetCliqTo(x,y);

 }



 void JtreeTable::ResetCliqTo(int i, int* x, int* y){

  cliqs[i]->ResetCliqTo(x,y);

 }



  JtreeTable::JtreeTable(JtreeTable* other, int _Marca){

  int i;

  NoVars = other->getNoVars();

  NoCliques = other->getNoCliques();

  cliqs =  new mpcjtclique*[NoCliques];

  solaps = new mpcjtclique*[NoCliques-1];



PrevCliq = new int[NoCliques];

PrevSolap = new int[NoCliques];

__cliq  = other->getcliqc();

__solp =  other->getsolpc();



ordering = new int[NoCliques];

BestConf = new int[NoVars];

cards = new unsigned int[NoVars];

for(i=0;i<NoVars;i++) cards[i] = getcard(i) ;

  Marca = _Marca;

  for(i=0;i<NoCliques;i++) {

	 cliqs[i] = new mpcjtclique(other->getCliq(i));

	 PrevCliq[i] =other->getPrevCliq(i);

	 ordering[i] = other->getordering(i);

	 PrevSolap[i] = other->getPrevSolap(i);

	}

	for(i=0;i<__solp;i++) solaps[i] = new mpcjtclique(other->getSolap(i));

  }








// Converts a tree structure into a junction propagation tree structure

void JtreeTable::convert(IntTreeModel* OtherTree){

   int auxvars[2];
   int auxcard[2];
   int i,j,solps;
  
   //We assume the number of cliques of the OtherTree has been already passed during construction
 
for( j=0; j < NoCliques; j++) 
  {

    i =   OtherTree->Queue[j];


if(OtherTree->Tree[i]> -1)    
 { 
   auxvars[0] =  OtherTree->Tree[i];
   auxvars[1] =  i;    
   auxcard[0] =  OtherTree->Card[OtherTree->Tree[i]];
   auxcard[1] =  OtherTree->Card[i];
 
   cliqs[__cliq] = new mpcjtclique(auxvars, 2 ,auxcard);
   solps = OtherTree->Tree[i];
   
   solaps[__solp]  = new mpcjtclique(&solps,1,&OtherTree->Card[solps]); //clique para solapamientos
   PrevCliq[__cliq] = WhereIsStore(&solps,1); // Se busca cual es el padre.
   PrevSolap[__cliq++] = __solp++;   

   //cout<<i<<" "<<auxvars[0]<<" "<<auxvars[1]<<endl;
 }
 else  
 { 
    auxvars[0] =  i;
    auxcard[0] =  OtherTree->Card[i];
    cliqs[__cliq] = new mpcjtclique(auxvars, 1 ,auxcard);
   
    //cout<<i<<" "<<auxvars[0]<<endl;
   
    PrevSolap[__cliq] = -1; // NO hay solapamiento
    PrevCliq[__cliq++] = -1; // Al no tener solapamiento este clique no tiene padre
 }  


  }

}

// Converts a MPM into a junction propagation tree structure

 void JtreeTable::convertMPM(unsigned long** listclusters){

  int* auxvars = new int[NoVars];
  int* auxcard = new int[NoVars];
   
   int a,i,j,solps;
   
   //We assume the number of cliques of the OtherTree has been already passed during construction
   //a=0;
for( i=0; i < NoCliques; i++) 
  {    
    //cout<<"( "<<__cliq<<") :";
    for( j=1; j < listclusters[i][0]+1; j++)  //for( j=1; j < NoVars/NoCliques +1; j++)   
     {
       auxvars[j-1] = listclusters[i][j]; //  = a
       auxcard[j-1] = cards[listclusters[i][j]]; // cards[a]; 
       // cout<<auxvars[j-1]<<" ";
     }
     //cout<<endl;
    //cliqs[__cliq] = new mpcjtclique(auxvars,NoVars/NoCliques,auxcard);      
   cliqs[__cliq] = new mpcjtclique(auxvars,listclusters[i][0] ,auxcard);        
   PrevSolap[__cliq] = -1; // NO hay solapamiento
   PrevCliq[__cliq++] = -1; // Al no tener solapamiento este clique no tiene padre    
}

  delete[] auxvars;
  delete[] auxcard;
}




/*
void JtreeTable::convert(jtree* tree){

 jtclique *auxcliq; 
   int *auxvars;
   int *auxsvars;
   int *auxscard;
   int *auxcards;
   int *solps;
   int i,j;

for( i=0; i < NoCliques; i++) 
  {
   auxcliq = tree->getCliq(tree->getindex(i));
 
   auxvars = auxcliq->getvars();

   auxcards = new int[auxcliq->getsize()];  // La cardinalidad  se extrae de JtreeTable
  
   for (j=0; j<auxcliq->getsize();j++) auxcards[j] = getcard(auxvars[j]);

   cliqs[__cliq] = new mpcjtclique( auxvars,auxcliq->getsize(),auxcards);

 
   if(auxcliq->gets_size()){                 // Si tiene solapamientos
      
      solps = new int(auxcliq->gets_size());  

      auxscard  = new int(auxcliq->gets_size()); 

      auxsvars = auxcliq->getsvars();
     
      for(j=0; j<auxcliq->gets_size(); j++) {
        
        solps[j] = auxvars[auxsvars[j]];       // Se cambian las posiciones relativas por absolutas
      
        auxscard[j] = auxcards[auxsvars[j]];
      } 

      solaps[__solp]  = new mpcjtclique( solps,auxcliq->gets_size(),auxscard); //clique para solapamientos

      PrevCliq[__cliq] = WhereIsStore(solps,auxcliq->gets_size()); // Se busca cual es el padre.

      PrevSolap[__cliq++] = __solp++; 

      delete[] solps;
        
      delete[] auxscard;

   } else {
           
           PrevSolap[__cliq] = -1; // NO hay solapamiento

           PrevCliq[__cliq++] = -1; // Al no tener solapamiento este clique no tiene padre
   
   }
    
   delete[] auxcards; 
}

}

*/









