#include "jtree.h" 
#include <string.h> 
#include <stdio.h> 
//----------------------------------------------------------------- 
// clique does not overlapp 
jtclique::jtclique(int *_vars, int _size){ 
  overl      = (marg  *)0; 
  TableCount = (marg **)0; 
  GenSUS     = (SUS  **)0; 
  SValues    = (int   *)0; 
  BValues    = (int   *)0; 
  svars      = (int   *)0; 
  bvars      = (int   *)0; 
  vars       = (int   *)0; 
  cards      = (int   *)0; 
 
 
  size = b_size = _size; 
  s_size = 0; 
 
  NoTables = 1; 
 
  vars = new int[size]; 
  memcpy(vars, _vars, size * sizeof(int) ); 
 
  bvars = new int[b_size]; 
  for(int i = 0; i < b_size; i++) bvars[i] = i; 
  
  BValues = new int[b_size]; 
 
} 
//----------------------------------------------------------------- 
// assume that _s_size != 0 
jtclique::jtclique(int *_vars, int _size, int *_svars, int _s_size){ 
  overl      = (marg  *)0; 
  TableCount = (marg **)0; 
  GenSUS     = (SUS  **)0; 
  SValues    = (int   *)0; 
  BValues    = (int   *)0; 
  svars      = (int   *)0; 
  bvars      = (int   *)0; 
  vars       = (int   *)0; 
  cards      = (int   *)0; 
 
  NoTables   = 0;  // para el caso de no body 
 
  size   = _size; 
  s_size = _s_size; 
  b_size = size - s_size; 
 
  SValues = new int[s_size]; 
 
  vars = new int[size]; 
  memcpy(vars, _vars, size * sizeof(int) ); 
 
  // Note that svars points to numbers relative to vars  
  // not to the whole vector  
  svars = new int[s_size]; 
  memcpy(svars, _svars, s_size * sizeof(int) ); 
 
  if( b_size ){ 
    bvars = new int[b_size]; 
    b_size = 0; 
    for(int i = 0; i < size; i++) 
      if( not_in_svars(i) ) bvars[b_size++] = i; 
      BValues = new int[b_size]; 
  } 
} 
 
//----------------------------------------------------------------- 
void 
jtclique::clear(){ 
int i; 
// return if there is not associated tables 
if(b_size == 0) return;   
for(i = 0; i < NoTables; i++) 
    if( TableCount[i] ) TableCount[i]->clear(); 
} 
//----------------------------------------------------------------- 
// return true if the ith-variable  
// of the clique is NOT overlapped 
int 
jtclique::not_in_svars(int i){ 
  for(int k=0; k<s_size; k++) 
    if( svars[k]==i ) return 0; 
return 1; 
} 
 
//------------------------------------------------------------------ 
jtclique::~jtclique(){ 
        if( overl   ) delete overl; 
	if( SValues ) delete[] SValues; 
	if( BValues ) delete[] BValues; 
	if( TableCount ){ 
	  for(int i=0; i < NoTables; i++) 
	    if( TableCount[i] ) delete TableCount[i]; 
	  delete[] TableCount; 
	} 
	if( GenSUS ){ 
	  for(int i=0; i < NoTables; i++) 
	    if( GenSUS[i]     ) delete GenSUS[i];	 
	  delete[] GenSUS; 
	} 
	if( vars  ) delete[] vars; 
	if( svars ) delete[] svars; 
	if( bvars ) delete[] bvars; 
	if( cards ) delete[] cards; 
} 
//----------------------------------------------------------------- 
void 
jtclique::compute(pop &Pop){ 
  int i; 
 
  // copying cardinalities 
  int *___card = Pop.get_card(); 
  if ( cards ) delete[] cards; 
  cards = new int[size]; 
  for(i = 0; i < size; i++) cards[i] = ___card[ vars[i] ];  
 
  if( b_size == 0 ) return; 
 
  if( s_size == 0 ){ // not overlapped clique 
    if ( TableCount ){ 
      for(i=0; i < NoTables; i++) 
        if( TableCount[i] ) delete TableCount[i]; 
        delete[] TableCount; 
    } 
	 TableCount = new marg*[1]; 
     
    if( GenSUS ){ 
      for(i=0; i < NoTables; i++) 
        if( GenSUS[i]     ) delete GenSUS[i];	 
        delete[] GenSUS; 
    } 
    GenSUS     = new SUS*[1]; 
     
    TableCount[0] = new marg(vars, cards, size); 
 
    TableCount[0] -> compute(&Pop); 
    int sizeTable =  TableCount[0] -> getC_size(); 
    double sample_size = double( TableCount[0] -> getS_size() ); 
	 double *D= new double[sizeTable]; 
	 for(int k = 0; k < sizeTable; k++) 
		D[k] = (TableCount[0] -> get(k)) / sample_size; 
	 GenSUS[0] = new SUS( sizeTable, D, __TAMANO_SUS__ ); 
	 delete []D; 
    return; 
  } 
 
  NoTables = 1; 
  for(i = 0; i < s_size; i++) 
	NoTables *= cards[ svars[i] ]; 
 
  if ( TableCount ){ 
    for(i=0; i < NoTables; i++) 
      if( TableCount[i] ) delete TableCount[i]; 
      delete[] TableCount; 
  } 
  TableCount = new marg*[NoTables]; 
     
  if( GenSUS ){ 
    for(i=0; i < NoTables; i++) 
      if( GenSUS[i]     ) delete GenSUS[i];	 
      delete[] GenSUS; 
  } 
  GenSUS     = new SUS*[NoTables]; 
 
  int* __bvars=new int [b_size]; 
  int* __bcard=new int[b_size]; 
 
  for(i = 0; i < b_size; i++){ 
	 // __bvars contains the true overlapped values, 
	 // not the indices as bvars does 
	 __bvars[i] = vars [ bvars[i] ]; 
	 __bcard[i] = cards[ bvars[i] ]; 
  } 
 
  int* __scard = new int[s_size]; 
  int* __svars = new int[s_size]; 
 
  for(i = 0; i < s_size; i++){ 
	 // __svars contains the true overlapped values, 
	 // not the indices as svars does 
	 __svars[i] = vars [ svars[i] ]; 
	 __scard[i] = cards[ svars[i] ]; 
  } 
 
  // to make conversions 
  if ( overl ) delete overl; 
  overl = new marg(svars, __scard, s_size); 
 
  // caso 0 
  overl -> getCase( 0, SValues); 
  TableCount[0]      = new marg(__bvars, __bcard, b_size); 
  TableCount[0]      -> compute(&Pop, __svars, SValues, s_size); 
  int sizeTable      = TableCount[0] -> getC_size(); 
  double sample_size = double( TableCount[0] -> getS_size() ); 
  double* D= new double [sizeTable]; 
  for(int k = 0; k < sizeTable; k++) 
	 D[k] = (TableCount[0] -> get(k)) / sample_size; 
  GenSUS[0] = new SUS( sizeTable, D, __TAMANO_SUS__ ); 
 
  for(i = 1; i < NoTables; i++){ 
	 TableCount[i] = new marg(__bvars, __bcard, b_size); 
	 overl -> getCase( i, SValues); 
	 TableCount[i] -> compute(&Pop, __svars, SValues, s_size); 
 
	 for(int k = 0; k < sizeTable; k++) 
		D[k] = (TableCount[i] -> get(k)) / sample_size; 
	 GenSUS[i] = new SUS( sizeTable, D, __TAMANO_SUS__ ); 
  } 
delete []__bvars; 
delete []__bcard; 
delete []__scard; 
delete []__svars; 
delete [] D; 
 
} 
 
//------------------------------------------------------- 
void 
jtclique::generate(int *vector){ 
int i, ind; 
 
if ( b_size == 0 ) return; 
 
ind = 0; 
if( s_size > 0 ){ 
  for(i=0; i < s_size; i++) 
    SValues[i] = vector[ vars[ svars[i] ] ]; 
  ind = overl -> index(SValues); 
} 
 
int sampledPoint = GenSUS[ind ] -> sample(); 
 
TableCount[ind] -> getCase(sampledPoint, BValues); 
 
for(i=0; i < b_size; i++) 
  vector[ vars[ bvars[i] ] ] =  BValues[i]; 
 
} 
 
//---------------------------------------------- 
// para experimento de Most Probable Configuration 
double  
jtclique::ReturnProbOfConfiguration(int *vector){ 
int i, ind; 
 
if ( b_size == 0 ) return 1.0; 
 
ind = 0; 
if( s_size > 0 ){ 
  for(i=0; i < s_size; i++) 
    SValues[i] = vector[ vars[ svars[i] ] ]; 
  ind = overl -> index(SValues); 
} 
 
for(i=0; i < b_size; i++) 
    BValues[i] = vector[ vars[ bvars[i] ] ]; 
 
return ( TableCount[ind] -> get(BValues) ) 
            / double( TableCount[ind] -> getS_size() ); 
 
} 
//------------------------------------------------------- 
void 
jtclique::shortPrint(){ 
int i; 
printf("vars: "); 
for(i=0; i<size;i++)printf("%d ",vars[i]); 
printf("\nsvars(indices): "); 
for(i=0; i<s_size;i++)printf("%d ",svars[i]); 
printf("\nsvars: "); 
for(i=0; i<s_size;i++)printf("%d ",vars[svars[i]]); 
printf("\nsize: %d, s_size: %d, b_size: %d, NoTables: %d\n",size,s_size,b_size,NoTables); 
} 
 
void 
jtclique::print(){ 
int i, j; 
 
printf("\n"); 
if( b_size == 0 ){ 
  printf("absorbed clique.\n"); 
  return; 
} 
 
if(s_size == 0){ 
  printf("not overlapped clique.\n"); 
  TableCount[0]->print(); 
  return; 
} 
 
for(i=0; i < NoTables; i++){ 
	overl -> getCase(i, SValues); 
	for(j=0; j < s_size; j++) 
	   printf("%d", SValues[j]); 
	printf("\n"); 
	TableCount[i]->print(); 
} 
} 
 
void 
jtclique::StructPrint(){ 
  if(s_size) printf("< "); 
  for(int j=0; j < s_size; j++) printf("%d ",vars[ svars[j] ]); 
  if(s_size) printf("> "); 
  for(int i=0; i < b_size; i++) printf("%d ",vars[ bvars[i] ]); 
} 
 
//================================================================= 
jtree::jtree(int _nvars, int _NoCliques){ 
nvars = _nvars; 
 
NoCliques = _NoCliques; 
cliqs = new jtclique*[NoCliques]; 
__cliq  = 0; 
 
ordering = new int[NoCliques]; 
for(int i=0; i < NoCliques; i++) ordering[i]=i; 
} 
 
jtree::~jtree(){ 
int i; 
  for(i = 0; i < NoCliques; i++) 
    if( cliqs[i] )  delete cliqs[i]; 
 
  delete[] cliqs; 
  delete[] ordering; 
} 
 
void 
jtree::clear(){ 
  for(int i=0; i < NoCliques; i++) cliqs[i]->clear(); 
} 
 
void 
jtree::add(int *vars, int size, int *Over, int over_size){ 
  cliqs[__cliq++] = new jtclique(vars, size, Over, over_size); 
} 
 
void 
jtree::add(int *vars, int size){ 
  cliqs[__cliq++] = new jtclique(vars, size); 
} 
 
void 
jtree::add(c Cliq, c Over){ 
  cliqs[__cliq++] = new jtclique(Cliq.vars, Cliq.size, Over.vars, Over.size); 
} 
 
void 
jtree::add(c Cliq){ 
  cliqs[__cliq++] = new jtclique(Cliq.vars, Cliq.size); 
} 
 
void  
jtree::compute(pop &Pop){ 
  for(int i = 0; i < NoCliques; i++) cliqs[i] -> compute(Pop); 
} 
 
void 
jtree::generate(int *vector){ 
  for(int i = 0; i < NoCliques; i++) cliqs[ordering[i]] -> generate(vector); 
} 
 
void 
jtree::generate(pop &Pop){ 
int i, j, k; 
int* vector = new int[nvars]; 
int N = Pop.Getsize(); 
 
memset(vector,0,nvars*sizeof(int)); 
for(j = 0; j < N; j++){ 
  for(k = 0; k < NoCliques; k++) cliqs[ordering[k]] -> generate(vector); 
  for(i = 0; i < nvars; i++) Pop.set( j, i, vector[i]); 
} 
delete []vector; 
} 
 
void jtree::print(){ 
  printf("\n"); 
  for(int i=0; i < NoCliques; i++){ 
    printf("\nClique #%d\n",i); 
    cliqs[i]->print(); 
  } 
} 
 
void jtree::shortPrint(){ 
  for(int i=0; i<NoCliques; i++){ 
    printf("\n\nClique No: %d\n",i); 
    cliqs[i]->shortPrint(); 
  } 
} 
 
void jtree::StructPrint(){ 
  for(int i=0; i<NoCliques; i++){ 
    cliqs[i]->StructPrint(); 
    printf("\n"); 
  } 
} 
 
//-------------------------------------------------------------------- 
/* 
void  
jtree::MakeJt( cl &O ){ 
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
//---------------------------------------------------------------- 
// para experimento de Most Probable Configuration 
double  
jtree::ReturnProbOfConfiguration(int *vector){ 
double P = 1; 
for(int i = 0; i < NoCliques; i++) 
    P *= cliqs[ordering[i]] -> ReturnProbOfConfiguration(vector); 
return P; 
} 
 
 
 
