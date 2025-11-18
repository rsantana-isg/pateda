#include "jtree.h" 
#include "cliques.h" 
 
c **l1, **l2; 
//--------------------------------------------  
 
 
// Order 3, sin solapamiento  
 
void jt1(int nvars){   
 
int CantCliques;  
int s, i; 
 
l1[0]= new c(0,1,2); 
l2[0]= (c *)0; 
 
s=3; 
CantCliques=nvars/3; 
 
for( i=1; i < CantCliques; i++, s+=3){ 
  l1[i]= new c(s,s+1,s+2); 
  l2[i]= (c *)0; 
} 
} 
 
//-------------------------------------------- 
// Order 3, con solapamiento 1 
 
void jt2(int nvars){           
int s, i; 
int CantCliques;  
 
 
  CantCliques=(nvars-1)/2; 
  
  l1[0]= new c(0,1,2); 
  l2[0]= (c *)0; 
 
  s=2; 
  for( i=1; i < CantCliques; i++, s+=2){ 
    l1[i]= new c(s,s+1,s+2); 
    l2[i]= new c(0); 
  } 
 
} 
 
//-------------------------------------------- 
// Order 2, con solapamiento 1 
 
void jt3(int nvars){   
 
int s, i; 
int CantCliques;  
 
 
CantCliques=nvars-1; 
 
  l1[0]= new c(0,1); 
  l2[0]=  (c *)0; 
 
s=1; 
for( i=1; i < CantCliques; i++, s+=1){ 
  l1[i]= new c(s,s+1); 
  l2[i]= new c(0); 
} 
 
 
} 
 
//-------------------------------------------- 
// Order 1, sin solapamiento  
 
void jt4(int nvars){   
 
int CantCliques; 
int s, i; 
s=0; 
CantCliques=nvars; 
for( i=0; i < CantCliques; i++, s+=1){ 
  l1[i]= new c(s); 
  l2[i]= (c *)0; 
} 
} 
 
//--------------------------------------------------------- 
// Order 5, con solapamiento 1 
 
void jt7(int nvars){           
int s, i; 
int CantCliques = (nvars-1)/4; 
  
  l1[0]= new c(0,1,2,3,4); 
  l2[0]= (c *)0; 
 
  s=4; 
  for( i=1; i < CantCliques; i++, s+=4){ 
    l1[i]= new c(s,s+1,s+2,s+3,s+4); 
    l2[i]= new c(0); 
  } 
} 
//.................. 
jtree *set_JT_Function(int NoJT, int nvars, int CantCliques ){ 
int i,s; 
 
l1 = new c*[CantCliques]; 
l2 = new c*[CantCliques]; 
 
//============================================== 
 
switch(NoJT){ 
case 1: jt1(nvars); break; 
case 2: jt2(nvars); break; 
case 3: jt3(nvars); break; 
case 4: jt4(nvars); break; 
case 7: jt7(nvars); break; 
} 
 
//============================================== 
jtree *jt = new jtree(nvars, CantCliques); 
for( i=0; i < CantCliques; i++) 
   if( l2[i] ) 
       jt->add( *l1[i], *l2[i] ); 
   else  
       jt->add( *l1[i] ); 
 
for( i=0; i < CantCliques; i++) // Anhadido por Roberto 
  if( l2[i] ) 
  {delete l1[i];delete l2[i];} 
  else delete l1[i]; 
       
delete []l1; // Anhadido por Roberto 
delete []l2; 
 
return jt; 
} 
 
 
