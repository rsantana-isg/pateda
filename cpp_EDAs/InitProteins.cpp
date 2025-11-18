#include <math.h> 
#include <time.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <iostream> 
#include <fstream> 
#include "auxfunc.h"  
#include "ProteinClass.h"  

  
HPProtein* FoldingProtein;

int  ReadProtein(int a, int modeprotein){  

  int i,k,j,u;
  int sizeProtein,Card;
  double Max;

 int* IntConf;
 
   int  IntConf1[23] = {1,0,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,1,0,1,1,0,0}; // 1 - 20 
   int  IntConf2[23] ={1,0,1,1,0,1,1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,1,0}; // 2 - 17
   int  IntConf3[23] ={0,1,0,1,0,1,0,0,0,1,1,0,1,1,1,0,1,0,0,1,1,0,0}; // 3 - 16
   int  IntConf4[23] ={0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,0}; // 4 - 20
   int  IntConf5[23] ={1,0,1,1,1,1,1,1,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0}; // 5 - 17
   int  IntConf6[23] ={0,0,1,0,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0}; // 6 - 13
   int  IntConf7[23] ={1,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0}; // 7 - 26
   int  IntConf8[23] ={0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,0,1,0,0,0}; // 8 - 16
   int  IntConf9[23] ={1,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,1,1,1,1,0}; // 9 - 15
   int  IntConf10[23] ={0,1,0,1,0,1,1,1,1,1,0,0,1,1,1,0,1,0,1,0,1,0,0}; // 10 - 14
   int  IntConf11[23] ={1,0,1,1,0,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,1,1,0}; // 11 - 15
 


   /*
   int  IntConf1[23] = {1,0,1,0,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0}; // 1 - 11
   int  IntConf2[23] = {1,0,1,0,1,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0}; // 2 - 11 
   int  IntConf3[23] = {1,0,1,0,1,0,1,1,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0}; // 3 - 16
   int  IntConf4[23] = {1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0}; // 4 - 14
   int  IntConf5[23] = {1,0,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0}; // 5 - 14
   int  IntConf6[23] = {0,1,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,0,1,0,0,0,0}; // 6 - 15
   int  IntConf7[23] = {1,0,1,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}; // 7 - 16
   int  IntConf8[23] = {0,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,0,0}; // 8 - 18
   int  IntConf9[23] = {0,1,1,0,1,1,0,0,1,0,1,0,0,0,1,0,1,0,0,0,0,1,0}; // 9 - 18
    */


  int  IntConfa[20] = {0,1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,1,0}; 
  int  IntConfb[24] = {0,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0}; 
  int  IntConfc[25] ={1,1,0,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0};
  int  IntConfd[36] ={1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,1,1};
  int  IntConfe[48] ={1,1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,0,0,0};
  int  IntConff[50] = {0,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,0,1,0,0,0,0,1,0,1,0,1,0,1,0,0} ;

  int  IntConfg[60] ={1,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,1};

  int  IntConfh[64] = {0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0};
  //  int  IntConfi[80] ={1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0};
  int  IntConfj[85] ={0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0};
  int  IntConfk[100]= {1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,1,1,1,1,0,1,0,0};
  int  IntConfl[100] = {1,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,1,1,1,1,1,0,0,0};  // ModelB 
   int  IntConfm[80] ={1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0};
 int  IntConfn[58] = {1,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0};
 int  IntConfo[103] = {1,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,0,0,0,1,1,1,1,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1};
 int  IntConfp[124] = {1,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1,0,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,1,0,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,1,1,1,1,0,1,1,1,0,0,0,0,0,1,1,1,1,0,0,1,0,1,0,1,0,1,0};
 int  IntConfq[136] = {0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,0,1,1,0,1,0,1,0,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1};

  //  unsigned int vector[64] = {0,0,0,2,0,0,0,2,0,0,0,2,0,0,1,2,0,2,2,2,0,2,2,2,0,2,2,2,2,1,2,1,2,0,2,2,2,0,2,0,0,0,2,0,0,0,2,0,0,0,0,0,0,2,1,2,2,0,2,2,2,0,2,2};

 
  //  unsigned int vector60[60] = {0,0,1,1,1,1,2,1,0,2,2,1,1,0,0,0,1,2,0,0,0,0,2,1,0,0,0,0,2,2,2,1,0,2,2,0,0,1,0,2,1,2,2,0,0,2,1,0,1,1,2,2,1,1,0,2,1,2,0,0};
  double eval;

  IntConf = IntConfa;

 if(a==1)
    {
       IntConf = IntConfa;
       sizeProtein = 20;
       Max = 9.0;
    }
   else  if(a==2) 
    {
     IntConf = IntConfb;
     sizeProtein = 24; 
     Max = 9.0;
    }
   else if(a==3) 
    {
    IntConf = IntConfc;
    sizeProtein = 25; 
     Max = 8.0;
    }
   else if(a==4) 
    {
     IntConf = IntConfd;
     sizeProtein = 36; 
     Max = 14.0;
    }
   else if(a==5) 
    {
     IntConf = IntConfe;
     sizeProtein = 48;
     Max = 23.0;
    }
   else if(a==6) 
    {
     IntConf = IntConff;
     sizeProtein = 50; 
     Max = 21.0;
    }
   else if(a==7) 
    {
     IntConf = IntConfg;
     sizeProtein = 60;
     Max = 36.0;
    }
  else if(a==8) 
    {
     IntConf = IntConfh;
     sizeProtein = 64; 
     Max = 42.0;
    }
 else if(a==9) 
    {
     IntConf = IntConfj;
     sizeProtein = 85; 
     Max = 53.0;
    }
   else if(a==10) 
    {
     IntConf = IntConfk;
     sizeProtein = 100;
     Max = 48.0;
    }
  else if(a==11) 
    {
     IntConf = IntConfl;
     sizeProtein = 100; 
     Max = 52.0;
    }
else if(a==12) 
    {
     IntConf = IntConfm;
     sizeProtein = 80; 
     Max = 1000.0;
    }
   else if(a==13) 
    {
     IntConf = IntConfn;
     sizeProtein = 58;
     Max = 1000.0;
    }
  else if(a==14) 
    {
     IntConf = IntConfo;
     sizeProtein = 103; 
     Max = 1000.0;
    }
  else if(a==15) 
    {
     IntConf = IntConfp;
     sizeProtein = 124; 
     Max = 1000.0;
    }
   else if(a==16) 
  {
     IntConf = IntConfq;
     sizeProtein = 136; 
     Max = 1000.0;
    }

 if(a>=20)   sizeProtein = 23;

 /*
 if(a==20)
    {
       IntConf = IntConf1;
        Max = 11.0; 
    }
 else if(a==21)
    {
       IntConf = IntConf2;
        Max = 11.0; 
    }
else if(a==22)
    {
       IntConf = IntConf3;
        Max = 14.0; 
    }
 else if(a==23)
    {
       IntConf = IntConf4;
        Max = 14.0; 
    }
else if(a==24)
    {
       IntConf = IntConf5;
        Max = 14.0; 
    }
 else if(a==25)
    {
       IntConf = IntConf6;
        Max = 15.0; 
    }
else if(a==26)
    {
       IntConf = IntConf7;
        Max = 16.0; 
    }
 else if(a==27)
    {
       IntConf = IntConf8;
        Max = 18.0; 
    }
else if(a==28)
    {
       IntConf = IntConf9;
        Max = 18.0; 
    }
 

 */

 if(a==20)
    {
       IntConf = IntConf1;
        Max = 20.0; 
    }
 else if(a==21)
    {
       IntConf = IntConf2;
        Max = 17.0; 
    }
else if(a==22)
    {
       IntConf = IntConf3;
        Max = 16.0; 
    }
 else if(a==23)
    {
       IntConf = IntConf4;
        Max = 20.0; 
    }
else if(a==24)
    {
       IntConf = IntConf5;
        Max = 17.0; 
    }
 else if(a==25)
    {
       IntConf = IntConf6;
        Max = 13.0; 
    }
else if(a==26)
    {
       IntConf = IntConf7;
        Max = 26.0; 
    }
 else if(a==27)
    {
       IntConf = IntConf8;
        Max = 16.0; 
    }
else if(a==28)
    {
       IntConf = IntConf9;
        Max = 15.0; 
    }
else if(a==29)
    {
       IntConf = IntConf10;
        Max = 14.0; 
    }
 else if(a==30)
    {
        IntConf = IntConf11;
        Max = 15.0; 
    }
 

 // This part is for 
       int nNextMax,NextMax;
       char auxchar;  
       FILE *stream;
       stream = fopen( "2dfmp.dat", "r+" ); 

       fseek(stream,47*(a),0);
       for (j=0;j<23;j++) 
        {
         fscanf( stream, "%c", &auxchar);
         IntConf[j] = 1-(auxchar-48);
         //cout<<auxchar;          
        }
       fscanf( stream, "%c", &auxchar);
       //cout<<" "<<auxchar<<" "; 
       fscanf( stream, "%c", &auxchar);
       //cout<<" "<<auxchar<<" "; 
       fscanf( stream, "%d", &j);
       //cout<<j<<" "; 
       Max = -1.0*j; //OJO QUITAR +1
       fscanf( stream, "%d", &j);
       //cout<<j<<" "; 
       fscanf( stream, "%d", &j); 
       //cout<<j<<" "; 
       NextMax = -1.0*j ; 
       fscanf( stream, "%d\n", &nNextMax);  
       //cout<<nNextMax<<endl; 
       fclose( stream );
       sizeProtein = 23;

       //cout<<modeprotein<<" Inst. "<<a<<endl;

    if(modeprotein == 2)
     {        
       Card = 3;
      
       FoldingProtein = new HPProtein(sizeProtein,IntConf);
       //FoldingProtein->create_contact_weights();  //Create matrix of weights for dynamic function evaluation
      
      
     //FoldingProtein = new HPProtein3D(sizeProtein,IntConf);  
     }
   else if(modeprotein == 3)
     {
      Card = 5;    
      FoldingProtein = new HPProtein3D(sizeProtein,IntConf);  
      Max = 1000;
    }
    else if(modeprotein == 4)
     {
      Card = 3;    
      FoldingProtein = new HPProtein3Diamond(sizeProtein,IntConf);  
      Max = 1000;
    }

  
  FoldingProtein->SetAlpha(1.0);
 
 

  //cout<<"alpha "<< FoldingProtein->alpha;
  //return FoldingProtein;

}      

  /*
eval = FoldingProtein->EvalOnlyVector(sizeProtein,vector);
 cout<< eval<<endl;
 for(i=2;i<vars;i++) vector[i] = 2-vector[i];
eval = FoldingProtein->EvalOnlyVector(sizeProtein,vector);  
 cout<< eval<<endl;
  */

 /*  
 if(a==1)
    {
       IntConf = IntConf1;
       Max = 20;
    }
   else  if(a==2) 
    {
     IntConf = IntConf2;
     Max = 17.0;
    }
   else if(a==3) 
    {
    IntConf = IntConf3;
     Max = 16.0;
    }
   else if(a==4) 
    {
     IntConf = IntConf4;
     Max = 20.0;
    }
   else if(a==5) 
    {
     IntConf = IntConf5;
     Max = 17.0;
    }
   else if(a==6) 
    {
     IntConf = IntConf6;
     Max = 13.0;
    }
   else if(a==7) 
    {
     IntConf = IntConf7;
     Max = 26.0;
    }
   else if(a==8) 
    {
     IntConf = IntConf8;
     Max = 16.0;
    }
   else if(a==9) 
    {
     IntConf = IntConf9;
     Max = 15.0;
    }
  else if(a==10) 
    {
     IntConf = IntConf10;
     Max = 14.0;
    }
   else if(a==11) 
    {
     IntConf = IntConf11;
     Max = 15.0;
    }
   */






