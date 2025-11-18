#include <stdio.h> 
#include <memory.h> 
#include "auxfunc.h"
#include "FProteinClass.h" 
 
 


void HPProtein::SetInitMoves() // Init all possible ordering of legal moves
{
    moves[0][0] = 0; moves[1][0] = 0;   moves[2][0] = 1; moves[3][0] = 1;  moves[4][0] = 2; moves[5][0] = 2;
    moves[0][1] = 1; moves[1][1] = 2;   moves[2][1] = 0; moves[3][1] = 2;  moves[4][1] = 0; moves[5][1] = 1;
    moves[0][2] = 2; moves[1][2] = 1;   moves[2][2] = 2; moves[3][2] = 0;  moves[4][2] = 1; moves[5][2] = 0;
}

void HPProtein::SetInitPos(int** aPos) // Init the first positions of the grid
{
    aPos[0][0] = 0;
    aPos[0][1] = 0;
    aPos[1][0] = 1;
    aPos[1][1] = 0;
}


void HPProtein::SetInitPosAt_i(int** aPos, int i) // Init the first positions of the grid
{
    aPos[i][0] = 0;
    aPos[i][1] = 0;
    aPos[i+1][0] = 1;
    aPos[i+1][1] = 0;
}


void  HPProtein::CreatePos() //Creates the first positions of the grid
 {
  int i;
  
 Pos = new int*[sizeProtein];
 NewPos = new int*[sizeProtein];

  for (i=0;i<sizeProtein;i++)
   {
    Pos[i] = new int[2];
    NewPos[i] = new int[2];
   } 

  SetInitPos(Pos); 
  SetInitPos(NewPos); 
  SetInitMoves();
 }


void  HPProtein::DeletePos() // Deletes the grid
 {
  int i;
  for (i=0;i<sizeProtein;i++) 
   {
     delete[] Pos[i];
     delete[] NewPos[i];
   }
  delete[] Pos;   
  delete[] NewPos;   
 }


void HPProtein::InitClass(int sizeP, int* HConf, int nmvs, int mvscomb)
{
  int i;
  sizeProtein = sizeP;
  IntConf =  HConf;
  nmoves = nmvs;
  combmoves = mvscomb;
  moves = new unsigned int*[combmoves];
  for (i=0;i<combmoves;i++) moves[i] = new unsigned int[nmoves];
  gains = new double[3*sizeProtein];
  svals = new int[sizeProtein];
  bestallpos  = new int[sizeProtein];
  newvector = new unsigned int[sizeProtein];
  auxnewvector = new unsigned int[sizeProtein];
  statvector = new  int[sizeProtein];
  Memory = new int [3*sizeProtein];
  for (i=0;i<sizeProtein;i++) statvector[i] = 0;
 
}

HPProtein::HPProtein(int sizeP, int* HConf, int nmvs, int mvscomb)
{
  InitClass(sizeP,HConf,3,6);
}


HPProtein::HPProtein(int sizeP, int* HConf)
{
  InitClass(sizeP,HConf,3,6);
  CreatePos();
  //cout<<"Finished here"<<endl;
}


HPProtein::HPProtein(int sizeP, int* HConf, double pMax, double pNextMax, int pnNextMax)
{
  Max = pMax;
  NextMax =  pNextMax; 
  nNextMax = pnNextMax;
  InitClass(sizeP,HConf,3,6);
  CreatePos();
  //cout<<"Finished here"<<endl;
}



HPProtein::~HPProtein()
{
  int i;
  DeletePos();
  delete[]  gains;
  delete[]  svals;
  delete[]  bestallpos;
  delete[]  newvector;
  delete[]  auxnewvector;
  delete[]  statvector;
  delete[]  Memory;
  for (i=0;i<combmoves;i++) delete[] moves[i];
  delete[] moves;
}


void  HPProtein::FindPos(int sizeChain, unsigned* vector) //Translates the vector to the  positions in the grid 
 {
  int i;
  SetInitPos(Pos);
   
   for (i=2;i<sizeChain;i++)
     {
      
       if(Pos[i-1][1]==Pos[i-2][1])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] + (Pos[i-1][0] - Pos[i-2][0]);
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               Pos[i][0] = Pos[i-1][0] +  (Pos[i-1][0] - Pos[i-2][0]); 
               Pos[i][1] = Pos[i-1][1];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] - (Pos[i-1][0] - Pos[i-2][0]);
             }
	 }
        
       if(Pos[i-1][0]==Pos[i-2][0])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0]  -  (Pos[i-1][1] - Pos[i-2][1]);
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               Pos[i][1] = Pos[i-1][1] +  (Pos[i-1][1] - Pos[i-2][1]); 
               Pos[i][0] = Pos[i-1][0];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0] +  (Pos[i-1][1] - Pos[i-2][1]);
             }
	 }
       //cout<<i<<"-->"<<Pos[i][0]<<" "<<Pos[i][1]<<endl;  
     }
   
}


void HPProtein::TranslatePos(int sizeChain, unsigned* newvector) // Translate an array of grid positions to a vector of movements
{

  int i;
  newvector[0] = 0;
  newvector[1] = 0;
 
  
 for (i=2;i<sizeChain;i++)
     {
        if(Pos[i-1][1]==Pos[i-2][1])
         {
           if (  (Pos[i][0] == Pos[i-1][0])  && (Pos[i][1] == Pos[i-1][1] + (Pos[i-1][0] - Pos[i-2][0]))  )                  //UP MOVE
	     newvector[i] = 0;
           else if ( (Pos[i][0] == Pos[i-1][0]  +  (Pos[i-1][0] - Pos[i-2][0])) && (Pos[i][1] == Pos[i-1][1]))   //FORWARD MOVE
             newvector[i] = 1;
           else if  ( (Pos[i][0] ==  Pos[i-1][0]) && (Pos[i][1] == Pos[i-1][1] - (Pos[i-1][0] - Pos[i-2][0]))) // DOWN MOVE
              newvector[i] = 2;
          }

        if(Pos[i-1][0]==Pos[i-2][0])
         {
           if (  (Pos[i][1] == Pos[i-1][1])  && (Pos[i][0] == Pos[i-1][0] - (Pos[i-1][1] - Pos[i-2][1]))  )                  //UP MOVE
	     newvector[i] = 0;
           else if ( (Pos[i][1] == Pos[i-1][1]  +  (Pos[i-1][1] - Pos[i-2][1])) && (Pos[i][0] == Pos[i-1][0]))   //FORWARD MOVE 
             newvector[i] = 1;
           else if  ( (Pos[i][1] ==  Pos[i-1][1]) && (Pos[i][0] == Pos[i-1][0] + (Pos[i-1][1] - Pos[i-2][1]))) // DOWN MOVE
              newvector[i] = 2;
          }
    
    }
}






  
int  HPProtein::CheckFree(int sizeChain, int** aPos, int P1, int P2)   // Checks if coordinate at_i_Pos is occuppied in the grid
{ 
int i,Found;

 i = 0;
 Found = 0;

 while(i<sizeChain && !Found)
 {
  Found = ( (aPos[i][0] == P1) && (aPos[i][1] == P2));
  i = i + 1; 
 }
 return (Found==0);
}
 

int  HPProtein::CheckValidity(int at_i_Pos, int** aPos) //Checks if the chain is connected at position at_i_Pos
 { 
   int resp  = 1;
   if(at_i_Pos > 0)   resp = ( (abs(aPos[at_i_Pos][0] - aPos[at_i_Pos-1][0]) +  abs(aPos[at_i_Pos][1] - aPos[at_i_Pos-1][1])) == 1);
   return resp;
 } 




void HPProtein::AssignPos(int sizeChain, int** Pos, int** OtherPos)     // Copy the position of one grid into another
 {
  int i;
   for (i=0;i<sizeChain;i++)
   {
    OtherPos[i][0] = Pos[i][0];
    OtherPos[i][1] = Pos[i][1];
    }
 }

void HPProtein::PrintPos(int sizeChain, int** aPos)       // Prints all or part of the protein's grid
{
  int i;

  for (i=0;i<sizeChain;i++) 
    { 
      cout<<i<<"-->"<<aPos[i][0]<<" "<<aPos[i][1]<<endl;  
    } 
  //cout<<"Pass one"<<endl;
  //cout<<"Pass two"<<endl;
}

 //cout<<"i "<<i<<" tot "<<tot<<" cut "<<cutoff<<endl;   

int HPProtein::PullMoves(int sizeChain,int at_i, int sval) // Given a chain of molecules, and a position makes a pull move 
{
int i,resp,i_minus_1;
 int L1,L2,C1,C2;

AssignPos(sizeChain,Pos,NewPos);

// cout<<"Here it is "<<at_i<<"  "<<sval<<endl;

if(Pos[at_i][0]==Pos[at_i+1][0])
 {
  L1 = Pos[at_i][0] + sval*(Pos[at_i+1][1]-Pos[at_i][1]);
  L2 = Pos[at_i+1][1];
  C1 = Pos[at_i][0] + sval*(Pos[at_i+1][1]-Pos[at_i][1]);
  C2 = Pos[at_i][1];
 } 
else if(Pos[at_i][1]==Pos[at_i+1][1])
 {
  L1 = Pos[at_i+1][0];
  L2 = Pos[at_i][1] + sval*(Pos[at_i+1][0] - Pos[at_i][0]);
  C1 = Pos[at_i][0];
  C2 = Pos[at_i][1] + sval*(Pos[at_i+1][0] - Pos[at_i][0]);
 }


resp = CheckFree(sizeChain, Pos, L1, L2);
//cout<<" L1 "<<L1<<" L2 "<<L2<<" C1 "<<C1<<" C2 "<<C2<<" resp "<<resp<<endl;
if(resp==0)  return resp;  // Position L is not free
else if(at_i==0)
 {
    NewPos[at_i][0] = L1;
    NewPos[at_i][1] = L2;
    resp = 1;
    return  resp;
 }
 
resp =  CheckFree(sizeChain, Pos, C1, C2);;
  
 if(resp == 0)  // Position C is not free
  {
   i_minus_1 = ( (Pos[at_i-1][0]==C1) && (Pos[at_i-1][1]==C2));
   if (i_minus_1 == 1) // Molecule i-1 is at position C  
     {
       NewPos[at_i][0] = L1;
       NewPos[at_i][1] = L2;
       resp = 1;
     }
     return resp; // if C is not free and is not occupied by i-1 then return.
  }

  NewPos[at_i][0] = L1;
  NewPos[at_i][1] = L2;
  NewPos[at_i-1][0] = C1;
  NewPos[at_i-1][1] = C2;
  i = at_i-2;

  while ( (i>=0) && !(CheckValidity(i,NewPos)==0))
  {
   NewPos[i][0] = Pos[i+2][0];
   NewPos[i][1] = Pos[i+2][1];
   i = i-1;  
  } 

  // PrintPos(sizeChain,NewPos);
  return resp;
}





double HPProtein::EvalChain(int sizeChain, unsigned int* vector, int** aPos)
{
  // Given a chain of molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 

  int i,j,Collisions,Overlappings;
  double result;
 Collisions = 0;
 Overlappings = 0;
  

   for (i=2;i<sizeChain;i++)
     {
       if(aPos[i-1][1]==aPos[i-2][1])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               aPos[i][0] = aPos[i-1][0];
               aPos[i][1] = aPos[i-1][1] +  (aPos[i-1][0] - aPos[i-2][0]);
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               aPos[i][0] = aPos[i-1][0] +  (aPos[i-1][0] - aPos[i-2][0]); 
               aPos[i][1] = aPos[i-1][1];
             }
	   else                                //DOWN  MOVE
	     {
               aPos[i][0] = aPos[i-1][0];
               aPos[i][1] = aPos[i-1][1] -  (aPos[i-1][0] - aPos[i-2][0]);
             }
	 }
        
       if(aPos[i-1][0]==aPos[i-2][0])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               aPos[i][1] = aPos[i-1][1];
               aPos[i][0] = aPos[i-1][0] -  (aPos[i-1][1] - aPos[i-2][1]);
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               aPos[i][1] = aPos[i-1][1] +  (aPos[i-1][1] - aPos[i-2][1]); 
               aPos[i][0] = aPos[i-1][0];
             }
	   else                                //DOWN  MOVE
	     {
               aPos[i][1] = aPos[i-1][1];
               aPos[i][0] = aPos[i-1][0] +  (aPos[i-1][1] - aPos[i-2][1]);
             }
	 }
     

        for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1])) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])) (Collisions)++;        
          }    
        }
     

     }
 
  result = (Collisions/(1.0+Overlappings));
  return (result);

 }





double HPProtein::EvalChain(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 
 int i,j,Collisions,Overlappings;

 Collisions = 0;
 Overlappings = 0;
 double result;

 for (i=2;i<sizeChain;i++)
   for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1])) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])) (Collisions)++;        
          }    
        }

     

 result = (Collisions/(1.0+Overlappings));
  return (result);
}



double HPProtein::EvalChainModel(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 
 int i,j,Collisions,Overlappings,numbint,numboptint;
 Collisions = 0; 
 Overlappings = 0;
 numbint=0;
 numboptint = 0;
 double result;

 for (i=2;i<sizeChain;i++)
   for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
	  
             if ( ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) ||  ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1]))  || ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1))  || ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])))  
	       {
                 Collisions+=2;
                 numbint++;
                 if  (optmatrix[j][i]>-1) numboptint++;
               }
          }    
         else 
         {
	    if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)--;
            if ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1])) (Collisions)--;
            if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)--;
            if ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])) (Collisions)--;        
          }   
        }

     

   result = (Collisions/(1.0+Overlappings));
   energymatrix[numbint][numboptint] += result;
   freqmatrix[numbint][numboptint] ++; 
   totnumboptint += numboptint;
   //cout<<numbint<<" "<<numboptint<<" "<<result<<endl;
  return (result);
}


double HPProtein::ContactOrderVector(int sizeChain, int* NC, unsigned int* vector)   //Evaluates the chain receiving only the  vector
{
  FindPos(sizeChain,vector);
  return ContactOrderChain(sizeChain,NC,Pos);
} 


int  HPProtein::CompactnessVector(int sizeChain, unsigned int* vector)   //Evaluates the chain receiving only the  vector
{
  FindPos(sizeChain,vector);
  return Compactness(sizeChain,Pos);
} 

double HPProtein::ContactOrderChain(int sizeChain, int* NC, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Contact order of the best solution
 int Corder = 0; 
 int i,j;
 int Collisions = 0;
 double result;

 for (i=2;i<sizeChain;i++)
   {
     //cout<<i <<" pos "<<aPos[i][0]<<" "<<aPos[i][1]<<endl;

   for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
         
         if (IntConf[i]==0  && IntConf[j]==0) 
          {
	    if ( ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) ||  ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1]))  || ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1))  || ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])))  
              {
	          Corder += (i-j);
                  Collisions ++;
                  //cout<<j<<" "<<i<<" "<<i-j<<" "<<aPos[j][0]<<" "<<aPos[j][1]<<" "<<aPos[i][0]<<" "<<aPos[i][1]<<" "<<Collisions<<endl; 
              }  
          }    
        }
   }
 *NC = Collisions;
 result = (Corder/(1.0*Collisions));
  return (result);
}




int HPProtein::Compactness(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Compactness  of the best solution
 int i;
 int minx = 0;  int maxx = 0;
 int miny = 0;  int maxy = 0;

 for (i=0;i<sizeChain;i++)
   {
     if(aPos[i][0]<minx) minx = aPos[i][0];
     if(aPos[i][1]<miny) miny = aPos[i][1];
     if(aPos[i][0]>maxx) maxx = aPos[i][0];
     if(aPos[i][1]>maxy) maxy = aPos[i][1];
   }
   if(maxx-minx>maxy-miny)  return (maxx-minx);
    return (maxy-miny);
}




double HPProtein::EvalChainLong(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 
 int i,j,Collisions,Overlappings;

 Collisions = 0;
 Overlappings = 0;
 double result;

 for (i=2;i<sizeChain;i++)
   for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0  && (i-j)>3) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)+=2;
            if ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1])) (Collisions)+=2;
            if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)+=2;
            if ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])) (Collisions)+=2;        
          }    
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;        
          }    
        }

     

 result = (Collisions/(2.0*(1.0+Overlappings)));
  return (result);
}


double HPProtein::EvalOnlyVector(int sizeChain, unsigned int* vector)   //Evaluates the chain receiving only the  vector
{
  FindPos(sizeChain,vector);
  return EvalChain(sizeChain,Pos);
} 



double HPProtein::EvalOnlyVectorModel(int sizeChain, unsigned int* vector)   //Evaluates the chain receiving only the  vector
{
  FindPos(sizeChain,vector);
  return EvalChainModel(sizeChain,Pos);
} 

double HPProtein::EvalOnlyVectorLong(int sizeChain, unsigned int* vector)   //Evaluates the chain receiving only the  vector
{
  FindPos(sizeChain,vector);
  return EvalChainLong(sizeChain,Pos);
} 



void HPProtein::CollectStat(int sizeChain, unsigned int* vector, double** freqvect)   //Evaluates the chain receiving only the  vector
{
  FindPos(sizeChain,vector);
  return FindFreq(sizeChain,Pos,freqvect);
} 


void HPProtein::FindFreq(int sizeChain, int** aPos,double** freqvect)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 
 int i,j,Collisions,Overlappings,numbint,numboptint;

 for (i=2;i<sizeChain;i++)
   for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if (IntConf[i]==0  && IntConf[j]==0) 
           {
	         if ( ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) ||  ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1]))  || ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1))  || ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])))  
	       {
		 // cout<<i<<" "<<j<<" "<<optmatrix[j][i]<<endl;
                                if(optmatrix[j][i]!=-1) freqvect[1][optmatrix[j][i]]++; 
               }
          }    
        }
}

int HPProtein::VectorContactMatrix(int sizeChain, unsigned int* vector, double** freqvect)   //Evaluates the chain receiving only the  vector
{
  FindPos(sizeChain,vector);
  return CreateContactMatrix(sizeChain,Pos,freqvect);
} 


int HPProtein::CreateContactMatrix(int sizeChain, int** aPos,double** freqvect)   //Evaluates the chain but using the grid positions and not the vector
{
 int i,j,a;
 a = 0;

 for (i=2;i<sizeChain;i++)
   for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if (IntConf[i]==0  && IntConf[j]==0) 
           {
	         if ( ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] - 1)) ||  ( (aPos[i][0]==aPos[j][0] + 1) && (aPos[i][1]==aPos[j][1]))  || ( (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1] + 1))  || ( (aPos[i][0]==aPos[j][0] - 1) && (aPos[i][1]==aPos[j][1])))  
	       {
                                optmatrix[j][i] =a; 
                                freqvect[0][a] = i-j;
                                freqvect[1][a] = 0;
                                a++;
                                //cout<<i<<" "<<j<<" "<<a<<" "<<i-j<<" "<<a<<endl;
               }
          }    
        }
 return a;
}



double  HPProtein::FindNewEval(int sizeChain, int pos, int sval, int* eval_local) //Evaluates a vector perturbed at a given position 
{
  int legal;
  double neweval; 
  legal = PullMoves(sizeChain,pos,sval);

    if(legal!=0)
	  { 
            neweval =  EvalChain(sizeChain,NewPos); 
            (*eval_local)++;
          }
   else neweval = -1;

   return neweval;
 
}


double  HPProtein::TabuFindNewEval(int sizeChain, int pos, int sval, int* eval_local) //Evaluates a vector perturbed at a given position 
{
  int legal;
  double neweval; 

  if (Memory[(sval+1)*sizeChain/2 + pos] > 0) return -1;

  legal = PullMoves(sizeChain,pos,sval);

    if(legal!=0)
	  { 
            neweval =  EvalChain(sizeChain,NewPos); 
            (*eval_local)++;
          }
   else neweval = -1;

   return neweval;
 
}



double  HPProtein::FindIsLegal(int sizeChain, int pos, int sval, int* eval_local, double currentEval) //Evaluates a vector perturbed at a given position 
{
  int legal;
  double neweval; 
  legal = PullMoves(sizeChain,pos,sval);

  if(legal!=0) neweval = currentEval;	 
  else neweval = -1;

   return neweval;
 
}
 
    
double  HPProtein::FindNewEvalSimple(int sizeChain, int pos, int sval, int* eval_local, unsigned int* currentvector) //Evaluates a vector perturbed at a given position 
{
  int a;
  double neweval; 
  
  a = currentvector[pos];
  if(sval == -1)
    {
      if (currentvector[pos] == 0) currentvector[pos] = 1;
      else  if (currentvector[pos] == 1) currentvector[pos] = 2;
      else currentvector[pos] = 0;              
    }
  else
    {
      if (currentvector[pos] == 0) currentvector[pos] = 2;
      else  if (currentvector[pos] == 1) currentvector[pos] = 0;
      else currentvector[pos] = 1;    
    }
  neweval = EvalOnlyVector(sizeChain,currentvector);  
  currentvector[pos] = a;
      (*eval_local)++;
     
   return neweval;
 
}



void HPProtein::SetGainsAndVals(int pos, double neweval1, double neweval2, double currentEval) //Sets the perturbation best value
{    

   
      if (neweval1 > neweval2)
	  { 
            gains[pos] = neweval1 - currentEval;
            svals[pos] = -1;
          }
        else if (neweval2 > neweval1)
	  { 
            gains[pos] = neweval2 - currentEval;
            svals[pos] = 1;
          }
	else if (neweval1==-1)
          { 
            gains[pos] = -1;
            svals[pos] = 0;
          }
        else
	  { 
            if(myrand()>0.5) 
              { 
               gains[pos] = neweval1 - currentEval;
               svals[pos] = -1;
              }
            else 
	     { 
               gains[pos] = neweval2 - currentEval;
               svals[pos] = 1;
             }
          }

      //  cout<<pos<<" gains  "<<gains[pos]<<" svals "<<svals[pos]<<endl;
        
}       



    



int  HPProtein::SetNewBestPos(int sizeChain, int* maxval, double* currentEval,int maxvalconstraint) // Modifies the best current grid in local optimizer
{    
  int i,howmany, bestpos, whichone, legal;
   
    howmany = 1;
    bestpos = 0;
    bestallpos[0] = 0;

    for (i=1;i<sizeChain-1;i++)
      {
	if(gains[i]>gains[bestpos])
	  {
            bestpos = i;
            howmany = 1;
            bestallpos[0] = bestpos;
          }
        else if(gains[i]==gains[bestpos])    bestallpos[howmany++] = i;           
      }            
    
    //cout<<"howmany"<<howmany<<endl;
    if(gains[bestpos]  > 0 || ((*maxval)<maxvalconstraint && fabs(gains[bestpos]) < 0.0000001)) 
     {
       if(fabs(gains[bestpos])  < 0.0000001)  (*maxval)++;
       if (howmany==1) whichone = 0;
       else  whichone = randomint(howmany);  
       //cout<<"Grid Pos is "<<endl;
       //PrintPos(sizeChain,Pos);
       legal = PullMoves(sizeChain,bestallpos[whichone],svals[bestallpos[whichone]]);
       //cout<<"whichone "<<whichone<<" bestpos "<<bestallpos[whichone]<<" svals "<<svals[bestallpos[whichone]]<<" legal "<<legal<<endl;
       //cout<<"pos "<<bestallpos[whichone]+2<<" svals " <<svals[bestallpos[whichone]]<<endl;
       //cout<<"Grid NewPos is "<<endl;
       //PrintPos(sizeChain,NewPos);
       
       AssignPos(sizeChain,NewPos,Pos);
       TranslatePos(sizeChain,newvector);
       FindPos(sizeChain,newvector);
       
       /*     
  cout<<"howmany"<<howmany<<" "<<gains[bestpos]<<endl;
 cout<<"whichone "<<whichone<<" bestpos "<<bestallpos[whichone]<<" svals "<<svals[bestallpos[whichone]]<<" legal "<<legal<<endl;
       for(int l=0;l<sizeChain;l++) cout<<newvector[l]<<" ";
             cout<<endl;  
  cout<<"CurrentVal "<<*currentEval<<" NewCurrentVal "<< (*currentEval) + gains[bestpos]<<" realeval "<<EvalOnlyVector(sizeChain,newvector)<<endl;  
       */
       
       (*currentEval) = (*currentEval) + gains[bestpos]; 
       //cout<<"CurrentVal "<<*currentEval<<" NewCurrentVal "<< (*currentEval) + gains[bestpos]<<" realeval "<<EvalOnlyVector(sizeChain,newvector)<<endl; 
       //cout<< "Once converted NewPos is "<<endl;
       //PrintPos(sizeChain,Pos);
     }
     else whichone = -1;

 return whichone;
}


int  HPProtein::TabuSetNewBestPos(int sizeChain, int* maxval, double* currentEval,int maxvalconstraint,int duration) // Modifies the best current grid in local optimizer
{    
  int i,howmany, bestpos, whichone,legal,bpos;
   
    howmany = 1;
    bestpos = 0;
    bestallpos[0] = 0;

    for (i=0;i<2*sizeChain-1;i++)
      {
	if(gains[i]>gains[bestpos])
	  {
            bestpos = i;
            howmany = 1;
            bestallpos[0] = bestpos;
          }
        else if(gains[i]==gains[bestpos])    bestallpos[howmany++] = i;
             
      }            
    
    if(gains[bestpos]>-1 && (*maxval)<maxvalconstraint) 
     {
        (*maxval)++;
       if (howmany==1) whichone = 0;
       else  whichone = randomint(howmany);  

       bpos =  bestallpos[whichone];  
       Memory[bpos] = duration;  
    
       if(bpos<sizeChain)  legal = PullMoves(sizeChain,bpos,-1);
       else   legal = PullMoves(sizeChain,bpos-sizeChain,1);
      
       //cout<<"whichone "<<whichone<<" bestpos "<<bestallpos[whichone]<<" howmany "<<howmany<<" legal "<<legal<<endl;
       //cout<<"maxval"<<(*maxval)<<" pos "<<bestallpos[whichone]+2<<endl;
      
       AssignPos(sizeChain,NewPos,Pos);
       TranslatePos(sizeChain,newvector);
       FindPos(sizeChain,newvector);   
        if (gains[bpos]>=(*currentEval)) 
          {
             (*currentEval) =  gains[bpos];
             for (i=0;i<sizeChain;i++) statvector[i] = newvector[i]; 
          }    

	//cout<<"CurrentVal "<<*currentEval<<" realeval "<<EvalOnlyVector(sizeChain,newvector)<<endl;   
     }
     else whichone = -1;

 return whichone;
}



int  HPProtein::SetNewBestPosSimple(int sizeChain, int* maxval, double* currentEval, unsigned int* currentvector) // Modifies the best current grid in local optimizer
{    
  int i,howmany, bestpos, whichone, legal,pos;
   
    howmany = 1;
    bestpos = 0;

    for (i=2;i<sizeChain-1;i++)
      {
	if(gains[i-2]>gains[bestpos])
	  {
            bestpos = i-2;
            howmany = 1;
            bestallpos[0] = bestpos;
          }
        else if(i>2 && gains[i-2]==gains[bestpos])    bestallpos[howmany++] = i-2;
             
      }            
    
 
    if(gains[bestpos]  > 0 || ((*maxval)<=1 && gains[bestpos] == 0)) 
     {
       if(gains[bestpos]  == 0)  (*maxval)++;
       if (howmany==1) whichone = 0;
       else  whichone = randomint(howmany);  
       pos = bestallpos[whichone]+2;
   if( svals[bestallpos[whichone]] == -1)
     {
      if (currentvector[pos] == 0) currentvector[pos] = 1;
      else  if (currentvector[pos] == 1) currentvector[pos] = 2;
      else currentvector[pos] = 0;              
    }
  else
    {
      if (currentvector[pos] == 0) currentvector[pos] = 2;
      else  if (currentvector[pos] == 1) currentvector[pos] = 0;
      else currentvector[pos] = 1;    
    }

       (*currentEval) = (*currentEval) + gains[bestpos];     
       FindPos(sizeChain,currentvector);   
     }
     else whichone = -1;

 return whichone;
}

double  HPProtein::ProteinLocalOptimizer(int sizeChain, unsigned int* vector, double currentEval, int* eval_local, int maxvalconstraint) // Given a chain of molecules, and a position makes a pull move 
{   
  int i,Continue;
  int maxval;
  double neweval1, neweval2;
  Continue = 1;
  maxval = 0;
  (*eval_local) = 0;
  

  FindPos(sizeChain,vector); // Puts in Pos the grid positions contained in vector


  while ( (Continue)  && maxval<=maxvalconstraint)
    {
      for (i=0;i<sizeChain-1;i++)
       {
	 neweval1 = FindNewEval(sizeChain,i,-1,eval_local);
       	 neweval2 = FindNewEval(sizeChain,i,1,eval_local);
	 // cout<<i<<" --> "<<neweval1<<" "<<neweval2<<endl;
         SetGainsAndVals(i,neweval1,neweval2,currentEval);
       }     
      Continue =  (SetNewBestPos(sizeChain,&maxval,&currentEval,maxvalconstraint) != -1);
     }    

  TranslatePos(sizeChain,vector);
  return currentEval; 
}


double  HPProtein::TabuOptimizer(int sizeChain, unsigned int* vector, double currentEval, int* eval_local, int maxvalconstraint, int duration) // Given a chain of molecules, and a position makes a pull move 
{   
  int i,Continue;
  int maxval;
  double neweval1, neweval2;
  Continue = 1;
  maxval = 0;
  (*eval_local) = 0;
  double InitVal = currentEval;
  
  gains[sizeChain-1] = -1;
  gains[2*sizeChain-1] = -1;
  FindPos(sizeChain,vector); // Puts in Pos the grid positions contained in vector

  for (i=0;i<2*sizeChain;i++) Memory[i] = 0;

  for (i=0;i<sizeChain;i++)  statvector[i] =  vector[i];

  while ( (Continue)  && maxval<=maxvalconstraint)
    {
      for (i=0;i<sizeChain-1;i++)
       {
	 neweval1 = TabuFindNewEval(sizeChain,i,-1,eval_local);
         gains[i] =  neweval1;                 
       	 neweval2 = TabuFindNewEval(sizeChain,i,1,eval_local);
         gains[sizeChain+i] =  neweval2;       
	 //  cout<<i<<" --> "<<neweval1<<" "<<neweval2<<endl;       
       }     

      Continue =  (TabuSetNewBestPos(sizeChain,&maxval,&currentEval,maxvalconstraint,duration) != -1);
      for (i=0;i<2*sizeChain;i++)
	if(Memory[i]>0) Memory[i]--;
     }    

   if(currentEval>InitVal) for (i=0;i<sizeChain;i++)  vector[i] =  statvector[i];
              
  //TranslatePos(sizeChain,vector);
  return currentEval; 
}


double  HPProtein::ProteinLocalPerturbation(int sizeChain, unsigned int* vector, double currentEval, int* eval_local, int maxvalconstraint) // Given a chain of molecules, and a position makes a pull move 
{   
  int i,Continue;
  int maxval;
  double neweval1, neweval2;
  Continue = 1;
  maxval = 0;
  (*eval_local) = 0;
  
  FindPos(sizeChain,vector); // Puts in Pos the grid positions contained in vector


  while ( (Continue)  && maxval<=maxvalconstraint)
    {
      for (i=0;i<sizeChain-1;i++)
       {
	 neweval1 = FindIsLegal(sizeChain,i,-1,eval_local,currentEval);
       	 neweval2 = FindIsLegal(sizeChain,i,1,eval_local,currentEval);
         //cout<<i<<" --> "<<neweval1<<" "<<neweval2<<endl;          
         SetGainsAndVals(i,neweval1,neweval2,currentEval);
       }     
      Continue =  (SetNewBestPos(sizeChain,&maxval,&currentEval,maxvalconstraint) != -1);
     }    
  TranslatePos(sizeChain,vector);
  return currentEval; 
}


double  HPProtein::ProteinLocalOptimizerSimple(int sizeChain, unsigned int* vector, double currentEval, int* eval_local) // Given a chain of molecules, and a position makes a pull move 
{   
  int i,Continue;
  int maxval;
  double neweval1, neweval2;
  Continue = 1;
  maxval = 1;
  (*eval_local) = 0;
  

  FindPos(sizeChain,vector); // Puts in Pos the grid positions contained in vector


  while ( (Continue)  && maxval<=1)
    {
      for (i=2;i<sizeChain-1;i++)
       {
	 neweval1 = FindNewEvalSimple(sizeChain,i,-1,eval_local,vector);
       	 neweval2 = FindNewEvalSimple(sizeChain,i,1,eval_local,vector);
         //cout<<i<<" --> "<<neweval1<<" "<<neweval2<<endl;
         SetGainsAndVals(i,neweval1,neweval2,currentEval);
       }     
      Continue =  (SetNewBestPosSimple(sizeChain,&maxval,&currentEval,vector) != -1);
     }    
  //FindPos(sizeChain,vector);   
  //TranslatePos(sizeChain,vector);
  return currentEval; 
}


/*

int  HPProtein::Feasible(unsigned int* s, int slength, int pos) //Checks if there are not overlappings in the grid
{ 
  int  Overlappings = 0;
  int j;
  
 if (slength  <=2) return 1;

  PutMoveAtPos(pos,s[pos]); 

  for (j=0;j<pos;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
    Overlappings +=  (Pos[pos][0]==Pos[j][0] && Pos[pos][1]==Pos[j][1]);  
  return ( Overlappings == 0);
}

*/



int  HPProtein::Feasible(unsigned int* s, int pos) //Checks if there are not overlappings in the grid
{ 
  int  Overlappings = 0;
  int j;
 

 if (pos  <2) return 1;

  PutMoveAtPos(pos,s[pos]); 
  j = 0;

  while (j<pos && Overlappings==0)   // Check for Overlappings and Collissions in all the    molecules except the previous one
    {
      Overlappings +=  (Pos[pos][0]==Pos[j][0] && Pos[pos][1]==Pos[j][1]); 
      j++;
    }  
  return (Overlappings == 0);
}


void  HPProtein::PutMoveAtPos(int pos, int mov)  //Sets a given move in the grid
{
  int i;
  if (pos<2) return;
  i = pos;

    if(Pos[i-1][1]==Pos[i-2][1])
         {
           if (mov==0)                  //UP MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] +  (Pos[i-1][0] - Pos[i-2][0]);
             }
           else if (mov==1)             //FORWARD MOVE
	     {
               Pos[i][0] = Pos[i-1][0] +  (Pos[i-1][0] - Pos[i-2][0]); 
               Pos[i][1] = Pos[i-1][1];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] -   (Pos[i-1][0] - Pos[i-2][0]);
             }
	 }
        
       if(Pos[i-1][0]==Pos[i-2][0])
         {
           if (mov==0)                  //UP MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0] -  (Pos[i-1][1] - Pos[i-2][1]);
             }
           else if (mov==1)             //FORWARD MOVE
	     {
               Pos[i][1] = Pos[i-1][1] +  (Pos[i-1][1] - Pos[i-2][1]); 
               Pos[i][0] = Pos[i-1][0];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0] + (Pos[i-1][1] - Pos[i-2][1]);
             }
	 }
}



/*
int HPProtein::Repair(int sizeChain, int pos, unsigned int* s,int slength, unsigned int* vector)    //Recursive function for repairing a protein 
  {
    int solutionFound,i,indexmove;
      
    if (Feasible(s, slength,pos-1)==1)
      {
        solutionFound  = (slength == sizeChain+1);
        i = 1;
        indexmove = randomint(6);
        while (solutionFound==0  && i<=3)
	  {
             if (i==1) 
	      {
                s[pos] = vector[pos];
                PutMoveAtPos(pos,vector[pos]);
                solutionFound = Repair(sizeChain,pos+1,s,slength+1,vector);
              }
             else if(moves[indexmove][i-1] != vector[pos])      
              {
                if (i==2) statvector[pos]++;
                s[pos] = moves[indexmove][i-1];
                PutMoveAtPos(pos,s[pos]);
                solutionFound = Repair(sizeChain,pos+1,s,slength+1,vector);
              } 
             i++;
	  } 
        // if (solutionFound==1)  s = new_s;
      }
    else  solutionFound = 0;
    return solutionFound;
  }


*/


/*
int HPProtein::Repair(int sizeChain, int pos, unsigned int* s, unsigned int* vector)    //Recursive function for repairing a protein 
  {
    int solutionFound,i,indexmove;
      

    solutionFound  = (pos == sizeChain);
       if( !solutionFound)
	 {
          indexmove = randomint(6);
          i = 1;  
          s[pos] = vector[pos];
          while (Feasible(s,pos)==0  && i<4)
           {    
	     //  if (i==1) statvector[pos]++;
             if(moves[indexmove][i-1] != vector[pos])  s[pos] = moves[indexmove][i-1];
             i++;
	   }
          if (i<4)  solutionFound =  Repair(sizeChain,pos+1,s,vector);
          else  solutionFound = 0;
         }
       return solutionFound;     
  }
*/


int  HPProtein::DownFeasible(int mov, int slength,   int pos) //Checks if there are not overlappings in the grid
{ 
  int  Overlappings = 0;
  int j;
  
 
  DownPutMoveAtPos(pos,mov); 

  j = pos + 1;
  while (j<slength && Overlappings==0)   // Check for Overlappings and Collissions in all the    molecules except the previous one
    {
     Overlappings +=  (Pos[pos][0]==Pos[j][0] && Pos[pos][1]==Pos[j][1]);  
     j++;
    } 
 return ( Overlappings == 0);
}



void  HPProtein::DownPutMoveAtPos(int pos, int mov)  //Sets a given move in the grid
{
  int i;

  i = pos;
  //cout<<i<<"  mov "<<mov<<"  "<<Pos[i][0]<<"  "<<Pos[i][1]<<endl;

    if(Pos[i+1][1]==Pos[i+2][1])
         {
           if (mov==0)                  //UP MOVE
	     {
               Pos[i][0] = Pos[i+1][0];
               Pos[i][1] = Pos[i+1][1] +  (Pos[i+1][0] - Pos[i+2][0]);
             }
           else if (mov==1)             //FORWARD MOVE
	     {
               Pos[i][0] = Pos[i+1][0] +  (Pos[i+1][0] - Pos[i+2][0]); 
               Pos[i][1] = Pos[i+1][1];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][0] = Pos[i+1][0];
               Pos[i][1] = Pos[i+1][1] -   (Pos[i+1][0] - Pos[i+2][0]);
             }
	 }
        
       if(Pos[i+1][0]==Pos[i+2][0])
         {
           if (mov==0)                  //UP MOVE
	     {
               Pos[i][1] = Pos[i+1][1];
               Pos[i][0] = Pos[i+1][0] -  (Pos[i+1][1] - Pos[i+2][1]);
             }
           else if (mov==1)             //FORWARD MOVE
	     {
               Pos[i][1] = Pos[i+1][1] +  (Pos[i+1][1] - Pos[i+2][1]); 
               Pos[i][0] = Pos[i+1][0];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][1] = Pos[i+1][1];
               Pos[i][0] = Pos[i+1][0] + (Pos[i+1][1] - Pos[i+2][1]);
             }
	 }

       //     cout<<i<<"  mov "<<mov<<"  "<<Pos[i][0]<<"  "<<Pos[i][1]<<endl;
}




int HPProtein::DownRepair(int sizeChain, int pos, unsigned int* s, unsigned int* vector)    //Recursive function for repairing a protein 
   {
    int solutionFound,i,indexmove;
      
    solutionFound  = (pos == 1);
       if( !solutionFound)
	 {
          indexmove = randomint(combmoves);
          i = 1;         
          while (! solutionFound  && i<=(nmoves+1))
           {    
             if (i==1) 
	       {  
                 statvector[pos]++;
                 s[pos] = vector[pos];
                 if(DownFeasible(s[pos],sizeChain,pos-2)==1) solutionFound =  DownRepair(sizeChain,pos-1,s,vector);
               }
             else if(moves[indexmove][i-1] != vector[pos]) 
               {
                 s[pos] = moves[indexmove][i-1];
                 if(DownFeasible(s[pos],sizeChain,pos-2)==1) solutionFound =  DownRepair(sizeChain,pos-1,s,vector);
               }
             i++;
	   }
         }
       return solutionFound;     
 
  }


int HPProtein::Repair(int sizeChain, int pos, unsigned int* s, unsigned int* vector)    //Recursive function for repairing a protein 
  {
    int solutionFound,i,indexmove;
      
    solutionFound  = (pos == sizeChain);
       if( !solutionFound)
	 {
          indexmove = randomint(combmoves);
          i = 1;  
         
          while (!solutionFound  && i<=(nmoves+1))
           {    
	     //  if (i==1) statvector[pos]++;
             if (i==1) 
	       { 
                 s[pos] = vector[pos];  
                 if(Feasible(s,pos)==1) solutionFound =  Repair(sizeChain,pos+1,s,vector);
               }
             else if(moves[indexmove][i-1] != vector[pos]) 
                 {
                   s[pos] = moves[indexmove][i-1];
                   if(Feasible(s,pos)==1) solutionFound =  Repair(sizeChain,pos+1,s,vector);
                 }
             i++;
	   }         
         }
       return solutionFound;     
  }


int HPProtein::PartialRepair(int sizeChain, int pos, unsigned int* s, unsigned int* vector)    //Recursive function for repairing a protein 
  {
      int solutionFound,i,indexmove;
      
    solutionFound  = (pos == sizeChain);
        
       if( !solutionFound)
	 {
          indexmove = randomint(combmoves);
          i = 1;  
         
          while (!solutionFound  && i<=(nmoves+1))
           {    

	   if (statvector[pos] < 500)
	    { 
	      statvector[pos]++;
             if (i==1) 
	       { 
                 s[pos] = vector[pos];
                 if(Feasible(s,pos)==1) solutionFound =  PartialRepair(sizeChain,pos+1,s,vector);
               }
             else if(moves[indexmove][i-2] != vector[pos]) 
                 {
                   s[pos] = moves[indexmove][i-2];
                   if(Feasible(s,pos)==1) solutionFound =  PartialRepair(sizeChain,pos+1,s,vector);
                 }
	    }
            else 
             {
               solutionFound = 1;
               //int eval_local;
               //double CVal = ProteinLocalOptimizer(sizeChain,vector,0,&eval_local,20);
	       //cout<<pos<<" "<<statvector[pos]<<" CVal= "<<CVal<<endl;
             }
              i++;
	   }         
         }
       return solutionFound;     
  }

int HPProtein::BackTracking(int sizeChain, int pos, unsigned int* s)    //Recursive function for creating protein by backtracking 
  {
    int solutionFound,i,indexmove;

       solutionFound  = (pos == sizeChain);
       if( !solutionFound)
	 {
          indexmove = randomint(combmoves);
          i = 0;  
          while (!solutionFound  && i<nmoves)
           {    
             s[pos] = moves[indexmove][i];
             if(Feasible(s,pos)==1) solutionFound =  BackTracking(sizeChain,pos+1,s);
             i++;
	   }
         }
       return solutionFound; 
  }






void  HPProtein::CallRepair(int* vector, int sizeChain) // Calls to the recursive Repair procedure
{  
  int i,solutionFound;
  auxnewvector[0] = 0;  auxnewvector[1] = 0;
 for (i=0;i<sizeChain; i++)  statvector[i] = 0;
  for (i=2;i<sizeChain;i++)  
    {
      auxnewvector[i] = vector[i];
    }
   solutionFound = PartialRepair(sizeChain,0,newvector,auxnewvector);
   for (i=0;i<sizeChain;i++) vector[i] = newvector[i]; 

  
}


void  HPProtein::CallRepair(unsigned int* vector, int sizeChain) // Calls to the recursive Repair procedure
{  
  int i,solutionFound;
  vector[0] = 0;  vector[1] = 0;
 for (i=0;i<sizeChain; i++)  statvector[i] = 0;
         
   SetInitPosAt_i(Pos,0);
   solutionFound = PartialRepair(sizeChain,0,newvector,vector);
   for (i=0;i<sizeChain;i++) vector[i] = newvector[i]; 
   //for (i=0;i<sizeChain;i++) cout<<vector[i]<<" "; 
   // cout<<endl;
}



void  HPProtein::DownCallRepair(int pos, unsigned int* vector, int sizeChain) // Calls to the recursive Repair procedure
{  
  int i,solutionFound;
   vector[0] = 0;  vector[1] = 0;

   if(pos>=2)
   {
     SetInitPosAt_i(Pos,pos-2);
     solutionFound = Repair(sizeChain,pos,newvector,vector);
     for (i=pos;i<sizeChain;i++) vector[i] = newvector[i]; 
     solutionFound = DownRepair(sizeChain,sizeChain-1,newvector,vector);  
     for (i=0;i<sizeChain;i++) vector[i] = newvector[i];    
   }
     
}


void  HPProtein::DownCallRepair(int pos, int* vector, int sizeChain) // Calls to the recursive Repair procedure
{  
  int i,solutionFound;
   
 auxnewvector[0] = 0;  auxnewvector[1] = 0;
 
  for (i=2;i<sizeChain;i++)   auxnewvector[i] = vector[i];
  
   if(pos>=2)
   {
     SetInitPosAt_i(Pos,pos-2);
     solutionFound = Repair(sizeChain,pos,newvector,auxnewvector);   
     for (i=pos;i<sizeChain;i++) auxnewvector[i] = newvector[i];
     solutionFound = DownRepair(sizeChain,sizeChain-1,newvector,auxnewvector);
     for (i=0;i<sizeChain;i++) vector[i] = newvector[i];   
    }

}


void  HPProtein::CallBackTracking(unsigned int* vect) // Calls to the recursive BackTracking procedure
{  
  int solutionFound;
  int i;
  for(i=0;i<sizeProtein; i++)  statvector[i] = 0;
  vect[0] = 0;  vect[1] = 0;
  for (i=2;i<sizeProtein;i++)  vect[i] = randomint(nmoves); 
  solutionFound = BackTracking(sizeProtein,2,vect);
}



void  HPProtein::CallBackTracking(int* vect) // Calls to the recursive BackTracking procedure
{  
  int solutionFound;
  int i;
  for(i=0;i<sizeProtein; i++)  statvector[i] = 0;
  vect[0] = 0;  vect[1] = 0;
  for (i=2;i<sizeProtein;i++)  newvector[i] = randomint(nmoves); 
  solutionFound = BackTracking(sizeProtein,2,newvector);
  for (i=0;i<sizeProtein;i++)  vect[i] = newvector[i]; 
}





// ***************************** IMPLEMENTACION PARA EL CASO 3D ******************************************************



void HPProtein3D::SetInitMoves() // Init all possible ordering of legal moves
{
    moves[0][0] = 0;  moves[0][1] = 1;   moves[0][2] = 2;   moves[0][3] = 3;  moves[0][4] = 4;
  
  for(int i=1;i<combmoves;i++) 
       {
        nextperm(nmoves,moves[i-1],moves[i]);   
	//      cout<<moves[i][0]<<moves[i][1]<<moves[i][2]<<moves[i][3]<<moves[i][4]<<endl;
       } 
}



void HPProtein3D::SetInitPos(int** aPos) // Init the first positions of the grid
{
    aPos[0][0] = 0;
    aPos[0][1] = 0;
    aPos[0][2] = 0;
    aPos[1][0] = 1;
    aPos[1][1] = 0;
    aPos[1][2] = 0;  
}


void HPProtein3D::SetInitPosAt_i(int** aPos, int i) // Init the first positions of the grid
{

    aPos[i][0] = 0;
    aPos[i][1] = 0;
    aPos[i][2] = 0;
    aPos[i+1][0] = 1;
    aPos[i+1][1] = 0;
    aPos[i+1][2] = 0;  
}


void  HPProtein3D::CreatePos() //Creates the first positions of the grid
 {
  int i;
  
 Pos = new int*[sizeProtein];
 NewPos = new int*[sizeProtein];

  for (i=0;i<sizeProtein;i++)
   {
    Pos[i] = new int[3];
    NewPos[i] = new int[3];
   } 

  SetInitPos(Pos); 
  SetInitPos(NewPos); 
  SetInitMoves();
 }



HPProtein3D::HPProtein3D(int sizeP, int* HConf):HPProtein(sizeP,HConf,5,120)
{ 
 
  CreatePos();
  
}




HPProtein3D::~HPProtein3D()
{
}


void  HPProtein3D::FindPos(int sizeChain, unsigned* vector) //Translates the vector to the  positions in the grid 
 {
  int i;
  SetInitPos(Pos);
   
   for (i=2;i<sizeChain;i++)
     {
      
       if(Pos[i-1][1]==Pos[i-2][1] && Pos[i-1][2]==Pos[i-2][2])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] + (Pos[i-1][0] - Pos[i-2][0]);
               Pos[i][2] = Pos[i-1][2];
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               Pos[i][0] = Pos[i-1][0] +  (Pos[i-1][0] - Pos[i-2][0]); 
               Pos[i][1] = Pos[i-1][1];
               Pos[i][2] = Pos[i-1][2];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] - (Pos[i-1][0] - Pos[i-2][0]);
               Pos[i][2] = Pos[i-1][2];
             }
           else if (vector[i]==3)             //FORWARD MOVE in Z
	     {
               Pos[i][0] = Pos[i-1][0]; 
               Pos[i][1] = Pos[i-1][1];
               Pos[i][2] = Pos[i-1][2] - (Pos[i-1][0] - Pos[i-2][0]);
              }
	   else  if (vector[i]==4)            //DOWN  MOVE  in Z
	     {
               Pos[i][0] = Pos[i-1][0]; 
               Pos[i][1] = Pos[i-1][1];
               Pos[i][2] = Pos[i-1][2] + (Pos[i-1][0] - Pos[i-2][0]);
             }

	 }
        
 if(Pos[i-1][0]==Pos[i-2][0] && Pos[i-1][2]==Pos[i-2][2])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0] - (Pos[i-1][1] - Pos[i-2][1]);
               Pos[i][2] = Pos[i-1][2];
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               Pos[i][1] = Pos[i-1][1] +  (Pos[i-1][1] - Pos[i-2][1]); 
               Pos[i][0] = Pos[i-1][0];
               Pos[i][2] = Pos[i-1][2];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0] + (Pos[i-1][1] - Pos[i-2][1]);
               Pos[i][2] = Pos[i-1][2];
             }
           else if (vector[i]==3)             //FORWARD MOVE in Z
	     {
               Pos[i][0] = Pos[i-1][0]; 
               Pos[i][1] = Pos[i-1][1];
               Pos[i][2] = Pos[i-1][2] - (Pos[i-1][1] - Pos[i-2][1]);
              }
	   else  if (vector[i]==4)            //DOWN  MOVE  in Z
	     {
               Pos[i][0] = Pos[i-1][0]; 
               Pos[i][1] = Pos[i-1][1];
               Pos[i][2] = Pos[i-1][2] + (Pos[i-1][1] - Pos[i-2][1]);
             }

	 }


 if(Pos[i-1][0]==Pos[i-2][0] && Pos[i-1][1]==Pos[i-2][1])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               Pos[i][2] = Pos[i-1][2];
               Pos[i][1] = Pos[i-1][1] + (Pos[i-1][2] - Pos[i-2][2]);
               Pos[i][0] = Pos[i-1][0];
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               Pos[i][2] = Pos[i-1][2] +  (Pos[i-1][2] - Pos[i-2][2]); 
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               Pos[i][2] = Pos[i-1][2];
               Pos[i][1] = Pos[i-1][1] - (Pos[i-1][2] - Pos[i-2][2]);
               Pos[i][0] = Pos[i-1][0];
             }
           else if (vector[i]==3)             //FORWARD MOVE in Z
	     {
               Pos[i][1] = Pos[i-1][1]; 
               Pos[i][2] = Pos[i-1][2];
               Pos[i][0] = Pos[i-1][0] - (Pos[i-1][2] - Pos[i-2][2]);
              }
	   else  if (vector[i]==4)            //DOWN  MOVE  in Z
	     {
               Pos[i][1] = Pos[i-1][1]; 
               Pos[i][2] = Pos[i-1][2];
               Pos[i][0] = Pos[i-1][0] + (Pos[i-1][2] - Pos[i-2][2]);
             }

	 }
    
   
} 
 }


void HPProtein3D::TranslatePos(int sizeChain, unsigned* newvector) // Translate an array of grid positions to a vector of movements
{

  int i;
  newvector[0] = 0;
  newvector[1] = 0;
 
  
 for (i=2;i<sizeChain;i++)
     {
        if(Pos[i-1][1]==Pos[i-2][1] && Pos[i-1][2]==Pos[i-2][2])
         {
           if (  (Pos[i][0] == Pos[i-1][0])  && (Pos[i][1] == Pos[i-1][1] + (Pos[i-1][0] - Pos[i-2][0])) && (Pos[i][2] == Pos[i-1][2])  )                  //UP MOVE
	     newvector[i] = 0;
           else if ( (Pos[i][0] == Pos[i-1][0]  +  (Pos[i-1][0] - Pos[i-2][0])) && (Pos[i][1] == Pos[i-1][1]) && (Pos[i][2] == Pos[i-1][2]) )   //FORWARD MOVE
             newvector[i] = 1;
           else if  ( (Pos[i][0] ==  Pos[i-1][0]) && (Pos[i][1] == Pos[i-1][1] - (Pos[i-1][0] - Pos[i-2][0])) && (Pos[i][2] == Pos[i-1][2]) ) // DOWN MOVE
              newvector[i] = 2;
           else if ( (Pos[i][2] == Pos[i-1][2]  +  (Pos[i-1][0] - Pos[i-2][0])) && (Pos[i][1] == Pos[i-1][1]) && (Pos[i][0] == Pos[i-1][0]) )   //FORWARD MOVE
             newvector[i] = 3;
           else if  ( (Pos[i][0] ==  Pos[i-1][0]) && (Pos[i][2] == Pos[i-1][2] - (Pos[i-1][0] - Pos[i-2][0])) && (Pos[i][1] == Pos[i-1][1]) ) // DOWN MOVE
              newvector[i] = 4;
          }

        if(Pos[i-1][0]==Pos[i-2][0] && Pos[i-1][2]==Pos[i-2][2])
         {
           if (  (Pos[i][1] == Pos[i-1][1])  && (Pos[i][0] == Pos[i-1][0] - (Pos[i-1][1] - Pos[i-2][1])) && (Pos[i][2] == Pos[i-1][2]) )                  //UP MOVE
	     newvector[i] = 0;
           else if ( (Pos[i][1] == Pos[i-1][1]  +  (Pos[i-1][1] - Pos[i-2][1])) && (Pos[i][0] == Pos[i-1][0])  && (Pos[i][2] == Pos[i-1][2]))   //FORWARD MOVE 
             newvector[i] = 1;
           else if  ( (Pos[i][1] ==  Pos[i-1][1]) && (Pos[i][0] == Pos[i-1][0] + (Pos[i-1][1] - Pos[i-2][1]))  && (Pos[i][2] == Pos[i-1][2])) // DOWN MOVE
              newvector[i] = 2;
           else if ( (Pos[i][2] == Pos[i-1][2]  -  (Pos[i-1][1] - Pos[i-2][1])) && (Pos[i][1] == Pos[i-1][1]) && (Pos[i][0] == Pos[i-1][0]) )   //FORWARD MOVE
             newvector[i] = 3;
           else if  ( (Pos[i][0] ==  Pos[i-1][0]) && (Pos[i][2] == Pos[i-1][2] + (Pos[i-1][1] - Pos[i-2][1])) && (Pos[i][1] == Pos[i-1][1]) ) // DOWN MOVE
              newvector[i] = 4;

          }

      if(Pos[i-1][0]==Pos[i-2][0] && Pos[i-1][1]==Pos[i-2][1])
         {
           if (  (Pos[i][0] == Pos[i-1][0])  && (Pos[i][1] == Pos[i-1][1] + (Pos[i-1][2] - Pos[i-2][2])) && (Pos[i][2] == Pos[i-1][2]) )                  //UP MOVE
	     newvector[i] = 0;
           else if ( (Pos[i][2] == Pos[i-1][2]  +  (Pos[i-1][2] - Pos[i-2][2])) && (Pos[i][0] == Pos[i-1][0])  && (Pos[i][1] == Pos[i-1][1]))   //FORWARD MOVE 
             newvector[i] = 1;
           else if  ( (Pos[i][0] ==  Pos[i-1][0]) && (Pos[i][1] == Pos[i-1][1] - (Pos[i-1][2] - Pos[i-2][2]))  && (Pos[i][2] == Pos[i-1][2])) // DOWN MOVE
              newvector[i] = 2;
           else if ( (Pos[i][2] == Pos[i-1][0]  +  (Pos[i-1][2] - Pos[i-2][2])) && (Pos[i][1] == Pos[i-1][1]) && (Pos[i][2] == Pos[i-1][2]) )   //FORWARD MOVE
             newvector[i] = 3;
           else if  ( (Pos[i][1] ==  Pos[i-1][1]) && (Pos[i][2] == Pos[i-1][0] - (Pos[i-1][2] - Pos[i-2][2])) && (Pos[i][2] == Pos[i-1][2]) ) // DOWN MOVE
              newvector[i] = 4;

          }
    
    }
}

  
int  HPProtein3D::CheckFree(int sizeChain, int** aPos, int P1, int P2, int P3)   // Checks if coordinate at_i_Pos is occuppied in the grid
{ 
int i,Found;

 i = 0;
 Found = 0;

 while(i<sizeChain && !Found)
 {
  Found = ((aPos[i][0] == P1) && (aPos[i][1] == P2) &&  (aPos[i][2] == P3));
  i = i + 1; 
 }
 return (Found==0);
}
 

int  HPProtein3D::CheckValidity(int at_i_Pos, int** aPos) //Checks if the chain is connected at position at_i_Pos
 { 
   int resp  = 1;
   if (at_i_Pos > 0)   resp = ( (abs(aPos[at_i_Pos][0] - aPos[at_i_Pos-1][0]) +  abs(aPos[at_i_Pos][1] - aPos[at_i_Pos-1][1]) +  (abs(aPos[at_i_Pos][2] - aPos[at_i_Pos-1][2]))) == 1);
   return resp;
 } 




void HPProtein3D::AssignPos(int sizeChain, int** Pos, int** OtherPos)     // Copy the position of one grid into another
 {
  int i;
   for (i=0;i<sizeChain;i++)
   {
    OtherPos[i][0] = Pos[i][0];
    OtherPos[i][1] = Pos[i][1];
    OtherPos[i][2] = Pos[i][2];
    }
 }

void HPProtein3D::PrintPos(int sizeChain, int** aPos)       // Prints all or part of the protein's grid
{
  int i;

  for (i=0;i<sizeChain;i++) 
    { 
      cout<<i<<"-->"<<aPos[i][0]<<" "<<aPos[i][1]<<" "<<aPos[i][2]<<endl;  
    } 
}



int HPProtein3D::PullMoves(int sizeChain,int at_i, int sval) // Given a chain of molecules, and a position makes a pull move 
{
int i,resp,i_minus_1;
 int L1,L2,L3,C1,C2,C3;

AssignPos(sizeChain,Pos,NewPos);

// cout<<"Here it is "<<at_i<<"  "<<sval<<endl;



if(Pos[at_i][0]==Pos[at_i+1][0] && Pos[at_i][2]==Pos[at_i+1][2])
 {
   L2 = Pos[at_i+1][1];
   C2 = Pos[at_i][1];

  if (sval==0) 
   {
    L1 = Pos[at_i][0] + (Pos[at_i+1][1]-Pos[at_i][1]);
    L3 = Pos[at_i+1][2]; 
    C1 = Pos[at_i][0] + (Pos[at_i+1][1]-Pos[at_i][1]);
    C3 = Pos[at_i][2]; 
   }
   else if (sval==2) 
   {
    L1 = Pos[at_i][0] - (Pos[at_i+1][1]-Pos[at_i][1]);
    L3 = Pos[at_i+1][2]; 
    C1 = Pos[at_i][0] - (Pos[at_i+1][1]-Pos[at_i][1]);
    C3 = Pos[at_i][2]; 
   }
   else if (sval==3) 
   {
    L1 = Pos[at_i+1][0]; 
    L3 = Pos[at_i][2] - (Pos[at_i+1][1]-Pos[at_i][1]); 
    C1 = Pos[at_i][0];
    C3 = Pos[at_i][2] - (Pos[at_i+1][1]-Pos[at_i][1]); 
   }
   else if (sval==3) 
   {
    L1 = Pos[at_i+1][0]; 
    L3 = Pos[at_i][2] + (Pos[at_i+1][1]-Pos[at_i][1]); 
    C1 = Pos[at_i][0];
    C3 = Pos[at_i][2] + (Pos[at_i+1][1]-Pos[at_i][1]); 
   }
 }
else if(Pos[at_i][1]==Pos[at_i+1][1] && Pos[at_i][2]==Pos[at_i+1][2])
 {
   L1 = Pos[at_i+1][0];
   C1 = Pos[at_i][0];
  if (sval==0) 
   {
    L2 = Pos[at_i][1] - (Pos[at_i+1][0]-Pos[at_i][0]);
    L3 = Pos[at_i+1][2]; 
    C2 = Pos[at_i][1] - (Pos[at_i+1][0]-Pos[at_i][0]);
    C3 = Pos[at_i][2]; 
   }
   else if (sval==2) 
   {
    L2 = Pos[at_i][1] + (Pos[at_i+1][0]-Pos[at_i][0]);
    L3 = Pos[at_i+1][2]; 
    C2 = Pos[at_i][1] + (Pos[at_i+1][0]-Pos[at_i][0]);
    C3 = Pos[at_i][2]; 
   }
   else if (sval==3) 
   {

    L2 = Pos[at_i+1][1]; 
    L3 = Pos[at_i][2] - (Pos[at_i+1][0]-Pos[at_i][0]); 
    C2 = Pos[at_i][1];
    C3 = Pos[at_i][2] - (Pos[at_i+1][0]-Pos[at_i][0]); 
   }
   else if (sval==3) 
   {
    L2 = Pos[at_i+1][1]; 
    L3 = Pos[at_i][2] + (Pos[at_i+1][0]-Pos[at_i][0]); 
    C2 = Pos[at_i][1];
    C3 = Pos[at_i][2] + (Pos[at_i+1][0]-Pos[at_i][0]); 
   }
 }
else if(Pos[at_i][1]==Pos[at_i+1][1] && Pos[at_i][0]==Pos[at_i+1][0])
 {
   L3 = Pos[at_i+1][2];
   C3 = Pos[at_i][2];
  if (sval==0) 
   {
    L1 = Pos[at_i+1][0];
    L2 = Pos[at_i][2]  + (Pos[at_i+1][2]-Pos[at_i][2]);
    C1 = Pos[at_i][0];
    C2 = Pos[at_i][2]  + (Pos[at_i+1][2]-Pos[at_i][2]); 
   }
   else if (sval==2) 
   {
    L1 = Pos[at_i+1][0];
    L2 = Pos[at_i][2]  - (Pos[at_i+1][2]-Pos[at_i][2]);
    C1 = Pos[at_i][0];
    C2 = Pos[at_i][2]  - (Pos[at_i+1][2]-Pos[at_i][2]); 
   }
   else if (sval==3) 
   {
    L1 = Pos[at_i][0]  + (Pos[at_i+1][2]-Pos[at_i][2]);
    L2 = Pos[at_i+1][2];
    C1 = Pos[at_i][0] + (Pos[at_i+1][2]-Pos[at_i][2]);
    C2 = Pos[at_i][2];     
   }
   else if (sval==3) 
   {
    L1 = Pos[at_i][0]  - (Pos[at_i+1][2]-Pos[at_i][2]);
    L2 = Pos[at_i+1][2];
    C1 = Pos[at_i][0] -  (Pos[at_i+1][2]-Pos[at_i][2]);
    C2 = Pos[at_i][2];   
   }
 }


resp = CheckFree(sizeChain, Pos, L1, L2,L3);
//cout<<" L1 "<<L1<<" L2 "<<L2<<" C1 "<<C1<<" C2 "<<C2<<" resp "<<resp<<endl;
if(resp==0)  return resp;  // Position L is not free
else if(at_i==0)
 {
    NewPos[at_i][0] = L1;
    NewPos[at_i][1] = L2;
    NewPos[at_i][1] = L3;
    resp = 1;
    return  resp;
 }
 
resp =  CheckFree(sizeChain, Pos, C1, C2, C3);
  
 if(resp == 0)  // Position C is not free
  {
   i_minus_1 = ( (Pos[at_i-1][0]==C1) && (Pos[at_i-1][1]==C2) && (Pos[at_i-1][2]==C3) );
   if (i_minus_1 == 1) // Molecule i-1 is at position C  
     {
       NewPos[at_i][0] = L1;
       NewPos[at_i][1] = L2;
       NewPos[at_i][2] = L3;
       resp = 1;
     }
     return resp; // if C is not free and is not occupied by i-1 then return.
  }

  NewPos[at_i][0] = L1;
  NewPos[at_i][1] = L2;
  NewPos[at_i][2] = L3;
  NewPos[at_i-1][0] = C1;
  NewPos[at_i-1][1] = C2;
  NewPos[at_i-1][2] = C3;
  i = at_i-2;

  while ( (i>=0) && !(CheckValidity(i,NewPos)==0))
  {
   NewPos[i][0] = Pos[i+2][0];
   NewPos[i][1] = Pos[i+2][1];
   NewPos[i][2] = Pos[i+2][2];
   i = i-1;  
  } 
  // PrintPos(sizeChain,NewPos);
  return resp;
}



double HPProtein3D::EvalChain(int sizeChain, unsigned int* vector, int** aPos)
{
  // Given a chain of molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 

  int i,j,Collisions,Overlappings;
  double result;
 Collisions = 0;
 Overlappings = 0;
  

   for (i=2;i<sizeChain;i++)
     {

    if(aPos[i-1][1]==aPos[i-2][1] && aPos[i-1][2]==aPos[i-2][2])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               aPos[i][0] = aPos[i-1][0];
               aPos[i][1] = aPos[i-1][1] + (aPos[i-1][0] - aPos[i-2][0]);
               aPos[i][2] = aPos[i-1][2];
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               aPos[i][0] = aPos[i-1][0] +  (aPos[i-1][0] - aPos[i-2][0]); 
               aPos[i][1] = aPos[i-1][1];
               aPos[i][2] = aPos[i-1][2];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               aPos[i][0] = aPos[i-1][0];
               aPos[i][1] = aPos[i-1][1] - (aPos[i-1][0] - aPos[i-2][0]);
               aPos[i][2] = aPos[i-1][2];
             }
           else if (vector[i]==3)             //FORWARD MOVE in Z
	     {
               aPos[i][0] = aPos[i-1][0]; 
               aPos[i][1] = aPos[i-1][1];
               aPos[i][2] = aPos[i-1][2] - (aPos[i-1][0] - aPos[i-2][0]);
              }
	   else  if (vector[i]==4)            //DOWN  MOVE  in Z
	     {
               aPos[i][0] = aPos[i-1][0]; 
               aPos[i][1] = aPos[i-1][1];
               aPos[i][2] = aPos[i-1][2] + (aPos[i-1][0] - aPos[i-2][0]);
             }

	 }
        
 if(aPos[i-1][0]==aPos[i-2][0] && aPos[i-1][2]==aPos[i-2][2])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               aPos[i][1] = aPos[i-1][1];
               aPos[i][0] = aPos[i-1][0] - (aPos[i-1][1] - aPos[i-2][1]);
               aPos[i][2] = aPos[i-1][2];
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               aPos[i][1] = aPos[i-1][1] +  (aPos[i-1][1] - aPos[i-2][1]); 
               aPos[i][0] = aPos[i-1][0];
               aPos[i][2] = aPos[i-1][2];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               aPos[i][1] = aPos[i-1][1];
               aPos[i][0] = aPos[i-1][0] + (aPos[i-1][1] - aPos[i-2][1]);
               aPos[i][2] = aPos[i-1][2];
             }
           else if (vector[i]==3)             //FORWARD MOVE in Z
	     {
               aPos[i][0] = aPos[i-1][0]; 
               aPos[i][1] = aPos[i-1][1];
               aPos[i][2] = aPos[i-1][2] - (aPos[i-1][1] - aPos[i-2][1]);
              }
	   else  if (vector[i]==4)            //DOWN  MOVE  in Z
	     {
               aPos[i][0] = aPos[i-1][0]; 
               aPos[i][1] = aPos[i-1][1];
               aPos[i][2] = aPos[i-1][2] + (aPos[i-1][1] - aPos[i-2][1]);
             }

	 }


 if(aPos[i-1][0]==aPos[i-2][0] && aPos[i-1][1]==aPos[i-2][1])
         {
           if (vector[i]==0)                  //UP MOVE
	     {
               aPos[i][2] = aPos[i-1][2];
               aPos[i][1] = aPos[i-1][1] + (aPos[i-1][2] - aPos[i-2][2]);
               aPos[i][0] = aPos[i-1][0];
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               aPos[i][2] = aPos[i-1][2] +  (aPos[i-1][2] - aPos[i-2][2]); 
               aPos[i][0] = aPos[i-1][0];
               aPos[i][1] = aPos[i-1][1];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               aPos[i][2] = aPos[i-1][2];
               aPos[i][1] = aPos[i-1][1] - (aPos[i-1][2] - aPos[i-2][2]);
               aPos[i][0] = aPos[i-1][0];
             }
           else if (vector[i]==3)             //FORWARD MOVE in Z
	     {
               aPos[i][1] = aPos[i-1][1]; 
               aPos[i][2] = aPos[i-1][2];
               aPos[i][0] = aPos[i-1][0] - (aPos[i-1][2] - aPos[i-2][2]);
              }
	   else  if (vector[i]==4)            //DOWN  MOVE  in Z
	     {
               aPos[i][1] = aPos[i-1][1]; 
               aPos[i][2] = aPos[i-1][2];
               aPos[i][0] = aPos[i-1][0] + (aPos[i-1][2] - aPos[i-2][2]);
             }
         }
     

        for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1]  && aPos[i][2]==aPos[j][2])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) &&  (aPos[i][2]==aPos[j][2])  && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] + 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0])  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] - 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)++;       
            if ( (aPos[i][2]==aPos[j][2] + 1)  &&  (aPos[i][1]==aPos[j][1]) && (aPos[i][0]==aPos[j][0])) (Collisions)++;
            if ( (aPos[i][2]==aPos[j][2] - 1)  &&  (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1])) (Collisions)++; 
          }    
        }
     

     }
 
  result = (Collisions/(1.0+Overlappings));
  return (result);
     
}    




double HPProtein3D::EvalChain(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 
 int i,j,Collisions,Overlappings;

 Collisions = 0;
 Overlappings = 0;
 double result;

 for (i=2;i<sizeChain;i++)
     for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1]  && aPos[i][2]==aPos[j][2])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) &&  (aPos[i][2]==aPos[j][2])  && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] + 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0])  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)++;
            if ( (aPos[i][0]==aPos[j][0] - 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)++;       
            if ( (aPos[i][2]==aPos[j][2] + 1)  &&  (aPos[i][1]==aPos[j][1]) && (aPos[i][0]==aPos[j][0])) (Collisions)++;
            if ( (aPos[i][2]==aPos[j][2] - 1)  &&  (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1])) (Collisions)++; 
          }    
        }
   
 result = (Collisions/(1.0+Overlappings));
  return (result);
}



double HPProtein3D::EvalChainModel(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 

 int i,j,Collisions,Overlappings;
 Collisions = 0;
 Overlappings = 0;
 double result;




 for (i=2;i<sizeChain;i++)
  for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1]  && aPos[i][2]==aPos[j][2])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) &&  (aPos[i][2]==aPos[j][2])  && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] + 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0])  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] - 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;       
            if ( (aPos[i][2]==aPos[j][2] + 1)  &&  (aPos[i][1]==aPos[j][1]) && (aPos[i][0]==aPos[j][0])) (Collisions)+=1;
            if ( (aPos[i][2]==aPos[j][2] - 1)  &&  (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1; 
          }    
          else if (IntConf[i]==1  && IntConf[j]==1) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) &&  (aPos[i][2]==aPos[j][2])  && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] + 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0])  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] - 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;       
            if ( (aPos[i][2]==aPos[j][2] + 1)  &&  (aPos[i][1]==aPos[j][1]) && (aPos[i][0]==aPos[j][0])) (Collisions)+=1;
            if ( (aPos[i][2]==aPos[j][2] - 1)  &&  (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1; 
          }    
        }

     

 result = (Collisions/(1.0+Overlappings));
  return (result);
}




double HPProtein3D::EvalChainLong(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 
 int i,j,Collisions,Overlappings;

 Collisions = 0;
 Overlappings = 0;
 double result;

 for (i=2;i<sizeChain;i++)
   for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1]  && aPos[i][2]==aPos[j][2])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0  && (i-j)>3)  
          {
	    if ( (aPos[i][0]==aPos[j][0]) &&  (aPos[i][2]==aPos[j][2])  && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)+=2;
            if ( (aPos[i][0]==aPos[j][0] + 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=2;
            if ( (aPos[i][0]==aPos[j][0])  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)+=2;
            if ( (aPos[i][0]==aPos[j][0] - 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=2;       
            if ( (aPos[i][2]==aPos[j][2] + 1)  &&  (aPos[i][1]==aPos[j][1]) && (aPos[i][0]==aPos[j][0])) (Collisions)+=2;
            if ( (aPos[i][2]==aPos[j][2] - 1)  &&  (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=2; 
          }    
          else if (IntConf[i]==0  && IntConf[j]==0 ) 
          {
	    if ( (aPos[i][0]==aPos[j][0]) &&  (aPos[i][2]==aPos[j][2])  && (aPos[i][1]==aPos[j][1] - 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] + 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0])  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1] + 1)) (Collisions)+=1;
            if ( (aPos[i][0]==aPos[j][0] - 1)  &&  (aPos[i][2]==aPos[j][2]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1;       
            if ( (aPos[i][2]==aPos[j][2] + 1)  &&  (aPos[i][1]==aPos[j][1]) && (aPos[i][0]==aPos[j][0])) (Collisions)+=1;
            if ( (aPos[i][2]==aPos[j][2] - 1)  &&  (aPos[i][0]==aPos[j][0]) && (aPos[i][1]==aPos[j][1])) (Collisions)+=1; 
          }    
         }
   

 result = (Collisions/(2*(1.0+Overlappings)));
  return (result);
}


double  HPProtein3D::TabuFindNewEval(int sizeChain, int pos, int sval, int* eval_local) //Evaluates a vector perturbed at a given position 
{
  int legal;
  double neweval; 
  int tab[5]={0,-1,1,2,3};

  if (Memory[(tab[sval]*sizeChain + pos)] > 0) return -1;

  legal = PullMoves(sizeChain,pos,sval);

    if(legal!=0)
	  { 
            neweval =  EvalChain(sizeChain,NewPos); 
            (*eval_local)++;
          }
   else neweval = -1;

   return neweval;
 
}

    
int  HPProtein3D::TabuSetNewBestPos(int sizeChain, int* maxval, double* currentEval,int maxvalconstraint,int duration) // Modifies the best current grid in local optimizer
{    
  int i,howmany, bestpos, whichone,legal,bpos;
   
    howmany = 1;
    bestpos = 0;
    bestallpos[0] = 0;

    for (i=0;i<4*sizeChain-1;i++)
      {
	if(gains[i]>gains[bestpos])
	  {
            bestpos = i;
            howmany = 1;
            bestallpos[0] = bestpos;
          }
        else if(gains[i]==gains[bestpos])    bestallpos[howmany++] = i;
             
      }            
    
    if(gains[bestpos]>-1 && (*maxval)<maxvalconstraint) 
     {
        (*maxval)++;
       if (howmany==1) whichone = 0;
       else  whichone = randomint(howmany);  

       bpos =  bestallpos[whichone];  
       Memory[bpos] = duration;  
    
       if(bpos<sizeChain)  legal = PullMoves(sizeChain,bpos,-1); //NECESARIO DETERMINAR AQUI COMO LLAMAR A PULLMOVES, SI SE LOGRA PUEDE SER VIRTUAL
       else   legal = PullMoves(sizeChain,bpos-sizeChain,1);
      
       //cout<<"whichone "<<whichone<<" bestpos "<<bestallpos[whichone]<<" howmany "<<howmany<<" legal "<<legal<<endl;
       //cout<<"maxval"<<(*maxval)<<" pos "<<bestallpos[whichone]+2<<endl;
      
       AssignPos(sizeChain,NewPos,Pos);
       TranslatePos(sizeChain,newvector);
       FindPos(sizeChain,newvector);   
        if (gains[bpos]>=(*currentEval)) 
          {
             (*currentEval) =  gains[bpos];
             for (i=0;i<sizeChain;i++) statvector[i] = newvector[i]; 
          }    

	//cout<<"CurrentVal "<<*currentEval<<" realeval "<<EvalOnlyVector(sizeChain,newvector)<<endl;   
     }
     else whichone = -1;

 return whichone;
}




double  HPProtein3D::TabuOptimizer(int sizeChain, unsigned int* vector, double currentEval, int* eval_local, int maxvalconstraint, int duration) // Given a chain of molecules, and a position makes a pull move 
{   
  int i,Continue;
  int maxval;
  double neweval1, neweval2;
  Continue = 1;
  maxval = 0;
  (*eval_local) = 0;
  double InitVal = currentEval;
  
  gains[sizeChain-1] = -1;
  gains[2*sizeChain-1] = -1;
  FindPos(sizeChain,vector); // Puts in Pos the grid positions contained in vector

  for (i=0;i<2*sizeChain;i++) Memory[i] = 0;

  for (i=0;i<sizeChain;i++)  statvector[i] =  vector[i];

  while ( (Continue)  && maxval<=maxvalconstraint)
    {
      for (i=0;i<sizeChain-1;i++)
       {
	 neweval1 = TabuFindNewEval(sizeChain,i,-1,eval_local);
         gains[i] =  neweval1;                 
       	 neweval2 = TabuFindNewEval(sizeChain,i,1,eval_local);
         gains[sizeChain+i] =  neweval2;       
	 //  cout<<i<<" --> "<<neweval1<<" "<<neweval2<<endl;       
       }     

      Continue =  (TabuSetNewBestPos(sizeChain,&maxval,&currentEval,maxvalconstraint,duration) != -1);
      for (i=0;i<2*sizeChain;i++)
	if(Memory[i]>0) Memory[i]--;
     }    

   if(currentEval>InitVal) for (i=0;i<sizeChain;i++)  vector[i] =  statvector[i];
              
  //TranslatePos(sizeChain,vector);
  return currentEval; 
}



int  HPProtein3D::Feasible(unsigned int* s, int pos) //Checks if there are not overlappings in the grid
{ 
  int  Overlappings = 0;
  int j;
  

 if (pos  <2) return 1;

  PutMoveAtPos(pos,s[pos]); 
  j = 0;
 
  while (j<pos && Overlappings==0)   // Check for Overlappings and Collissions in all the    molecules except the previous one
    {
      Overlappings +=  (Pos[pos][0]==Pos[j][0] && Pos[pos][1]==Pos[j][1] && Pos[pos][2]==Pos[j][2]); 
      j++;
    }  
  return (Overlappings == 0);

}



void  HPProtein3D::PutMoveAtPos(int pos, int mov)  //Sets a given move in the grid ME QUEDE AQUI
{
  int i;
  if (pos<2) return;
  i = pos;

    if(Pos[i-1][1]==Pos[i-2][1])
         {
           if (mov==0)                  //UP MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] +  (Pos[i-1][0] - Pos[i-2][0]);
             }
           else if (mov==1)             //FORWARD MOVE
	     {
               Pos[i][0] = Pos[i-1][0] +  (Pos[i-1][0] - Pos[i-2][0]); 
               Pos[i][1] = Pos[i-1][1];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][0] = Pos[i-1][0];
               Pos[i][1] = Pos[i-1][1] -   (Pos[i-1][0] - Pos[i-2][0]);
             }
	 }
        
       if(Pos[i-1][0]==Pos[i-2][0])
         {
           if (mov==0)                  //UP MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0] -  (Pos[i-1][1] - Pos[i-2][1]);
             }
           else if (mov==1)             //FORWARD MOVE
	     {
               Pos[i][1] = Pos[i-1][1] +  (Pos[i-1][1] - Pos[i-2][1]); 
               Pos[i][0] = Pos[i-1][0];
             }
	   else                                //DOWN  MOVE
	     {
               Pos[i][1] = Pos[i-1][1];
               Pos[i][0] = Pos[i-1][0] + (Pos[i-1][1] - Pos[i-2][1]);
             }
	 }
}






// ***************************** IMPLEMENTACION PARA EL CASO 3D DIAMOND ******************************************************


void HPProtein3Diamond::SetInitPos(int** aPos) // Init the first positions of the grid
{
    aPos[0][0] = 0;
    aPos[0][1] = 0;
    aPos[0][2] = 2;
    aPos[1][0] = 1;
    aPos[1][1] = 1;
    aPos[1][2] = 1;  
}


void HPProtein3Diamond::SetInitPosAt_i(int** aPos, int i) // Init the first positions of the grid
{

    aPos[i][0] = 0;
    aPos[i][1] = 0;
    aPos[i][2] = 2;
    aPos[i+1][0] = 1;
    aPos[i+1][1] = 1;
    aPos[i+1][2] = 1;  
}


void  HPProtein3Diamond::CreatePos() //Creates the first positions of the grid
 {
  int i;
  
 Pos = new int*[sizeProtein];
 NewPos = new int*[sizeProtein];

  for (i=0;i<sizeProtein;i++)
   {
    Pos[i] = new int[3];
    NewPos[i] = new int[3];
   } 

  SetInitPos(Pos); 
  SetInitPos(NewPos); 
  SetInitMoves();
 }



HPProtein3Diamond::HPProtein3Diamond(int sizeP, int* HConf):HPProtein(sizeP,HConf,3,6)
{ 
 
  CreatePos();
  
}




HPProtein3Diamond::~HPProtein3Diamond()
{
}


void  HPProtein3Diamond::FindPos(int sizeChain, unsigned* vector) //Translates the vector to the  positions in the grid 
 {
  int i;
  SetInitPos(Pos);
   

for (i=2;i<sizeChain;i++)
     {      
           if (vector[i]==0)                  //UP MOVE
	     {
               Pos[i][0] = Pos[i-2][0]; 
               if(Pos[i-2][1] > Pos[i-1][1]) Pos[i][1] = Pos[i-2][1] - 2;
               else  Pos[i][1] = Pos[i-2][1] +  2;
               if(Pos[i-2][2] > Pos[i-1][2]) Pos[i][2] = Pos[i-2][2] - 2;
               else  Pos[i][2] = Pos[i-2][2] +  2;
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               if(Pos[i-2][0] > Pos[i-1][0]) Pos[i][0] = Pos[i-2][0] - 2;
               else  Pos[i][0] = Pos[i-2][0] +  2;
               if(Pos[i-2][1] > Pos[i-1][1]) Pos[i][1] = Pos[i-2][1] - 2;
               else  Pos[i][1] = Pos[i-2][1] +  2;
               Pos[i][2] = Pos[i-2][2];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               if(Pos[i-2][0] > Pos[i-1][0]) Pos[i][0] = Pos[i-2][0] - 2;
               else  Pos[i][0] = Pos[i-2][0] +  2;
               Pos[i][1] = Pos[i-2][1];
               if(Pos[i-2][2] > Pos[i-1][2]) Pos[i][2] = Pos[i-2][2] - 2;
               else  Pos[i][2] = Pos[i-2][2] +  2;                      
             }
     }

         
 }

void HPProtein3Diamond::TranslatePos(int sizeChain, unsigned* newvector) // Translate an array of grid positions to a vector of movements
{

  int i;
  newvector[0] = 0;
  newvector[1] = 0;
 
  
 for (i=2;i<sizeChain;i++)
     {

       if(Pos[i][0] == Pos[i-2][0])  newvector[i]=0;
       else if(Pos[i][1] == Pos[i-2][1])  newvector[i]=2;
       else if(Pos[i][2] == Pos[i-2][2])  newvector[i]=1;
     }
 }
 
 
int  HPProtein3Diamond::CheckFree(int sizeChain, int** aPos, int P1, int P2, int P3)   // Checks if coordinate at_i_Pos is occuppied in the grid
{ 
int i,Found;

 i = 0;
 Found = 0;

 while(i<sizeChain && !Found)
 {
  Found = ((aPos[i][0] == P1) && (aPos[i][1] == P2) &&  (aPos[i][2] == P3));
  i = i + 1; 
 }
 return (Found==0);
}
 

int  HPProtein3Diamond::CheckValidity(int at_i_Pos, int** aPos) //Checks if the chain is connected at position at_i_Pos
 { 
   int resp  = 1;
   if (at_i_Pos > 0)   resp = ( (abs(aPos[at_i_Pos][0] - aPos[at_i_Pos-1][0]) +  abs(aPos[at_i_Pos][1] - aPos[at_i_Pos-1][1]) +  (abs(aPos[at_i_Pos][2] - aPos[at_i_Pos-1][2]))) == 3);
   return resp;
 } 




void HPProtein3Diamond::AssignPos(int sizeChain, int** Pos, int** OtherPos)     // Copy the position of one grid into another
 {
  int i;
   for (i=0;i<sizeChain;i++)
   {
    OtherPos[i][0] = Pos[i][0];
    OtherPos[i][1] = Pos[i][1];
    OtherPos[i][2] = Pos[i][2];
    }
 }

void HPProtein3Diamond::PrintPos(int sizeChain, int** aPos)       // Prints all or part of the protein's grid
{
  int i;

  for (i=0;i<sizeChain;i++) 
    { 
      cout<<i<<"-->"<<aPos[i][0]<<" "<<aPos[i][1]<<" "<<aPos[i][2]<<endl;  
    } 
}


double HPProtein3Diamond::EvalChain(int sizeChain, unsigned int* vector, int** aPos)
{
  // Given a chain of molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 

  int i,j,Collisions,Overlappings;
  double result;
 Collisions = 0;
 Overlappings = 0;
   SetInitPos(Pos);
   
   for (i=2;i<sizeChain;i++)
     {      
           if (vector[i]==0)                  //UP MOVE
	     {
               Pos[i][0] = Pos[i-2][0]; 
               if(Pos[i-2][1] > Pos[i-1][1]) Pos[i][1] = Pos[i-2][1] - 2;
               else  Pos[i][1] = Pos[i-2][1] +  2;
               if(Pos[i-2][2] > Pos[i-1][2]) Pos[i][2] = Pos[i-2][2] - 2;
               else  Pos[i][2] = Pos[i-2][2] +  2;
             }
           else if (vector[i]==1)             //FORWARD MOVE
	     {
               if(Pos[i-2][0] > Pos[i-1][0]) Pos[i][0] = Pos[i-2][0] - 2;
               else  Pos[i][0] = Pos[i-2][0] +  2;
               if(Pos[i-2][1] > Pos[i-1][1]) Pos[i][1] = Pos[i-2][1] - 2;
               else  Pos[i][1] = Pos[i-2][1] +  2;
               Pos[i][2] = Pos[i-2][2];
             }
	   else  if (vector[i]==2)            //DOWN  MOVE
	     {
               if(Pos[i-2][0] > Pos[i-1][0]) Pos[i][0] = Pos[i-2][0] - 2;
               else  Pos[i][0] = Pos[i-2][0] +  2;
               Pos[i][1] = Pos[i-2][1];
               if(Pos[i-2][2] > Pos[i-1][2]) Pos[i][2] = Pos[i-2][2] - 2;
               else  Pos[i][2] = Pos[i-2][2] +  2;                      
             }
     }
     
  
  for (i=2;i<sizeChain;i++)
     for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1]  && aPos[i][2]==aPos[j][2])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
              if ((Pos[j][0] == Pos[i-1][0])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)))  (Collisions)++;
               if ((Pos[j][2] == Pos[i-1][2])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)))  (Collisions)++;    
              if ((Pos[j][1] == Pos[i-1][1])  &&  (Pos[j][1] == Pos[i-1][1]) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)) && ((Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)) )  (Collisions)++; 
           }    
        }
      
  
 
  result = (Collisions/(1.0+Overlappings));
  return (result);
     
}    




double HPProtein3Diamond::EvalChain(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 
 int i,j,Collisions,Overlappings;

 Collisions = 0;
 Overlappings = 0;
 double result;
 //PrintPos(sizeChain,Pos);
 for (i=2;i<sizeChain;i++)
     for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{
          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1]  && aPos[i][2]==aPos[j][2])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
              if ((Pos[j][0] == Pos[i-1][0])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)))  (Collisions)++;
               if ((Pos[j][2] == Pos[i-1][2])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)))  (Collisions)++;    
              if ((Pos[j][1] == Pos[i-1][1])  &&  (Pos[j][1] == Pos[i-1][1]) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)) && ((Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)) )  (Collisions)++; 
           }    
        }
 // cout<<Collisions<<" "<<Overlappings<<endl;  
 result = (Collisions/(1.0+Overlappings));
  return (result);
}



double HPProtein3Diamond::EvalChainModel(int sizeChain, int** aPos)   //Evaluates the chain but using the grid positions and not the vector
{
  // Given the positions of the  molecules, calculates the numer of Collisions with 
  // neighboring same sign molecules, and the number of Overlappings molecules.
  // InitConf is the configuration of the Chain of molecules
  //  vector is the set of moves for all 

 int i,j,Collisions,Overlappings;
 Collisions = 0;
 Overlappings = 0;
 double result;


 for (i=2;i<sizeChain;i++)
     for (j=0;j<i-2;j++)   // Check for Overlappings and Collissions in all the    molecules except the previous one
	{


          if(aPos[i][0]==aPos[j][0] && aPos[i][1]==aPos[j][1]  && aPos[i][2]==aPos[j][2])  (Overlappings)++;
          else if (IntConf[i]==0  && IntConf[j]==0) 
          {
              if ((Pos[j][0] == Pos[i-1][0])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)))  (Collisions)+=2;
               if ((Pos[j][2] == Pos[i-1][2])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)))  (Collisions)+=2;    
              if ((Pos[j][1] == Pos[i-1][1])  &&  (Pos[j][1] == Pos[i-1][1]) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)) && ((Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)) )  Collisions+=2; 
           }   
          else if (IntConf[i]==1  && IntConf[j]==1) 
          {
              if ((Pos[j][0] == Pos[i-1][0])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)))  (Collisions)-=1;
               if ((Pos[j][2] == Pos[i-1][2])  && ( (Pos[i-1][1] > Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] - 2) || (Pos[i-1][1] < Pos[i][1]  &&  Pos[j][1] == Pos[i-1][1] + 2)) && ( (Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)))  (Collisions)-=1;    
              if ((Pos[j][1] == Pos[i-1][1])  &&  (Pos[j][1] == Pos[i-1][1]) && ( (Pos[i-1][2] > Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] - 2) || (Pos[i-1][2] < Pos[i][2]  &&  Pos[j][2] == Pos[i-1][2] + 2)) && ((Pos[i-1][0] > Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] - 2) || (Pos[i-1][0] < Pos[i][0]  &&  Pos[j][0] == Pos[i-1][0] + 2)) )  Collisions-=1; 
           }   

        }
   

 result = (Collisions/(1.0+Overlappings));
  return (result);
}



int  HPProtein3Diamond::Feasible(unsigned int* s, int pos) //Checks if there are not overlappings in the grid
{ 
  int  Overlappings = 0;
  int j;
  

 if (pos  <2) return 1;

  PutMoveAtPos(pos,s[pos]); 
  j = 0;
 
  while (j<pos && Overlappings==0)   // Check for Overlappings and Collissions in all the    molecules except the previous one
    {
      Overlappings +=  (Pos[pos][0]==Pos[j][0] && Pos[pos][1]==Pos[j][1] && Pos[pos][2]==Pos[j][2]); 
      j++;
    }  
  return (Overlappings == 0);

}


