#include "auxfunc.h"
#include "Treeprob.h"
#include <math.h>
#include <time.h>

 

ProbTree::ProbTree(int vars, int* AllInd, int psize, int clusterkey,Popul* pop)
 {
  int i,j;
  length = vars; 
  Pop = pop;
  AllProb = new double[length]; 
  AllSecProb = new double*[4];  
  for(i=0; i < 4; i++) AllSecProb[i] = new double[length*(length-1)/2];
  MutualInf = new double [length*(length-1)/2];
  Tree = new int[length];
  Queue =  new int [length];
  actualpoolsize = 0;
  for(i=0; i < psize; i++) actualpoolsize += ( AllInd[i] == clusterkey );
  actualindex = new int[actualpoolsize];
  j = 0;
  for(i=0; i < psize; i++) 
  {
	 if ( AllInd[i] == clusterkey ) actualindex[j++] = i;
  }
 }

void ProbTree::CalProb(Popul* Pop)
 {
   int aux,i,j,k,l;
	// Primer paso
	// Se calculan todas las probabilidades de primer y segundo orden
	// para todas las variables del cromosoma, teniendo en cuenta solo
	// aquellos individuos incluidos en el conjunto seleccionado

    
	 for(j=0; j<length-1; j++)
	  {
		AllProb[j]=0;

		for(k=j+1 ; k<length; k++)
			{
              aux = j*(2*length-j+1)/2 +k-2*j-1;
			  AllSecProb[0][aux]=0;
			  AllSecProb[1][aux]=0;
			  AllSecProb[2][aux]=0;
			  AllSecProb[3][aux]=0;
if(actualpoolsize>0)
{
			  for(l=0; l<actualpoolsize; l++)
				  {
					  //Se calcula la probabilidad de cada gen en genepool		
				    i = actualindex[l];
					++AllSecProb[2*Pop->P[i][j]+Pop->P[i][k]][aux];
					if (k==j+1) AllProb[j]+=Pop->P[i][j];

				  }

              AllSecProb[0][aux]= AllSecProb[0][aux] / actualpoolsize;
			  AllSecProb[1][aux]= AllSecProb[1][aux] / actualpoolsize;
              AllSecProb[2][aux]= AllSecProb[2][aux] / actualpoolsize;
			  AllSecProb[3][aux]= AllSecProb[3][aux] / actualpoolsize;
}

			 }

	 }

  AllProb[length-1]=0;
  
if (actualpoolsize >0)
{
  for(i=0; i<actualpoolsize; i++) AllProb[length-1]+=Pop->P[actualindex[i]][length-1]; 
  for(i=0; i<length; i++) AllProb[i] = AllProb[i] / actualpoolsize; 
 }
}

void ProbTree::CalProbFvect(Popul* Pop, double *vector)
 {
   int aux,i,j,k;
	// Primer paso
	// Se calculan todas las probabilidades de primer y segundo orden
	// para todas las variables del cromosoma, teniendo en cuenta solo
	// aquellos individuos incluidos en el conjunto seleccionado

    genepoollimit = Pop->psize;

	 for(j=0; j<length-1; j++)
	  {
		AllProb[j]=0;
        
		for(k=j+1 ; k<length; k++)
			{
              aux = j*(2*length-j+1)/2 +k-2*j-1;
			  AllSecProb[0][aux]=0;
			  AllSecProb[1][aux]=0;
			  AllSecProb[2][aux]=0;
			  AllSecProb[3][aux]=0;

			  for(i=0; i<genepoollimit; i++)
				  {
					  //Se calcula la probabilidad de cada gen en genepool		
						AllSecProb[2*Pop->P[i][j]+Pop->P[i][k]][aux]+=vector[i];
						if ((k==j+1) && (Pop->P[i][j]==1)) AllProb[j]+=vector[i];
				  }

			  
			 }

	 }

  AllProb[length-1]=0;
  for(i=0; i<genepoollimit; i++)
  {
	  if (Pop->P[i][length-1]==1) AllProb[length-1]+=vector[i]; 
  } 
 genepoollimit = 1;  
 actualpoolsize = 1;
}

void ProbTree::CalMutInf()
 {
   int j,k;
	 // Segundo paso
	 // Se construye un arbol que contendra un conjunto optimo
	 // de las dependencias de primer orden. El criterio de optimalidad
	 // para incorporar una relacion al conjunto es maximizar la
	 // la informacion mutua entre las variables. ( VER Kullback-Lieber divergence )

	 // Se halla la informacion mutua.

   double aux1,aux2;
   int aux;


      aux1 =0;

	 for(j=0; j<length-1; j++)
	  	for(k=j+1 ; k<length; k++)
			{

               
				aux = j*(2*length-j+1)/2 +k-2*j-1;
				MutualInf[aux]=0; // (AllProb[j] !=1 && AllProb[k] !=1 )

				aux2=(AllSecProb[0][aux]+aux1)/(actualpoolsize);
				if (aux2 > 0.000000001) 
                MutualInf[aux]+=aux2*(log(aux2/((1-AllProb[j])*(1-AllProb[k]))));
				

				aux2=(AllSecProb[1][aux]+aux1)/(actualpoolsize);
				if (aux2 > 0.000000001 ) //.000000001
				MutualInf[aux]+=aux2*(log(aux2/((1-AllProb[j])*(AllProb[k]))));

				aux2=(AllSecProb[2][aux]+aux1)/(actualpoolsize);
				if (aux2 > 0.000000001 ) 
				MutualInf[aux]+=aux2*(log(aux2/((AllProb[j])*(1-AllProb[k]))));

				aux2=(AllSecProb[3][aux]+aux1)/(actualpoolsize);
				if (aux2 > 0.000000001) 
				MutualInf[aux]+=aux2*(log(aux2/((AllProb[j])*(AllProb[k]))));
		}
}	           			
 
 int ProbTree::FindRootNode()
 {
	 // El nodo raiz del arbol se puede escoger aleatoriamente,
	 // Siguiendo a De Bonet aqui se escoge el de menor entropia incondicional
	// Se determina la variable con menor entropia incondicional
    int j;
	double min;
    int minindex=0;
	double aux=0;

	min=-((AllProb[0]*log(AllProb[0])+ (1-AllProb[0])*log(1-AllProb[0])));

	for(j=0; j<length; j++)
	 {
			aux= -((AllProb[j]*log(AllProb[j])+ (1-AllProb[j])*log(1-AllProb[j])));
		  if (aux<=min)
			 {
				minindex=j;
				min=aux;
			 }
	 }


  return minindex;

 }

 int ProbTree::RandomRootNode()
 {
  return int((length-1)*myrand());
 }

void ProbTree::MakeTree(int rootn)
 {
  // En cada paso se incorpora el nodo que no estando en el arbol
  // tiene el mayor valor de informacion mutua con alguno de los
  // nodos que ya estan en el arbol, el cual sera ademas su padre

	double max,threshhold,auxm;
	int maxsonindex;
	int maxfatherindex;
	int i,j,k,aux;


	 for(i=0; i<length; i++) Tree[i]=i;
	 Tree[rootn]=-1;
	 threshhold=-100;//0.005;


	for(i=0; i<length-1; i++)  // Para los n-1 nodos que faltan por incorporar
	 {
		max=-10;
		for(j=0; j<length; j++)
		 for(k=0; k<length; k++)
		  {
			 if (Tree[j]==j && Tree[k]!=k )
				  {

					  aux = j*(2*length-j+1)/2 +k-2*j-1;
					  
					  if (j<k) aux = j*(2*length-j+1)/2 +k-2*j-1;
					  else aux = k*(2*length-k+1)/2 +j-2*k-1;
					  
					  auxm=MutualInf[aux];
					  
					  if (auxm>max)
						 {
								maxsonindex=j;
								maxfatherindex=k;
								max=auxm;
						 }
				  }

		  }
		 if (max>=threshhold) Tree[maxsonindex]=maxfatherindex;
		 else Tree[maxsonindex]=-1;
	 } // Llegado este punto se supone que todos los nodos esten en el arbol
  ArrangeNodes();
 }	

void ProbTree::MakeRandomTree(int rootn)
 {
  
	int* Unconnected;
	int* Connected;
	int i, conn, unconn, uind, cind;

	Unconnected = new int[length];
    Connected = new int[length];

	
	Connected[0] = rootn;
	Tree[rootn] = -1;
	conn = 1;

    for(i=0; i<length; i++) Unconnected[i] = i;
    Unconnected[rootn] = Unconnected[length-1];
	unconn = length - 1;
	
	while (unconn>0)
	{
	  uind = int((unconn-1)*myrand());
	  cind = int((conn-1)*myrand());
      Connected[conn++] = Unconnected[uind];
	  Tree[Unconnected[uind]] = Connected[cind];
	  Unconnected[uind] = Unconnected[-1 + unconn-- ];
	}

	delete[] Unconnected;
    delete[] Connected;

	ArrangeNodes();
 }	


void ProbTree::GenPop(int From, Popul* NewPop)
 {
  // Tercer paso
   //Se procede a generar todos los individuos de la nueva poblacion
  // Se utiliza para ello el arbol

double auxprob, cutoff,aux2,aux1;
 int aux,i,j,p,current;
 int Npopsize;

 Npopsize=NewPop->psize;
 
 for (p=0;p<length; p++ )
	 {
        j = NextInOrder(p);
		for(i=From; i<Npopsize; i++)
		  { 
		    cutoff = myrand();			
			if (Tree[j] == -1)
            {
			 if (cutoff > AllProb[j]) NewPop->P[i][rootnode]=0;
		     else NewPop->P[i][j]=1;
			}
			else
            {
		     if (NewPop->P[i][Tree[j]]==1)
			  {
				if (j<Tree[j]) 
				 {
  				  aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1;
				  aux2=(AllSecProb[3][aux]+aux1);
				 }
				 else 
				 {
 				  aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1;
				  aux2=(AllSecProb[3][aux]+aux1);
				 }
				 auxprob=aux2/AllProb[Tree[j]];
				}
			 else
				{
                    if(j<Tree[j]) 
					{
						aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1;
						aux2=(AllSecProb[2][aux]+aux1);
					}
					else 
					{
						aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1;
						aux2=(AllSecProb[1][aux]+aux1);
					}
					auxprob=aux2/(1-AllProb[Tree[j]]);
				}
				
				if (cutoff > auxprob) NewPop->P[i][j]=0;
				else NewPop->P[i][j]=1;
			  }
		} //first for
	}//second for
  
}



void ProbTree::ResetProb()
 {
   int aux,j,k;
	// Primer paso
	// Se calculan todas las probabilidades de primer y segundo orden
	// para todas las variables del cromosoma, teniendo en cuenta solo
	// aquellos individuos incluidos en el conjunto seleccionado

	 for(j=0; j<length-1; j++)
	  {
		AllProb[j]=0;

		for(k=j+1 ; k<length; k++)
			{
              aux = j*(2*length-j+1)/2 +k-2*j-1;
			  
			  AllSecProb[0][aux]=0;
			  AllSecProb[1][aux]=0;
			  AllSecProb[2][aux]=0;
			  AllSecProb[3][aux]=0;
			}

	 }

  AllProb[length-1]=0;
  genepoollimit=0;
}

void ProbTree::PrintMut()
{
  int aux,j,k;

	for(j=0; j<length; j++)
	{
	  	for(k=0 ; k<length; k++)
		{
          if (k==j)
		  {
 	        fprintf(f1,"%f ",0);  	       
			printf("%f ",0);  	       
			  
		  }	  
		   else if(j<k)
		   { 
  			aux = j*(2*length-j+1)/2 +k-2*j-1;
			printf("%f ",MutualInf[aux]);
 	        fprintf(f1,"%f ",MutualInf[aux]); 
		   }
		   else
			{ 
			aux = k*(2*length-k+1)/2 +j-2*k-1;
			printf("%f ",MutualInf[aux]);
 	        fprintf(f1,"%f ",MutualInf[aux]); 
		   }
		}	
		fprintf(f1,"\n");
		printf("\n");
	}
fclose(f1);
}

void ProbTree::ArrangeNodes()
{ 
	int j,p,current;

    Queue[0] = rootnode;
	current = 0;
    p = 0;
	while (p < length)
	{
		for(j=0; j<length; j++)
			if (Tree[j]==Queue[p]) Queue[++current]=j;
		   p++; 
    }
}

double ProbTree::Prob (unsigned* vector)
 {
  // Se determina cual es la probabilidad del
 // vector dado el arbol

 double auxprob,aux2,aux1,prob;
 int aux,j;

 aux1= 0;
 prob = 1;

	for (j=0;j<length; j++ )
	 {
	  if (Tree[j]==-1) prob = (vector[j]==1)?prob*AllProb[j]:prob*(1-AllProb[j]); 
	  else 
	  {	  
		 if (j<Tree[j]) 
		 {
				 aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1;
	        	 aux2=(AllSecProb[2*vector[j]+vector[Tree[j]]][aux]+aux1);
		 }
		 else 
		 { 
 		 	      aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1;
				  aux2=(AllSecProb[2*vector[Tree[j]]+vector[j]][aux]+aux1);
		 }
		
		 //auxprob= (vector[Tree[j]]==1)?(aux2/AllProb[Tree[j]]):(aux2/(1-AllProb[Tree[j]]));  
		 if( vector[Tree[j]]==1 && AllProb[Tree[j]]>0) auxprob = aux2/AllProb[Tree[j]];
		 else if( vector[Tree[j]]==0 && AllProb[Tree[j]] < 1) auxprob = aux2/(1-AllProb[Tree[j]]);
		 else auxprob = 0;
         
		 if (auxprob == 0) return 0;
		 else prob*=auxprob;
	  }
	
	}
return prob; 
}


ProbTree::~ProbTree()
 {
  int i;
  for(i=0; i < 4; i++) delete[] AllSecProb[i];  
  delete[] AllSecProb;
  delete[] AllProb;
  delete[] Tree;
  delete[] Queue;
  delete[] MutualInf;
  delete[] actualindex;
 }


int ProbTree::NextInOrder(int previous)
{
	return Queue[previous];
}

void ProbTree::GenIndividual (Popul* NewPop, int pos)
 {
  // The vector in position pos is generated
 
 double auxprob,aux2,cutoff,aux1;
 int aux,j,i;
 

 	for (i=0;i<length; i++ )
	 {
      j = NextInOrder(i);
	  cutoff = myrand();
	  if (Tree[j]==-1) 
	  {
		if (cutoff > AllProb[j]) NewPop->P[pos][rootnode]=0;
		else NewPop->P[pos][rootnode]=1;
      } 
	  else 
	  {	  
	   if (NewPop->P[pos][Tree[j]]==1)
		 {
	   	  if (j<Tree[j]) 
		  {
		   aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1;
		   aux2=(AllSecProb[3][aux]+aux1);
		  }
		  else 
		  {
 		   aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1;
		   aux2=(AllSecProb[3][aux]+aux1);
		  }
		  auxprob=aux2/AllProb[Tree[j]];
		 }
		else
		{
         if(j<Tree[j]) 
		 {
		  aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1;
		  aux2=(AllSecProb[2][aux]+aux1);
		 }
		 else 
		 {
		  aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1;
		  aux2=(AllSecProb[1][aux]+aux1);
		 }
		auxprob=aux2/(1-AllProb[Tree[j]]);
		}
		if (cutoff > auxprob) NewPop->P[pos][j]=0;
  	    else NewPop->P[pos][j]=1;
	  }
	}
}


