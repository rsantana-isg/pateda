#include <stdio.h> 
#include <memory.h> 
#include "CNF.h" 
#include "auxfunc.h" 
 
CNF::CNF(int cantvars, int cantc,int dimclaus) 
{ 
  SetVariables(cantvars,cantc,dimclaus); 
} 
 
void CNF::SetVariables(int cantvars, int cantc,int dimclaus) 
{ 
	int i; 
	cantclauses = cantc; 
	NumberVars = cantvars; 
	dimclause = dimclaus; 
	clauses = new int*[cantclauses]; 
	adjmatrix = new int*[NumberVars]; 
	clause_weights = new double[cantclauses]; 
        nsastindex = new unsigned[NumberVars]; 
	for(i=0;i<NumberVars;i++)  
	{ 
		adjmatrix[i] = new int[NumberVars]; 
		memset(adjmatrix[i] , 0, sizeof(int)*(NumberVars)); 
	} 
    
	for(i=0;i<cantclauses;i++) clause_weights[i] = 1; 
	Tot_Weight = cantclauses; 
} 
 
CNF::~CNF() 
{  
	int  i; 
	for(i=0;i<cantclauses;i++)  
	{ 
		delete[] clauses[i]; 
	} 
 
	delete[] clauses; 
	for(i=0;i<NumberVars;i++) 	delete[] adjmatrix[i]; 
	delete[] adjmatrix; 
	delete[] clause_weights; 
        delete[] nsastindex;
} 
 
void CNF::Addclause(int index, int* clau) 
{ 
	clauses[index] = clau; 
} 
 
double CNF::SatClauses(unsigned* vector) 
{ 
  int i,j; 
  unsigned claus1,aux1; 
  int satisfied; 
  double numbersatisfied; 
   Tot_Weight = 0; 
    numbersatisfied = 0; 
    Satisfied = 0;
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  satisfied = 0; 
	  while ( ! satisfied && j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  satisfied = (vector[claus1]==aux1); 
		  j++; 
	  } 
	  numbersatisfied += satisfied*clause_weights[i]; 
          Satisfied += satisfied;
          Tot_Weight += clause_weights[i]; 
	} 
    
    return (numbersatisfied/Tot_Weight)*cantclauses;  
} 

double CNF::SatClausesClean(unsigned* vector) 
{ 
  int i,j; 
  unsigned claus1,aux1; 
  int satisfied; 
  double numbersatisfied; 
   Tot_Weight = 0; 
    numbersatisfied = 0; 
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  satisfied = 0; 
	  while ( ! satisfied && j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  satisfied = (vector[claus1]==aux1); 
		  j++; 
	  } 
	  numbersatisfied += satisfied; 
	 
	} 
	Satisfied = numbersatisfied;
    return numbersatisfied;  
} 
 



double CNF::SatClausesChange(unsigned* vector) 
{ 
  int i,j,ns; 
  int claus1,aux1; 
  int satisfied; 
  double numbersatisfied; 
  double bval,auxval;
   Tot_Weight = 0;
   ns = 0; 
    numbersatisfied = 0; 
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  satisfied = 0; 
	  while ( ! satisfied && j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  satisfied = (vector[claus1]==aux1); 
		  j++; 
	  } 
	  if (!satisfied) nsastindex[ns++] = i;
	 
	  numbersatisfied += satisfied*clause_weights[i]; 
	  Tot_Weight += clause_weights[i]; 
	} 

	 bval = numbersatisfied;
	for(int k=0;k<ns;k++)
        { i =  nsastindex[randomint(ns)];
         j = 0; 
	 //while (j<dimclause) 
	   j=randomint(3);
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
                  claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  vector[claus1]=1-vector[claus1];
                  auxval = SatClauses(vector);
                  if (auxval<bval) vector[claus1]=1-vector[claus1];
                  else bval = auxval;
		  //if(myrand()>0.5) vector[claus1]=1-vector[claus1];
		  j++; 
	  } 
        }
	 return SatClauses(vector);  
} 


double CNF::SatClauses(int* vector) 
{ 
  int i,j; 
  int claus1,aux1; 
  int satisfied; 
  double numbersatisfied; 
   Tot_Weight = 0; 
    numbersatisfied = 0; 
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  satisfied = 0; 
	  while ( ! satisfied && j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  satisfied = (vector[claus1]==aux1); 
		  j++; 
	  } 
	  numbersatisfied += satisfied*clause_weights[i]; 
	  Tot_Weight += clause_weights[i]; 
	} 
    return (numbersatisfied/Tot_Weight)*cantclauses;  
} 


// This function calculate the ave. sum of the  
// univ. prob. of the variables that do not 
// satisfy a clause and multiply it by the  
// the weight of the clause 
// i.e. for the weight to be maximum all no 
// variable in the population satify the given clause 
 
void CNF::UpdateWeights(double* weights) 
{ 
  int i,j; 
  unsigned claus1,aux1; 
  double sum; 
 
     
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  sum = 0; 
	  while (j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  if (aux1 == 0) sum += (weights[claus1]); 
		  else sum += (1-weights[claus1]); 
		  j++; 
	  } 
	  clause_weights[i] += (sum/dimclause); 
	}  
} 
 
 
 
void CNF::EqualUpdateWeights(double* weights) 
{ 
  int i,j; 
  unsigned claus1,aux1; 
  double allsum,sum; 
 
  allsum = 0; 
     
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  sum = 0; 
	  while (j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  if (aux1 == 0) sum += (weights[claus1]); 
		  else sum += (1-weights[claus1]); 
		  j++; 
	  } 
	  allsum += (sum/dimclause); 
	}  
	for(i=0;i<cantclauses;i++) clause_weights[i] *= (allsum/cantclauses); 
} 
 
// This function calculate the clauses' weights  
// given a total reward to be shared 
// clauses that are almost not satisfied in the population 
// have the maximum weight 
 
void CNF::AdaptWeights(double reward, unsigned* vector) 
{  
	int i,j,aux1,claus1,numbersatisfied,satisfied; 
 
	numbersatisfied = 0; 
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  satisfied = 0; 
	  while ( ! satisfied && j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  satisfied = (vector[claus1]==aux1); 
		  j++; 
	  } 
	  numbersatisfied += satisfied; 
	} 
     
	Satisfied = numbersatisfied; 
	for(i=0;i<cantclauses;i++)  
	{ 
	  j = 0; 
	  satisfied = 0; 
	  while ( ! satisfied && j<dimclause) 
	  { 
		  aux1 = (clauses[i][2*j]!=-1); 
          claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		  satisfied = (vector[claus1]==aux1); 
		  j++; 
	  } 
	  if (!satisfied) clause_weights[i] += reward/(cantclauses-numbersatisfied); 
	} 
  Tot_Weight += reward; 
} 
 
 
void CNF::FillMatrix() 
{ 
  int i,j,k; 
  int claus1,claus2; 
 
   for(i=0;i<cantclauses;i++)  
	 for(j=0;j<dimclause-1;j++)  
       for(k=j+1;k<dimclause;k++)  
	   { 
		   claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		   claus2 = (clauses[i][2*k]==-1)?clauses[i][2*k+1]:clauses[i][2*k]; 
           //(adjmatrix[claus1][claus2])++; 
		   //(adjmatrix[claus2][claus1])++; 
		   adjmatrix[claus2][claus1]=1; 
		   adjmatrix[claus1][claus2]=1; 
	   } 
 
		 
} 
 
// typeconst==0, negated literals 
// typeconst==1 positive literals 
int CNF::FindConstraints(int typeconst) 
{ 
  int i,j,k,ConstNum,samesign; 
  int claus1,claus2; 
  int* auxadjmatrix;  
	auxadjmatrix = new int[NumberVars]; 
	ConstNum = 0; 
	memset(auxadjmatrix , 0, sizeof(int)*(NumberVars)); 
	 
 
   for(i=0;i<cantclauses;i++)  
   { 
	   samesign = 0; 
 
	 for(j=0;j<dimclause-1;j++)  
       for(k=j+1;k<dimclause;k++)  
	   { 
		   claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		   claus2 = (clauses[i][2*k]==-1)?clauses[i][2*k+1]:clauses[i][2*k]; 
		   samesign += ((clauses[i][2*j+1-typeconst]==clauses[i][2*k+1-typeconst]) && (clauses[i][2*j+typeconst]>0) && (auxadjmatrix[claus1]+auxadjmatrix[claus2]==0) ) ;            
		} 
 
    if (samesign==dimclause) 
	{ 
	  ConstNum++; 
  	  for(j=0;j<dimclause;j++) 
	  {   
           claus1 = (clauses[i][2*j]==-1)?clauses[i][2*j+1]:clauses[i][2*j]; 
		   auxadjmatrix[claus1] =1;	    
	  } 
   } 
  } 
   delete[] auxadjmatrix; 
 return ConstNum;	 
} 
 
 
void CNF::AssignUnitations() 
{ 
   MaxUnit=NumberVars-FindConstraints(1); 
   MinUnit=FindConstraints(0);   
} 
 
CNF::CNF(FILE * f,int dim) 
{ 
	char c; 
	char * tmp; 
	int i,indclauses, d1, d2,j,k; 
	int auxnumber; 
    char number[10]; 
    int* clause; 
	 
 
	indclauses = 0; 
 
	tmp = (char *)calloc(100, sizeof(char)); 
	 
	while ((c = fgetc(f)) != EOF){ 
	   
		switch (c) 
		  { 
			case '\n': 
			case ' ': 
			case '\t': 
			  while ((c = fgetc(f)) == ' ' || c == '\t'|| c == '\n'); 
			  ungetc(c, f); 
			  break; 
 
			case 'c': 
			    for (i = 0;  
				   (c = fgetc(f)) != '\n' && i < LINE_LENGTH - 1 && c != EOF 
				   ; i++);  
				   
			  if ((i == LINE_LENGTH - 1 && c != '\n') || c == EOF){ 
				  ungetc(c, f); 
				  ungetc(' ', f); 
				  ungetc('c', f); 
			  } 
			  break; 
			   
			case 'p': 
			  fscanf(f, "%s %d %d\n", tmp, &d1, &d2); 
			  SetVariables(d1,d2,dim); 
              break;  
			case '0': 
			  break; 
	  		case '%': 
			  break; 
	 
			   
			default: j = 0; 
				     clause = new int[2*dim]; 
					 do  
					 {	      
					    if (c == ' ') while  ((c = fgetc(f)) == ' '); 
						if (c != '0')  
						{ 
						k = 0; 
					    number[k] = c; 
				        while  ((c = fgetc(f)) != ' ') number[++k]=c; 
						number[++k]=0; 
						auxnumber = atoi(number); 
						if (auxnumber<0)  
						{ 
                          clause[j++] = -1; 
						  clause[j++] = -1*(auxnumber)-1; 
						} else   
						{ 
						  clause[j++] = auxnumber-1; 
						  clause[j++] = -1; 
						}  
					} 
				 } 
				  while  ( (c!= '%') && (c!= '0') &&(c = fgetc(f)) != '0'); 
                  Addclause(indclauses,clause); 
				  //printf( "%d %d %d %d %d %d %d \n",indclauses,clause[0],clause[1],clause[2],clause[3],clause[4],clause[5]); 
                  indclauses++; 
			  break; 
		  } 
	} 
	 
} 
 


CNF_Generator::CNF_Generator(int N,int C,int cN,int cC,int card,int dimclaus):CNF(N*cN,C*cN+cC,dimclaus) 
{ 
  int i; 
  cantclusters = cN; 
  Vars_per_Cluster = N; 
  Clauses_per_cluster = C; 
  Clauses_between_clusters = cC; 
  card_sol = card; 
  internal_clauses = cantclauses - Clauses_between_clusters; 
 
  Clusters = new int*[cantclusters]; 
 
  for(i=0;i<cantclusters;i++) Clusters[i] = new int[Vars_per_Cluster]; 
  solution = new unsigned[NumberVars]; 
} 
 
CNF_Generator::~CNF_Generator() 
{ 
  int i; 
  for(i=0;i<cantclusters;i++) delete[] Clusters[i];  
  delete[] Clusters; 
  delete[] solution; 
} 
 
void CNF_Generator::Generate_Solution() 
{ 
  int* index; 
  int i,chosen_var,current_total; 
 
  index = new int[NumberVars]; 
 
for (i=0; i<NumberVars; i++)  
{ 
 index[i] = i; 
 solution[i] = 0; 
} 
 
current_total = NumberVars; 
i=0; 
 
while ( i< card_sol) 
{ 
  chosen_var = randomint(current_total);	 
  solution[index[chosen_var]] = 1; 
  index[chosen_var] = index[current_total--]; 
  i++; 
} 
  delete[] index; 
 
} 
 
 
void CNF_Generator::Generate_Clusters() 
{ 
  int* index; 
  int i,j,chosen_var,current_total; 
 
  index = new int[NumberVars]; 
 
for (i=0; i<NumberVars; i++) index[i] = i; 
current_total = NumberVars; 
 
for (j=0; j<cantclusters; j++)  
{ 
  i=0; 
  while ( i< Vars_per_Cluster) 
  { 
   chosen_var = randomint(current_total);	 
   Clusters[j][i] = index[chosen_var]; 
   index[chosen_var] = index[current_total--]; 
   i++; 
  } 
} 
 
delete[] index; 
 
} 
 

 
int CNF_Generator::EvalClause(int* clause) 
{ 
    int satisfied,j,aux1,claus1;   
 
	satisfied = 0; 
	j = 0; 
    while (j<dimclause) 
	  { 
		  aux1 = (clause[2*j]!=-1); 
          claus1 = (clause[2*j]==-1)?clause[2*j+1]:clause[2*j]; 
		  satisfied += (solution[claus1]==aux1); 
		  j++; 
	  } 
   return satisfied;  
} 

 
void CNF_Generator::Generate_Internal_Clauses() 
{ 
  int* clause; 
  int addedclauses,i,j; 
 
  addedclauses = 0; 
  for (i=0; i<cantclusters; i++) 
  { 
    for (j=0; j<Clauses_per_cluster; j++) 
	{ 
 	  clause = new int[2*dimclause]; 
      Generate_OneInternal_Clause(clause,i); 
	  Addclause(addedclauses++,clause); 
    } 
  } 
} 
 
void CNF_Generator::Generate_External_Clauses() 
{ 
  int* clause; 
  int i; 
      
  for (i=internal_clauses; i<cantclauses; i++) 
  { 
      clause = new int[2*dimclause]; 
      Generate_OneExternal_Clause(clause); 
	  Addclause(i,clause); 
  } 
} 
 
void CNF_Generator::Generate_Clauses() 
{ 
  Generate_Internal_Clauses(); 
  Generate_External_Clauses(); 
} 
 
void CNF_Generator::Generate_OneInternal_Clause(int* clause, int cluster) 
{ 
  int i,j,auxnumber,Eval; 
 
	do  
	{ 
      j = 0; 
	  for (i=0; i<dimclause; i++) 
	  { 
	   auxnumber = randomint(Vars_per_Cluster);  
	    
	   if (myrand()>0.5) 
	   { 
         clause[j++] = -1; 
   	     clause[j++] = Clusters[cluster][auxnumber]; 
	   } else   
	   { 
	    clause[j++] = Clusters[cluster][auxnumber]; 
	    clause[j++] = -1; 
	   }  
	  } 
	  Eval = EvalClause(clause); 
    }while (Eval==0 || Eval==2); 
	 
} 
 
void CNF_Generator::Generate_OneExternal_Clause(int* clause) 
{ 
  int i,j, Eval,auxnumber; 
  int *clusters; 
  clusters = new int[dimclause]; 
 
   do  
	{ 
		do 
		{ 
			for (i=0; i<dimclause; i++) clusters[i]= randomint(cantclusters); 
		}   while (!All_Clusters_Different(clusters)); 
	  j = 0; 
	  for (i=0; i<dimclause; i++) 
	  { 
       auxnumber = randomint(Vars_per_Cluster);  
	    
	   if (myrand()<0.5) 
	   { 
         clause[j++] = -1; 
   	     clause[j++] = Clusters[clusters[i]][auxnumber]; 
	   } else   
	   { 
	    clause[j++] = Clusters[clusters[i]][auxnumber]; 
	    clause[j++] = -1; 
	   }  
	  } 
	  Eval = EvalClause(clause); 
    } while (Eval==0 || Eval==2); 
	 
	delete[] clusters; 
	 
} 
 
int CNF_Generator::All_Clusters_Different(int* clusters) 
{ 
 int i,j, test;  
 test = dimclause*(dimclause-1)/2; 
  
 for (i=0; i<dimclause-1; i++) 
	for (j=i+1; j<dimclause; j++) 
		 test -= (clusters[i] != clusters[j]); 
 return (test==0); 
  
} 
 
void CNF_Generator::Create() 
{ 
  Generate_Solution(); 
  Generate_Clusters(); 
  Generate_Clauses(); 
} 
 
void CNF_Generator::SaveInstance(char* filename)  
{ 
	FILE *stream; 
	int i,j; 
 
    stream = fopen(filename, "w+" ); 
	fprintf(stream,"c %s\n", filename);    
	fprintf(stream,"c Data: cantclusters %d,Vars_per_Cluster %d, Clauses_per_cluster %d, Clauses_between_clusters %d, card %d\n",cantclusters,Vars_per_Cluster,Clauses_per_cluster,Clauses_between_clusters,card_sol);  
   
	fprintf(stream,"c Solution ");    
 
    for (i=0; i<NumberVars; i++) fprintf(stream,"%d ",solution[i]); 
    fprintf(stream,"\n"); 
 
    fprintf(stream,"p cnf %d %d\n",NumberVars,cantclauses);    
    for (i=0; i<cantclauses; i++)  
	{ 
	 for (j=0; j<dimclause; j++)  
	 { 
  		if (clauses[i][2*j]==-1) fprintf(stream,"-%d ",clauses[i][2*j+1]+1); 
		else fprintf(stream,"%d ",clauses[i][2*j]+1);	 
	 }		 
		 fprintf(stream,"0\n"); 
	} 
	fclose(stream); 
}  
