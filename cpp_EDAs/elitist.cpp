#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

//------------------------------------------------------------------------
//tipo de codificación
#define LONGITUD_MAX 250   //Longitud maxima de los individuos
int LONGITUD=250;          //Longitud real de los individuos

typedef int TListaEntera [LONGITUD_MAX];

//tipo de corrección de individuos
int CORRECTION_METHOD=0;
#define NONE            0
#define PENALISATION    1
#define CORRECTION      2

//------------------------------------------------------------------------
//Código de EDA.cpp
int EVALUATIONS = 0;

//------------------------------------------------------------------------
//cabeceras de las funciones usadas en el algoritmo
double Metric (TListaEntera* sol);
void CruzarListaEnteraUnPunto (TListaEntera* Pobl[], int i1, int i2, int* h1, int* h2, double pc);
void MutaSwapListaEntera (TListaEntera* indiv, double pm);
void CrearListaEntera (TListaEntera** sol);
void InicListaEnteraAleat  (TListaEntera* sol);
void MuestraListaEntera  (char* nom, TListaEntera* sol, double result);
void Elitista1 (void* Pobl[], double* Eval, int mejor, int tampob, int numHijos);
void FreeListaEntera (TListaEntera** sol);
void SelecFuncObjMax (double* eval, int tampob, double sumafo,  int* h1, int* h2);

//------------------------------------------------------------------------
//My file
typedef struct{
  int u, v;
}arc;

double ALPHA=0.8;
int FITNESS_OPTION=3;
char *ADJG1_NAME="g1.adj";
char *ADJG2_NAME="g2.adj";
char *RHOSIGMA_NAME="g1g2.nsm";
char *RHOMU_NAME="g1g2.esm";
int  g1_nodes,g1_arcs,g2_nodes,g2_arcs; //variables to store the number of
                                        //nodes and arcs of g1 and g2
unsigned char  **adg1, **adg2; //these variables will store the adjacency
                               //values of nodes (g1) and arcs (g2)
arc *arcnodesg1, *arcnodesg2;
float  **rho_sigma, **rho_mu;

//Includes of the file
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <stream.h>
#include <fstream.h>
#include "Timer.h"

char *OUTPUTFILE=NULL;
ofstream foutput;
unsigned char *c;

/********************************************************************/
//Functions defined by Claudia
/********************************************************************/

unsigned char  **CreateAdjacencyMatrix(FILE *fileadjg, int *xdim, int *ydim)
{
  unsigned char**adjg;
  int i,j,height,width;

  fscanf(fileadjg, "%d %d", &height,&width);
  fscanf(fileadjg, "\n");

  //reserve memory for adjacency matrix
  adjg = new unsigned char *[height];
  for (i=0; i<=height; i++)
      adjg[i]=new unsigned char [width];

  //fill in the matrix
  for(i=0; i<height; i++){
    for(j=0; j<width; j++)
      fscanf(fileadjg, "%c ", &adjg[i][j]);
    fscanf(fileadjg, "\n");
  }
  fscanf(fileadjg,"\n");

  *xdim = height;
  *ydim = width;

  return adjg;
}


/******** Rho_sigma matrix creation with the association's degree values for
 *****each pair of nodes (u,v), u of the model graph and v of the scene graph**/

float **Create_rho_matrix(char *rho_matrix_file)
{
  int i,j, height, width;
  float **matrix;
  FILE *frho;

  frho = fopen(rho_matrix_file,"r");
  if (frho==NULL) {
      cerr << "Could not open file " << rho_matrix_file << endl;
      return (NULL);
  }

  fscanf(frho, "%d ", &height);
  fscanf(frho, "%d ", &width);

  //reserve memory for rhosigma matrix
  matrix = new float *[height];
  for (i=0; i<=height; i++)
      matrix[i]=new float [width];

  fscanf(frho, "\n");
  for(i=0; i<height; i++){
    for(j=0; j<width; j++)
      fscanf(frho, "%f ", &matrix[i][j]);
    fscanf(frho, "\n");
  }

  fclose(frho);

  return(matrix);
}


arc *Create_arcnodes(FILE *madj, int eset_size)
{
  int i,w;
  arc *arcnode;

  if (eset_size)
    arcnode = new arc [eset_size];

  for(i=0; i<eset_size; i++){
      fscanf(madj, "%d ", &w);
      fscanf(madj, "%d %d",&arcnode[i].u,&arcnode[i].v);
      fscanf(madj, "\n");
  }

  return(arcnode);
}

int Input(void)
{

  FILE *madjg1, *madjg2;
  int aux;

  /* Data input: adjacence matrices (G1 and G2), edge sets sizes and edge-nodes information */
  madjg1 = fopen(ADJG1_NAME,"r");
  if (madjg1==NULL) {
      cerr << "Could not open file " << ADJG1_NAME << endl;
      return (-1);
  }
  adg1 = CreateAdjacencyMatrix(madjg1, &g1_nodes, &aux);
  fscanf(madjg1,"%d %d\n",&g1_arcs, &aux);
  arcnodesg1 = Create_arcnodes(madjg1,g1_arcs);
  fclose(madjg1);

  madjg2 = fopen(ADJG2_NAME,"r");
  if (madjg2==NULL) {
      cerr << "Could not open file " << ADJG2_NAME << endl;
      return (-1);
  }
  adg2 = CreateAdjacencyMatrix(madjg2, &g2_nodes, &aux);
  fscanf(madjg2,"%d %d\n",&g2_arcs, &aux);
  arcnodesg2 = Create_arcnodes(madjg2,g2_arcs);
  fclose(madjg2);


  /* Rho_sigma: matrix with the degree association values for all pair of nodes (u,v)
     u of model graph and v of scene graph.
     Rho_mu: matrix with the degree association values for all pair of arcs (e1,e2)
     e1 of model graph and e2 of scene graph.*/
  rho_sigma = Create_rho_matrix(RHOSIGMA_NAME);
  rho_mu = Create_rho_matrix(RHOMU_NAME);

  if (!(rho_sigma)&&(rho_mu)) return (-1);

  return (0);
}



double Metric(TListaEntera* sol)
{
    // This function returns the value of an individual
    // It consists in applying one of the fitness functions
    //  developed by Claudia at ENST.

  int i, j, s=0,k,l,u1,v1,u2,v2,c1,c2,node_assoc,edge_assoc;
  double fit1, fit2;
  double f;


  int *genes = (int *)sol;

    /*
    genes[0]=0; genes[1]=0; genes[2]=1; genes[3]=1; genes[4]=1; genes[5]=2;
    genes[6]=2; genes[7]=3; genes[8]=3; genes[9]=4; genes[10]=5;genes[11]=5;
    genes[12]=7;genes[13]=7;genes[14]=7;genes[15]=6;genes[16]=6;
    */

    /*
    genes[0]=6; genes[1]=2; genes[2]=7; genes[3]=2; genes[4]=7; genes[5]=0;
    genes[6]=2; genes[7]=7; genes[8]=7; genes[9]=1; genes[10]=3;genes[11]=0;
    genes[12]=4;genes[13]=4;genes[14]=7;genes[15]=3;genes[16]=7;
    */

  //aldaketa 1 edo 0 balioetara pasatzeko, cij sortuz:
    for(i=0; i<g1_nodes*g2_nodes; i++)
        c[i]=0;

    for (i=0; i<g2_nodes; i++)
        c[genes[i] * g2_nodes + i] = 1;


  /* Fitness computed considering all chromosome values */
  if(FITNESS_OPTION == 1 || FITNESS_OPTION == 2 || FITNESS_OPTION == 6) {
    fit1 = 0.0;
    for(i=0; i<g1_nodes; i++)
      for(j=0; j<g2_nodes; j++){

    s = i * g2_nodes + j;
    fit1 += 1.0 - fabs(c[s] - rho_sigma[i][j]);
      }

    fit1 = fit1/(g1_nodes*g2_nodes);

    fit2 = 0.0;
    for(k=0; k<g1_arcs; k++){
      u1 = arcnodesg1[k].u;
      v1 = arcnodesg1[k].v;
      for(l=0; l<g2_arcs; l++){
    u2 = arcnodesg2[l].u;
    v2 = arcnodesg2[l].v;
    c1 = u1 * g2_nodes + u2;
    c2 = v1 * g2_nodes + v2;
    fit2 += 1.0  - fabs(c[c1]*c[c2] - rho_mu[k][l]);
      }
    }

    fit2 = fit2/(g1_arcs*g2_arcs);
  }

  /* Fitness computed considering only chromossome values equal to 1 */
  if(FITNESS_OPTION == 3 ||FITNESS_OPTION == 4 ||FITNESS_OPTION == 5){
    fit1 = 0.0;
    node_assoc = 0;
    for(i=0; i<g1_nodes; i++)
      for(j=0; j<g2_nodes; j++){
        s = i * g2_nodes + j;
        if(c[s] == 1){
            node_assoc++;
            fit1 += 1.0 - fabs(c[s] - rho_sigma[i][j]);
        }
      }

    fit1 = fit1/(node_assoc);

    fit2 = 0.0;
    edge_assoc = 0;
    for(k=0; k<g1_arcs; k++){
      u1 = arcnodesg1[k].u;
      v1 = arcnodesg1[k].v;
      for(l=0; l<g2_arcs; l++){
        u2 = arcnodesg2[l].u;
        v2 = arcnodesg2[l].v;
        c1 = u1 * g2_nodes + u2;
        c2 = v1 * g2_nodes + v2;
        if (rho_mu[k][l] != 0.0)
            if(rho_sigma[u1][u2] > 0.0 && rho_sigma[v1][v2] > 0.0)
                if(c[c1]*c[c2] == 1){
                    edge_assoc++;
                    fit2 += 1.0  - fabs(c[c1]*c[c2] - rho_mu[k][l]);
                }
      }
    }
    if (edge_assoc>0) fit2 = fit2/(edge_assoc);
//  else fit2 = 0.0;
  }

  f = ALPHA * fit1 + (1 - ALPHA) * fit2;

  // This allows us to keep track of the number
  // of evaluations.
  EVALUATIONS++;

//  delete c;

  return(f);

}

//------------------------------------------------------------------------
//funciones similares a la modificación de la simulación en los EDA
int POPCORRECT=0;
int POPMISS1=0;
int POPMISS2=0;
int POPMISS3=0;
int POPMISSMORE=0;

int faltan(TListaEntera* sol)
{
  int i;
    // This function returns the number of values still to appear in the individual.
  int *genes = (int *)sol;

  int values_left = 0;
  int *Indiv = new int[g1_nodes];

    //Initialise Indiv
    for (i=0; i<g1_nodes; i++) Indiv[i]=0;
    //Check which values are already in the individual
    for (i=0; i<g2_nodes; i++) Indiv[genes[i]]++;
    //Calculate how many are missing
    for (i=0; i<g1_nodes; i++)
        if (Indiv[i]==0) values_left++;

    delete [] Indiv;

    return (values_left);
}

void Correct_Individual(TListaEntera* *sol_p)
{
    // This function returns the number of values still to appear in the individual.
  int i, j, v;
  TListaEntera* sol = *sol_p;
  int *genes = (int *)sol;

  int values_left = 0;
  int *Indiv = new int[g1_nodes];

    //Initialise Indiv
    for (i=0; i<g1_nodes; i++) Indiv[i]=0;
    //Check which values are already in the individual
    for (i=0; i<g2_nodes; i++) Indiv[genes[i]]++;
    //Calculate how many are missing
    for (i=0; i<g1_nodes; i++)
        if (Indiv[i]==0) values_left++;

    while (values_left>0) {
            //select a random position that contains a variable already appeared at least twice
            do {j=rand()%g2_nodes;} while (Indiv[genes[j]]<2);

            //substitute this variable by the next missing value.
            for (v=0; v<g1_nodes; v++) {if (Indiv[v]==0) break;}
            Indiv[genes[j]]--;
            genes[j]=v;
            Indiv[v]++;
            values_left--;
    }

    delete [] Indiv;

    return;
}

void DisplayCorrectnessInfo(int generation, int POP_SIZE)
{
    //Display statistics about the generation
    cout << "-Generation "<< generation << ": "
         << "Correct: " << POPCORRECT << "(" << (float)POPCORRECT/POP_SIZE*100 << "%), "
         << "missing: 1: " << POPMISS1 << "(" << (float)POPMISS1/POP_SIZE*100 << "%), "
         << "2: " << POPMISS2 << "(" << (float)POPMISS2/POP_SIZE*100 << "%), "
         << "3: " << POPMISS3 << "(" << (float)POPMISS3/POP_SIZE*100 << "%), "
         << "more: " << POPMISSMORE << "(" << (float)POPMISSMORE/POP_SIZE*100 << "%) " << endl;
    if (foutput)
        foutput << "-Generation "<< generation << ": "
                << "Correct: " << POPCORRECT << "(" << (float)POPCORRECT/POP_SIZE*100 << "%), "
                << "missing: 1: " << POPMISS1 << "(" << (float)POPMISS1/POP_SIZE*100 << "%), "
                << "2: " << POPMISS2 << "(" << (float)POPMISS2/POP_SIZE*100 << "%), "
                << "3: " << POPMISS3 << "(" << (float)POPMISS3/POP_SIZE*100 << "%), "
                << "more: " << POPMISSMORE << "(" << (float)POPMISSMORE/POP_SIZE*100 << "%) " << endl;
    POPCORRECT=0;
    POPMISS1=0;
    POPMISS2=0;
    POPMISS3=0;
    POPMISSMORE=0;
}

//------------------------------------------------------------------------
//funciones que no se pueden configurar
void GrabarResultado(char* nom, double result)
{
    FILE *out;
    char archivo[256];
    int i=strlen(nom)-1;
    int j;

    while(nom[i] != '.'){i--;}

    for(j=0;j<i;j++)
        archivo[j] = nom[j];
    archivo[i] = '\0';

    strcat(archivo,".est");

    if ((out = fopen (archivo,"at")) != NULL)
    {
          fprintf(out, "%g\n", result);
          fclose(out);
     }
}


void InicSemilla ()
{
  time_t t;
  srand((unsigned) time(&t));
}

int intRand(int n)
{
    return rand() % (n+1);
}

double doubleRand (int n)
{
    return (((double)rand() / (double)(RAND_MAX+1.0))*(double)(n));
}

double normalRand(double media, double desv)
{
    double gauss;
    double uniforme[2];
    double numpi = 3.1415926535897932385;

    /*
       Código obtenido a partir de la pagina Web de Horst Meyerdierks, de la
       University of Glasgow en la dirección: http://www.roe.ac.uk/hmewww/Entropic/Stats.html
       Código basado en el manual Standard Software Module de la calculadora
       programable TI-59.
    */

    //Primero se obtiene un valor que sigue una distribución con media 0 y desv. 1
    //En segundo lugar se escala la desviación típica
    //Y en tercer lugar se escala la media.

    uniforme[0] = (double)rand() / ( (double)RAND_MAX + 1.0 );
    uniforme[1] = (double)rand() / ( (double)RAND_MAX + 1.0 );
    gauss = sqrt( -2.0 * log(uniforme[0]) )
                                       * cos( 2.0 * numpi * uniforme[1] );
    gauss *= desv;
    gauss += media;

    return gauss;
}


//------------------------------------------------------------------------
//datos auxiliares, propios de cada problema


int main(int argc,char* argv[])
{

    int iter, indiv1, indiv2, hijo1, hijo2, topeHijos, mejor, i;

    int TamPob;
    double ValBusq;
    int maxIter;
    double pc;
    double pm;

    //la población es el doble que el tamaño de la población, para poder contener a los hijos.
    TListaEntera** Pobl;
    double* Eval;
    double sumaFO;
    int informar;
    CTimer timer;

    // Seed for random numbers.
    srand(time(NULL));

    //Cargar las matrices con los datos.
    if (Input()<0) return (-1);
    c = new unsigned char [g1_nodes*g2_nodes];

    // An invidual of g2_nodes binary genes is defined.
    LONGITUD = g2_nodes;
    cout<<"elitist: Size of the individual: " << LONGITUD << endl;

    //Inicializar los valores de las variables
    TamPob = 2000;
    ValBusq = 2;
    maxIter = 100;
    pc = 1;
    pm = 1.0/LONGITUD;
    informar = 1;

    if (argv[1]) {
        CORRECTION_METHOD = atoi(argv[1]);

        // open the output file (if exists)
        OUTPUTFILE = argv[2];
        if (OUTPUTFILE) {
            foutput.open(OUTPUTFILE);
            if (!foutput) {
                cerr << "Could not open file " << OUTPUTFILE << ". Ignoring this file." << endl;
                OUTPUTFILE=NULL;
            }
        }
    }

    InicSemilla();


    Pobl = (TListaEntera**)calloc(TamPob*2, sizeof(TListaEntera));
    Eval = (double*)calloc(TamPob*2, sizeof(double));

    //inicializo la población, creándola, inicializándola y obteniendo su valor de función objetivo
    //así como el mejor indiv de la población
    mejor = 0;
    sumaFO = 0;
    for(i=0;i<TamPob;i++)
    {
        CrearListaEntera(&Pobl[i]);
        InicListaEnteraAleat(Pobl[i]);

        //Corrección y Evaluación de los individuos
        switch (CORRECTION_METHOD) {
        case NONE:
            Eval[i] = Metric(Pobl[i]);
            break;
        case PENALISATION:
            Eval[i] = Metric(Pobl[i])/(faltan(Pobl[i])+1);
            break;
        case CORRECTION:
            Correct_Individual(&Pobl[i]);
            Eval[i] = Metric(Pobl[i]);
            break;
        }

        sumaFO = sumaFO + Eval[i];
        if (Eval[i] > Eval[mejor])
            mejor = i;
    }

    //Calculate the correctness of the population
    for(i=0; i<TamPob ;i++) {
            switch(faltan(Pobl[i])) {
            case 0: POPCORRECT++; break;
            case 1: POPMISS1++; break;
            case 2: POPMISS2++; break;
            case 3: POPMISS3++; break;
            default: POPMISSMORE++;
            }
    }
    iter = 0;
    if(informar == 1) {
        printf("Generacion = [%d]\tactual = [%f]\n",iter,Eval[mejor]);
        if ((foutput)&&(iter%5==0)) {
            foutput << "Generation "<< iter << ": " << Eval[mejor] << endl;
        }
    }
    DisplayCorrectnessInfo(iter, TamPob);


        //creo los individuos que servirán como hijos.
    for(i=TamPob;i<(2*TamPob);i++)
        CrearListaEntera(&Pobl[i]);

    while ((Eval[mejor] < ValBusq) &&(iter < maxIter))
    {
        iter++;

        //con topeHijos controlo cuantos hijos llevo generados, es decir, cual
        //es la posición dentro de la población que ocupará el nuevo hijo
        topeHijos = TamPob;
        for(i=0;i<(TamPob/2);i++)
        {

            SelecFuncObjMax(Eval, TamPob, sumaFO, &indiv1, &indiv2);
            hijo1 = topeHijos;
            hijo2 = hijo1+1;

            CruzarListaEnteraUnPunto(Pobl,indiv1,indiv2,&hijo1,&hijo2,pc);

            //si hijo1 o hijo2 son -1, es que ese hijo no se ha creado
            if(hijo1 != -1)
            {
                MutaSwapListaEntera(Pobl[hijo1],pm);
                topeHijos++;
            }
            if(hijo2 != -1)
            {
                MutaSwapListaEntera(Pobl[hijo2],pm);
                topeHijos++;
            }
        }

        //Corrección y Evaluación de los individuos
        switch (CORRECTION_METHOD) {
        case NONE:
            for(i=TamPob;i<topeHijos;i++)
                Eval[i] = Metric(Pobl[i]);
            break;
        case PENALISATION:
            for(i=TamPob;i<topeHijos;i++)
                Eval[i] = Metric(Pobl[i])/(faltan(Pobl[i])+1);
            break;
        case CORRECTION:
            for(i=TamPob;i<topeHijos;i++) {
                Correct_Individual(&Pobl[i]);
                Eval[i] = Metric(Pobl[i]);
            }
            break;
        }

        //reduzco la población
        Elitista1((void**)Pobl,Eval,mejor,TamPob,topeHijos-TamPob);

        //Calculate the correctness of the population
        for(i=0; i<TamPob ;i++) {
            switch(faltan(Pobl[i])) {
            case 0: POPCORRECT++; break;
            case 1: POPMISS1++; break;
            case 2: POPMISS2++; break;
            case 3: POPMISS3++; break;
            default: POPMISSMORE++;
            }
        }
        DisplayCorrectnessInfo(iter, TamPob);

        //hallo el sumatorio de las funciones objetivos de la población actual
        //y el mejor individuo de la nueva población
        sumaFO = 0;
        mejor = 0;
        for(i=0;i<TamPob;i++)
        {
            sumaFO = sumaFO + Eval[i];
            if(Eval[mejor] < Eval[i])
                mejor = i;
        }

        if(informar == 1) {
            printf("Generacion = [%d]\tactual = [%f]\n",iter,Eval[mejor]);
            if ((foutput)&&(iter%5==0)) {
                foutput << "Generation "<< iter << ": " << Eval[mejor] << endl;
            }
        }

    }

    timer.End();

    printf("Numero de Iteraciones: %d\n", iter);
    if ((foutput)&&(iter % 5 != 0)) {
            foutput << "Generation "<< iter << ": " << Eval[mejor] << endl;
    }
    cout << "Generation: " << iter << endl
         << "Evaluations: " << EVALUATIONS << endl
         << "Total time: " << timer.TimeString() << endl
         << "Execution time: " << timer.ExecutionTimeString() << endl
         << "Best individual: " << Eval[mejor] << " ";
    if (foutput) {
        foutput << "== final RESULTS =="<< endl
                << "Generation: " << iter << endl
                << "Evaluations: " << EVALUATIONS << endl
                << "Total time: " << timer.TimeString() << endl
                << "Execution time: " << timer.ExecutionTimeString() << endl
                << "Best individual: " << Eval[mejor] << " ";
    }
    MuestraListaEntera(argv[0], Pobl[mejor], Eval[mejor]);
    //GrabarResultado(argv[0], Eval[mejor]);

    //destruyo la población
    for(i=0;i<(TamPob*2);i++)
        FreeListaEntera(&Pobl[i]);
    free(Pobl);
    free(Eval);

    //Write the information of the solution in the file (if exists)
    if (foutput) {
        foutput.close();
    }
    //Free the memory for the matrixes;
    for (i=0; i<g1_nodes; i++)
        delete [] adg1[i];
    //delete [] adg1;
    delete [] arcnodesg1;
    for (i=0; i<g2_nodes; i++)
        delete [] adg2[i];
    //delete [] adg2;
    delete [] arcnodesg2;
    for (i=0; i<g1_nodes; i++)
        delete [] rho_sigma[i];
    //delete [] rho_sigma;
    for (i=0; i<g1_arcs; i++)
        delete [] rho_mu[i];
    //delete [] rho_mu;

    delete c;

}

//------------------------------------------------------------------------
//funciones referentes al tipo de codificación
void CrearListaEntera (TListaEntera** sol)
{
    *sol = (TListaEntera*)malloc(sizeof(TListaEntera));
}


void FreeListaEntera (TListaEntera** sol)
{
    free(*sol);
    *sol = NULL;
}


void InicListaEnteraAleat  (TListaEntera* sol)
{
    int i;
    for(i=0;i<LONGITUD;i++)
    {
        (*sol)[i] = intRand(g1_nodes-1);
    }
}


void MuestraListaEntera  (char* nom, TListaEntera* sol, double result)
{
    int i;
    printf("[");
    for(i=0;i<LONGITUD-1;i++) {
        printf("%d, ", (*sol)[i]);
        if (foutput) {
	  foutput <<  (*sol)[i] << ", ";
        }
    }
    printf("%d]", (*sol)[LONGITUD-1]);
    printf(" --> %g\n", result);
    if (foutput) {
        foutput <<  (*sol)[LONGITUD-1] << ". --> " << result << endl;
    }
}


//------------------------------------------------------------------------
//funciones del algoritmo genético
void CruzarListaEnteraUnPunto (TListaEntera* Pobl[], int i1, int i2, int* h1, int* h2, double pc)
{
    int i, punto;

    if(pc > doubleRand(1))
    {
        punto = intRand(LONGITUD-1);
        for(i=0;i<punto;i++)
        {
            (*Pobl[*h1])[i] = (*Pobl[i1])[i];
            (*Pobl[*h2])[i] = (*Pobl[i2])[i];
        }
        for(i=punto;i<LONGITUD;i++)
        {
            (*Pobl[*h1])[i] = (*Pobl[i2])[i];
            (*Pobl[*h2])[i] = (*Pobl[i1])[i];
        }
    }
    else
    {
        *h1 = -1;
        *h2 = -1;
    }
}


void MutaSwapListaEntera (TListaEntera* indiv, double pm)
{
    int p1, p2, iguales, aux;

    if(pm > doubleRand(1))
    {
        p1 = intRand(LONGITUD-1);
        iguales = 1;
        while(iguales == 1)
        {
            p2 = intRand(LONGITUD-1);
            if(p2 != p1)
                iguales = 0;
        }
        aux = (*indiv)[p1];
        (*indiv)[p1] = (*indiv)[p2];
        (*indiv)[p2] = aux;
    }
}


void Elitista1 (void* Pobl[], double* Eval, int mejor, int tampob, int numHijos)
{
    int i, peor;
    void* paux;
    double daux;
    peor = tampob;

    //para elegir el que no pasa a la siguiente generación, obtengo
    //un numero aleatorio entre 0 y el numero de hijos - 1.
    //a continuacion se le suma el tamaño de la poblacion, para
    //que el indice corresponda a un hijo y no a un padre.
    if (numHijos > 0)
        peor = intRand(numHijos-1)+tampob;

    for(i=0;i<numHijos;i++)
    {
        if(i != mejor)
        {
            paux = Pobl[i];
            Pobl[i] = Pobl[i+tampob];
            Pobl[i+tampob] = paux;
            daux = Eval[i];
            Eval[i] = Eval[i+tampob];
            Eval[i+tampob] = daux;
        }
        else
        {
            paux = Pobl[peor];
            Pobl[peor] = Pobl[i+tampob];
            Pobl[i+tampob] = paux;
            daux = Eval[peor];
            Eval[peor] = Eval[i+tampob];
            Eval[i+tampob] = daux;
        }
    }
}


void SelecFuncObjMax (double* eval, int tampob, double sumafo,  int* h1, int* h2)
{
    double* Prob;
    double aleat;
    int i;
    int iguales;

    Prob = (double*)calloc(tampob, sizeof(double));

    //determino la probabilidad de selección, basándome en la función
    //objetivo de cada individuo. Acumulo las probabilidades, para
    //crear un conjunto de tampob subintervalos, y así generando un
    //número aleatorio, localizar cual es el individuo "afortunado"

    Prob[0] = eval[0] / sumafo;
    for(i=1;i<tampob-1;i++)
        Prob[i] = Prob[i-1] + eval[i]/sumafo;
    Prob[tampob-1] = 1;
    //al último individuo, le asigno un 1, pues si hago la acumulación
    //dará un valor cercano a uno, pero no 1, por lo que al generar
    //el número aleatorio, puede que este quede por encima, aunque
    //sea muy poco probable.

    //determino dos individuos a cruzar, obteniendo un número
    //aleatorio y viendo en que intervalo de Prob, cae.
    *h1 = 0;
    aleat = doubleRand(1);
    while((*h1<tampob)&&(Prob[*h1]<aleat))
        (*h1)++;

    //para el segundo padre, compruebo que no sea igual que el primero
    iguales = 1;
    while(iguales)
    {
        *h2 = 0;
        aleat = doubleRand(1);
        while((*h2<tampob)&&(Prob[*h2]<aleat))
            (*h2)++;
        if(*h1 != *h2)
            iguales = 0;
    }
    free(Prob);
}



//------------------------------------------------------------------------
//función para la inicialización de los datos auxiliares
