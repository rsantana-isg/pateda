// Parallel.cpp: implementation of the CParallel class.
//
#ifdef PARALLEL

#include "Parallel.h"
#include "EDA.h"
#include "BayesianNetwork.h"
#include "Solution.h"
#include <limits.h>

#ifndef WIN32
    #include <semaphore.h>
    #include <pthread.h>
    //pthread_t thread, NextFinishedThread;
    //int count;
    //  int status;
    //pthread_mutex_t MutexQueue;
    //bool WaitingAnyThread = false;
    //void *WaitingAnyThreadReturn;

#endif


#ifdef sun
        #include <thread.h>
#endif


CParallel::CParallel()
{

#ifdef sun
    /*
     * On Solaris 2.5, threads are not timesliced. To ensure
     * that our threads can run concurrently, we need to
     * increase the concurrency level to CREW_SIZE.
     */
    #ifdef VERBOSE
        cout <<  "Setting concurrency level to " <<  MAX_THREADS << endl;
    #endif
    thr_setconcurrency (MAX_THREADS);
#endif

    m_num_threads=0;

    //Initialise semaphore and mutex
#ifdef WIN32

        hSemaChild = CreateSemaphore( NULL,0, MAX_THREADS, NULL );
        hMutexChild = CreateMutex( NULL, TRUE, NULL );

    //Tin=0;Tout=0;Telem=0;
#else

    sem_init(&SemMaxChildren, 0 /*the semaphore is only for this process*/, MAX_THREADS);
    sem_init(&SemWaitingAnyThread, 0 /*the semaphore is only for this process*/, 0);
    pthread_mutex_init(&MutexQueue, NULL) ;
    //pthread_mutex_init(&hMutexWaitingAnyThread, NULL) ;
    //for (int i=0; i<MAX_THREADS; i++) m_Workerid[i] = (pthread_t) -1;
#endif
}


CParallel::~CParallel()
{
    //Delete semaphore and mutex for Win32
#ifdef WIN32
        CloseHandle( hSemaChild );
        CloseHandle( hMutexChild );
#else
    sem_destroy( &SemMaxChildren );
    sem_destroy( &SemWaitingAnyThread );
    pthread_mutex_destroy( &MutexQueue ); 
    //    pthread_mutex_destroy( &hMutexWaitingAnyThread ); 
    //si la cola fuera dinámica habría que liberar el espacio aqui
#endif

    return;
}

/*********************************************************************
 *********************************************************************
 *********************************************************************
    FUNCTIONS FOR PARALLELISING CALCULATE A NODE
 *********************************************************************
 *********************************************************************
 ********************************************************************/
void ParallelWorkerCalculateANode(CBayesianNetwork *bn)
{
    int node;
    double new_metric;
    int FirstJob, LastJob;

 #ifdef VERBOSE
    cout << "worker: sortua!! " << endl;
 #endif

    //Calculate the part of the work to do - critical section

    bn->m_paralelo.wait();
    node = bn->m_paralelo.m_Node;
    FirstJob = (bn->m_paralelo.m_CurrentTask) * (bn->m_paralelo.m_JobSize);  //Get the task number and continue  
    LastJob = ((bn->m_paralelo.m_CurrentTask + 1) * (bn->m_paralelo.m_JobSize)) -1;
    bn->m_paralelo.m_CurrentTask++;
    bn->m_paralelo.signal();

 #ifdef VERBOSE
    cout << "worker: lana " << node << " hastear." << endl;
 #endif

      //Check that the end of the work is not reached
      if (LastJob > bn->m_paralelo.m_MaxNumTasks)
          LastJob = bn->m_paralelo.m_MaxNumTasks;

 #ifdef VERBOSE
    cout << "worker: lana " << FirstJob << " - " << LastJob << " hastear." << endl;
 #endif

/* We will use the bn->m_ActualMetric[node] value
    switch(SCORE)
    {
    case BIC_SCORE:
      old_metric = bn->BIC(node,bn->m_cases);
      break;
    case K2_SCORE:
      old_metric = bn->K2(node,bn->m_cases);
      break;
    }
*/

    for(int i=FirstJob;i<LastJob;i++)
     if ((i!=node)/*&&(bn->m_paths[i][node]==0)*/)
      {
          switch(SCORE)
          {
          case BIC_SCORE:
              new_metric = bn->deltaBIC(node,i, bn->m_cases);     
              break;
          case K2_SCORE:
              new_metric = bn->deltaK2(node,i, bn->m_cases);
          }
          bn->m_A[node][i] = new_metric - bn->m_ActualMetric[node];
      }
      else bn->m_A[node][i] = INT_MIN;    //bic funtzioa...

 #ifdef VERBOSE
    cout << "worker: amaitua1!!!" << endl;
 #endif

    bn->m_paralelo.AmaituThread(NULL);
 #ifdef VERBOSE
    cout << "worker: amaitua2!!!" << endl;
 #endif
    return;
}

void CParallel::ParallelCalculateANode(CBayesianNetwork *bn, int NumTasks, int node)
{

 #ifdef VERBOSE
    cout << "manager: hasiera Calculate A Node!! " << endl;
 #endif

    int CreatedIndividuals=0;
    m_num_threads=0; //this variable is updated automatically in SortuThread and AmaituThread

    m_CurrentTask =0; //we start from the node 0
    m_MaxNumTasks = NumTasks; //there are NumTasks nodes to treat
    m_Node = node; //number of the node that we are treating
    m_JobSize = ((NumTasks-1) / MAX_THREADS)+1;

    // Creation of the parallel crew by using threads
    for(int i=0;i<MAX_THREADS;i++)
    {
        //    bn->m_paralelo.m_WorkerNum=m_num_threads;
        //bn->m_paralelo.m_NumTask=i;
 #ifdef VERBOSE
    cout << "manager: worker " << i << " sortzen..." << endl;
 #endif
    SortuThread((void * (*)(void *)) ParallelWorkerCalculateANode, bn, i);
 #ifdef VERBOSE
    cout << "manager: worker " << i << " sortua!!" << endl;
 #endif
    }

    //Ume guztien zain geratu
 #ifdef VERBOSE
    cout << "manager: ume guztien zain" << endl;
 #endif
 
    WaitForAllThreads(MAX_THREADS);
 
 #ifdef VERBOSE
    cout << "manager: ume guztiak amaituta!" << endl;
 #endif

    return;
}

/*********************************************************************
 *********************************************************************
 *********************************************************************
    FUNCTIONS FOR PARALLELISING BIC METRIC
 *********************************************************************
 *********************************************************************
 ********************************************************************/

void ParallelWorkerBIC(CBayesianNetwork *bn)
{
    int node;
    double old_metric, new_metric;

 #ifdef VERBOSE
    cout << "worker: sortua!! " << endl;
 #endif
    while (1) {

      //Calculate the part of the work to do

/*
 #ifdef VERBOSE
    cout << "worker: lan bila" << endl;
 #endif
*/
      bn->m_paralelo.wait();
      node = bn->m_paralelo.m_CurrentTask++;  //Get the task number and continue  
      //printf("node: %d\n", node);
      bn->m_paralelo.signal();
 #ifdef VERBOSE
    cout << "worker: lana " << node << " hastear." << endl;
 #endif

      //Check that the end of the work is not reached
      if (node >= bn->m_paralelo.m_MaxNumTasks)
          break;

      switch(SCORE)
    {
    case BIC_SCORE:
      old_metric = bn->BIC(node,bn->m_cases);
      break;
    case K2_SCORE:
      old_metric = bn->K2(node,bn->m_cases);
      break;
    }

      for(int i=0;i<IND_SIZE;i++)
    if ((i!=node)/*&&(bn->m_paths[i][node]==0)*/)
      {
            switch(SCORE)
          {
          case BIC_SCORE:
          new_metric = bn->deltaBIC(node,i, bn->m_cases);     
              break;
          case K2_SCORE:
        new_metric = bn->deltaK2(node,i, bn->m_cases);
          }
        bn->m_A[node][i] = new_metric - old_metric;
/*
#ifdef VERBOSE
//Show metrics
cout << "i: " << node << " j: " << i <<  " new m: " << new_metric << " old m. " << old_metric << " diff:" << bn->m_A[node][i] << endl;
#endif
*/
      }
        else bn->m_A[node][i] = INT_MIN;    //bic funtzioa...
    }

 #ifdef VERBOSE
    cout << "worker: amaitua1!!!" << endl;
 #endif

    bn->m_paralelo.AmaituThread(NULL);
 #ifdef VERBOSE
    cout << "worker: amaitua2!!!" << endl;
 #endif
    return;

}

void CParallel::ParallelBIC(CBayesianNetwork *bn, int NumTasks)
{

 #ifdef VERBOSE
    cout << "manager: hasiera!! " << endl;
 #endif

    int CreatedIndividuals=0;
    m_num_threads=0; //this variable is updated automatically in SortuThread and AmaituThread

    m_CurrentTask =0; //we start from the node 0
    m_MaxNumTasks = NumTasks; //there are NumTasks nodes to treat

    // Creation of the parallel crew by using threads
    for(int i=0;i<MAX_THREADS;i++)
    {
        //    bn->m_paralelo.m_WorkerNum=m_num_threads;
        //bn->m_paralelo.m_NumTask=i;
 #ifdef VERBOSE
    cout << "manager: worker " << i << " sortzen..." << endl;
 #endif
    SortuThread((void * (*)(void *)) ParallelWorkerBIC, bn, i);
 #ifdef VERBOSE
    cout << "manager: worker " << i << " sortua!!" << endl;
 #endif
    }

    //Ume guztien zain geratu
 #ifdef VERBOSE
    cout << "manager: ume guztien zain" << endl;
 #endif
 
    WaitForAllThreads(MAX_THREADS);
 
 #ifdef VERBOSE
    cout << "manager: ume guztiak amaituta!" << endl;
 #endif

    return;
}

/*********************************************************************
 *********************************************************************
 *********************************************************************
    FUNCTIONS FOR PARALLELISING SIMULATION
 *********************************************************************
 *********************************************************************
 ********************************************************************/

void ParallelWorkerSimulation(CSolution *sol)
{
    int numindividual;
    
    while (1) {

      //Calculate the part of the work to do

      sol->m_paralelo.wait();
      numindividual = sol->m_paralelo.m_CurrentTask++;  //Get the task number and continue  
      //printf("individual: %d\n", numindividual);
      sol->m_paralelo.signal();

      //Check that the end of the work is not reached
      if (numindividual >= sol->m_paralelo.m_MaxNumTasks)
          break;

      CIndividual * individual = sol->m_bayesian_network.Simulate();
      
      // sol->m_total += individual->Value();

      sol->m_paralelo.wait();
      sol->AddToPopulation(individual);
      sol->m_paralelo.signal();

    }
    return;

}

void CParallel::ParallelSimulation(CSolution *sol, int NumTasks)
{

    int CreatedIndividuals=0;
    m_num_threads=0; //this variable is updated automatically in SortuThread and AmaituThread

    m_CurrentTask =0; //we start from the individual to create # 0
    m_MaxNumTasks = NumTasks; //there are NumTasks individuals to Simulate

    // Creation of the parallel crew by using threads
    for(int i=0;i<MAX_THREADS;i++)
    {
        //    bn->m_paralelo.m_WorkerNum=m_num_threads;
        //bn->m_paralelo.m_NumTask=i;
    SortuThread((void * (*)(void *)) ParallelWorkerSimulation, sol, i);
    }

    //Ume guztien zain geratu
    WaitForAllThreads(MAX_THREADS);

    return;
}




//
// BESTEENTZAKO FUNTZIOAK (PUBLIKOAK)
//

/*
void CParallel::Parallelise(void( __cdecl *boss_func)( void * ), void( __cdecl *worker_func)( void * ), int NumWorkers)
{
}
*/

/*
void CParallel::ParallelSimulation(CSolution *sol, int NumIndividuals)
{
    int CreatedIndividuals=0;
    m_num_threads=0; //this variable is updated automatically in SortuThread and AmaituThread

    // Creation of the population.in parallel by using threads
    for(int i=0;i<NumIndividuals;i++)
    {
        while (m_num_threads>MAX_THREADS) {
            ItxoinThread(0);    //edozein thread baten zain gelditu
            //jaso emaitza: ez dugu ezer egiten
            CreatedIndividuals++;
        //printf("WWW-> eginak %d, num_threads: %d\n", CreatedIndividuals,  m_num_threads);
        }
//        SortuThread( (void * (*)(void *)) ParallelWorker, sol);
    }

    //Ume guztien zain geratu
    while (ItxoinThread(0)!=-1) {CreatedIndividuals++;};
    return;
}
*/

/*
#ifndef WIN32
    void CParallel::ParallelSimulation(void( __cdecl *boss_func)( void * ), void( __cdecl *worker_func)( void * ), int NumWorkers)
    {

    }
#else
    void CParallel::ParallelSimulation(void( __cdecl *boss_func)( void * ), void( __cdecl *worker_func)( void * ), int NumWorkers)
    {

    }
#endif
*/


/***********************************************************************/

//
// OINARRIZKO FUNTZIOAK (PRIBATUAK)
//

/***********************************************************************/
void CParallel::WaitForAllThreads(int number_worker_threads)
{
int i;

//printf("itxoiten duena:%d\n", m_num_threads);

    //wait for all threads. If there is no left, return.
//    if (m_num_threads==0) return;

#ifndef WIN32
    for ( i = 0; i < number_worker_threads; i++){

    #ifdef VERBOSE
      printf("worker number %d, id:%d \n", i, (unsigned long) m_Workerid[i]);
      printf("martxan %d thread oraindik -- before join\n", m_num_threads);
    #endif
      
      pthread_join(m_Workerid[i], NULL);

    #ifdef VERBOSE
      printf("martxan %d thread oraindik -- after join\n", m_num_threads);
    #endif

      sem_wait (&SemWaitingAnyThread);

    #ifdef VERBOSE
      printf("martxan %d thread oraindik -- after sem\n", m_num_threads);
    #endif
     
    }

    return;
#else
    // there's no correspondence in the Win32's API

    /* Wait for next thread to finish. */
//printf("WWW-> barruan %d\n", m_num_threads);
    for ( i = 0; i < MAX_THREADS; i++)
      WaitForSingleObject( hSemaChild, INFINITE );
//printf("WWW-> kanpoan %d\n", m_num_threads);

    return(0);

#endif
}

unsigned long CParallel::SortuThread(void * (*start_routine)(void *), void *arglist, int NumWorker)
{
#ifndef WIN32
/*
    int pthread_create(pthread_t *thread,const pthread_attr_t  *attr,
              void * (*start_routine)(void *),
              void *arg);
*/
    pthread_t NumChildThread;
    pthread_attr_t thread_attr;
    int ThreadNr;
    void *(*funtzioa)(void *) = (void *(*)( void * )) start_routine;

    //    sem_wait( &SemMaxChildren );

    int status = pthread_attr_init (&thread_attr);
    if (status != 0)
        cerr << "Error " << status << ": Create attr" << endl;

    //create a new thread
    m_num_threads++;
    //printf("num_threads:%d\n", m_num_threads);

    /*
     * Create a detached thread.
     */
    status = pthread_attr_setdetachstate (
        &thread_attr, PTHREAD_CREATE_DETACHED);
    if (status != 0)
        cerr << "Error " << status << ": Set detach" << endl;

    //ThreadNr = pthread_create(&NumChildThread, NULL, funtzioa, arglist);
    status = pthread_create (&NumChildThread, &thread_attr, funtzioa, arglist);
    if (status!=0) 
        cerr << "Error: the new thread could not be created!" << endl; 

    //Set the worker number in the table
    m_Workerid[NumWorker] = NumChildThread;
    //    printf("worker %d id:%d", NumWorker, (unsigned long) NumChildThread);

    return ((unsigned long) NumChildThread);

#else
    int ThreadNr;

    WaitForSingleObject( hSemaChild, INFINITE );

    //create a new thread
    m_num_threads++;

    ThreadNr = _beginthread( (void ( __cdecl *)( void * )) *start_routine, 0, arglist );

    //Set the worker number in the table
    //m_Workerid[NumWorker] = NumChildThread; In Win32 it is not necessary

    if (ThreadNr==0) printf("Error: the new thread could not be created!\n");
//  _beginthread( start_routine, 0, arglist );
    return ((unsigned long) ThreadNr);

#endif

}

/***********************************************************************/
/*
void CParallel::AmaituThreadEmaitzarekin(void *value_ptr)
{
    //exit a thread
    m_num_threads--;

#ifndef WIN32
printf("num_threads:%d\n", m_num_threads);
    
    //Record the end of the thread event to synchronize
    wait(&MutexQueue);
    //NextFinishedThread = pthread_self();
    int i=0; while (m_Queue[i] != (pthread_t) -1) i++;
    m_Queue[i] = pthread_self();
    //    WaitingAnyThreadReturn =  value_ptr;
    signal(&MutexQueue);
    sem_post(&SemWaitingAnyThread);
        
    pthread_exit(value_ptr);
    return;

#else

    ReleaseSemaphore( hSemaChild, 1, NULL);
//printf("%d\n", m_num_threads);
    _endthread();
    return;

#endif
}
*/

/***********************************************************************/
void CParallel::AkatuThread(unsigned long ThreadNumber)
{
    //kill a thread

    m_num_threads--;

#ifndef WIN32
/*
  int pthread_kill(pthread_t thread, int sig);
*/
    pthread_kill( (pthread_t) ThreadNumber, SIGKILL);

#else
    // There's no correspondence in the Win32's API
    cerr << "There's no correspondence in the Win32's API for pthread_kill" << endl;

#endif
}


/***********************************************************************/
/*     THREAD BAT GERATZEKO ETA BERRABIATZEKO FUNTZIO LAGUNTZAILEAK    */
/*     void CParallel::GeratuThread(unsigned long ThreadNumber)        */




void CParallel::GeratuThread(unsigned long ThreadNumber)
{
    //freeze a thread

#ifndef WIN32

    // In POSIX a thread cannot be suspended

#else
/*
DWORD SuspendThread(HANDLE hThread // handle to the thread );
*/
    SuspendThread( (HANDLE) ThreadNumber);

#endif
}


/***********************************************************************/
void CParallel::BerrabiatuThread(unsigned long ThreadNumber)
{
    //unfreeze a frozen thread

#ifndef WIN32

    // In POSIX a thread cannot be suspended


#else
/*
DWORD ResumeThread(HANDLE hThread // identifies thread to restart);
*/
    ResumeThread( (HANDLE) ThreadNumber);
#endif
}


/***********************************************************************
int CParallel::ItxoinThreadEmaitzarekin(unsigned long ThreadNumber)
{
    //wait for a thread. if there is no other, return -1.
    if (m_num_threads==0) return(-1);

#ifndef WIN32

// int pthread_join(pthread_t target_thread, void **status);

void **status;
int i;
pthread_t tid;
//int ReturnValue;
printf("itxoiten duena0:%d\n", m_num_threads);

    if (ThreadNumber==0) {//wait for any thread!
      sem_wait(&SemWaitingAnyThread);
      //there is at least a finished thread
printf("itxoiten duena1:%d\n", m_num_threads);
      wait (&MutexQueue);
printf("itxoiten duena2:%d\n", m_num_threads);
      i=0; while (m_Queue[i]== (pthread_t) -1) i++;
      tid = m_Queue[i]; m_Queue[i] = (pthread_t) -1;
      printf("%d->%d\n", (int) tid, i);
      signal (&MutexQueue);
printf("itxoiten duena3:%d\n", m_num_threads);
      sem_post( &SemMaxChildren );
      return (pthread_join(tid,NULL));
    }
    else {
      //emaitza = thr_join(ThreadNumber, NULL, NULL);
      //return (emaitza);
      return (pthread_join((pthread_t) ThreadNumber,NULL));
    }

#else

    // there's no correspondence in the Win32's API

    // Wait for next thread to finish. 
//printf("WWW-> barruan %d\n", m_num_threads);
    WaitForSingleObject( hSemaChild, INFINITE );
//printf("WWW-> kanpoan %d\n", m_num_threads);

    return(0);

#endif
}
*/

/***********************************************************************/
void CParallel::AmaituThread(void *value_ptr)
{
    //exit a thread
    m_num_threads--;

#ifndef WIN32
/*
  void  pthread_exit(void *value_ptr);
*/
    //printf("num_threads:%d\n", m_num_threads);
    
    //Record the end of the thread event to synchronize
    sem_post(&SemWaitingAnyThread);
    pthread_exit(value_ptr);
    return;

#else
    ReleaseSemaphore( hSemaChild, 1, NULL);
//printf("%d\n", m_num_threads);
    _endthread();
    return;

#endif
}

int CParallel::ItxoinThread(unsigned long ThreadNumber)
{
    //wait for a thread. if there is no other, return -1.
    if (m_num_threads==0) return(-1);

#ifndef WIN32
/*
 int pthread_join(pthread_t target_thread, void **status);
*/
void **status;
int i;
pthread_t tid;
//int ReturnValue;

    if (ThreadNumber==0) {//wait for any thread!
      sem_wait(&SemWaitingAnyThread);

      /*
      //there is at least a finished thread
      wait (&MutexQueue);
      i=0; while (m_Queue[i]== (pthread_t) -1) i++;
      tid = m_Queue[i]; m_Queue[i] = (pthread_t) -1;
      printf("%d->%d\n", (int) tid, i);
      signal (&MutexQueue);
printf("itxoiten duena3:%d\n", m_num_threads);
      */
      sem_post( &SemMaxChildren );
      return (pthread_join(tid,NULL));
    }
    else {
      //emaitza = thr_join(ThreadNumber, NULL, NULL);
      //return (emaitza);
      return (pthread_join((pthread_t) ThreadNumber,NULL));
    }

#else

    // there's no correspondence in the Win32's API

    /* Wait for next thread to finish. */
//printf("WWW-> barruan %d\n", m_num_threads);
    WaitForSingleObject( hSemaChild, INFINITE );
//printf("WWW-> kanpoan %d\n", m_num_threads);

    return(0);

#endif
}

/***********************************************************************/
//wait Semaphore/Mutex
#ifndef WIN32
int CParallel::wait(void) {wait ( &MutexQueue );}
int CParallel::signal(void) {signal( &MutexQueue );}
int CParallel::wait(pthread_mutex_t *hMutex)     //Mutex to synchronize the threads
{
    pthread_mutex_lock ( hMutex );
}
int CParallel::signal(pthread_mutex_t *hMutex)   //Mutex to synchronize the threads
{
    pthread_mutex_unlock ( hMutex );
}
#else
int CParallel::wait(void)
{
    //wait for a mutex
    return WaitForSingleObject( hMutexChild, INFINITE );
}
int CParallel::signal(void)
{
    //signal a mutex
    return ReleaseMutex( hMutexChild );
}
#endif

#endif /*PARALLEL*/
