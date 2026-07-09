// Parallel.h: interface for the CParallel class.
//
// Description:
//      The class to use when parallelising
//      with different schemas (workers).
//
// Author:  Endika Bengoetxea
// Date:    2000-03-27
//
// Notes:
//      This class intends to serve for both UNIX and PC-based
//      programs. It can be used to parallelise
//      slow programs.
//
// To compile:
//   POSIX
//      cc [ flag ... ] file ...  -lpthread [ library ... ]
//
//   WINDOWS
//      CL /MT file

#ifdef PARALLEL

#ifndef _PARALLEL_
#define _PARALLEL_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_THREADS 4                   /* Number of workers */


#ifndef WIN32
    #include <unistd.h>
    #include <sys/types.h>
    #include <pthread.h>
    #include <semaphore.h>
    #include <signal.h>
    #include <math.h>

#else
    #ifndef _MT
        #define _MT
    #endif

    #include <conio.h>
    #include <windows.h>
    #include <process.h>

#endif


extern class CSolution s;
extern class CBayesianNetwork b;
//extern "C" void ParallelWorker(CSolution *sol);

class CParallel
{
public:

    // It creates the master-slave scheme.
//    void Parallelise(void( __cdecl *boss_func)( void * ), void( __cdecl *worker_func)( void * ), int NumWorkers);
    void CParallel::ParallelSimulation(CSolution *sol, int NumIndividuals);
    void CParallel::ParallelBIC(CBayesianNetwork *bn, int NumTasks);
    void CParallel::ParallelCalculateANode(CBayesianNetwork *bn, int NumTasks, int node);

    // The constructor. It creates the Parallel.
    CParallel();

    // The destructor.
    virtual ~CParallel();

    //Functions to work with threads.
    void AmaituThread(void *value_ptr);
    int m_WorkerNum;
    int m_NumTask;
    int m_CurrentTask;
    int m_Node;
    int m_JobSize;

#ifndef WIN32
    //Functions to work with mutex
    int CParallel::wait(pthread_mutex_t *hMutex);
    int CParallel::signal(pthread_mutex_t *hMutex);
#endif
    //Functions to work with mutex (generic)
    int CParallel::wait(void);
    int CParallel::signal(void);

    //Maximum number of tasks to execute
    int m_MaxNumTasks;

private:

     //Functions to work with threads.
     unsigned long SortuThread( void * (*start_routine)(void *), void *arglist, int NumWorker);
     void AkatuThread(unsigned long ThreadNumber);
     void GeratuThread(unsigned long ThreadNumber);
     void BerrabiatuThread(unsigned long ThreadNumber);
     int  ItxoinThread(unsigned long ThreadNumber);

     void WaitForAllThreads(int);

#ifdef WIN32

     //Semaphore to control the number of children
     HANDLE  hSemaChild;
     //Mutex to synchronize the threads
     HANDLE  hMutexChild;

     //Table to control termination of each thread
/*   const int SizeTable = MAX_THREADS;
     long TableThreads[SizeTable];
   int Tin, Tout, Telem;*/
#else

     //Mutex to synchronize the threads
     //pthread_mutex_t hMutexChild;
     pthread_mutex_t MutexQueue;

     //Semaphore to control the number of children
     sem_t SemMaxChildren, SemWaitingAnyThread;

     //Array of worker thread ids
         pthread_t m_Workerid[MAX_THREADS];

#endif

     //m_num_threads stores the number of threads created so far.
     int m_num_threads;

};

#endif

#endif /*PARALLEL*/
