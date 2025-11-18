// Solution.cpp: implementation of the CSolution class.
//

#include <stdio.h>
#include <limits.h>

#ifndef WIN32
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/times.h>
#endif

#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include "Timer.h"

CTimer::CTimer()
{
    //set the time at the beginning
    Reset();

    m_seconds_end = -1;
    m_execution_end = -1;
}

/*
CTimer::~CTimer()
{
}
*/

void CTimer::Reset()
{
    //reset the execution time
#ifndef WIN32
    struct tms buf;
    times(&buf)/CLOCKS_PER_SEC;
    //printf("hasi: %f, %f, %f\n", float(buf.tms_utime + buf.tms_stime)/100, float(CLK_TCK), 1/float(CLK_TCK));
//    m_execution_beginning = (long double) (buf.tms_utime + buf.tms_stime)/float(CLK_TCK); //In seconds
    m_execution_beginning = (long double) (buf.tms_utime + buf.tms_stime)/float(CLOCKS_PER_SEC); //In seconds
#else
    clock();
    m_execution_beginning = ((long double) clock()) / CLOCKS_PER_SEC; //to pass the value from microseconds to seconds
#endif

    //store the actual time
    m_seconds_beginning = time(NULL);
}

void CTimer::End()
{
    //store the execution time
#ifndef WIN32
    struct tms buf;
    times(&buf);
    //printf("bukatu: %f\n", float(buf.tms_utime + buf.tms_stime)/100);
//    m_execution_end = (long double) (buf.tms_utime + buf.tms_stime)/float(CLK_TCK); //In seconds
    m_execution_end = (long double) (buf.tms_utime + buf.tms_stime)/float(CLOCKS_PER_SEC); //In seconds
#else
    m_execution_end = ((long double) clock()) / CLOCKS_PER_SEC; //to pass the value from microseconds to seconds
#endif

    //store the actual time
    m_seconds_end = time(NULL);
}


//Functions to control the total time
float CTimer::Time()
{
    time_t TimeEnd=FinalTime();
    return ((float)(TimeEnd - m_seconds_beginning));
}

float CTimer::Time(int *hours, int *minutes, int *seconds)
{
    time_t TimeEnd=FinalTime();
    *hours   = (TimeEnd - m_seconds_beginning)/3600;
    *minutes = ((TimeEnd - m_seconds_beginning)%3600)/60;
    *seconds = (TimeEnd - m_seconds_beginning)%60;
    return ((float)(TimeEnd - m_seconds_beginning));
}

char *CTimer::TimeString()
{
    static char TimeString[50];
    time_t TimeEnd=FinalTime();
    sprintf(TimeString, "%d hours, %d minutes, and %d seconds",
                            (TimeEnd - m_seconds_beginning)/3600,
                            ((TimeEnd - m_seconds_beginning)%3600)/60,
                            (TimeEnd - m_seconds_beginning)%60);
    return TimeString;
}

char *CTimer::TimeStringFormatted(char *FormatString)
{
    static char TimeString[50];
    time_t TimeEnd=FinalTime();
    sprintf(TimeString, FormatString,
                            (TimeEnd - m_seconds_beginning)/3600,
                            ((TimeEnd - m_seconds_beginning)%3600)/60,
                            (TimeEnd - m_seconds_beginning)%60);
    return TimeString;
}

time_t CTimer::FinalTime()
{
    if (m_seconds_end!=-1) return m_seconds_end;
    else return time(NULL);
}


//Functions to control the execution time

long double CTimer::ExecutionTime()
{
    long double TimeStart = m_execution_beginning;
    long double TimeEnd = (long double)FinalExecutionTime();
    return ((long double)(TimeEnd - TimeStart));
}

long double CTimer::ExecutionTime(int *hours, int *minutes, float *seconds)
{
    long double TimeStart = m_execution_beginning;
    long double TimeEnd = (long double)FinalExecutionTime();

    *hours   = int((TimeEnd - TimeStart)/3600);
    *minutes = int((int((TimeEnd - TimeStart))%3600)/60);
    *seconds = (float) fmod(double(TimeEnd - TimeStart),60);
    return ((long double)(TimeEnd - TimeStart));
}

void CTimer::AddExecutionTime(long double MoreTime)
{
  if (m_execution_end!=-1) m_execution_end += MoreTime;
}

char *CTimer::ExecutionTimeString()
{
    static char TimeString[50];
    long double TimeStart = m_execution_beginning;
    long double TimeEnd = (long double)FinalExecutionTime();
    sprintf(TimeString, "%d hours, %d minutes, and %.2f seconds",
                            int((TimeEnd - TimeStart)/3600),
                            int((int(TimeEnd - TimeStart)%3600)/60),
                            (float) fmod(double(TimeEnd - TimeStart),60));
    return TimeString;
}
char *CTimer::ExecutionTimeString(long double GivenTime)
{
    static char TimeString[50];
    sprintf(TimeString, "%d hours, %d minutes, and %.2f seconds",
                            int((GivenTime)/3600),
                            int((int(GivenTime)%3600)/60),
                            (float) fmod(double(GivenTime),60));
    return TimeString;
}

char *CTimer::ExecutionTimeStringFormatted(char *FormatString)
{
    static char TimeString[50];
    long double TimeStart = m_execution_beginning;
    long double TimeEnd = (long double)FinalExecutionTime();
    sprintf(TimeString, FormatString,
                            int((TimeEnd - TimeStart)/3600),
                            int((int(TimeEnd - TimeStart)%3600)/60),
                            (float) fmod(double(TimeEnd - TimeStart),60));
    return TimeString;
}

long double CTimer::FinalExecutionTime()
{
    if (m_execution_end!=-1) return m_execution_end;
    else return ((long double)(clock()/ CLOCKS_PER_SEC));
}

