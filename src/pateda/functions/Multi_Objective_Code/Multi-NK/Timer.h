// Timer.h: interface for the CTimer class.
//
// Description:
//		The class that contains all the time-related 
//		functions and variables.
//
// Author:	Endika Bengoetxea
// Date:	1999-12-24
//
// Notes: 
//		This class intends to serve for both UNIX and PC-based
//		programs. It can be used to calculate total times, and
//		also process execution times.

#ifndef _TIMER_
#define _TIMER_

#include <time.h>

class CTimer 
{
public:

	// It sets the timer to 0.
	void Reset();

	// It sets the final time.
	void End();

	// Both functions return the total time in seconds since the timer was set to 0.
	// The second one also returns the values translated to hours, minutes and seconds.
	float Time();
	float Time(int *hours, int *minutes, int *seconds);

	// It returns the execution time in seconds since the timer was set to 0.
	// The second one also returns the values translated to hours, minutes and seconds.
	long double ExecutionTime();
	long double ExecutionTime(int *hours, int *minutes, float *seconds);

	// It returns a message with the total time in seconds since the timer was set to 0.
	char *TimeString();

	// It returns a message with the execution time in since the timer was set to 0.
	char *ExecutionTimeString();
	char *ExecutionTimeString(long double GivenTime);

	// It returns a message with the total time in seconds since the timer was set to 0.
	// It gives the possibility to pass as a parameter the format string of a 'sprintf' function.
	char *TimeStringFormatted(char *FormatString);

	// It returns a message with the execution time in seconds since the timer was set to 0.
	// It gives the possibility to pass as a parameter the format string of a 'sprintf' function.
	char *ExecutionTimeStringFormatted(char *FormatString);

	// It adds execution time. Used by Parallelization rutines in order
	// to add properly the execution time of all the processes.
	void AddExecutionTime(long double MoreTime);

	// The constructor. It creates the timer.
	CTimer();

/*
	// The destructor.
	virtual ~CSTimer();
*/
private:

	 //m_seconds_beginning is used to store the initial time,
	 //which corresponds to the number of seconds since the first january 1970
	 time_t m_seconds_beginning;

	 //m_execution_start is used to store at which time the execution time
	 //is considered to be 0
	 long double m_execution_beginning;

	 //m_seconds_beginning is used to store the time at which the execution finished,
	 //which corresponds to the number of seconds since the first january 1970
	 time_t m_seconds_end;

	 //m_execution_end is used to store at which time the execution finished.
	 //It is considered to be 0
	 long double m_execution_end;

	 //methods to return the value of the final time, or the actual time if
	 //the timer has not been stopped.
	 time_t FinalTime();
	 long double FinalExecutionTime();
};

#endif 

