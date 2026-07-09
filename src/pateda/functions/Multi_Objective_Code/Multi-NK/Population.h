// Population.h: interface for the CPopulation class.
//
// Description:
//		A list of individuals.
//
// Author:	Ramon Etxeberria
// Date:	1999-09-25

#ifndef _POPULATION_
#define _POPULATION_

#include "EDA.h"
#include "Individual.h"

class CPopulation;

typedef CPopulation * POSITION;

class CPopulation  
{
	// The individual at the current node.
	CIndividual * m_individual;

	// The next node in the list.
	CPopulation * m_next;

	// The previous node in the list.
	CPopulation * m_prev;

public:
	// The constructor. It creates an empty list.
	CPopulation();

	// The destructor. 
	virtual ~CPopulation();

	// It returns the individual at the head of the list.
	CIndividual * & GetHead();

	// It returns the individual in the node pointed by pos.
	CIndividual * & GetAt(POSITION pos);

	// It returns the individual at the tail of the list.
	CIndividual * & GetTail();

	// It moves forward in the list.
	POSITION GetNext(POSITION pos);

	// It return the node at the head of the list.
	POSITION GetHeadPosition();

	// It returns the individual at the tail of the list and
	// removes its node.
	void RemoveTail();

	// Function to add an individual to the population
	void AddToPopulation(CIndividual * individual);

	//Import/Export the population from/to a stream or file
	friend ostream & operator<<(ostream & os,CPopulation & population);
	friend istream & operator>>(istream & is,CPopulation & population);
	void ImportPopulation(char * FileName);
	void ExportPopulation(char * FileName);

 private:

	// It inserts the given individual before the node pointed
	// by pos.
	void InsertBefore(POSITION pos, CIndividual * individual);

	// It adds the given individual at the end of the list.
	void AddTail(CIndividual * individual);

};

#endif

