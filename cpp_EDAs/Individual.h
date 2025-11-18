// Individual.h: interface for the CIndibiduoa class.
//
// Description:
//		The class which handles the individual's information:
//		its genes and its value.
//
// Author:	Ramon Etxeberria
// Date:	1999-09-25
//
// Note:
//		The problem's information must be loaded before any
//		individual is created.


#ifndef _INDIVIDUAL_
#define _INDIVIDUAL_

#include <iostream.h>

class CIndividual
{
public:

	// The constructor. The constructed individual has
	// all zeroes as its genes.
	CIndividual();

	// The destructor. It frees the memory allocated at
	// the construction of the individual.
	virtual ~CIndividual();

	// It returns an array with the genes of the individual.
	int * Genes();

	// It returns the value of the individual.
	double Value();

	// Input and output of an individual: its value and genes.
	friend ostream & operator<<(ostream & os,CIndividual * & individual);
	friend istream & operator>>(istream & is,CIndividual * & individual);

	// Number of variables not appeared in the solution
	int m_values_left;

private:

	// The variable storing the individual's value. 
	double m_value;

	// The genes of the individual.
	int * m_genes;
};

#endif
