/***************************************************************************
                          fun.h  -  description                              
                             -------------------                                         
    begin                : Mon Dec 27 1999                                           
    copyright            : (C) 1999 by                          
    email                :                                      
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   * 
 *                                                                         *
 ***************************************************************************/

#ifndef __FUN__
#define __FUN__

#include "pop.h"
#include "pada.h"
#include "dist.h"

void evalua(pop &pPop, double *fv, pada_params *W);
void funname(int Fun, char *dest);
double eval(int , int *,int );
int StopConditions(int Fun, double BestFv, double AvgFv, int StopGen, pop *Pop, dist1 *D1, dist2 *D2, dist3 *D3);

#endif
