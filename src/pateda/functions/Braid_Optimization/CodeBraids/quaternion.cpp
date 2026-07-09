#include "quaternion.h"
#include <iostream>
#include <fstream>
#include <stdlib.h> 
#include <math.h>
#include <assert.h>

//using namespace quaternion;
using namespace std;

Complex I = (Complex){0,1};
Complex minusI = (Complex){0,-1};
Complex U = (Complex){1,0};
Complex minusU = (Complex){-1,0};
Complex Z = (Complex){0,0};

void Complex_Add(Complex a, Complex b, Complex* result)
{
	result->r = a.r + b.r;
	result->i = a.i + b.i;
	
}

double Complex_NormSquared(Complex a)
{
	return a.r*a.r + a.i*a.i;
}

void Complex_Subtract(Complex a, Complex b, Complex* result)
{

	result->r = a.r - b.r;
	result->i = a.i - b.i;
	
}

void Complex_Multiply(Complex a, Complex b, Complex* result)
{
	
	result->r = a.r * b.r - a.i * b.i;
	result->i = a.r * b.i + a.i * b.r;
	
}

Complex Complex_Inverse(Complex a)
{
	double r2 = Complex_NormSquared(a);
	return (Complex){a.r/r2, -a.i/r2};
}

#ifndef FAKE_QUATERNIONS


void Quaternion_Scale(Complex s, Quaternion q, Quaternion* r)
{
	int i;
	for(i=0; i<16; i++)
	  Complex_Multiply(s, q.elements[i],&r->elements[i]);
       
}

void Quaternion_Multiply(Quaternion a, Quaternion b, Quaternion* result)
{
	int i,j,k;
        Complex auxcomplex;
        Quaternion AuxQuat;
  
	for(i=0; i<4; i++)
	{
		for(j=0; j<4; j++)
		{
			AuxQuat.elements[i*4+j] = Z;
			for(k=0; k<4; k++)
			{
			  Complex_Multiply(a.elements[i*4+k], b.elements[k*4+j],&auxcomplex);
			  Complex_Add(AuxQuat.elements[i*4+j],auxcomplex,&(AuxQuat.elements[i*4+j]));
			}
		}
	}

    for(i=0; i<4; i++)
	for(j=0; j<4; j++)
	  result->elements[i*4+j] = AuxQuat.elements[i*4+j];
	
}

double Quaternion_LengthSquared(Quaternion q)
{
	double len2 = 0;
	int i;
	for(i=0; i<16; i++)
		len2 += Complex_NormSquared(q.elements[i]);
	return len2;
}


double Quaternion_Distance(Quaternion a, Quaternion b)
{
	Quaternion diff;
	int i;
	for(i=0; i<16; i++)
	  Complex_Subtract(a.elements[i], b.elements[i],&diff.elements[i]);
	return Quaternion_Length(diff);
}

void Quaternion_Print(Quaternion q)
{
	int i,j;
	for(i=0; i<4; i++)
	{
		for(j=0; j<4; j++)
			printf("%f+i%f  ", q.elements[4*i+j].r, q.elements[4*i+j].i);
		printf("\n");
	}
}

void Quaternion_Copy(Quaternion* Dest, Quaternion orig)
{
  int i;
    for(i=0; i<16; i++)
      {	
	Dest->elements[i] = orig.elements[i];
      }
	
}

#else
void Quaternion_Print(Quaternion q)
{
	printf("%f+i%f+j%f+k%f\n", q.s, q.i, q.j, q.k);
}

void Quaternion_Multiply(Quaternion a, Quaternion b, Quaternion* result)
{
  Quaternion AuxQuat;
 
	AuxQuat.s = a.s*b.s - (a.i*b.i + a.j*b.j + a.k*b.k);
	AuxQuat.i = a.s*b.i + b.s*a.i + (a.j*b.k - a.k*b.j);
	AuxQuat.j = a.s*b.j + b.s*a.j + (a.k*b.i - a.i*b.k);
	AuxQuat.k = a.s*b.k + b.s*a.k + (a.i*b.j - a.j*b.i);

       	result->s = AuxQuat.s;
        result->i = AuxQuat.i;
        result->j = AuxQuat.j;
        result->k = AuxQuat.k;    
     
	
}



void Quaternion_Conjugate(Quaternion q, Quaternion* result)
{
	result->s = q.s;
	result->i = -q.i;
	result->j = -q.j;
	result->k = -q.k;
	
	
}

double Quaternion_LengthSquared(Quaternion q)
{
	return q.s*q.s + q.i*q.i + q.j*q.j + q.k*q.k;
}

double Quaternion_Distance(Quaternion a, Quaternion b)
{
	Quaternion difference;
	difference.s = a.s - b.s;
	difference.i = a.i - b.i;
	difference.j = a.j - b.j;
	difference.k = a.k - b.k;
	
	return Quaternion_Length(difference);
}

void Quaternion_Copy(Quaternion* Dest, Quaternion orig)
{
	
	Dest->s = orig.s;
	Dest->i = orig.i;
	Dest->j = orig.j;
	Dest->k = orig.k;	
	
}


//double Quaternion_Length(Quaternion q)
//{
//	return sqrt(q.s*q.s + q.i*q.i + q.j*q.j + q.k*q.k);
//}

#endif

double Quaternion_Length(Quaternion q)
{
	return sqrt(Quaternion_LengthSquared(q));
}


Quaternion Quaternion_FromAxisAngleBurrello(double x,double y,double z, double phi) 
{

  Quaternion q;
  double norm = sqrt(x * x + y* y + z* z);
  assert(norm != 0.0);
 
  Complex a = (Complex){cos(phi / 2.0), -z / norm * sin(phi / 2.0)};
  Complex b = (Complex){-y / norm * sin(phi / 2.0), -x / norm * sin(phi / 2.0)};

  q.s = a.r;
  q.i = a.i;
  q.j = b.r;
  q.k = b.i;
  
  cout<<q.s<< " "<<q.i<<" "<<q.j<<" "<<q.k<<endl;

  //Quaternion_Print(q);

  return q;
}



void Quaternion_FromAxisAngle(double axisX, double axisY, double axisZ, double angle, Quaternion* q)
{
       //Quaternion q;
	q->s = cos(angle/2);
	
	double s = sin(angle/2);
	double l = sqrt(axisX*axisX + axisY*axisY + axisZ*axisZ);
	q->i = axisX * s / l;
	q->j = axisY * s / l;
	q->k = axisZ * s / l;
	
       
	//return q;
}


