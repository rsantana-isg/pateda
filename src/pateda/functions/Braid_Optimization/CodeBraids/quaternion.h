#ifndef QUATERNION_H
#define QUATERNION_H

//namespace quaternion
//{


typedef struct
{
	double r,i;
} Complex;


extern Complex I;
extern Complex minusI;
extern Complex U;
extern Complex minusU;
extern Complex Z;

void Complex_Add(Complex a, Complex b, Complex* result);
void Complex_Multiply(Complex a, Complex b, Complex* result);
Complex Complex_Invert(Complex a);


#ifndef FAKE_QUATERNIONS 

typedef struct
{
	Complex elements[16];
} Quaternion;

void Quaternion_Scale(Complex s, Quaternion q,  Quaternion* result);

#else
typedef struct
{
	double s;
	double i,j,k;
} Quaternion;

void Quaternion_FromAxisAngle(double axisX, double axisY, double axisZ, double angle, Quaternion* q);
Quaternion Quaternion_FromAxisAngleBurrello(double axisX, double axisY, double axisZ, double angle);
#endif

void Quaternion_Multiply(Quaternion a, Quaternion b, Quaternion* result);
void Quaternion_Conjugate(Quaternion q, Quaternion* result);
double Quaternion_LengthSquared(Quaternion q);
double Quaternion_Length(Quaternion q);
double Quaternion_Distance(Quaternion a, Quaternion b);
void Quaternion_Print(Quaternion q);
void Quaternion_Copy(Quaternion* Dest, Quaternion orig);

//}
#endif

