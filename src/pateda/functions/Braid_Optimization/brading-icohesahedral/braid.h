/*****************************************************************************
*
* TQC Project: Topological Quantum Compiling/Computation
*
* braid.h: header file for the braiding library
*
* Copyright (C) 2009 by Xin Wan <xinwan@zimp.zju.edu.cn>
*
*
* This software is part of the TQC libraries, published under the TQC
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1.0 or (at your option) any
* later version.
* 
* You should have received a copy of the TQC Library License along with
* the TQC Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://sites.google.com/site/braidanyons/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
* NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
* DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER
* LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
* OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef _BRAID_H
#define _BRAID_H

#include <iostream>
#include <cstdlib>
#include <complex>
#include <vector>
#include <iterator>
#include <assert.h>
using namespace std;

class UMatrix {
private:
//********************************
//
// U =  a  b   
//     -b* a*  * e^{i theta}
//
// Note: we do not require, here, 
//       1 = |a|^2 + |b|^2
//
//********************************
  complex<double> a;
  complex<double> b;
  double theta; // overall phase
public:
  UMatrix() : a(1.0), b(0.0), theta(0.0) {}
  UMatrix(double ua, double ub, double ut = 0) : a(ua), b(ub), theta(ut) {}
  UMatrix(complex<double> ua, complex<double> ub, double ut = 0) : a(ua), b(ub), theta(ut) {}

  UMatrix& set_axis_angle(const double x, const double y, const double z, const double phi);
  void matrixElements(complex<double>* m) const;

  UMatrix& operator=(const UMatrix& u) { a = u.a; b = u.b; theta = u.theta; return *this; }
  UMatrix& operator/=(const double s) { a /= s; b /= s; return *this; }
  
  UMatrix operator~() { return UMatrix(conj(a), -b, -theta); }  // Hermitian conjugate
  UMatrix operator-() { return UMatrix(-a, -b, theta); }  // negative
  UMatrix operator-(const UMatrix& u) { return UMatrix(a - u.a, b - u.b); }  //phase information gone
  UMatrix operator*(const UMatrix& u) { 
    return UMatrix(a * u.a - b * conj(u.b), a * u.b + b * conj(u.a), theta + u.theta);
  }
  
  double quasidistance(const UMatrix& u);
  double quasidistance() { return abs(a.real()) + abs(a.imag()) + abs(b.real()) + abs(b.imag()); }
  double distance(const UMatrix& u);
  double distance() { return sqrt(norm(a) + norm(b)); }
  double phase() { return theta; }
  void init() { a = 1.0; b = 0.0; theta = 0.0; return; } 

  void print_ab_format() const { cout << UMatrix(a, b, 0.0); }
  void print_abt() const { cout << a << ", " << b << " | " << theta / M_PI << " Pi\n"; }

  friend ostream& operator<<(ostream&, const UMatrix&);

};
  
ostream& operator<<(ostream&, const UMatrix&);

class Braid {
private:
  static const UMatrix sigma[4];
  vector<int> s;
  UMatrix u;
public:
  Braid() {};
  Braid(const int bl, const int* bs); // Initialize with 
  // a sequence of fundamental braids labeled by {1,2,3,4} or {1,2,-2,-1}
  Braid(const vector<int> bs, UMatrix bu) : s(bs), u(bu) {}; // Initialize with vector sequence and known matrix representation
  
  Braid& operator=(const Braid& b) { s = b.s; u = b.u; return *this; }
  Braid& operator+=(const Braid& b);
  
  Braid operator~();

  int length() const { return s.size(); }
  UMatrix rep() const { return u; }
  double quasidistance(UMatrix& u0) { return u.quasidistance(u0); }
  double distance(UMatrix& u0) { return u.distance(u0); }

  bool redundant(const Braid& b) { return s.back() == (3 - b.s.front());} 
  
  Braid& simplify();
  Braid& clear() { s.clear(); u.init(); return *this; }

  void print_abt() { u.print_abt(); }
  void print_seq() { for (int i = 0; i < s.size(); i++) cout << s[i] << " "; cout << endl; }

  friend ostream& operator<<(ostream&, const Braid&);
};

ostream& operator<<(ostream&, const Braid&);

#endif // _BRAID_H
