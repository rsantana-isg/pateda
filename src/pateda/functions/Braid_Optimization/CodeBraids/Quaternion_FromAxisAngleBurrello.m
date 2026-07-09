function[q] = Quaternion_FromAxisAngleBurrello(x,y,z,phi)

norm = sqrt(x * x + y* y + z* z);  
a = cos(phi / 2.0) + (-z / norm * sin(phi / 2.0))*i;
b =  -y / norm * sin(phi / 2.0) + (-x / norm * sin(phi / 2.0))*i;

q(1) = real(a);
q(2) = imag(a);
q(3) = real(b);
q(4) = imag(b);
	



