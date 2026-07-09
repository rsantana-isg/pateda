function[q] = Quaternion_FromAxisAngle(axisX,axisY,axisZ,angle)

q(1) = cos(angle/2);
s = sin(angle/2);
l = sqrt(axisX*axisX + axisY*axisY + axisZ*axisZ);
q(2) = axisX * s / l;
q(3) = axisY * s / l;
q(4) = axisZ * s / l;
	
