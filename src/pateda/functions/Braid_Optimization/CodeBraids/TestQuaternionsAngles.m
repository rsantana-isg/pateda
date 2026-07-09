
% Test of the computation of the quaternions from the axis angles
% The multiplication of the quaterions from the unitary matrices 
% should be equal to the quaternion of the result of the multiplication
% of the matrices, and that is the case.

axisX =-1.0; axisY = tau; axisZ = 0; angle = 0.4*pi;
q = Quaternion_FromAxisAngle(axisX,axisY,axisZ,angle)
quatmultiply(q,q)

mat = [q(1)+q(2)*i,q(3)+q(4)*i;-(q(3)-q(4)*i),q(1)-q(2)*i]

