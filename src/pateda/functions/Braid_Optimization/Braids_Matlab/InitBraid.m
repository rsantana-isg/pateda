function[] = InitBraid(lambda)

 global Sigma
 global Target;   
 global LAMBDA;

 
 LAMBDA = lambda;
 
 tau = (sqrt(5)-1)/2;
 theta1 = -7*pi/10;
 theta2 = 9*pi/10;
 Sigma{1} = [exp(i*theta1),0;0,exp(-i*theta1)];
 Sigma{2} = [-tau*exp(-i*pi/10),-i*sqrt(tau);-i*sqrt(tau),-tau*exp(i*pi/10)];
 tquat{1} = [cos(theta1),sin(theta1),0,0];   
 tquat{2} = [tau*cos(theta2),tau*sin(theta2),0,-sqrt(tau)];   

 for j=1:2,
   Sigma{j+2} = inv(Sigma{j});  
  end
 
Target = [0,i;i,0];
