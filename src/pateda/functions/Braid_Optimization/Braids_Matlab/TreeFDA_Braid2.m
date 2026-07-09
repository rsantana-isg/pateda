 % EXAMPLE 18: 
 
 global Sigma
 global Target;   
 global LAMBDA;
 global BoltzMannProb;
 
 
 tau = (sqrt(5)-1)/2;
 theta1 = -7*pi/10;
 theta2 = 9*pi/10;
 Sigma{1} = [exp(i*theta1),0;0,exp(-i*theta1)];
 Sigma{2} = [-tau*exp(-i*pi/10),-i*sqrt(tau);-i*sqrt(tau),-tau*exp(i*pi/10)];
 tquat{1} = [cos(theta1),sin(theta1),0,0];   
 tquat{2} = [tau*cos(theta2),tau*sin(theta2),0,-sqrt(tau)];   
 
 % quatmultiply(tquat{1},tquat{2})
 % Sigma{1}*Sigma{2}
 
  for j=1:2,
   Sigma{j+2} = inv(Sigma{j});  
  end
 
 Target = [0,i;i,0];
 LAMBDA = 0.001; 
 
 % The number of variables is equal to the sequence length and each
 % variables takes values in {0,1,2}
 PopSize = 500; NumbVar = 50; cache  = [0,0,0,0,0]; Card = 4*ones(1,NumbVar);   maxgen = 1000;
 % The Markov chain model(Cliques) is constructed specifying the number of
 % conditioned (previous) variables. In the example below this number is
 % 1., i.e. p(x) = p(x0)p(x1|x0) ... p(xn|xn-1) 
 BoltzMannProb = zeros(PopSize,1); 
 F = 'evalMajorana2'; % HP protein evaluation function
 edaparams{1} = {'learning_method','LearnTreeModel',{}};
 edaparams{2} = {'sampling_method','SampleFDA',{PopSize}};
 edaparams{3} = {'selection_method','truncation_selection',{0.15,'fitness_ordering'}};
 
 %Cliques = CreateMarkovModel(NumbVar, 0);
 %edaparams{1} = {'learning_method','LearnFDA',{Cliques}};
 %edaparams{2} = {'sampling_method','SampleFDA',{PopSize}};

 %edaparams{1} = {'learning_method','LearnTreeModelFromVector',{}};
 %edaparams{2} = {'sampling_method','SampleFDA',{PopSize}};        
 %edaparams{3} = {'selection_method','Boltzmann_selection',{1}};
 edaparams{4} = {'replacement_method','elitism',{PopSize/10,'fitness_ordering'}};
 edaparams{5} = {'stop_cond_method','max_gen',{maxgen}};
 
 [AllStat,Cache]=RunEDA(PopSize,NumbVar,F,Card,cache,edaparams);
 
 