 % EXAMPLE 18:  Tree-FDA for the HP protein model (The tree structure is
 % learned from the mutual information applying a threshold on the minimum
 % MI value to consider a dependence. Therefore the learned structure
 % actually corresponds to a forest.
 global B
 global Target;   % This is the HP protein instance, defined as a sequence of zeros and ones
 global LAMBDA;
 global BoltzMannProb;
 
 B{1}=           [i,0,0,0;0,i,0,0;0,0,1,0;0,0,0,1]; 
 B{2}= 1/sqrt(2)*[1,0,i,0;0,1,0,i;i,0,1,0;0,i,0,1]; 
 B{3}=           [i,0,0,0;0,1,0,0;0,0,1,0;0,0,0,i]; 
 B{4}= 1/sqrt(2)*[1,i,0,0;i,1,0,0;0,0,1,-i;0,0,-i,1]; 
 B{5}= [i,0,0,0;0,1,0,0;0,0,i,0;0,0,0,1]; 
 for j=1:5,
   B{j+5} = inv(B{j});  
 end
 
 Target = [1,0,0,0;0,1,0,0;0,0,0,1;0,0,1,0];
 LAMBDA = 0.01; 
 
 % The number of variables is equal to the sequence length and each
 % variables takes values in {0,1,2}
 PopSize = 1500; NumbVar = 25; cache  = [0,0,0,0,0]; Card = 10*ones(1,NumbVar);   maxgen = 1000;
 % The Markov chain model(Cliques) is constructed specifying the number of
 % conditioned (previous) variables. In the example below this number is
 % 1., i.e. p(x) = p(x0)p(x1|x0) ... p(xn|xn-1) 
 BoltzMannProb = zeros(PopSize,1); 
 F = 'evalMajoranaPlus'; % HP protein evaluation function
 %edaparams{1} = {'learning_method','LearnTreeModel',{}};
 %edaparams{2} = {'sampling_method','SampleFDA',{PopSize}};
 edaparams{3} = {'selection_method','truncation_selection',{0.15,'fitness_ordering'}};
 
 Cliques = CreateMarkovModel(NumbVar, 0);
 edaparams{1} = {'learning_method','LearnFDA',{Cliques}};
 edaparams{2} = {'sampling_method','SampleFDA',{PopSize}};

 %edaparams{1} = {'learning_method','LearnTreeModelFromVector',{}};
 %edaparams{2} = {'sampling_method','SampleFDA',{PopSize}};        
 %edaparams{3} = {'selection_method','Boltzmann_selection',{1}};
 %edaparams{4} = {'replacement_method','elitism',{1,'fitness_ordering'}};
 edaparams{4} = {'replacement_method','elitism',{PopSize/10,'fitness_ordering'}};
 edaparams{5} = {'stop_cond_method','max_gen',{maxgen}};
 
 [AllStat,Cache]=RunEDA(PopSize,NumbVar,F,Card,cache,edaparams);
 
 