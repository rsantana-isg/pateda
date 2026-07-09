


corrs = [-0.5,0,0.5];             % Possible correlations for the mUBQP instances
dens = 0.2*[1:5];                 % Possible densities for the mUBQP instances    


ninst = 50;                       % Number of instances
M = 2;                            % Number of objectives


for n=11:14,                       % We generate problems with dimension between n=5 and n=14
 Card = 2*ones(1,n);              % All variables are binary
 AccCard = cumprod(Card)/2;
 AccCard = AccCard([n:-1:1]); 
 N = 2^n;                         % Total number of solutions in the search space
 Pop = zeros(N,n);                % Pop contains all the solutions from the search space          

                             
 for j=1:N,                       %  Pop is filled with all possible binary vectors, size n  
  num = IndexconvertCard(j-1,n,AccCard);
  Pop(j,:) = num;
  IndPop{j} = find(Pop(j,:)==1);               % Save the indices in each solution that are ones
                                               % Used for efficiency of the function evaluation                                               
 end

% We have previously generated 50 instances of each combination of the
% correlations and indexes values for all values of n
% These intances were generated using random generator proposed by the
% authors and programmed in R
% Now we evaluate the full space of solutions for each of the instances and
% we store in the variable    AllFs{n,i,d,k} where indices correspond to n:
% problem size, i:correlation, d:density, k:instance
 clear therank
 clear sols_ranks;       
 clear RankBestHeus;
 clear DistToBest;
 for i=1:3,
  for d=1:5,
     [n,i,d]   
     for k=1:ninst,     
       fname = ['N_',num2str(n),'_d_',num2str(dens(d)),'_ro_',num2str(corrs(i)),'_',num2str(k),'.dat'];
       the_f = load(fname);
       F = zeros(N,2);      
       for s=1:M,
         mat = reshape(the_f(:,s),n,n);  
         for l=1:N,                                % For each pair of ones in the solution l
           for t=IndPop{l},                        % the corresponding cells of the UBQP matrices  
              for q=IndPop{l},                     % are saved
                F(l,s) = F(l,s) + mat(t,q) + mat(t,q);  
              end    
           end 
         end   
         [vals,positions] = sort(F(:,s),'descend');
         therank(:,s) = positions;         
         sols_ranks(positions,s) = 1:N;       
         TheBest = IndexconvertCard(therank(1,s)-1,n,AccCard);
         BestHeus = (sum(mat) + sum(mat'))>0;
         postBestHeus = NumconvertCard(BestHeus,n,AccCard)+1;
         RankBestHeus(s) = sols_ranks(postBestHeus);   
         DistToBest(s) = sum(abs(BestHeus-TheBest));
       end      
       
       AllFs{n,i,d,k} = F;
       AllRanks{n,i,d,k} = therank;
       AllSolRanks{n,i,d,k} = sols_ranks;
       AllBestHeusPos{n,i,d,k} = RankBestHeus;
       AllBestDist{n,i,d,k} = DistToBest;
     end
  end
 end
end         

for n=5:14 
 MeanBestDist{n-4} = zeros(3,5);   
 for i=1:3,
  for d=1:5,   
    for k=1:ninst,     
     MeanBestDist{n-4}(i,d) = MeanBestDist{n-4}(i,d) + AllBestDist{n,i,d,k}(1) + AllBestDist{n,i,d,k}(2);
    end    
  end
 end
 MeanBestDist{n-4} = MeanBestDist{n-4}/(2*ninst);
end
 
save ResultsMUBQPAnalysis.mat  AllFs  AllRanks   AllSolRanks  AllBestHeusPos   AllBestDist

