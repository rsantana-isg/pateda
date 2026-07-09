
load BraidLandScape.mat
MI = zeros(n);
Card = 4*ones(1,n);
BivProb = Biv;
UnivProb = Univ;
for i=1:n-1,
 for j=i+1:n,
   for k=1:Card(i),
      for l=1:Card(j),
        if(BivProb{i,j}(k,l)>0)
            MI(i,j) = MI(i,j) + BivProb{i,j}(k,l)* log(BivProb{i,j}(k,l)/(UnivProb(k,i)*UnivProb(l,j)));
        end
      end,
   end,
   MI(i,j) = MI(i,j)/(Card(i)*Card(j)); %Normalization of the mutual information 
   MI(j,i) = MI(i,j);
 end,
end,

save BraidLandScape.mat Fit Pop Univ BoltzMannProb Biv N MI



load ELBraidLandScape.mat
MI = zeros(n);
Card = 4*ones(1,n);
BivProb = ELBiv;
UnivProb = ELUniv;
for i=1:n-1,
 for j=i+1:n,
   for k=1:Card(i),
      for l=1:Card(j),
        if(BivProb{i,j}(k,l)>0)
            MI(i,j) = MI(i,j) + BivProb{i,j}(k,l)* log(BivProb{i,j}(k,l)/(UnivProb(k,i)*UnivProb(l,j)));
        end
      end,
   end,
   MI(i,j) = MI(i,j)/(Card(i)*Card(j)); %Normalization of the mutual information 
   MI(j,i) = MI(i,j);
 end,
end,

ELMI = MI;
save ELBraidLandScape.mat ELFit Pop ELUniv ELBoltzMannProb ELBiv N ELMI





load BestBraidLandScape.mat
MI = zeros(n);
Card = 4*ones(1,n);
BivProb = BESTBiv;
UnivProb = BESTUniv;
for i=1:n-1,
 for j=i+1:n,
   for k=1:Card(i),
      for l=1:Card(j),
        if(BivProb{i,j}(k,l)>0)
            MI(i,j) = MI(i,j) + BivProb{i,j}(k,l)* log(BivProb{i,j}(k,l)/(UnivProb(k,i)*UnivProb(l,j)));
        end
      end,
   end,
   MI(i,j) = MI(i,j)/(Card(i)*Card(j)); %Normalization of the mutual information 
   MI(j,i) = MI(i,j);
 end,
end,

BESTMI = MI
save BestBraidLandScape.mat BESTFit Pop BESTUniv BESTBoltzMannProb BESTBiv N BESTMI