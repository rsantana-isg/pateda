function[Cor] =  DifficultyMetrics(thefunc,Pop)

for i=1:size(thefunc,2),
   bestval = max(thefunc);
   nbest = find(thefunc==bestval);

   for j=1:size(Pop,1)
    dist = HillClimbing(j,Pop,thefunc);
    Alldist(j) = dist;   
   end 
end