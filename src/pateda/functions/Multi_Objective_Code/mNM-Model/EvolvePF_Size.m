function [ListFactors,TheMax] = EvolvePF_Size(nfactors,ListOfLists,nsteps,tsign)

global Tables
global PE
global AllVectors

n = size(AllVectors,2);  % Number of variables

ListFactors = ListOfLists;
for nf=2:nfactors,
  ListFactors{nf} = ListOfLists{nf};
  for ll=1:nsteps,    
    a = 0;
    for i=1:n,    
     set1 = ListFactors{nf}{i}(:)';
     set2 = setdiff([1:n],set1);
     for j=2:size(set1,2),
      for l=1:size(set2,2),  
       ListFactors{nf} = ListOfLists{nf};
       ListFactors{nf}{i}(j) = set2(l);
       for nm=1:nfactors        
         fvals(:,nm) = EvaluatePopNK(AllVectors,ListFactors{nm});
       end
       a = a + 1;
       [Index]=FindParetoSet(fvals,fvals);
       TheVals{a} = fvals(Index,:);     
       ThePairs(a,:) = [i,j,ListOfLists{nf}{i}(j),set2(l),size(Index,2)];
      end
     end
    end
   
  
 if tsign==-1,
     [b1,c1] = min(ThePairs(:,5));   % Maximum decrease in PF size;
 elseif tsign==1,
     [b1,c1] = max(ThePairs(:,5));   % Maximum increase in PF size;
 end  
 
 TheMax(nf,ll) = b1                

 [ac1] = find(ThePairs(:,5)==b1);  % For which arc changes as the maximum PF size achieved?
 s_ac1 = size(ac1,1);
  if s_ac1==1
    c1 = ac1;
  else
    pos = randint(1,1,s_ac1)+1;
    c1 = ac1(pos);
  end
  ListOfLists{nf}{ThePairs(c1,1)}(ThePairs(c1,2)) = ThePairs(c1,4); % The new landscape has tha
  ListFactors{nf} = ListOfLists{nf};
  end
end
 
  
  return
  
  
  
  

  