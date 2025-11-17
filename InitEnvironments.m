function[] = InitEnvironments()
% [] = InitEnvironments()
%
% InitEnvironments:           Initialize the environment of mateda
%                   After installing the BNT, BNT_SLP learning matlab, and
%                   mateda toolboxs, update the paths below according the
%                   location of the programs in your computer.             
%
% Last version 8/26/2008. Roberto Santana (roberto.santana@ehu.es)       

path_mateda =  '~/Dropbox/ExperimentsAndCode/Mateda';
path_FullBNT = '~/Dropbox/ExperimentsAndCode/Mateda/bnt';
path_BNT_SLP = '~/Dropbox/ExperimentsAndCode/Mateda/BNT_SLP';

%path_mateda =  'C:\WorkDirectory\Mateda2.2';
%path_FullBNT = 'C:\WorkDirectory\FullBNT-1.0.4';
%path_BNT_SLP = 'C:\WorkDirectory\FullBNT-1.0.4\BNT_StructureLearning_v1[1].4c\BNT_SLP';


   cd(path_FullBNT);
   addpath(genpathKPM(pwd));
   cd(path_BNT_SLP);
   add_SLP;   


P = genpath(path_mateda);
addpath(P);
cd(path_mateda);



% Last version 8/26/2008. Roberto Santana (roberto.santana@ehu.es) 
