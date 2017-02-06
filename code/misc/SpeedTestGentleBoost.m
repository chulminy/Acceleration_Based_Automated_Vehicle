%% Implementation of Gentleboost Algorithm
feature     = rands(1000,200);
nRounds     = 100;

classLabel(1,1:100) = 1;
classLabel(1,101:200) = -1;

%% 0 : m-file
tic;
[clss.k, clss.th, clss.a, clss.b] = ...
    GentleBoost_original(feature', classLabel, nRounds);
fprintf('pure m-file : %f s \n', toc);

%% 1 : parellel computing implemented in Matlab 
tic;
poolobj = gcp('nocreate');
if isempty(poolobj) % checking to see if my pool is already open
    parpool('local',feature('numCores'));
end; clearvars poolobj;

[clss.k, clss.th, clss.a, clss.b] = ...
        GentleBoost_parallel(feature', classLabel, nRounds);
fprintf('parellel in Matlab : %f s \n', toc);
delete(poolobj);

%% 2 : C++ implemementation using mex
tic;
if ~exist(['GentleBoost_mex.' mexext],'file')
    mex GentleBoost_mex.cpp
end
[clss.k, clss.th, clss.a, clss.b] = ...
        GentleBoost_mex(feature, classLabel, nRounds);
fprintf('mex implementation : %f s \n', toc);
   
%% 3 : C++ implemementation using mex + OpenMP 
tic;
if  ~exist(['GentleBoost_mex_openmp.' mexext],'file')
    mex GentleBoost_mex_openmp.cpp COMPFLAGS="/openmp $COMPFLAGS" LINKFALGS="$LINKFALGS -openmp";
end
[clss.k, clss.th, clss.a, clss.b] = ...
        GentleBoost_mex_openmp(feature, classLabel, nRounds);   
fprintf('mex+openmp implementation : %f s \n', toc);