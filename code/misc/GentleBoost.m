function clss = GentleBoost(feature, classLabel, nRounds, flag_comp)

clssIdx = unique(classLabel);

if numel(clssIdx) ~= 2
    error('gentleboost algorithm only support binary classification');
else
    classLabel(classLabel==clssIdx(1)) = 1;
    classLabel(classLabel==clssIdx(2)) = -1;
end

clss.idx = clssIdx';

switch flag_comp
    case 0 % pure m-file
    [clss.k, clss.th, clss.a, clss.b] = ...
        GentleBoost_original(feature', classLabel, nRounds);
    case 1 % parfor matlab distribute computing
    [clss.k, clss.th, clss.a, clss.b] = ...
        GentleBoost_parallel(feature', classLabel, nRounds);
    case 2 % mex
    [clss.k, clss.th, clss.a, clss.b] = ...
        GentleBoost_mex(feature, classLabel, nRounds);
    case 3 % mex + openmp
    [clss.k, clss.th, clss.a, clss.b] = ...
        GentleBoost_mex_openmp(feature, classLabel, nRounds);    
end







