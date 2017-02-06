function featureVec = CompFeature(integSpect, haarPatchMat, nFeature, sizeSpect)

featureVec   = zeros(nFeature,1);
for fIdx=1:nFeature
    posMat  = haarPatchMat{fIdx,1}(:,1:2);
    operMat = haarPatchMat{fIdx,1}(:,3);
    tmpIdx =(posMat(:,1)-1)*(sizeSpect.h + 1) + posMat(:,2);
    featureVec(fIdx)  = integSpect(tmpIdx)'*operMat;
end
