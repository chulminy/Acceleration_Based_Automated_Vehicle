%% Description 
% 
% This file is to extract features from each acceleartion signal. Unlike
% viola and jones method, the STFT images is considered as "window", not
% patch. Thus, a large number of features are needed to extract entire
% areas of a STFT image evenly.
%
% Acceleration signals were captured from different bounds, vehicles,
% accelerometers, and runs. 

% When training and testing are assigned, 
% To increase computational efficiency, I tweaked some codes, which may
% not be intuitively understood. If you don't understand the code, please
% email me and I will make a better description next to the corresponding
% code lines.

Parameter;

%% Load acceleartion data 
if ~exist(fullfile(folderPrc,'DataOrg.mat'),'file');
    count       = 1;
    for ii=1:nBound
        for jj=1:nVehicle
            bound   = prefBound{ii};
            vehicle = prefVehicle{jj};
            load(fullfile(folderAcc, [bound '_' vehicle '.mat']));
            for qq=1:nRun
                for kk=1:nSensor
                    DataOrg(count).bound    = ii;
                    DataOrg(count).vehicle  = jj;
                    DataOrg(count).acc      = rawSig{qq,kk};
                    DataOrg(count).idSensor = kk;
                    DataOrg(count).idRun    = qq;
                    count = count + 1;
                end
            end
        end
    end
    save(fullfile(folderPrc,'DataOrg.mat'),'DataOrg');
else
    load(fullfile(folderPrc,'DataOrg.mat'),'DataOrg');
end

%% Generate Haar patches
if ~exist(fullfile(folderPrc,'DataHaar.mat'),'file');
    haarPatch = cell(nFeature,1);
    for ii= 1:nFeature
        [posMat, operMat]   = ...
            CompHaarPatch([1 1], sizeSpect.w, sizeSpect.h);
        haarPatch{ii}    = [posMat operMat];
    end; clearvars ii posMat operMat;
    
    % for example: a feature is computed using a featIdx.haar(i) haar
    % patch on featIdx.channel(i) channel image
    save(fullfile(folderPrc,'DataHaar.mat'),'haarPatch');
else
    load(fullfile(folderPrc,'DataHaar.mat'),'haarPatch');
end

%% threshold generation ---------------------------------------------------
% threshold determination (extracting last 1000 points) assuming that 
% signal tails have pure noise.
%
% we assume that noise characteristics are sensitive to sensors (idSensor)
% and bridge instrallations (bound).
if ~exist(fullfile(folderPrc,'DataThresh.mat'),'file');
    nSig        = numel(DataOrg);
    noiseSig    = cell(nSensor,nBound);
    for ii=1:nSig
        noiseSig{DataOrg(ii).idSensor,DataOrg(ii).bound} = ...
            [noiseSig{DataOrg(ii).idSensor,DataOrg(ii).bound}; ...
            DataOrg(ii).acc(end-1000:end)];
    end
    
    % threshold amplitude estimation
    thresh.amp       = zeros(nSensor,nBound);
    thresh.meanAmp   = zeros(nSensor,nBound);
    thresh.RMS       = zeros(nSensor,nBound);
    for ii=1:nSensor
        for jj=1:nBound
            thresh.amp(ii,jj)       = std(noiseSig{ii,jj})*5;
            thresh.meanAmp(ii,jj)   = mean(noiseSig{ii,jj});
            tmp = filter(b_filt_h, a_filt_h, ...
                noiseSig{DataOrg(ii).idSensor,DataOrg(ii).bound});
            thresh.RMS(ii,jj) = std(tmp)*5;
        end
    end
    save(fullfile(folderPrc,'DataThresh.mat'),'thresh');
else
    load(fullfile(folderPrc,'DataThresh.mat'),'thresh');
end

%% Data range generation --------------------------------------------------
if ~exist(fullfile(folderPrc,'DataRange.mat'),'file')
    nSig            = numel(DataOrg);
    rangeCrop       = zeros(nSig,2);
    for ii=1:nSig
        dataTmp1 = DataOrg(ii).acc - ...
            thresh.meanAmp(DataOrg(ii).idSensor,DataOrg(ii).bound);
        
        % in reallity, signal is triggered based on threshold acc
        % amplitude. i1 time is not needed in real implementation.
        
        % estimate a starting point
        tmp1    = MovingRMS(dataTmp1, windRMS);
        i1      = find(tmp1> ...
            thresh.amp(DataOrg(ii).idSensor,DataOrg(ii).bound)); 
        
        % estimate an end point
        dataTmp2    = filter(b_filt_h, a_filt_h, dataTmp1);
        tmp1        = MovingRMS(dataTmp2, windRMS);
        % sometime, sudden noise is included after vehicle exist. Here, I
        % assume that all vehicles exist no more than 8 second.
        i2          = find(tmp1(i1(1):min(i1(1)+8*fs, numel(tmp1)))> ...
            thresh.RMS(DataOrg(ii).idSensor,DataOrg(ii).bound));
        
        rangeCrop(ii,:) = [i1(1) i1(1)+i2(end)];
    end
    
    % if gravels hit one of accelerometers, high peaks are prodcued 
    % a certain high peak is produced if gravelrs hit one of
    % accelerometers. Here, crop range is identifical for all
    % accelerometers in each case (combination of a vehicle and a bound)
    range       = zeros(nSig,2);
    for ii=1:nBound
        for jj=1:nVehicle
            for kk=1:nRun
                idx1 = ([DataOrg(:).bound] == ii);
                idx2 = ([DataOrg(:).vehicle] == jj);
                idx3 = ([DataOrg(:).idRun] == kk);
                
                idx  = idx1 & idx2 & idx3;
                
                tmp1 = round(median(rangeCrop(idx,1)));
                tmp2 = round(median(rangeCrop(idx,2)));

                range(idx,1) = tmp1;
                range(idx,2) = tmp2;
            end
        end
    end
    save(fullfile(folderPrc,'DataRange.mat'),'range');
    for ii=1:nSig; DataOrg(ii).range = range(ii,:); end
else
    nSig = numel(DataOrg);
    load(fullfile(folderPrc,'DataRange.mat'),'range');
    for ii=1:nSig; DataOrg(ii).range = range(ii,:); end
end

%% Feature computation ----------------------------------------------------
if ~exist(fullfile(folderPrc,'DataFeatureMat.mat'),'file');
    featureMat  = zeros(nFeature, nSig);
    for ii=1:nSig
        dataTmp1 = DataOrg(ii).acc - ...
            thresh.meanAmp(DataOrg(ii).idSensor,DataOrg(ii).bound);
        
        dataCrop = dataTmp1(DataOrg(ii).range(1):DataOrg(ii).range(2));
        
        tmp = resample(dataCrop, 1, floor(fs/(2*cutoff_freq)));
        
        dataCropSTFTL        = ...
            spectrogram([zeros(wlenL/2,1); tmp ; zeros(wlenL/2-1,1)], ...
            wL, wlenL-1, nfft*2);
        dataCropSTFTH        = ...
            spectrogram([zeros(wlenH/2,1); tmp ; zeros(wlenH/2-1,1)], ...
            wH, wlenH-1, nfft*2);
        
        dataCropSTFTL   = imresize(abs(dataCropSTFTL), [nfft+1, ntt]);
        dataCropSTFTH   = imresize(abs(dataCropSTFTH), [nfft+1, ntt]);
        
        dataIntegL      = integralImage(dataCropSTFTL);
        dataIntegH      = integralImage(dataCropSTFTH);
        
        featureMat(1:nFeature/2,ii)        = CompFeature(...
          dataIntegL, haarPatch(1:nFeature/2), nFeature/2,sizeSpect);
        featureMat(nFeature/2+1:end,ii)    = CompFeature(...
          dataIntegH, haarPatch(nFeature/2+1:end),nFeature/2,sizeSpect);
    end
    save(fullfile(folderPrc,'DataFeatureMat.mat'),'featureMat');
else
    load(fullfile(folderPrc,'DataFeatureMat.mat'),'featureMat');
end