%% Generation of features
FeatureGeneration;

% cast studies (set 1 if you want to plot)
TABLE1 = 1;  % table 1 in the paper
TABLE2 = 1;  % table 2 in the paper
TABLE3 = 1;  % table 3 in the paper

%% Compute a confusion matrix of bridge setup classification uinsg V4 data
% 
% Please take a look at Table 1 in the paper to understand how to assign
% testing and training data. 
if TABLE1 % Table 1
    idVehicle   = 4; % using a V4 (reference vehicle)
    
    baseIdx  = ([DataOrg(:).vehicle] == idVehicle);

    nCase    = nRun*nBound;
    idTs     = zeros(nCase,nSensor);                % testing sample
    idTr     = zeros(nCase,sum(baseIdx)-nSensor);   % training sample
    clssTs   = zeros(nCase,1);               % class of the testing sample
    count    = 1;
    for ii=1:nBound
        for jj=1:nRun
            idx1 = ([DataOrg(:).bound] == ii);
            idx2 = ([DataOrg(:).idRun] == jj);
            idx  = baseIdx & idx1 & idx2;
            
            idTs(count,:)   = find(idx);
            idTr(count,:)   = find(baseIdx & ~idx);
            clssTs(count)   = ii; % true class of testing (bound)
            count = count + 1;
        end
    end
    
    confMat         = zeros(nBound,nBound+1);
    nTrSample       = size(idTr,2);
    for ii = 1: nCase
        boundIdx = [DataOrg(idTr(ii,:)).bound];
        
        % OVR setup
        clssPair = nchoosek(1:nBound,2);
        nPair    = size(clssPair,1);
        
        clss   = cell(nPair,1);
        for qq=1:nPair
            idx = or((boundIdx == clssPair(qq,1)), ...
                     (boundIdx == clssPair(qq,2)));
            clss{qq} = ...                
                GentleBoost(featureMat(:,idTr(ii,idx))', ...
                boundIdx(idx)', nRounds, flag_comp);
        end
        
        % testing
        clssEst = [];
        for qq=1:numel(clss)
            clssEst = [clssEst; ...
                PredictGentleBoost(clss{qq},featureMat(:,idTs(ii,:))')];
        end;    
        
        [a,~,c] = mode(clssEst);
        
        % 12 is UC threshold, which is (nBound-1)*nSensor*0.75
        if (length(c{1}) == 1) && (sum(clssEst==a) >= 12)
            confMat(clssTs(ii),a) = confMat(clssTs(ii),a) + 1;
        else
            confMat(clssTs(ii),nBound+1) = confMat(clssTs(ii),nBound+1)+ 1;
        end
    end
    save(fullfile(folderPrc,'ConfMat_Table1.mat'),'confMat');
end

%% Confusion matrix for vehicle classification 
% use of training and testing data collected from an identical bridge setup
% 
% Please look at Table 2 in the paper to understand how to assign testing
% and training data.

if TABLE2 % Table 2
    
    confMat  = zeros(nVehicle,nVehicle+1);
    
    clssIdx  = [1 3:6];     % let's remove V2.
    nCase    = nRun*nBound* (nVehicle-1);           
    idTs     = zeros(nCase,nSensor);                % testing sample
    
    % training sample 
    idTr     = zeros(nCase,nRun*nSensor*(nVehicle-1)-nSensor);  
    
    clssTs   = zeros(nCase,1);  % true class of testing data
    count    = 1;
    for ii=1:nBound
        for jj=1:numel(clssIdx)
            for kk=1:nRun
                idx1 = ([DataOrg(:).bound] == ii);
                idx2 = ([DataOrg(:).vehicle]  == clssIdx(jj));
                idx3 = ([DataOrg(:).idRun] == kk);                
                
                idxTsTmp  = idx1 & idx2 & idx3;                
                
                idx4 = ([DataOrg(:).vehicle] ~= 2);
                idxTrTmp  = idx1 & idx4 & (~idxTsTmp);
                
                idTs(count,:)   = find(idxTsTmp);
                idTr(count,:)   = find(idxTrTmp);
                clssTs(count)  = clssIdx(jj);
                count = count + 1;
            end
        end
    end
    
    for ii = 1: nCase
       vIdx = [DataOrg(idTr(ii,:)).vehicle];
       
       % training
       clssPair = nchoosek(clssIdx,2);
       nPair    = size(clssPair,1);
       
       clss     = cell(nPair,1);
       for qq=1:nPair
           idxV = or((vIdx == clssPair(qq,1)), ...
                    (vIdx == clssPair(qq,2)));
           clss{qq} = ...                
                GentleBoost(featureMat(:,idTr(ii,idxV))', ...
                vIdx(idxV)', nRounds, flag_comp); 
       end
        
        % testing
        clssEst = [];
        for qq=1:numel(clss)
            clssEst = [clssEst; ...
                PredictGentleBoost(clss{qq},featureMat(:,idTs(ii,:))')];
        end; clearvars clss;   
        
        % 24 is UC threshold, which is (nVehicle-1)*nSensor*0.75
        [a,~,c] = mode(clssEst);
        if (length(c{1}) == 1) && (sum(clssEst==a) >= 24)
            confMat(clssTs(ii),a) = confMat(clssTs(ii),a) + 1;
        else
            confMat(clssTs(ii),nVehicle+1) = ...
                confMat(clssTs(ii),nVehicle+1)+ 1;
        end
        fprintf('Complete case number %d \n',ii); 
    end
    
    % V2 vehicle testing
    baseIdx     = ([DataOrg(:).vehicle] ~= 2);
    vIdx        = [DataOrg(:).vehicle];
    for ii=1:nBound
        trIdx   = and(baseIdx,([DataOrg(:).bound] == ii));
                
        clssPair = nchoosek(clssIdx,2);
        nPair    = size(clssPair,1);
        
        clss     = cell(nPair,1);
        for qq=1:nPair
            idxV = or((vIdx == clssPair(qq,1)), ...
                     (vIdx == clssPair(qq,2)));
            clss{qq} = ...
                GentleBoost(featureMat(:,trIdx(idxV))', ...
                vIdx(idxV)', nRounds, flag_comp);
        end
        
        for jj=1:nRun
            tsIdx = and(([DataOrg(:).vehicle] == 2), ...
                        ([DataOrg(:).idRun] == jj));
            tsIdx = and(tsIdx,[DataOrg(:).bound] == ii);
            
            % testing
            clssEst = [];
            for qq=1:nPair
                clssEst = [clssEst; ...
                    PredictGentleBoost(clss{qq},featureMat(:,tsIdx)')];
            end; 
            
            % 12 is UC threshold, which is (nBound-1)*nSensor*0.75
            [a,~,c] = mode(clssEst);
            if (length(c{1}) == 1) && (sum(clssEst==a) >= 12)
                confMat(2,a) = confMat(2,a) + 1;
            else
                confMat(2,nVehicle+1) = confMat(2,nVehicle+1)+ 1;
            end
        end; clearvars clss;
    end
    save(fullfile(folderPrc,'ConfMat_Table2.mat'),'confMat');
end

%% Confusion matrix for vehicle classification 
% (training and testing data collected from different bridge setups)
% use of training and testing data collected from an identical bridge setup
% 
% Please look at Table 3 in the paper to understand how to assign testing
% and training data.

if TABLE3 % Table 3

    % bridge setup classifier
    idVehicle   = 4; % using a V4 (reference vehicle)
    
    baseIdx  = ([DataOrg(:).vehicle] == idVehicle);

    idTs     = zeros(nBound,nSensor*nRun);                % testing sample
    idTr     = zeros(nBound,sum(baseIdx)-nSensor*nRun);   % training sample
    clssTs   = zeros(nBound,1);
    for ii=1:nBound
        idx1 = ([DataOrg(:).bound] == ii); 
        idx  = baseIdx & idx1;
        
        idTs(ii,:)   = find(idx);
        idTr(ii,:)   = find(baseIdx & ~idx);
        clssTs(ii)   = ii;
    end
    closeBound = zeros(nBound,1);

    for ii = 1: nBound
        clss = GentleBoost(featureMat(:,idTr(ii,:))', ...
                [DataOrg(idTr(ii,:)).bound]', nRounds, flag_comp);
                
        clssEst =  PredictGentleBoost(clss,featureMat(:,idTs(ii,:))');      
        
        [closeBound(ii),~,~] = mode(clssEst); % do not consider UC
    end; clearvars clss;
     
    clssIdx  = [1 3:6];

    clssPair = nchoosek(clssIdx,2);
    nPair    = size(clssPair,1);
    clss     = cell(nPair,nBound);
    for qq=1:nPair
        idxV = or(([DataOrg(:).vehicle] == clssPair(qq,1)), ...
                  ([DataOrg(:).vehicle] == clssPair(qq,2)));  
        for rr = 1: nBound
            idxB    = ([DataOrg(:).bound] == rr);
            idx     = idxV & idxB;
            
            clss{qq,rr} = GentleBoost(featureMat(:,idx)', ...
                [DataOrg(idx).vehicle], nRounds, flag_comp);
        end
    end
    
    nCase   	= nRun* nBound* nVehicle;           
    idTs        = zeros(nCase,nSensor);  % testing sample
    
    clssTs      = zeros(nCase,1);  % true class of testing data
    boundTs     = zeros(nCase,1);  % true class of testing data
    
    count    = 1;
    for ii=1:nBound
        for jj=1:nVehicle
            for kk=1:nRun
                idx1 = ([DataOrg(:).bound] == ii);
                idx2 = ([DataOrg(:).vehicle] == jj);
                idx3 = ([DataOrg(:).idRun] == kk);                
                
                idTs(count,:)   = find(idx1 & idx2 & idx3);
                clssTs(count)   = jj;
                boundTs(count)  = ii;
                count = count + 1;
            end
        end
    end
    
    confMat  = zeros(nVehicle,nVehicle+1);
    for ii=1:nCase
        clssEst = [];
        for qq=1:nPair
            clssEst = [clssEst; ...
                PredictGentleBoost(clss{qq,closeBound(boundTs(ii))}, ...
                featureMat(:,idTs(ii,:))')];
        end
        
         % 24 is UC threshold, which is (nVehicle-1)*nSensor*0.75
        [a,~,c] = mode(clssEst);
        if (length(c{1}) == 1) && (sum(clssEst==a) >= 24)
            confMat(clssTs(ii),a) = confMat(clssTs(ii),a) + 1;
        else
            confMat(clssTs(ii),nVehicle+1) = ...
                confMat(clssTs(ii),nVehicle+1)+ 1;
        end
    end
    save(fullfile(folderPrc,'ConfMat_Table3.mat'),'confMat');
end
