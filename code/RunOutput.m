FeatureGeneration;

idx = 577;

%--------------------------------------------------------------------------
%% Training_Step1 ---------------------------------------------------------
figure('Name','Training_Step1');

% vehilce number
vehicleNumber    = prefVehicle{DataOrg(idx).vehicle}; 

% boundary condition
boundCondition  = prefBound{DataOrg(idx).bound}; 

% senosr number
sensorNumber    = DataOrg(idx).idSensor; 

% run number
runNumber       = DataOrg(idx).idRun; 

% acceleration signal
sigAcc          = DataOrg(idx).acc;

str = { ['Vehicle   : ' vehicleNumber],...
        ['Boundary  : ' boundCondition],...
        ['Sensor ID : ' int2str(sensorNumber)],...
        ['Run ID    : ' int2str(runNumber)]};

dim = [0.70 0.4 0.5 0.5];    
annotation('textbox',dim,'String',str,'FitBoxToText','on', ...
           'linewidth',1, 'FontSize',10,'FontWeight','bold');

plot([0:length(sigAcc)-1]./fs, sigAcc, 'b', 'linewidth',2);
xlabel('\bf Time (s)','FontSize',10, 'fontweight','bold');
ylabel('\bf Acceleration (m/s^2)','FontSize',10, 'fontweight','bold');
set(gca,'fontsize',10,'linewidth',2,'fontweight','bold');
set(gcf, 'pos',[10 10 900 250]);axis tight;
print(fullfile(folderOut,'Training_Step1'),'-djpeg','-r0')

%% Training_Step2 ---------------------------------------------------------
figure('Name','Training_Step2');

dataTmp1 = sigAcc - ...
    thresh.meanAmp(DataOrg(idx).idSensor,DataOrg(idx).bound);

% estimate a starting point
tmp1    = MovingRMS(dataTmp1, windRMS);
i1      = find(tmp1> ...
    thresh.amp(DataOrg(idx).idSensor,DataOrg(idx).bound));

% estimate an end point
dataTmp2    = filter(b_filt_h, a_filt_h, dataTmp1);
tmp1        = MovingRMS(dataTmp2, windRMS);
i2          = find(tmp1> ...
    thresh.RMS(DataOrg(idx).idSensor,DataOrg(idx).bound));

plot([0:length(dataTmp2)-1]./fs, dataTmp2, 'b', 'linewidth',2); hold on;
line([i1(1) i1(1)]./fs,[min(dataTmp2) max(dataTmp2)],'color',[1 0 0], ...
    'linestyle',':','linewidth',2); hold off;
line([i2(end) i2(end)]./fs,[min(dataTmp2) max(dataTmp2)],'color',[1 0 0], ...
    'linestyle',':','linewidth',2); hold off;
xlabel('\bf Time (s)','FontSize',10, 'fontweight','bold');
ylabel('\bf Acceleration (m/s^2)','FontSize',10, 'fontweight','bold');
set(gca,'fontsize',10,'linewidth',2,'fontweight','bold');
set(gcf, 'pos',[10 10 900 250]);axis tight;
print(fullfile(folderOut,'Training_Step2'),'-djpeg','-r0')

%% Training_Step3 ---------------------------------------------------------
figure('Name','Training_Step3');

sigCrop     = sigAcc(i1(1):i2(end));

plot([0:length(sigCrop)-1]./fs, sigCrop, 'b', 'linewidth',2); hold on;
xlabel('\bf Time (s)','FontSize',10, 'fontweight','bold');
ylabel('\bf Acceleration (m/s^2)','FontSize',10, 'fontweight','bold');
set(gca,'fontsize',10,'linewidth',2,'fontweight','bold');
set(gcf, 'pos',[10 10 900 250]);axis tight;
print(fullfile(folderOut,'Training_Step3'),'-djpeg','-r0')

%% Training_Step4 ---------------------------------------------------------
figure('Name','Training_Step4');

sigResamp   = resample(sigCrop, 1, floor(fs/(2*cutoff_freq)));

% Spectrogram
dataCropSTFTL        = abs(spectrogram([zeros(wlenL/2,1); sigResamp ; ...
    zeros(wlenL/2-1,1)], wL, wlenL-1, nfft*4));
dataCropSTFTH        = abs(spectrogram([zeros(wlenH/2,1); sigResamp ; ...
    zeros(wlenH/2-1,1)], wH, wlenH-1, nfft*4));
dataCropSTFTL        = imresize(dataCropSTFTL./max(dataCropSTFTL(:)), ...
    [nfft+1, ntt]);
dataCropSTFTH        = imresize(dataCropSTFTH./max(dataCropSTFTH(:)), ...
    [nfft+1, ntt]);

dataCropSTFT         = [dataCropSTFTL dataCropSTFTH];

imagesc(flipud(dataCropSTFT)); colorbar; colormap(jet);
xlabel('Time sample','FontSize',10, 'fontweight','bold');
ylabel('Frequency sample','FontSize',10, 'fontweight','bold');
set(gca,'fontsize',10,'linewidth',2,'fontweight','bold');
set(gcf, 'pos',[10 10 900 250]);axis tight;
print(fullfile(folderOut,'Training_Step4'),'-djpeg','-r0');

%% Spectrogram images generated from three different crossings of six vehicles under B1 

figure('Name','Spectrograms');

bIdx = ([DataOrg(:).bound] == 1);
vIdx = ([DataOrg(:).vehicle]);
rIdx = ([DataOrg(:).idRun]);
sIdx = ([DataOrg(:).idSensor] == 1);

sigIdx = zeros(3,6);
for ii=1:3
    for jj=1:6
        sigIdx(ii,jj) = find(bIdx & (vIdx == jj) & (rIdx ==ii) & sIdx);
    end
end

imgBase = [];
for ii=1:3
    imgTmp = [];
    for jj=1:6
        idx = sigIdx(ii,jj);
        
        dataTmp1 = DataOrg(idx).acc - ...
            thresh.meanAmp(DataOrg(idx).idSensor,DataOrg(idx).bound);
        
        dataCrop = dataTmp1(DataOrg(idx).range(1):DataOrg(idx).range(2));
        tmp = resample(dataCrop, 1, floor(fs/(2*cutoff_freq)));
        dataCropSTFTL   = ...
            spectrogram([zeros(wlenL/2,1); tmp ; zeros(wlenL/2-1,1)], ...
            wL, wlenL-1, nfft*2);
        
        dataCropSTFTL   = imresize(abs(dataCropSTFTL), [nfft+1, ntt]);
        dataCropSTFTL = dataCropSTFTL - min(dataCropSTFTL(:)); 
        dataCropSTFTL = dataCropSTFTL / max(dataCropSTFTL(:)); 
        imgTmp = cat(2,imgTmp,flipud(dataCropSTFTL));
    end
    imgBase = cat(1,imgBase,imgTmp);
end

imagesc(imgBase); colorbar; colormap(jet);
set(gca,'XTick',[512 512+1024*1 512+1024*2 512+1024*3 512+1024*4 512+1024*5])
set(gca,'XTickLabel',{'V1','V2','V3','V4','V5','V6'})
set(gca,'YTick',[128 256*1+128 256*2+128])
set(gca,'YTickLabel',{'Run1','Run2','Run4'})
set(gca,'fontsize',10,'linewidth',2,'fontweight','bold');
set(gcf, 'pos',[10 10 900 400]);axis tight;
print(fullfile(folderOut,'Spectrograms'),'-djpeg','-r0');


%% Table 1
load(fullfile(folderPrc,'ConfMat_Table1.mat'));

vClssCol = {'B1','B2','B3','UC','Accuracy'};
vClssRow = {'B1','B2','B3'};

accuracy  = zeros(3,1);
for ii=1:3
    accuracy(ii) = [confMat(ii,ii)./sum(confMat(ii,:))];
end

T = table(confMat(:,1),...
        confMat(:,2),...
        confMat(:,3),...
        confMat(:,4),accuracy, 'RowNames',vClssRow, ...
        'VariableNames',vClssCol);

% Get the table in string form.
TString = evalc('disp(T)');

% Use TeX Markup for bold formatting and underscores.
TString = strrep(TString,'<strong>','\bf');
TString = strrep(TString,'</strong>','\rm');
TString = strrep(TString,'_','\_');

% Get a fixed-width font.
FixedWidth = get(0,'FixedWidthFontName');

figure('Name','Table1');
% Output the table using the annotation command.
annotation(gcf,'Textbox','String',TString,'Interpreter','Tex', ...
    'FontName',FixedWidth,'Units','Normalized', ...
    'FontSize',14,'Position',[0 0 1 1],'EdgeColor','white');
set(gcf, 'pos',[10 10 550 180]);
print(fullfile(folderOut,'Table1'),'-djpeg','-r0');

%% Table 2
load(fullfile(folderPrc,'ConfMat_Table2.mat'));

% Set up an example table.
vClssRow = {'V1','V3','V4','V5','V6','V2'};
vClssCol = {'V1','V3','V4','V5','V6','UC','Accuracy'};


confMat(end+1,:) = confMat(2,:); confMat(2,:) = [];
confMat(:,2) = [];
accuracy  = zeros(6,1);

for ii=1:6
    if ii==6
        accuracy(ii) = [confMat(ii,1)./sum(confMat(ii,:))];
    else
        accuracy(ii) = [confMat(ii,ii)./sum(confMat(ii,:))];
    end
end

T = table(confMat(:,1),...
        confMat(:,2),...
        confMat(:,3),...
        confMat(:,4),...
        confMat(:,5),...
        confMat(:,6),accuracy, 'RowNames',vClssRow, ...
        'VariableNames',vClssCol);

% Get the table in string form.
TString = evalc('disp(T)');

% Use TeX Markup for bold formatting and underscores.
TString = strrep(TString,'<strong>','\bf');
TString = strrep(TString,'</strong>','\rm');
TString = strrep(TString,'_','\_');

% Get a fixed-width font.
FixedWidth = get(0,'FixedWidthFontName');

figure('Name','Table2');
% Output the table using the annotation command.
annotation(gcf,'Textbox','String',TString,'Interpreter','Tex', ...
    'FontName',FixedWidth,'Units','Normalized', ...
    'FontSize',14,'Position',[0 0 1 1],'EdgeColor','white');
set(gcf, 'pos',[10 10 700 250]);
print(fullfile(folderOut,'Table2'),'-djpeg','-r0');

%% Table 3
load(fullfile(folderPrc,'ConfMat_Table3.mat'));

% Set up an example table.
vClssRow = {'V1','V3','V4','V5','V6','V2'};
vClssCol = {'V1','V3','V4','V5','V6','UC','Accuracy'};


confMat(end+1,:) = confMat(2,:); confMat(2,:) = [];
confMat(:,2) = [];
accuracy  = zeros(6,1);

for ii=1:6
    if ii==6
        accuracy(ii) = [confMat(ii,1)./sum(confMat(ii,:))];
    else
        accuracy(ii) = [confMat(ii,ii)./sum(confMat(ii,:))];
    end
end

T = table(confMat(:,1),...
        confMat(:,2),...
        confMat(:,3),...
        confMat(:,4),...
        confMat(:,5),...
        confMat(:,6),accuracy, 'RowNames',vClssRow, ...
        'VariableNames',vClssCol);

% Get the table in string form.
TString = evalc('disp(T)');

% Use TeX Markup for bold formatting and underscores.
TString = strrep(TString,'<strong>','\bf');
TString = strrep(TString,'</strong>','\rm');
TString = strrep(TString,'_','\_');

% Get a fixed-width font.
FixedWidth = get(0,'FixedWidthFontName');

figure('Name','Table3');
% Output the table using the annotation command.
annotation(gcf,'Textbox','String',TString,'Interpreter','Tex', ...
    'FontName',FixedWidth,'Units','Normalized', ...
    'FontSize',14,'Position',[0 0 1 1],'EdgeColor','white');
set(gcf, 'pos',[10 10 700 250]);
print(fullfile(folderOut,'Table3'),'-djpeg','-r0');