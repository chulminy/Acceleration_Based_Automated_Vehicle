%% Vision-Based Automated Crack Detection for Bridge Inspection
%
%% Contact
%  Name  : Chul Min Yeum
%  Email : chulminy@gmail.com
%  Please contact me if you have a question or find a bug. You can use a
%  "issues" page in the github.

%% Description
%
% This code is used for the following paper: 
% Chul Min Yeum, Shirley J. Dyke, Ricardo E. Rovira, Christian Silva, and
% Jeff Demo."Acceleration-Based Automated Vehicle Classification on Mobile
% Bridges.” Computer-Aided Civil and Infrastructure Engineering 31
% (2016):813-825.
%
% All data and a package code are provided. I did my best to optimize the
% code in terms of speed and readability but, it is still slow to complete
% all steps. For future users for this code, please convert this code to
% C++ or other fater languages.
%
% The ouctome here is slightly different from the ones in the paper because
% Haar features were rege
% You can feel free to modify this code but I would recommend cite my
% paper.

%% Parameter setup
clear; clc; close all; format shortg; warning off;

% Gentleboost implementation
% Please test speed of the genetleboost algorithm in your machine
% see 'SpeedTestGentleBoost.m' in a folder of 'misc'
% 0 : m-file
% 1 : matlabpool (parfor) 
% 2 : mex 
% 3 : mex + openmp
% see a 'test_gentle_boost' folder

flag_comp = 3;

folderMisc = 'misc';
addpath(folderMisc);
if flag_comp==1
    poolobj = gcp('nocreate');
    if isempty(poolobj) % checking to see if my pool is already open
        parpool('local',feature('numCores')-2);
    end; clearvars poolobj;
elseif flag_comp==2
    if ~exist(fullfile(folderMisc, ['GentleBoost_mex.' mexext]),'file')
        cd(folderMisc);
        mex GentleBoost_mex.cpp
        cd('../');
    end
elseif flag_comp==3
    if  ~exist(fullfile(folderMisc, ['GentleBoost_mex_openmp.' mexext]),'file')
        cd(folderMisc);
        mex GentleBoost_mex_openmp.cpp COMPFLAGS="/openmp $COMPFLAGS" LINKFALGS="$LINKFALGS -openmp";
        cd('../');
    end
end; clearvars folderMisc;

% folder setup
folderBase  = fullfile(cd(cd('..')),'data'); 
folderAcc   = fullfile(folderBase,'acceleartion');
folderPrc   = fullfile(folderBase,'prc_data');
clearvars folderBase;

% result plot
folderOut   = fullfile(cd(cd('..')),'post');

% sampling freqeuency 
fs      = 1024;

% frequency and nfft (spectrogram resolution)
nfft    = 256; 

% time and sampling frequency (spectrogram resolution)
ntt     = 1024;

% window length
wlenL   = 32;
wL      = hann(wlenL, 'symmetric');

wlenH   = 128;
wH      = hann(wlenH, 'symmetric');

% size of spectrogram
sizeSpect.w = ntt; 
sizeSpect.h = nfft;

% # of feature winodws (for training)
nFeature    = 60000;

% Adaboost learning round
nRounds     = 100;

% cutoff frequency
cfreq_h     = 100; % high frequency (Hz)
[b_filt_h, a_filt_h] = butter(9,cfreq_h/(fs/2),'high'); clear cfreq_h;

cutoff_freq = 60; % low frequency for resampling

% sensor information
nSensor     = 8;

% vehicle information
nVehicle    = 6;
prefVehicle = {'V1','V2','V3','V4','V5','V6'};

% # of run
nRun        = 6;

% bound information
nBound      = 3; 
prefBound   = {'BG','BR','BW'};

% moving RMS parameters
windRMS     = 64;