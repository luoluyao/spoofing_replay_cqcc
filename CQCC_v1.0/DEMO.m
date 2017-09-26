clear

%% ADD CQT TOOLBOX TO THE PATH
addpath('CQT_toolbox_2013');

%% INPUT SIGNAL
[x,fs] = audioread('D18_1000001.wav'); % from ASVspoof2015 database

%% PARAMETERS
B = 96;
fmax = fs/2;
fmin = fmax/2^9;
d = 16;
cf = 19;
ZsdD = 'ZD';

%% COMPUTE CQCC FEATURES
[CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
    cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD);

