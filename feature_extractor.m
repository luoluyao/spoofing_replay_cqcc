clear; close all; clc;
%%
% TODO: переписать сахранение файлов в соответствующие директории
%%
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));

dir_data = 'D:\Data';
pathToDatabase = fullfile(dir_data,'ASVspoof2017_train_dev','wav');
trainProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_train.trn');
devProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_dev.trl');
evaProtocolFile = fullfile(dir_data,'ASVspoof2017_eval', 'protocol', 'KEY_ASVspoof2017_eval_V2.trl');

% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCellTrain = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    genuineFeatureCellTrain{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');
save ('./files/genuineFeatureCellTrain.mat', 'genuineFeatureCellTrain');


% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCellTrain = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'train',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofFeatureCellTrain{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');

save ('./files/spoofFeatureCellTrain.mat', 'spoofFeatureCellTrain');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% READ DEVELOPMENT SET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Feature extraction for development data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE development data...');
genuineFeatureCellDev = cell(size(genuineIdx));
parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,'dev',filelist{genuineIdx(i)}); %#ok<PFBNS>
    disp (filePath);
    [x,fs] = audioread(filePath);
    genuineFeatureCellDev{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');

save ('./files/genuineFeatureCellDev.mat', 'genuineFeatureCellDev');


% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF development data...');
spoofFeatureCellDev = cell(size(spoofIdx));
parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,'dev',filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofFeatureCellDev{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');

save ('./files/spoofFeatureCellDev.mat', 'spoofFeatureCellDev');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVALUATION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read train protocol
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for evaluation data...');
evaluationFeature = cell(size(genuineIdx));
parfor i=1:length(filelist)
    filePath = fullfile('D:\Science\Antispoofing Research (ipython notebook)\asvspoof2017\ASVspoof2017_eval\eval',filelist{i});
    [x,fs] = audioread(filePath);
    evaluationFeature{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');

save ('./files/evaluationFeature.mat', 'evaluationFeature');