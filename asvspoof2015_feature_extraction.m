clear; close all; clc;

addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('voicebox'));

%%  TODO: Поменять пути к базе и протоколам (На машине на диске D\data\asvspoof2015)
% Также предварительно посмотреть, чсоздана ли директория asvspoof2015_features
% Туда будем писать все наши новые фичи
% Если нет, то mkdir('asvspoof2015_features')

pathToDatabase = fullfile('F:\Science\Antispoofing Datasets\ASVSpoof2015\wav');

trainProtocolFile = fullfile('F:\Science\Antispoofing Datasets\','ASVSpoof2015', 'protocol', 'CM_protocol','cm_train.trn');
devProtocolFile = fullfile('F:\Science\Antispoofing Datasets\','ASVSpoof2015', 'protocol', 'CM_protocol','cm_develop.ndx');
evaProtocolFile = fullfile('F:\Science\Antispoofing Datasets\','ASVSpoof2015', 'protocol', 'CM_protocol','cm_evaluation.ndx');

%% Параметры MFCC

frame_length = 0.02; %20ms
frame_hop = 0.01; %10ms
n_MFCC = 13; %number of cepstral coefficients excluding 0'th coefficient [default 19]
delta_feature = '0'; % 0'th coefficient. Append any of the following for more options 
                     %'d': for single delta; 'D': for double delta; 'E': log energy
%% Читаем трейн. Получаем имена файлов и метки  
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

directory_name = protocol{1};
names = protocol{2};
labels = protocol{4};
filelist = strcat(directory_name, '/', names);

genuineIdx = find(strcmp(labels,'human'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Извлечение признаков для трейна

disp('Extracting features for GENUINE training data...');
genuineCqccTrain = cell(size(genuineIdx));
genuineMfccTrain = cell(size(genuineIdx));

parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,filelist{genuineIdx(i)});
    [x,fs] = audioread(filePath);
    genuineCqccTrain{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    genuineMfccTrain{i} = melcepst(x,fs, n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, 'dD');
end
disp('Done!');

save ('./asvspoof2015_features/genuineCqccTrain.mat', 'genuineCqccTrain');
save ('./asvspoof2015_features/genuineMfccTrain.mat', 'genuineMfccTrain');

%% Извлечение признаков для Spoof
disp('Extracting features for SPOOF training data...');
spoofCqccTrain = cell(size(spoofIdx));
spoofMfccTrain = cell(size(spoofIdx));

parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase, filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofCqccTrain{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    spoofMfccTrain{i} = melcepst(x,fs, n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, 'dD');

end
disp('Done!');

save ('./asvspoof2015_features/spoofCqccTrain.mat', 'spoofCqccTrain');
save ('./asvspoof2015_features/spoofMfccTrain.mat', 'spoofMfccTrain');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEVELOPMENT SET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

directory_name = protocol{1};
names = protocol{2};
labels = protocol{4};
filelist = strcat(directory_name, '/', names);

genuineIdx = find(strcmp(labels,'genuine'));
spoofIdx = find(strcmp(labels,'spoof'));

%% Извлечение development данных

disp('Extracting features for GENUINE development data...');
genuineCqccDev = cell(size(genuineIdx));
genuineMfccDev = cell(size(genuineIdx));

parfor i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,filelist{genuineIdx(i)}); 
    [x,fs] = audioread(filePath);
    genuineCqccDev{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    genuineMfccDev{i} = melcepst(x,fs, n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, 'dD');
end

disp('Done!');

save ('./asvspoof2015_features/genuineCqccDev.mat', 'genuineCqccDev');
save ('./asvspoof2015_features/genuineMfccDev.mat', 'genuineMfccDev');

%% Извлечение признаков для Spoof Dev
disp('Extracting features for SPOOF development data...');
spoofCqccDev = cell(size(spoofIdx));
spoofMfccDev = cell(size(spoofIdx));

parfor i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,filelist{spoofIdx(i)});
    [x,fs] = audioread(filePath);
    spoofCqccDev{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    spoofMfccDev{i} = melcepst(x,fs, n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, 'dD');
end
disp('Done!');

save ('./asvspoof2015_features/spoofCqccDev.mat', 'spoofCqccDev');
save ('./asvspoof2015_features/spoofMfccDev.mat', 'spoofMfccDev');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EVALUATION 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

directory_name = protocol{1};
names = protocol{2};
labels = protocol{4};

filelist = strcat(directory_name, '/', names);

%% Извлечение признаков

disp('Extracting features for evaluation data...');
evaluationCqcc = cell(size(genuineIdx));
evaluationMFCC = cell(size(genuineIdx));

parfor i=1:length(filelist)
    filePath = fullfile(pathToDatabase,filelist{i});
    [x,fs] = audioread(filePath);
    evaluationCqcc{i} = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    evaluationMFCC{i} = melcepst(x,fs, n_MFCC, floor(3*log(fs)) ,frame_length * fs, frame_hop * fs, 'dD');

end
disp('Done!');

save ('./asvspoof2015_features/evaluationCqcc.mat', 'evaluationCqcc');
save ('./asvspoof2015_features/evaluationMFCC.mat', 'evaluationMFCC');