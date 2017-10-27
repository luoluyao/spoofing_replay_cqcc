clear; close all; clc;

%% добавляем директории с тулкитами
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath('MSR Identity Toolkit v1.0/code');
addpath('voicebox');

%% PATH's для файлов с метками
dir_data = 'D:\Data';
pathToDatabase = fullfile(dir_data,'ASVspoof2017_train_dev','wav');
trainProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_train.trn');
devProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_dev.trl');
evaProtocolFile = fullfile(dir_data,'ASVspoof2017_eval', 'protocol', 'KEY_ASVspoof2017_eval_V2.trl');

%% Подгружаем фичи

feature = 'mfcc_i_vec';
directory_features = fullfile('D:','Filin','spoofing_replay_cqcc','extracted_features', feature);

load(fullfile(directory_features, 'genuineMFCCTrain.mat'));
load(fullfile(directory_features, 'genuineMFCCDev.mat'));
load(fullfile(directory_features, 'spoofMFCCDev.mat'));
load(fullfile(directory_features, 'spoofMFCCTrain.mat'));
load(fullfile(directory_features, 'evaluationMFCCFeature.mat'));

%% объединяем train и development
training_genuine = cat(1, genuineMFCCTrain, genuineMFCCDev);
training_spoof = cat(1, spoofMFCCTrain, spoofMFCCDev);
all_data = cat(1, training_genuine, training_spoof);

%%

clear genuineMFCCTrain
clear genuineMFCCDev
clear spoofMFCCTrain
clear spoofMFCCDev

%%  UBM-моделЬ 

nmix = 512; % number of Gaussian components
final_niter = 10;
ds_factor = 1;
nWorkers = 12; % max allowed in the toolkit

ubm = gmm_em(all_data(:),nmix, final_niter, ds_factor, nWorkers);

save(fullfile(directory_features, 'ubm.mat'), 'ubm');

%% Загружаем UBM-моделЬ 

load (fullfile(directory_features, 'ubm.mat'));

%% Вычислим статистики, необходимые для iVector  model. 
for i = 1:size(all_data,1)
    [N,F] = compute_bw_stats(all_data{i,1}, ubm);
    ubm_stats{i} = [N;F];
end

for i = 1:size(training_genuine,1)
    [N,F] = compute_bw_stats(training_genuine{i,1}, ubm);
    genuine_stats{i} = [N;F];
end

for i = 1:size(training_spoof,1)
    [N,F] = compute_bw_stats(training_spoof{i,1}, ubm);
    spoof_stats{i} = [N;F];
end

save(fullfile(directory_features, 'ubm_stats.mat'), 'ubm_stats');
save(fullfile(directory_features, 'genuine_stats.mat'),'genuine_stats');
save(fullfile(directory_features, 'spoof_stats.mat'), 'spoof_stats');

%% Загружаем статистики

load(fullfile(directory_features, 'ubm_stats.mat'));
load(fullfile(directory_features, 'genuine_stats.mat'));
load(fullfile(directory_features, 'spoof_stats.mat'));

%% Считаем Т (Total variability subspace)
tvDim = 100;
niter = 5;

disp('T!');
T_ubm = train_tv_space(ubm_stats(:), ubm, tvDim, niter, nWorkers);

save(fullfile(directory_features, 'T_ubm.mat'),'T_ubm');

%% Загружаем Т (Total variability subspace)
load(fullfile(directory_features, 'T_ubm.mat'));

%% Считаем i-vector для тренировочных данных

training_genuine_i_vectors = zeros(tvDim, size(training_genuine,1));
parfor i = 1:size(training_genuine,1)
       training_genuine_i_vectors(:,i) = extract_ivector(genuine_stats{i}, ubm, T_ubm);
end

training_spoof_i_vectors = zeros(tvDim, size(training_spoof,1));
parfor i = 1:size(training_spoof,1)
       training_spoof_i_vectors(:,i) = extract_ivector(spoof_stats{i}, ubm, T_ubm);
end

save(fullfile(directory_features, 'training_spoof_i_vectors.mat'),'training_spoof_i_vectors');
save(fullfile(directory_features, 'training_genuine_i_vectors.mat'),'training_genuine_i_vectors');

%% Загружаем i-vector для тренировочных данных

load(fullfile(directory_features, 'training_spoof_i_vectors.mat'));
load(fullfile(directory_features, 'training_genuine_i_vectors.mat'));

%% Считаем i-vector для тестовых данных

evaluation_i_vectors = zeros(tvDim, size(evaluationFeature,1));
parfor i = 1:size(evaluationFeature,1)
    [N,F] = compute_bw_stats(evaluationFeature{i,1}, ubm);
    evaluation_i_vectors(:,i) = extract_ivector([N;F], ubm, T_ubm);
end

save(fullfile(directory_features, 'evaluation_i_vectors.mat'),'evaluation_i_vectors');

%% Загружаем i-vector для тестовых данных

load(fullfile(directory_features, 'evaluation_i_vectors.mat'));


%% Читаем Evaluation метки
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s');
filenames = protocol{1};
labels = protocol{2};
fclose(fileID);

%% Обучем SVM с Gaussian ядром

training_i_vec = [training_genuine_i_vectors'; training_spoof_i_vectors'];
training_labels = [ones(1, size(training_genuine_i_vectors, 2)), -ones(1, size(training_spoof_i_vectors, 2))];

%%  Обучем SVM с Gaussian ядром

svmModel = fitcsvm(training_i_vec, training_labels, 'KernelFunction','linear');
[~, scoresSVM] = predict(svmModel, evaluation_i_vectors_ubm');

%%
scoresSVM = -log(scoresSVM(:,1));

[Pmiss,Pfa] = rocch(scoresSVM(strcmp(labels,'genuine')),scoresSVM(strcmp(labels,'spoof')));
[pm,pf] = compute_roc(scoresSVM(strcmp(labels,'genuine')),scoresSVM(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);
