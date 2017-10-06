clear; close all; clc;
%% добавляем директории с тулкитами
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath('MSR Identity Toolkit v1.0/code');
addpath('voicebox');
%% PATH's для файлов с метками

dir_data = 'D:\Data';
feature = 'cqcc_and_4_feature';% какие фичи хотим использовать
directory_features = fullfile('D:', 'Filin', 'spoofing_replay_cqcc', 'extracted_features',feature);
%% создаем директорию для фич
mkdir(directory_features)

%% чтение файлов с фичами
pathToDatabase = fullfile(dir_data,'ASVspoof2017_train_dev','wav');
trainProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_train.trn');
devProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_dev.trl');
evaProtocolFile = fullfile(dir_data,'ASVspoof2017_eval', 'protocol', 'KEY_ASVspoof2017_eval_V2.trl');

genuine_cqcc_train = load(fullfile(directory_features, 'genuineFeatureCellTrain.mat'));
genuine_cqcc_dev = load(fullfile(directory_features, 'genuineFeatureCellDev.mat'));
spoof_cqcc_dev = load(fullfile(directory_features, 'spoofFeatureCellDev.mat'));
spoof_cqcc_train = load(fullfile(directory_features, 'spoofFeatureCellTrain.mat'));
evaluationFeature = load(fullfile(directory_features, 'evaluationFeature.mat'));

%% подготовка загруженных данных (убираем вложенный фрагмент в cell)
genuine_cqcc_train = genuine_cqcc_train.genuineFeatureCellTrain;
spoof_cqcc_train = spoof_cqcc_train.spoofFeatureCellTrain;
genuine_cqcc_dev = genuine_cqcc_dev.genuineFeatureCellDev;
spoof_cqcc_dev = spoof_cqcc_dev.spoofFeatureCellDev;
evaluationFeature = evaluationFeature.evaluationFeature;

%% объединяем train и development
training_genuine = cat(1, genuine_cqcc_train, genuine_cqcc_dev);
training_spoof = cat(1, spoof_cqcc_train, spoof_cqcc_dev);
all_data = cat(1, training_genuine, training_spoof);

%% Чистим ненужное

clear genuine_cqcc_train;
clear spoof_cqcc_train;
clear genuine_cqcc_dev;
clear spoof_cqcc_dev;

%%  UBM-модель

nmix = 512; % number of Gaussian components
final_niter = 10;
ds_factor = 1;
nWorkers = 12; % max allowed in the toolkit

ubm = gmm_em(all_data(:),nmix, final_niter, ds_factor,nWorkers);

save (fullfile(directory_features,'ubm.mat'), 'ubm');

%%
ubm = load (fullfile(directory_features,'ubm.mat'));
ubm = ubm.ubm;

%% Вычислим статистики, необходимые для iVector  model. 
for i = 1:size(all_data,1)
    [N,F] = compute_bw_stats(all_data{i,1}, ubm);
    ubm_stats{i} = [N;F];
end

% GENUINE
for i = 1:size(training_genuine,1)
    [N,F] = compute_bw_stats(training_genuine{i,1}, ubm);
    genuine_stats{i} = [N;F];
end

% SPOOF
for i = 1:size(training_spoof,1)
    [N,F] = compute_bw_stats(training_spoof{i,1}, ubm);
    spoof_stats{i} = [N;F];
end

save(fullfile(directory_features,'ubm_stats.mat'), 'ubm_stats');
save(fullfile(directory_features,'genuine_stats.mat'), 'genuine_stats');
save(fullfile(directory_features,'spoof_stats.mat'), 'spoof_stats');

%%

load(fullfile(directory_features,'ubm_stats.mat'));
load(fullfile(directory_features,'genuine_stats.mat'));
load(fullfile(directory_features,'spoof_stats.mat'));


%% Считаем Т (Total variability subspace)
tvDim = 100;
niter = 5;
nWorkers = 12;

disp('T!');
T_ubm = train_tv_space(ubm_stats(:), ubm, tvDim, niter, nWorkers);

save(strcat(directory_features, 'T_ubm.mat'), 'T_ubm')

%%

load(fullfile(directory_features, 'T_ubm.mat'));

% T_ubm = load(strcat(directory_features, 'T_ubm.mat'))
% ubm_stats = load(strcat(directory_features, 'ubm_stats.mat'))
% T_genuine = load(strcat(directory_features, 'T_genuine.mat'))
% T_spoof = load(strcat(directory_features, 'T_spoof.mat'))
% genuine_stats = load(strcat(directory_features, 'genuine_stats.mat'))
% spoof_stats = load(strcat(directory_features, 'spoof_stats.mat'))

%T_ubm = T_ubm.T_ubm;
%ubm_stats = ubm_stats.ubm_stats;
%%
% T_genuine = T_genuine.T_genuine;
% T_spoof = T_spoof.T_spoof;
% genuine_stats = genuine_stats.genuine_stats;
% spoof_stats = spoof_stats.spoof_stats;
%% Считаем i-vector для тренировочных данных

% genuine

tvDim = 100;
niter = 5;
nWorkers = 12;

training_genuine_i_vectors = zeros(tvDim, size(training_genuine,1));
parfor i = 1:size(training_genuine,1)
       training_genuine_i_vectors(:,i) = extract_ivector(genuine_stats{i}, ubm, T_ubm);
end

save(fullfile(directory_features, 'training_genuine_i_vectors.mat'), 'training_genuine_i_vectors');
%%
% training_genuine_i_vectors = load(strcat(directory_features, 'training_genuine_i_vectors.mat'));
% training_genuine_i_vectors = training_genuine_i_vectors.training_genuine_i_vectors;


%%
training_spoof_i_vectors = zeros(tvDim, size(training_spoof,1));
parfor i = 1:size(training_spoof,1)
       training_spoof_i_vectors(:,i) = extract_ivector(spoof_stats{i}, ubm, T_ubm);
end

save(fullfile(directory_features, 'training_spoof_i_vectors.mat'), 'training_spoof_i_vectors');
%%
% training_spoof_i_vectors = load(strcat(directory_features, 'training_spoof_i_vectors.mat'));
% training_spoof_i_vectors = training_spoof_i_vectors.training_spoof_i_vectors;

%%

evaluation_i_vectors_ubm = zeros(tvDim, size(all_data,1));

parfor i = 1:size(evaluationFeature,1)
    [N,F] = compute_bw_stats(evaluationFeature{i,1},ubm);
    evaluation_i_vectors_ubm(:,i) = extract_ivector([N;F], ubm, T_ubm);
end

save(fullfile(directory_features, 'evaluation_i_vectors_ubm.mat'), 'evaluation_i_vectors_ubm')

%%
% evaluation_i_vectors_ubm = load(strcat(directory_features, 'evaluation_i_vectors_ubm.mat'));
% evaluation_i_vectors_ubm = evaluation_i_vectors_ubm.evaluation_i_vectors_ubm;
%% Чтение меток Training

fileID_train = fopen(trainProtocolFile);
protocol_train = textscan(fileID_train, '%s%s%s%s%s%s%s');
filenames_train = protocol_train{1};
labels_train = protocol_train{2};
fclose(fileID_train);

genuineIdx_train = find(strcmp(labels_train,'genuine'));
spoofIdx_train = find(strcmp(labels_train,'spoof'));

%% Чтение меток Developmenp

fileID_dev = fopen(devProtocolFile);
protocol_dev = textscan(fileID_dev, '%s%s%s%s%s%s%s');
filenames_dev = protocol_dev{1};
labels_dev = protocol_dev{2};
fclose(fileID_dev);
genuineIdx_dev = find(strcmp(labels_dev,'genuine'));
spoofIdx_dev = find(strcmp(labels_dev,'spoof'));

%% Работа с индексами
train_labels_genuine = cat(1, labels_train(1:length(genuineIdx_train)),  labels_dev(1:length(genuineIdx_dev)))
train_labels_spoof = cat(1,labels_train(length(genuineIdx_train)+1:length(labels_train)),labels_dev(length(genuineIdx_dev)+1:length(labels_dev)))
%%
ubm_data = cat(2, training_genuine_i_vectors,training_spoof_i_vectors);
ubm_labels = cat(1, train_labels_genuine, train_labels_spoof);

%%
niter = 10;
nphi = 90;
plda_ubm = gplda_em(ubm_data, ubm_labels, nphi, niter);

%%
save(strcat(directory_features, 'plda_ubm.mat'), 'plda_ubm')

%%
%plda_ubm = load(strcat(directory_features, 'plda_ubm.mat'))

%%
scores_genuine = score_gplda_trials(plda_ubm, training_genuine_i_vectors, evaluation_i_vectors_ubm);
scores_spoof = score_gplda_trials(plda_ubm, training_spoof_i_vectors, evaluation_i_vectors_ubm);

%%

% scores_genuine = load(strcat(directory_features, 'scores_genuine.mat'))
% scores_genuine = scores_genuine.scores_genuine;
% scores_spoof = load(strcat(directory_features, 'scores_spoof.mat'))
% scores_spoof=scores_spoof.scores_spoof;

%%

new_scores_genuine = mean(scores_genuine,1)
new_scores_spoof = mean(scores_spoof,1)
scores = new_scores_genuine - new_scores_spoof;

%%
save(strcat(directory_features, 'scores_genuine.mat'), 'scores_genuine')
save(strcat(directory_features, 'scores_spoof.mat'), 'scores_spoof')

%%
% чтение меток evaluation 
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s');
filenames = protocol{1};
labels = protocol{2};
fclose(fileID);
%% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
[pm,pf] = compute_roc(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

%% НА ВСЯКИЙ СЛУЧАЙ ПОСЛЕДИТЬ ЗА НАЗВАНИЕМ

% save(strcat(directory_features, 'scores_genuine_cqcc_i_vector_100_gplda_90.mat'), 'scores_genuine');
% save(strcat(directory_features, 'scores_spoof_cqcc_i_vector_100_gplda_90.mat'), 'scores_spoof');
%%
save(strcat(directory_features, 'score.mat'), 'score')