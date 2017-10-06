clear; close all; clc;

%% добавляем директории с тулкитами
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath('MSR Identity Toolkit v1.0/code');
addpath('voicebox');

%% PATH's для файлов с метками
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

%% Чтение меток Training и Developmenp

fileID_train = fopen(trainProtocolFile);
protocol_train = textscan(fileID_train, '%s%s%s%s%s%s%s');
filenames_train = protocol_train{1};
labels_train = protocol_train{2};
fclose(fileID_train);

genuineIdx_train = find(strcmp(labels_train,'genuine'));
spoofIdx_train = find(strcmp(labels_train,'spoof'));

fileID_dev = fopen(devProtocolFile);
protocol_dev = textscan(fileID_dev, '%s%s%s%s%s%s%s');
filenames_dev = protocol_dev{1};
labels_dev = protocol_dev{2};
fclose(fileID_dev);

genuineIdx_dev = find(strcmp(labels_dev,'genuine'));
spoofIdx_dev = find(strcmp(labels_dev,'spoof'));

train_labels_genuine = cat(1, labels_train(1:length(genuineIdx_train)),  labels_dev(1:length(genuineIdx_dev)));
train_labels_spoof = cat(1,labels_train(length(genuineIdx_train)+1:length(labels_train)),labels_dev(length(genuineIdx_dev)+1:length(labels_dev)));

%% Обучаем GPLDA модель
ubm_data = cat(2, training_genuine_i_vectors,training_spoof_i_vectors);
ubm_labels = cat(1, train_labels_genuine, train_labels_spoof);

niter = 10;
nphi = 90;

plda_ubm = gplda_em(ubm_data, ubm_labels, nphi, niter);

save(fullfile(directory_features, 'plda_ubm.mat'),'plda_ubm');

%% Загружаем GPLDA модель

load(fullfile(directory_features, 'plda_ubm.mat'));

%% Оцениваем Scores

scores_genuine = score_gplda_trials(plda_ubm, training_genuine_i_vectors, evaluation_i_vectors);
scores_spoof = score_gplda_trials(plda_ubm, training_spoof_i_vectors, evaluation_i_vectors);

new_scores_genuine = mean(scores_genuine,1);
new_scores_spoof = mean(scores_spoof,1);

scores = new_scores_genuine - new_scores_spoof;

save(fullfile(directory_features, 'scores_genuine_mfcc_i_vector_100_gplda_90.mat'), 'scores_genuine');
save(fullfile(directory_features, 'scores_spoof_mfcc_i_vector_100_gplda_90.mat'), 'scores_spoof');
save(fullfile(directory_features, 'scores_mfcc_i_vector_100_gplda_90.mat'), 'scores');

%% Читаем метки Evaluation 

fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s');
filenames = protocol{1};
labels = protocol{2};
fclose(fileID);

%% Вычисляем EER

[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
[pm,pf] = compute_roc(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

%% Строим кривые
plot(Pfa,Pmiss,'r-^',pf,pm,'g');
title (strcat('ROCCH and ROC plot for  ', model, ' model'));
axis('square');grid;legend('ROCCH','ROC');
xlabel('false alarm probability (%)') % x-axis label
ylabel('miss probability (%)') % y-axis label
plot_det_curve(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')), model)
