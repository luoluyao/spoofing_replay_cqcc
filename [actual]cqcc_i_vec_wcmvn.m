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

% какие фичи хотим использовать
feature = 'cqcc_features';
% чтение файлов с фичами
directory_features = fullfile('extracted_features',feature);
output_directory = fullfile('extracted_features','cqcc_i_vec_wcmvn_1_90');

%% чтение файлов с фичами

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

% объединяем train и development
training_genuine = cat(1, genuine_cqcc_train, genuine_cqcc_dev);
training_spoof = cat(1, spoof_cqcc_train, spoof_cqcc_dev);

%% Чистимся

clear genuine_cqcc_train;
clear spoof_cqcc_train;
clear genuine_cqcc_dev;
clear spoof_cqcc_dev;

% %% Выкидываем лишнее
% 
% for i = 1:length(training_genuine)
%     training_genuine{i} = training_genuine{i}(2:90, :);
% end
% 
% for i = 1:length(training_spoof)
%     training_spoof{i} = training_spoof{i}(2:90, :);
% end
% 
% for i = 1:length(evaluationFeature)
%     evaluationFeature{i} = evaluationFeature{i}(2:90, :);
% end

%% Предобработка (wcmvn, mean only)

parfor i = 1:length(training_genuine)
    training_genuine_wcmvn{i,1} = wcmvn(training_genuine{i}, 301, false);
end

parfor i = 1:length(training_spoof)
    training_spoof_wcmvn{i,1} = wcmvn(training_spoof{i}, 301, false);
end

parfor i = 1:length(evaluationFeature)
    evaluation_wcmvn{i,1} = wcmvn(evaluationFeature{i}, 301, false);
end

%%
save(fullfile(output_directory, 'training_genuine_wcmvn.mat'), 'training_genuine_wcmvn');
save(fullfile(output_directory, 'training_spoof_wcmvn.mat'), 'training_spoof_wcmvn');
save(fullfile(output_directory, 'evaluation_wcmvn.mat'), 'evaluation_wcmvn', '-v7.3');

%% Чистим лишнее

clear training_genuine;
clear training_spoof;
clear evaluationFeature;

%%

load(fullfile(output_directory, 'training_genuine_wcmvn.mat'));
load(fullfile(output_directory, 'training_spoof_wcmvn.mat'));
load(fullfile(output_directory, 'evaluation_wcmvn.mat'));

%%  UBM-модель

nmix = 512; % number of Gaussian components
final_niter = 10;
ds_factor = 1;
nWorkers = 12; % max allowed in the toolkit

all_data = cat(1, training_genuine_wcmvn, training_spoof_wcmvn);

ubm = gmm_em(all_data(:),nmix, final_niter, ds_factor, nWorkers);

save(fullfile(output_directory, 'ubm.mat'), 'ubm');

%%
load(fullfile(output_directory, 'ubm.mat'));

%% Вычислим статистики, необходимые для iVector  model. 

disp('BW stats beginning');

parfor i = 1:size(all_data,1)
    [N,F] = compute_bw_stats(all_data{i,1}, ubm);
    ubm_stats{i} = [N;F];
end

disp('All OK!');

% GENUINE
parfor i = 1:size(training_genuine_wcmvn,1)
    [N,F] = compute_bw_stats(training_genuine_wcmvn{i,1}, ubm);
    genuine_stats{i} = [N;F];
end

disp('Genuine OK!')

% SPOOF
parfor i = 1:size(training_spoof_wcmvn,1)
    [N,F] = compute_bw_stats(training_spoof_wcmvn{i,1}, ubm);
    spoof_stats{i} = [N;F];
end

disp('Spoofed OK!')

save(fullfile(output_directory, 'ubm_stats.mat'), 'ubm_stats');
save(fullfile(output_directory, 'genuine_stats.mat'), 'genuine_stats');
save(fullfile(output_directory, 'spoof_stats.mat'), 'spoof_stats');

%%

load(fullfile(output_directory, 'ubm_stats.mat'));
load(fullfile(output_directory, 'genuine_stats.mat'));
load(fullfile(output_directory, 'spoof_stats.mat'));

%% Считаем Т (Total variability subspace)

tvDim = 100;
niter = 5;

T_ubm = train_tv_space(ubm_stats(:), ubm, tvDim, niter, nWorkers);

save(fullfile(output_directory, 'T_ubm.mat'), 'T_ubm');

%%

load(fullfile(output_directory, 'T_ubm.mat'));

%% Считаем i-vector для тренировочных данных

% genuine
training_genuine_i_vectors = zeros(tvDim, size(training_genuine_wcmvn,1));
parfor i = 1:size(training_genuine_wcmvn,1)
       training_genuine_i_vectors(:,i) = extract_ivector(genuine_stats{i}, ubm, T_ubm);
end

training_spoof_i_vectors = zeros(tvDim, size(training_spoof_wcmvn,1));
parfor i = 1:size(training_spoof_wcmvn,1)
       training_spoof_i_vectors(:,i) = extract_ivector(spoof_stats{i},ubm, T_ubm);
end

save(fullfile(output_directory, 'training_genuine_i_vectors.mat'), 'training_genuine_i_vectors');
save(fullfile(output_directory, 'training_spoof_i_vectors.mat'), 'training_spoof_i_vectors');

%%

load(fullfile(output_directory, 'training_genuine_i_vectors.mat'));
load(fullfile(output_directory, 'training_spoof_i_vectors.mat'));

%% Считаем i-vector для тестовых данных

evaluation_i_vectors_ubm = zeros(tvDim, size(all_data,1));
parfor i = 1:size(evaluation_wcmvn,1)
    [N,F] = compute_bw_stats(evaluation_wcmvn{i,1},ubm);
    evaluation_i_vectors_ubm(:,i) = extract_ivector([N;F], ubm, T_ubm);
end

save(fullfile(output_directory, 'evaluation_i_vectors_ubm.mat'), 'evaluation_i_vectors_ubm');

%%

load(fullfile(output_directory, 'evaluation_i_vectors_ubm.mat'));

%% Подготовка данных для PLDA

% Чтение меток Training

fileID_train = fopen(trainProtocolFile);
protocol_train = textscan(fileID_train, '%s%s%s%s%s%s%s');
filenames_train = protocol_train{1};
labels_train = protocol_train{2};
fclose(fileID_train);

genuineIdx_train = find(strcmp(labels_train,'genuine'));
spoofIdx_train = find(strcmp(labels_train,'spoof'));

% Чтение меток Developmenp

fileID_dev = fopen(devProtocolFile);
protocol_dev = textscan(fileID_dev, '%s%s%s%s%s%s%s');
filenames_dev = protocol_dev{1};
labels_dev = protocol_dev{2};
fclose(fileID_dev);

genuineIdx_dev = find(strcmp(labels_dev,'genuine'));
spoofIdx_dev = find(strcmp(labels_dev,'spoof'));

% Работа с индексами

train_labels_genuine = cat(1,labels_train(1:length(genuineIdx_train)), labels_dev(1:length(genuineIdx_dev)));
train_labels_spoof   = cat(1,labels_train(length(genuineIdx_train)+1:length(labels_train)),labels_dev(length(genuineIdx_dev)+1:length(labels_dev)));

training_i_vectors = cat(2, training_genuine_i_vectors,training_spoof_i_vectors);
training_labels    = cat(1, train_labels_genuine, train_labels_spoof);

%% Считаем PLDA-модель

niter = 10;
nphi = 90;

plda_ubm = gplda_em(training_i_vectors, training_labels, nphi, niter);

save(fullfile(output_directory, 'plda_ubm.mat'), 'plda_ubm');

%%

load(fullfile(output_directory, 'plda_ubm.mat'));

%%  Считаем скоры

scores_genuine = score_gplda_trials(plda_ubm, training_genuine_i_vectors, evaluation_i_vectors_ubm);
scores_spoof = score_gplda_trials(plda_ubm, training_spoof_i_vectors, evaluation_i_vectors_ubm);

new_scores_genuine = mean(scores_genuine,1);
new_scores_spoof = mean(scores_spoof,1);

scores = new_scores_genuine - new_scores_spoof;

save(fullfile(output_directory, 'scores_genuine.mat'), 'scores_genuine');
save(fullfile(output_directory, 'scores_spoof.mat'), 'scores_spoof');
save(fullfile(output_directory, 'scores.mat'), 'scores');

%%

load(fullfile(output_directory, 'scores_genuine.mat'));
load(fullfile(output_directory, 'scores_spoof.mat'));
load(fullfile(output_directory, 'scores.mat'));

%% чтение меток evaluation 

fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s');
filenames = protocol{1};
labels = protocol{2};
fclose(fileID);

%% Вычисление EER при помощи Bosaris

prior = 0.5
[actdcf,mindcf,prbep,eer] = fastEval(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')), prior)

%% Построение DET-кривых при помощи Bosaris

plot_title = 'DET plot for i-vector CQCC model';
plot_type = Det_Plot.make_plot_window_from_string('old');
plot_obj = Det_Plot(plot_type,plot_title);
plot_obj.set_system(scores(strcmp(labels,'genuine')), scores(strcmp(labels,'spoof')),'i-vector CQCC');
plot_obj.plot_steppy_det({'k','LineWidth',2},' ');
plot_obj.plot_DR30_fa('k--','30 false alarms');
plot_obj.plot_DR30_miss('k--','30 misses');
plot_obj.plot_mindcf_point(prior,{'k*','MarkerSize',8},'mindcf');
plot_obj.display_legend();
