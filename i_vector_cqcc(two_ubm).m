clear; close all; clc;
%%
model = 'GMM';
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
feature = 'cqcc_i_vec/';
% чтение файлов с фичами
directory_features = fullfile('./extracted_features/' ,feature);

%%
% какие фичи хотим использовать
feature = 'cqcc_features/';
% чтение файлов с фичами
directory_features = fullfile('./extracted_features/' ,feature);

genuine_cqcc_train = load(strcat(directory_features, 'genuineFeatureCellTrain.mat'));
genuine_cqcc_dev = load(strcat(directory_features, 'genuineFeatureCellDev.mat'));
spoof_cqcc_dev = load(strcat(directory_features, 'spoofFeatureCellDev.mat'));
spoof_cqcc_train = load(strcat(directory_features, 'spoofFeatureCellTrain.mat'));
evaluationFeature = load(strcat(directory_features, 'evaluationFeature.mat'));

%% подготовка загруженных данных (убираем вложенный фрагмент в cell)
genuine_cqcc_train = genuine_cqcc_train.genuineFeatureCellTrain;
spoof_cqcc_train = spoof_cqcc_train.spoofFeatureCellTrain;

genuine_cqcc_dev = genuine_cqcc_dev.genuineFeatureCellDev;
spoof_cqcc_dev = spoof_cqcc_dev.spoofFeatureCellDev;

evaluationFeature = evaluationFeature.evaluationFeature;
%% объединяем train и development
training_genuine = cat(1, genuine_cqcc_train, genuine_cqcc_dev);
training_spoof = cat(1, spoof_cqcc_train, spoof_cqcc_dev);

%% загрузка UBM-моделей 

ubm_genuine = load(fullfile('./training_models/','cqcc_features', 'genuineGMM.mat'), 'genuineGMM');
ubm_genuine = ubm_genuine.genuineGMM;

ubm_spoof = load(fullfile('./training_models/','cqcc_features', 'spoofGMM.mat'), 'spoofGMM');
ubm_spoof = ubm_spoof.spoofGMM;

% т.к. функция compute_bw_stats работает с другими названиями полей -
% добавляем такие в нашу структуру. Причина - формат модели. возвращаемый
% функцией gmm_em - под нее написана библиотека. мы тренировались функцией
% gmm.

ubm_genuine.mu = ubm_genuine.m;
ubm_genuine.sigma = ubm_genuine.s;
ubm_genuine.w = ubm_genuine.w';

ubm_spoof.mu = ubm_spoof.m;
ubm_spoof.sigma = ubm_spoof.s;
ubm_spoof.w = ubm_spoof.w';

%% Вычислим статистики, необходимые для iVector  model. 
% Стоит отметить, что в коде по обучению i-vectors ребята использовали 10
% итераций для тренировки gmm_em модели. Базовая модель asvspoof
% предлагает 100 итераций. В данной реализации - используем 100 итер. При
% обучении ругался на sparsity matrix с 56 итер


% GENUINE
for i = 1:size(training_genuine,1)
    [N,F] = compute_bw_stats(training_genuine{i,1}, ubm_genuine);
    genuine_stats{i} = [N;F];
end
%%
% SPOOF
for i = 1:size(training_spoof,1)
    [N,F] = compute_bw_stats(training_spoof{i,1}, ubm_spoof);
    spoof_stats{i} = [N;F];
end

%% Считаем Т (Total variability subspace)
tvDim = 100;
niter = 5;
nWorkers = 12; % максимальное чисто параллельных потоков

disp('T_genuine!');
T_genuine = train_tv_space(genuine_stats(:), ubm_genuine, tvDim, niter, nWorkers);
%%
disp('T_spoof!');
T_spoof = train_tv_space(spoof_stats(:), ubm_spoof, tvDim, niter, nWorkers);
%%
try
    save(strcat(directory_features, 'T_genuine.mat'), 'T_genuine')
    save(strcat(directory_features, 'T_spoof.mat'), 'T_spoof')
    save(strcat(directory_features, 'genuine_stats.mat'), 'genuine_stats')
    save(strcat(directory_features, 'spoof_stats.mat'), 'spoof_stats')
catch
    mkdir(strcat(directory_features));
    save(strcat(directory_features, 'T_genuine.mat'), 'T_genuine')
    save(strcat(directory_features, 'T_spoof.mat'), 'T_spoof')
    save(strcat(directory_features, 'genuine_stats.mat'), 'genuine_stats')
    save(strcat(directory_features, 'spoof_stats.mat'), 'spoof_stats')
end

%% Считаем i-vector для тренировочных данных
% genuine
training_genuine_i_vectors = zeros(tvDim, size(training_genuine,1));
for i = 1:size(training_genuine,1)
       training_genuine_i_vectors(:,i) = extract_ivector(genuine_stats{i}, ubm_genuine, T_genuine);
end

save(strcat(directory_features, 'training_genuine_i_vectors.mat'), 'training_genuine_i_vectors')
%%
training_spoof_i_vectors = zeros(tvDim, size(training_spoof,1));
parfor i = 1:size(training_spoof,1)
       training_spoof_i_vectors(:,i) = extract_ivector(spoof_stats{i}, ubm_spoof, T_spoof);
end

save(strcat(directory_features, 'training_spoof_i_vectors.mat'), 'training_spoof_i_vectors')


