%% Чистимся

clear; close all; clc;

%% добавляем директории с тулкитами
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath('MSR Identity Toolkit v1.0/code');
addpath('voicebox');

%% PATH's для файлов с метками
dir_data = fullfile('D','Data');

pathToDatabase = fullfile(dir_data,'ASVspoof2017_train_dev','wav');
trainProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_train.trn');
devProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_dev.trl');
evaProtocolFile = fullfile(dir_data,'ASVspoof2017_eval', 'protocol', 'KEY_ASVspoof2017_eval_V2.trl');

%% Чтение меток 

fileID = fopen('D:\Data\ASVspoof2017_eval\protocol\KEY_ASVspoof2017_eval_V2.trl');
protocol = textscan(fileID, '%s%s');
filenames = protocol{1};
labels = protocol{2};
fclose(fileID);

%% Подгружаем фичи
feature = 'cqcc_features_two_ubm';
directory_features = fullfile('D:','Filin','spoofing_replay_cqcc','extracted_features', feature);

load(fullfile(directory_features, 'genuineFeatureCellTrain.mat'));
load(fullfile(directory_features, 'genuineFeatureCellDev.mat'));
load(fullfile(directory_features, 'spoofFeatureCellDev.mat'));
load(fullfile(directory_features, 'spoofFeatureCellTrain.mat'));
load(fullfile(directory_features, 'evaluationFeature.mat'));

%% объединяем train и development

training_genuine = cat(1, genuineFeatureCellTrain, genuineFeatureCellDev);
training_spoof = cat(1, spoofFeatureCellTrain, spoofFeatureCellDev);
all_data = cat(1, training_genuine, training_spoof);

%% чистимся от лишних данных

clear genuineFeatureCellTrain;
clear genuineFeatureCellDev;
clear spoofFeatureCellTrain;
clear spoofFeatureCellDev;

%% Выкидываем первую переменную из данных

for i = 1:length(all_data)
    all_data_1{i,1} = all_data{i}(2:90,:);
end

%%  UBM-моделЬ 

nmix = 512; % number of Gaussian components
final_niter = 10;
ds_factor = 1;
nWorkers = 12; % max allowed in the toolkit

ubm = gmm_em(all_data(:),nmix, final_niter, ds_factor, nWorkers);
ubm_genuine = gmm_em(training_genuine(:), nmix, final_niter, ds_factor, nWorkers);
ubm_spoof = gmm_em(training_spoof(:), nmix, final_niter, ds_factor, nWorkers);

save(fullfile(directory_features, 'ubm.mat'), 'ubm');
save(fullfile(directory_features, 'ubm_genuine.mat'), 'ubm_genuine');
save(fullfile(directory_features, 'ubm_spoof.mat'), 'ubm_spoof');

%% Сделаем адаптацию background модели ubm
map_tau = 19.0;
config = 'm';

ubm_adapted_genuine = mapAdapt(training_genuine(:), ubm, map_tau, config);
ubm_adapted_spoof   = mapAdapt(training_spoof(:), ubm, map_tau, config);

%%
save(fullfile(directory_features, 'ubm_adapted_genuine.mat'), 'ubm_adapted_genuine');
save(fullfile(directory_features, 'ubm_adapted_spoof.mat'), 'ubm_adapted_spoof');

%% Подсчет scores для неадаптированных моделей

scores_nonadapted = zeros(size(evaluationFeature));

parfor i = 1:length(evaluationFeature)
    ubm_genuine_llk = mean(compute_llk(evaluationFeature{i}, ubm_genuine.mu, ubm_genuine.sigma, ubm_genuine.w(:)));
    ubm_spoof_llk = mean(compute_llk(evaluationFeature{i}, ubm_spoof.mu, ubm_spoof.sigma, ubm_spoof.w(:)));
    scores_nonadapted(i) = ubm_genuine_llk - ubm_spoof_llk;
end

%% EER для неадаптированных моделей (EER is 24.90)

[Pmiss,Pfa] = rocch(scores_nonadapted(strcmp(labels,'genuine')),scores_nonadapted(strcmp(labels,'spoof')));
[pm,pf] = compute_roc(scores_nonadapted(strcmp(labels,'genuine')),scores_nonadapted(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

%% Подсчет scores для адаптированных моделей

scores_adapted = zeros(size(evaluationFeature));

for i = 1:length(evaluationFeature)
    ubm_genuine_llk = mean(compute_llk(evaluationFeature{i}, ubm_adapted_genuine.mu, ubm_adapted_genuine.sigma, ubm_adapted_genuine.w(:)));
    ubm_spoof_llk = mean(compute_llk(evaluationFeature{i}, ubm_adapted_spoof.mu, ubm_adapted_spoof.sigma, ubm_adapted_spoof.w(:)));
    scores_adapted(i) = ubm_genuine_llk - ubm_spoof_llk;
end

%% EER для адаптированных моделей 
% EER is 27.14 for map_tau = 10.0, config = 'mwv'
% EER is 32.00 for map_tau = 19.0, config = 'm'

[Pmiss,Pfa] = rocch(scores_adapted(strcmp(labels,'genuine')),scores_adapted(strcmp(labels,'spoof')));
[pm,pf] = compute_roc(scores_adapted(strcmp(labels,'genuine')),scores_adapted(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

%%
save(fullfile(directory_features, 'scores_nonadapted.mat'), 'scores_nonadapted');
