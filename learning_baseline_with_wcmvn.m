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

%%
% какие фичи хотим использовать
features = 'cqcc_features';
% чтение файлов с фичами
df = fullfile('extracted_features', features);

directory_features = fullfile('extracted_features', 'cqcc_features_wcmvn_two_ubm');

genuine_cqcc_train = load(fullfile(df, 'genuineFeatureCellTrain.mat'));
genuine_cqcc_dev   = load(fullfile(df, 'genuineFeatureCellDev.mat'));
spoof_cqcc_dev     = load(fullfile(df, 'spoofFeatureCellDev.mat'));
spoof_cqcc_train   = load(fullfile(df, 'spoofFeatureCellTrain.mat'));
evaluationFeature  = load(fullfile(df, 'evaluationFeature.mat'));

%%
% подготовка загруденных данных (убираем вложенный фрагмент в cell)
genuine_cqcc_train = genuine_cqcc_train.genuineFeatureCellTrain;
spoof_cqcc_train = spoof_cqcc_train.spoofFeatureCellTrain;

genuine_cqcc_dev = genuine_cqcc_dev.genuineFeatureCellDev;
spoof_cqcc_dev = spoof_cqcc_dev.spoofFeatureCellDev;

evaluation = evaluationFeature.evaluationFeature;

%% объединяем train и development
training_genuine = cat(1, genuine_cqcc_train, genuine_cqcc_dev);
training_spoof = cat(1, spoof_cqcc_train, spoof_cqcc_dev);

%% Чистимся

clear genuine_cqcc_train;
clear genuine_cqcc_dev;
clear spoof_cqcc_train;
clear spoof_cqcc_dev;
clear evaluationFeature;

%% Предобработка (wcmvn, mean only)

parfor i = 1:length(training_genuine)
    training_genuine_wcmvn{i,1} = wcmvn(training_genuine{i}, 301, false);
end

parfor i = 1:length(training_spoof)
    training_spoof_wcmvn{i,1} = wcmvn(training_spoof{i}, 301, false);
end

parfor i = 1:length(evaluation)
    evaluation_wcmvn{i,1} = wcmvn(evaluation{i}, 301, false);
end

save(fullfile(directory_features, 'training_genuine_wcmvn.mat'), 'training_genuine_wcmvn');
save(fullfile(directory_features, 'training_spoof_wcmvn.mat'), 'training_spoof_wcmvn');
save(fullfile(directory_features, 'evaluation_wcmvn.mat'), 'evaluation_wcmvn', '-v7.3');

%%

load(fullfile(directory_features, 'training_genuine_wcmvn.mat'));
load(fullfile(directory_features, 'training_spoof_wcmvn.mat'));
load(fullfile(directory_features, 'evaluation_wcmvn.mat'));

%% Чистимся

clear training_genuine;
clear training_spoof;
clear evaluation;

%% обучение GMM-модели 

% обучение GMM для GENUINE 
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([training_genuine_wcmvn{:}], 512, 'verbose', 'MaxNumIterations', 100);

% try
%     save(fullfile('./training_models/',feature, 'genuineGMM.mat'), 'genuineGMM')
% catch
%     mkdir(fullfile('./training_models/', feature));
%     save(fullfile('./training_models/',feature, 'genuineGMM.mat'), 'genuineGMM')
% end

disp('Done!');

% обучение GMM для SPOOF 
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([training_spoof_wcmvn{:}], 512, 'verbose', 'MaxNumIterations',100);

% try
%     save(fullfile('./training_models/',feature, 'spoofGMM.mat'), 'spoofGMM')
% catch
%     mkdir(fullfile('./training_models/', feature));
%     save(fullfile('./training_models/',feature, 'spoofGMM.mat'), 'spoofGMM')
% end

disp('Done!');

%%%%%%%%%%%%%%%%%%%%%%%% EVALUATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' Computing scores for evaluation trials...');

% чтение меток evaluation 
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
filenames = protocol{1};
labels = protocol{2};
fclose(fileID);

scores = zeros(size(filenames));

% скоры для evaluation

for i=1:length(filenames)
    llk_genuine = mean(compute_llk(evaluation_wcmvn{i},genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(evaluation_wcmvn{i},spoofGMM.m,spoofGMM.s,spoofGMM.w));
    scores(i) = llk_genuine - llk_spoof;
end
% %%
% try
%     save(fullfile('./scores/',feature, 'scores.mat'), 'scores')
% catch
%     mkdir(fullfile('./scores/',feature));
%     save(fullfile('./scores/',feature, 'scores.mat'), 'scores')
% end
% %%
% scores = load(fullfile('./scores/',feature, 'scores.mat'));
% scores = scores.scores;

% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
[pm,pf] = compute_roc(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);


% %% plots
% plot(Pfa,Pmiss,'r-^',pf,pm,'g');
% title (strcat('ROCCH and ROC plot for  ', model, ' model'));
% axis('square');grid;legend('ROCCH','ROC');
% xlabel('false alarm probability (%)') % x-axis label
% ylabel('miss probability (%)') % y-axis label
% %%
% plot_det_curve(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')), model)
%%

save(fullfile(directory_features, 'genuineGMM.mat'), 'genuineGMM');
save(fullfile(directory_features, 'spoofGMM.mat'), 'spoofGMM');
save(fullfile(directory_features, 'scores_gmm_wcmvn.mat'), 'scores');

