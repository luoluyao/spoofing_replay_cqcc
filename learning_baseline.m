clear; close all; clc;
%%
model = 'GMM';
%%
% добавляем директории с тулкитами
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));

% PATH's для файлов с метками

dir_data = 'D:\Data';
pathToDatabase = fullfile(dir_data,'ASVspoof2017_train_dev','wav');
trainProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_train.trn');
devProtocolFile = fullfile(dir_data,'ASVspoof2017_train_dev', 'protocol', 'ASVspoof2017_dev.trl');
evaProtocolFile = fullfile(dir_data,'ASVspoof2017_eval', 'protocol', 'KEY_ASVspoof2017_eval_V2.trl');
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

%%
% подготовка загруденных данных (убираем вложенный фрагмент в cell)
genuine_cqcc_train = genuine_cqcc_train.genuineFeatureCellTrain;
spoof_cqcc_train = spoof_cqcc_train.spoofFeatureCellTrain;

genuine_cqcc_dev = genuine_cqcc_dev.genuineFeatureCellDev;
spoof_cqcc_dev = spoof_cqcc_dev.spoofFeatureCellDev;

evaluationFeature = evaluationFeature.evaluationFeature;
%%
% объединяем train и development
training_genuine = cat(1, genuine_cqcc_train, genuine_cqcc_dev);
training_spoof = cat(1, spoof_cqcc_train, spoof_cqcc_dev);

%% обучение GMM-модели 

% обучение GMM для GENUINE 
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([training_genuine{:}], 512, 'verbose', 'MaxNumIterations',100);
try
    save(fullfile('./training_models/',feature, 'genuineGMM.mat'), 'genuineGMM')
catch
    mkdir(fullfile('./training_models/', feature));
    save(fullfile('./training_models/',feature, 'genuineGMM.mat'), 'genuineGMM')
end

disp('Done!');
%%
% обучение GMM для SPOOF 
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([training_spoof{:}], 512, 'verbose', 'MaxNumIterations',100);

try
    save(fullfile('./training_models/',feature, 'spoofGMM.mat'), 'spoofGMM')
catch
    mkdir(fullfile('./training_models/', feature));
    save(fullfile('./training_models/',feature, 'spoofGMM.mat'), 'spoofGMM')
end

disp('Done!');

%%
%%%%%%%%%%%%%%%%%%%%%%%% EVALUATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(' Computing scores for evaluation trials...');

% чтение меток evaluation 
fileID = fopen(evaProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
filenames = protocol{1};
labels = protocol{2};
fclose(fileID);

scores = zeros(size(filenames));
%% скоры для evaluation
for i=1:length(filenames)
    llk_genuine = mean(compute_llk(evaluationFeature{i},genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(evaluationFeature{i},spoofGMM.m,spoofGMM.s,spoofGMM.w));
    scores(i) = llk_genuine - llk_spoof;
end
%%
try
    save(fullfile('./scores/',feature, 'scores.mat'), 'scores')
catch
    mkdir(fullfile('./scores/',feature));
    save(fullfile('./scores/',feature, 'scores.mat'), 'scores')
end
%%
scores = load(fullfile('./scores/',feature, 'scores.mat'));
scores = scores.scores;
%% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
[pm,pf] = compute_roc(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);


%% plots
plot(Pfa,Pmiss,'r-^',pf,pm,'g');
title (strcat('ROCCH and ROC plot for  ', model, ' model'));
axis('square');grid;legend('ROCCH','ROC');
xlabel('false alarm probability (%)') % x-axis label
ylabel('miss probability (%)') % y-axis label
%%
plot_det_curve(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')), model)
