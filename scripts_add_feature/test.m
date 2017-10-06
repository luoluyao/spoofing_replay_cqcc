clear;
dir = fullfile('D:\Filin','spoofing_replay_cqcc','extracted_features', 'addition_features');

Feature1 = 'snrOriginalDEV';
Feature2 = 'excessOriginalDEV';
Feature3 = 'sigmaOriginalDEV';
Feature4 = 'rangeOriginalDEV';

%data_name = strcat(dir,'/files/',dataFrame,'.mat');
feature_1_name = strcat(dir,'/', Feature1, '.txt');
feature_2_name = strcat(dir,'/', Feature2, '.txt');
feature_3_name = strcat(dir,'/', Feature3, '.txt');
feature_4_name = strcat(dir,'/', Feature4, '.txt');

%%
    % load data
    feature_1 = load(feature_1_name);
    feature_2 = load(feature_2_name);
    feature_3 = load(feature_3_name);
    feature_4 = load(feature_4_name);
%%
allAdditionData = [feature_1,feature_2,feature_3,feature_4];

%% Нормалиация
mean_v = mean(allAdditionData, 1);
deviance = mean(power((allAdditionData - mean_v),2),1);
normalise = (allAdditionData - mean_v) ./ sqrt(deviance);
