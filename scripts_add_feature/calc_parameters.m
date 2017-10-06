dir = fullfile('D:\Filin','spoofing_replay_cqcc','extracted_features', 'addition_features');

genuineFeatureCellDev_add_name  = 'D:\Filin\spoofing_replay_cqcc\extracted_features\addition_features\genuineFeatureCellDev_add.mat';
genuineFeatureCellTrain_add_name = 'D:\Filin\spoofing_replay_cqcc\extracted_features\addition_features\genuineFeatureCellTrain_add.mat';
spoofFeatureCellDev_add_name = 'D:\Filin\spoofing_replay_cqcc\extracted_features\addition_features\spoofFeatureCellDev_add.mat';
spoofFeatureCellTrain_add_name = 'D:\Filin\spoofing_replay_cqcc\extracted_features\addition_features\spoofFeatureCellTrain_add.mat';

genuineFeatureCellDev_add = load(genuineFeatureCellDev_add_name);
genuineFeatureCellTrain_add = load(genuineFeatureCellTrain_add_name);
spoofFeatureCellDev_add = load(spoofFeatureCellDev_add_name);
spoofFeatureCellTrain_add = load(spoofFeatureCellTrain_add_name);

%%

genuineFeatureCellDev_add = genuineFeatureCellDev_add.allAdditionData;
genuineFeatureCellTrain_add = genuineFeatureCellTrain_add.allAdditionData;
spoofFeatureCellDev_add = spoofFeatureCellDev_add.allAdditionData;
spoofFeatureCellTrain_add = spoofFeatureCellTrain_add.allAdditionData;


%%
concat_data = cat(1, genuineFeatureCellDev_add, genuineFeatureCellTrain_add,spoofFeatureCellDev_add,spoofFeatureCellTrain_add);
%% Нормалиация
mean_v = mean(concat_data, 1);
deviance = mean(power((concat_data - mean_v),2),1);
%%
save('D:\Filin\spoofing_replay_cqcc\extracted_features\addition_features\parameters_for_norm.mat','mean_v', 'deviance' )