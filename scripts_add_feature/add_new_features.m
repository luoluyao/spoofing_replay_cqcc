function [frame] = add_new_features(dir, dataFrame, Feature1,Feature2,Feature3, Feature4)
    
    % make fullname
   
    data_name = strcat('D:\Filin\spoofing_replay_cqcc\extracted_features\cqcc_features\',dataFrame,'.mat');
    
    feature_1_name = strcat(dir, '/', Feature1, '.txt');
    feature_2_name = strcat(dir,'/', Feature2, '.txt');
    feature_3_name = strcat(dir,'/', Feature3, '.txt');
    feature_4_name = strcat(dir,'/', Feature4, '.txt');
    
    % load data
    data = load(data_name);
    feature_1 = load(feature_1_name);
    feature_2 = load(feature_2_name);
    feature_3 = load(feature_3_name);
    feature_4 = load(feature_4_name);
    
    allAdditionData = [feature_1,feature_2,feature_3,feature_4];
   %% Нормалиация -  подгружаем параметры.
    load('D:\Filin\spoofing_replay_cqcc\extracted_features\addition_features\parameters_for_norm.mat');
    
    allAdditionData = (allAdditionData - mean_v) ./ sqrt(deviance);
    
    if strcmp(dataFrame,'genuineFeatureCellDev')
        frame = data.genuineFeatureCellDev;
    elseif strcmp(dataFrame,'genuineFeatureCellTrain')
        frame = data.genuineFeatureCellTrain;
    elseif strcmp(dataFrame,'spoofFeatureCellTrain')
        frame = data.spoofFeatureCellTrain;        
    elseif strcmp(dataFrame,'spoofFeatureCellDev')
        frame = data.spoofFeatureCellDev;
    elseif strcmp(dataFrame,'evaluationFeature')
        frame = data.evaluationFeature;    
    end
    
    
    for i=1:length(frame)        
        [row, col] = size(frame{i});

        additionArray = zeros(4, col);
        additionArray(1,:) = allAdditionData(i,1);
        additionArray(2,:) = allAdditionData(i,2);
        additionArray(3,:) = allAdditionData(i,3);
        additionArray(4,:) = allAdditionData(i,4);

        cellToList = [frame{i}];
        frame{i}=[cellToList;additionArray];
    end
end


