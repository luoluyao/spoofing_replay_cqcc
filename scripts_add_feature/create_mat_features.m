function create_mat_features(dir, dataFrame, Feature1,Feature2,Feature3, Feature4)
    
    % make fullname
   
    data_name = strcat('D:\Filin\spoofing_replay_cqcc\extracted_features\cqcc_features\',dataFrame,'.mat');
    
    feature_1_name = strcat(dir, '/', Feature1, '.txt');
    feature_2_name = strcat(dir,'/', Feature2, '.txt');
    feature_3_name = strcat(dir,'/', Feature3, '.txt');
    feature_4_name = strcat(dir,'/', Feature4, '.txt');
    
    % load data
    feature_1 = load(feature_1_name);
    feature_2 = load(feature_2_name);
    feature_3 = load(feature_3_name);
    feature_4 = load(feature_4_name);
    
    allAdditionData = [feature_1, feature_2, feature_3,feature_4];
    
    save (strcat('../addition_features/', dataFrame,'_add.mat'), 'allAdditionData');
end
