%% Script for feature extraction

%% Clear all

clear; close all; clc;

%% Features to extract from database

feature_name = 'cqcc';
database_name = 'ASVSpoof2015';

%% Add toolbox paths

addpath(genpath('CQCC_v1.0'));
addpath('MSR Identity Toolkit v1.0/code');
addpath('voicebox');

%% Paths

root_folder = fullfile('D:', 'Data', 'ASVSpoof2015');
wav_folder = fullfile(root_folder, 'wav');
protocol_folder = fullfile(root_folder, 'protocol', 'CM_protocol');

output_folder = fullfile('extracted_features', database_name, feature_name);

%% Process all subfolders

all_files = dir(wav_folder);
is_folder = [all_files.isdir];
all_files_name = all_files.name;
folders = all_files_name(is_folder);


