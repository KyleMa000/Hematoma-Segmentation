%% Data Preparation
% This script is to read and save all patients' brain CT images and annotations
% Output: PatientsData
% Author: Heming Yao
% Platform: Linux/macOS
clear all
close all
clc
% Initialization
PatientsData = struct('brains', {},  'annots', {}, 'dicomImgs', {},...
    'meta', {}, 'intensity_mean', {}, 'Pid', {}, 'Datatype', {});
index = 0;
%% Extract brain imags from each patient in ProTECT dataset
save_path = 'Z:\Users\hemingy\TBI\';
prefix = 'Z:\Users\hemingy\TBI\RawDataset\';
pathProT = [prefix 'ProTECT/'];

subfolder = ["Craig", "Heming", "TrauImg"];
for ii = 1:length(subfolder)
    %%
    casePath = strcat(pathProT, subfolder(ii));
    Patients = dir(casePath); 
    Patients = Patients(~strncmpi('.', {Patients.name},1));
    %%
    for p = 1:length(Patients)
        index = index + 1;
        pid = Patients(p).name
        DcmDir = strcat(casePath, filesep, pid, filesep, 'DICOM', filesep);
        ImgDir = strcat(casePath, filesep, pid, filesep);
        pd = BrainImage_pid( 'Protected', DcmDir, ImgDir);
        pd.Pid = pid;
        if ii<3
            pd.Datatype = 'Protected';
        else
            pd.Datatype = 'TrauImg';
        end
        PatientsData(index) = pd;
        folder = strcat(save_path, pd.Pid, '_', pd.Datatype);
        if ~isfolder(folder)
            mkdir(folder);
        end
        brains = pd.brains;
        annots = pd.annots;
        for ind = 1:size(brains,3)
            img = cat(2, brains(:,:,ind), annots(:,:,ind)*256);
            imwrite(img, strcat(folder, filesep, num2str(ind), '.jpg'))
        end
    end
end

%% Extract brain imags from each patient in TrauImgdataset
pathImg = [prefix 'Data_TrauImg' filesep];
pathAnnot = [prefix 'Data_TrauImg_Annotation' filesep];

subfolder = ["Craig", "Heming"];
for ii = 1:length(subfolder)
    %%
    casePath = strcat(pathAnnot, subfolder(ii));
    Patients = dir(casePath); 
    Patients = Patients(~strncmpi('.', {Patients.name},1));
    %%
    for p = 1:length(Patients)
        index = index + 1;
        pid = Patients(p).name
        DcmDir = strcat(pathImg, filesep, pid, filesep);
        ImgDir = strcat(casePath, filesep, pid, filesep);
        pd = BrainImage_pid('TrauImg', DcmDir, ImgDir);
        pd.Pid = pid;
        pd.Datatype = 'TrauImg';
        PatientsData(index) = pd;
        folder = strcat(save_path, pd.Pid, '_', pd.Datatype);
        if ~isfolder(folder)
            mkdir(folder);
        end
        brains = pd.brains;
        annots = pd.annots;
        for ind = 1:size(brains,3)
            img = cat(2, brains(:,:,ind), annots(:,:,ind)*256);
            imwrite(img, strcat(folder, filesep, num2str(ind), '.jpg'))
        end
    end
end

%% Read Negative Cases
pathImg = [prefix 'Negative' filesep];
pathAnnot = [];

Patients = dir(pathImg); 
Patients = Patients(~strncmpi('.', {Patients.name},1));
%
for p = 1:length(Patients)
%for p=1
    index = index + 1;
    pid = Patients(p).name;
    DcmDir = strcat(pathImg, pid, filesep);
    ImgDir = "";
    pd = BrainImage_pid('Negative', DcmDir, ImgDir);
    pd.Pid = pid;
    pd.Datatype = 'Negative';
    PatientsData(index) = pd;
    folder = strcat(save_path, pd.Pid, '_', pd.Datatype);
    if ~isfolder(folder)
        mkdir(folder);
    end
    brains = pd.brains;
    annots = pd.annots;
    for ind = 1:size(brains,3)
        img = cat(2, brains(:,:,ind), annots(:,:,ind)*256);
        imwrite(img, strcat(folder, filesep, num2str(ind), '.jpg'))
    end
end

% %% Save the data
% PatientsData_total = PatientsData;
% %%
% for i = 1:5
%     fprintf(['Saving ' num2str(i) '....\n'])
%     start = (i-1)*25+1;
%     endi = min(i*25, length(PatientsData_total));
%     PatientsData = PatientsData_total(start:endi);
%     fname = fullfile(save_path, ['PatientsData_' num2str(i) '.mat']);
%     save(fname, 'PatientsData', '-v7.3')
% end