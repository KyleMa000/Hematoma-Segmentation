function preprocess(startI, endI, fileSize, fold, pernum)

% Paths
% data_path = 'A:\Projects\Polytrauma\Data\Raw\subarachnoid_hematoma\raw_images\';
% data_path = '/Volumes/med-kayvan-lab/Projects/Polytrauma/Data/Raw/subarachnoid_hematoma/raw_images/'
data_path = '/nfs/turbo/med-kayvan-lab/Projects/Polytrauma/Data/Raw/subarachnoid_hematoma/raw_images/';
% save_path = '/Volumes/med-kayvan-lab/Projects/Polytrauma/Data/Processed/hodgman/subarachnoid_hematoma/';
% save_path = 'A:\Projects\Polytrauma\Data\Processed\hodgman\subarachnoid_hematoma\';
save_path = '/nfs/turbo/med-kayvan-lab/Projects/Polytrauma/Data/Processed/hodgman/subarachnoid_hematoma/';

% Get all patient IDs in data path
Pids = dir(data_path);
Patients = Pids(~strncmpi('.', {Pids.name},1));
Patients = {Patients.name};
fprintf('number of patients: %d', length(Patients));

% Initialization
% PatientsData is the output struct array
PatientsData = struct('dicomImgs', {}, 'pid', {}, 'info', {}, 'path', {}, ...
'imgnameList', {}, 'sliceSpacing', {}, 'pixelSpacing', {}, 'time', {}); 

%% Find first appropriate axial brain CT series per patient
parfor ind = startI:endI
    fprintf('Processing %s\n', Patients{ind});
    % Find the dicom directory at the earliest date and time
    % Decide whether it is the brain CT scan based on the window center and
    % window width metadata stored in dicom file header
    
    % get all series folders
    pidPath = fullfile(data_path, Patients{ind});
    series = dir(pidPath);
    series(ismember({series.name},{'.','..'})) = [];
    series(endsWith({series.name},'.dir')) = [];
    series = series(~strncmpi('.', {series.name},1));
    
    % The series for brain soft tissue should have a 0<= WindowCenter<=120 &&
    % WindowWidth<=400, ....
    %%
    brain_struct = struct('path',{}, 'idx',{}, 'time',{});
    numSeries = 1;
    
    for j = 1:length(series)
    
        % get the individual DICOM files in the series
        sPath = fullfile(series(1).folder, series(j).name);
        fnames = dir(sPath);
        fnames(endsWith({fnames.name},'.dir')) = [];
        fnames = fnames(~strncmpi('.', {fnames.name},1));

        % skip if there are less than 20 images in the series
        if length(fnames)<20
            continue
        end
        
        % check first DICOM image in series
        info = dicominfo(fullfile(sPath, fnames(2).name));
    
        if isfield(info, 'WindowCenter') && isfield(info, 'WindowWidth') && ...
            isfield(info, 'SliceThickness') && ~isempty(info.SliceThickness) && ...
            isfield(info,'ImageOrientationPatient') && isfield(info,'PixelSpacing')
            
            orient = info.ImageOrientationPatient;
            if orient(6)==-0.5
                orient(6) = -0.49;
            elseif orient(6) == 0.5
                orient(6) = 0.49;
            end
            if orient(6) == 0.5
                orient(6) = 0.51;
            end                
            orient = round(orient);
    
            if (info.SliceThickness>=2)&&(info.WindowCenter(1)<=120)&&(info.WindowCenter(1)>=0)...
                &&info.WindowWidth(1)<=400 && info.PixelSpacing(1)> 0.3 && ...
                sum(orient(1:5)==[1;0;0;0;1])== 5 && (orient(5)*orient(6)==-1||orient(5)*orient(6)==0)
                %sum(orient==[1;0;0;0;1;0])==6
    
                if isfield(info, 'BodyPartExamined') && ...
                        ~isempty(info.BodyPartExamined) && ...
                        ~((contains(info.BodyPartExamined, 'HEAD') || contains(info.BodyPartExamined, 'BRAIN')) && ~(contains(info.BodyPartExamined, 'HEADNECK')))
                
                    continue
                end
    
                if isfield(info, 'SeriesDescription') && ...
                        ~isempty(info.SeriesDescription) && ...
                        (contains(info.SeriesDescription, 'BONE') || (contains(info.SeriesDescription, 'Bone')...
                        || contains(info.SeriesDescription, 'SPINE')|| contains(info.SeriesDescription, 'Spine')...
                        || contains(info.SeriesDescription, 'BOLUS') || contains(info.SeriesDescription, 'Bolus')...
                        || contains(info.SeriesDescription, 'FACE')) || contains(info.SeriesDescription, 'Face'))
                    continue
                end
                
                % save series information
                time = info.StudyTime;
                h = str2double(time(1:2));
                m = str2double(time(3:4));
                s = str2double(time(5:6));
                brain_struct(numSeries).path = sPath;
                brain_struct(numSeries).idx = fnames; % this is supposed to be a list of image file names?
                brain_struct(numSeries).time = h*60*60+m*60+s;
                brain_struct(numSeries).orientation = info.ImageOrientationPatient;
                
                numSeries = numSeries+1;
            end
        end
    end

    % check if patient doesn't have any qualifying series
    if numel(brain_struct)==0
        fprintf('No series for patient %d\n', ind);
        continue
    end
    
    % get the earliest series
    seriesIdx = 1;
    if numel(brain_struct)>1
       time_list = [brain_struct.time]; 
       [~,seriesIdx] = min(time_list);
    end

    %% Read dicom images
    fname = brain_struct(seriesIdx).idx;
    sPath = brain_struct(seriesIdx).path;
    instances = zeros([length(fname),1]);
    imgs = zeros([512,512,length(fname)]);
    imgname_list = strings([length(fname),1]);
    dists = zeros([length(fname),1]);
    for fi = 1:length(fname)
        imgname = fullfile(sPath, fname(fi).name);
        info = dicominfo(imgname);
        orient = info.ImageOrientationPatient;
        if orient(6)==-0.5
            orient(6) = -0.49;
        elseif orient(6) == 0.5
            orient(6) = 0.49;
        end
        if orient(6) == 0.5
            orient(6) = 0.51;
        end                
        orient = round(orient);
        %if sum(orient==[1;0;0;0;1;0])==6
        if sum(orient(1:5)==[1;0;0;0;1])== 5 && (orient(5)*orient(6)==-1||orient(5)*orient(6)==0) ...
                && sum(ismember(instances,info.InstanceNumber))<1
            imgname_list(fi) = fname(fi).name;
            dis = info.ImagePositionPatient;
            cosines = info.ImageOrientationPatient;
            normal = zeros([1,3]);
            normal(1) = cosines(2)*cosines(6) - cosines(3)*cosines(5);
            normal(2) = cosines(3)*cosines(4) - cosines(1)*cosines(6);
            normal(3) = cosines(1)*cosines(5) - cosines(2)*cosines(4);
            dist = 0;
            for d = 1:3
                dist = dist + normal(d)*dis(d);
            end
            dists(fi) = dist;    
            
            img = dicomread(imgname);
            if size(img,1)~=512|| size(img,2)~=512
                img = cropOrPadding(img, 512);
            end

            % Calculate CT values
            dicomImg = info.RescaleSlope *  img  + info.RescaleIntercept;
            imgs(:,:,fi) = dicomImg;
            % Read and save InstanceNumber
            instances(fi) = info.InstanceNumber;
        end
    end
    exist_index = sum(sum(imgs,1),2)~=0;
    brains = imgs(:,:,exist_index);
    dists = dists(exist_index);
    imgname_list = imgname_list(exist_index);
    instances = instances(exist_index);
    % Sort brain slices based on the instance number
    [~, oi] = sort(instances);
    brains = brains(:,:,oi);
    dists = dists(oi);
    
    % If the number of slices is more thant 70, we only keep the last 75
    % slices. This is because sometimes the scan start at neck regions.
    orig_brains = brains;
    if size(brains,3)>70
        brains = brains(:,:,end-69:end);
        dists = dists(end-69:end);
        imgname_list = imgname_list(end-69:end);
    end
    
    % Calculate slice spacing. If there are minus number 
    sliceSpacing = dists(2:end) - dists(1:end-1);
    temp = sliceSpacing<0;
    % Sometimes the series we extract scan the head more than once. In this
    % case, we only keep the first one.
    if sum(temp==0)>sum(temp==1)
        temp = find(temp==1);
        if ~isempty(temp)
            endInstanceIdx = temp(1)-1;
        end
    else
        temp = find(temp==0);
        if ~isempty(temp)
            endInstanceIdx = temp(1)-1;
        end
    end
    
    if ~isempty(temp)
        brains = brains(:,:,1:endInstanceIdx);
        dists = dists(1:endInstanceIdx);
        imgname_list = imgname_list(1:endInstanceIdx);   
    end
    
    % Rotate brain slices
    temp = brains>250;
    s = regionprops3(temp,'Volume','Orientation','Centroid');
    [~, index] = sort([s.Volume]);
    rotate_angle = -1*s.Orientation(index(end));
    center = s.Centroid(index(end),1:2);
    rota = img_rotate(brains, center, rotate_angle, 'bilinear');

    % adjust contrast of images
    for i = 1:size(rota,3)
        rota(:,:,i) = uint8(ContAdj(rota(:,:,i),0,140));
    end

    % Add the patient data to output stuct array
    PatientsData(ind).dicomImgs = rota;
    PatientsData(ind).pid = str2num(Patients{ind});
    PatientsData(ind).info = info;
    PatientsData(ind).path = sPath;
    PatientsData(ind).imgnameList = imgname_list;
    PatientsData(ind).sliceSpacing = dists(2:end) - dists(1:end-1);
    PatientsData(ind).pixelSpacing = info.PixelSpacing;
    PatientsData(ind).time = brain_struct(seriesIdx).time;
    
    %% Visualization
%     for i = 1:size(rota,3)
%         subplot(6,ceil(size(rota,3)/6),i);
%         img = rota(:,:,i);
%         I_adjust = ContAdj(img, 0, 160);
%         imshow(uint8(I_adjust))
%     end


%     figure;
%     for i = 1:min(size(rota,3),36)
%         subplot(6,6,i);
%         img = rota(:,:,i);
%         I_adjust = ContAdj(img, 0, 160);
%         imshow(uint8(I_adjust))
%     end
%     saveas(gcf, [save_path 'plots' filesep Patients{ind} '.png'])
%     close 
    
end
%% Save
all_patientId = [];

for ind = startI:endI
    all_patientId(ind-startI+1) = str2num(Patients{ind});
    if mod(ind,fileSize)==0
        num_case = (fold-1)*pernum +ind/fileSize;
        fname = [save_path 'AllData' filesep 'AllPatientsData_' num2str(num_case) '.mat'];
        pdata = PatientsData(ind-fileSize+1:ind);
        save(fname, 'pdata', '-v7.3')
    end
end

pidNotFound = setdiff(all_patientId, [PatientsData.pid]);
save([save_path 'AllData' filesep, ['fold' num2str(fold) '_notfound.mat']], 'pidNotFound')

end
