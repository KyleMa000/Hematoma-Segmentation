function [pd] = BrainImage_pid(dataset, DcmDir, ImgDir)
%Read patient brain CT images
%Input: 
%  dataset: 'TrauImg', 'Protected'. As the annotation color for
%              acute hematoma are different in these two datasets, the annotation reading
%              method is a little bit different.
%  DcmDir: Dicom directory
%  ImgDir: Annotation directory
%Output:
%  pd: A strcture. Fields include: brain (3D array), dicomImgs (3D array), 
%       meta (pixel spacing and slice spacing info), intensity_mean
%
% Author: Heming Yao
% Platform: Linux/macOS
    %%
    
    if strcmp(dataset,'TrauImg')
        mode = 2;
    else
        mode = 1;
    end
   
    %% Read Dicom Image List
    DcmList = dir(strcat(DcmDir, '*'));
    DcmList = DcmList(~strncmpi('.', {DcmList.name},1));
    
    ImgNew = [];
    
    for i= 1 : length(DcmList)
        fname = DcmList(i).name;
        if ~((strcmp(fname(end-2:end),'tif'))||(strcmp(fname(end-1:end),'db')))
            ImgNew = [ImgNew DcmList(i)];
        end
    end
    
    DcmList = ImgNew;
    
    if strcmp(dataset,'Negative')
        series = DcmList(1).name(1:4);
        ind = [];
        for j = 1:length(DcmList) 
            name = DcmList(j).name;
            if strncmpi(name,series,4)
                ind = [ind j];
            end
        end
        DcmList = DcmList(ind);
    end
    
    %% Read Dicom Images and Meta-data
    fname = DcmList(1).name;
    inf= dicominfo(strcat(DcmDir, fname));

    meta = [];
    meta.pixel_spacing = inf.PixelSpacing;
    meta.windom_width = inf.WindowWidth;
    meta.windom_center = inf.WindowCenter;
    meta.slice_thickness = inf.SliceThickness;
    
    [brains, dicomImgs, fnamelis, sliceThickness] = read_dicoms(DcmDir, DcmList);

    %% Read Annotations
    
    % get last 3 integers of first part of first dicom file name???
    % looks like this is just used later to keep track of the order or
    % index of the slices?
    first = split(fnamelis(1).fname, '.');
    first = first{1};
    first = str2num(first(end-3:end));

    Annots = zeros([512,512,size(brains,3)]);
    
    if ~strcmp(dataset,'Negative')
    ImgFiles = dir(ImgDir);
    ImgFiles = ImgFiles(~strncmpi('.', {ImgFiles.name},1));
    
    inxlist = [];
    for fidx = 1:length(ImgFiles)
        fname = ImgFiles(fidx).name;
        if length(fname)>8
            if strcmp(fname(end-2:end),'tif')||strcmp(fname(end-3:end),'tiff')
                X = imread(char(strcat(ImgDir, fname)));
                R = X(:,:,1);
                G = X(:,:,2);
                B = X(:,:,3);

                img_annot = cat(3,R,G,B);
                name = split(fname, '.');
                name = name{1};
                InstanceIdx = str2num(name(end-3:end));
                if first~=1
                    InstanceIdx = InstanceIdx - first+1;
                end
                    
                brain = brains(:,:,InstanceIdx);
                    
                [ImAnnot, cond] = find_annotation(img_annot, brain, mode);
                if cond
                    temp = Annots(:,:,InstanceIdx);
                    if sum(temp(:))>0
                        Annots(:,:,InstanceIdx) = (ImAnnot + temp)>0; 
                    else
                        Annots(:,:,InstanceIdx) = ImAnnot;
                    end
                    inxlist(end+1) = InstanceIdx;
                end
            end
        end
    end
    end
    %%
    temp = brains>250;
    s = regionprops3(temp,'Volume','Orientation','Centroid');
    [~, ind] = sort([s.Volume]);
    rotate_angle = -1*s.Orientation(ind(end));
    center = s.Centroid(ind(end),1:2);
    rota =img_rotate(brains, center, rotate_angle, 'bilinear');
    %[rota_brains, center, rotate_angle] =  rotate_method2(brains);
    Annots = Annots>=1;
    annots =img_rotate(Annots, center, rotate_angle, 'bilinear');
    annots = annots>0.5;
    rota_dicomImgs = img_rotate(dicomImgs, center, rotate_angle, 'bilinear');
    %%
    fname = DcmList(1).name;
    info= dicominfo(strcat(DcmDir, fname));
        
    meta.dicom_inf = info;
    meta.rotate_angle = rotate_angle;
    meta.sliceThickness = sliceThickness;

    pd = struct('brains', {}, 'annots', {}, 'dicomImgs', {}, 'meta', {}, 'intensity_mean', {});

    pd(1).brains = uint8(rota);
    pd(1).dicomImgs = rota_dicomImgs;
    %pd(1).raw_dicomImgs = dicomImgs;
    pd(1).annots = annots;
    pd(1).meta = meta;
    plist = rota(:);
    plist(plist==0) = [];
    intensity_mean = mean(plist);
    pd(1).intensity_mean = intensity_mean;
          
end

