function [adjustImgs, dicomImgs, fnamelis, sliceThickness] = read_dicoms(DcmDir, DcmList)
% Read all dicom images under the given dicom directory
% Input: 
%   DcmDir: dicom directory
%   DcmList: file name list
% Output:
%   adjustImgs: images after contrast adjustment
%   dicomImgs: raw dicom images
%   fnamelis: file name list after sorting based on the instance index
%
% Author: Heming Yao
% Platform: Linux/macOS
    adjustImgs = [];
    fnamelis = [];
    
   %% Sort the file name list based on the dicom file's instance index.
    for i= 1 : length(DcmList)
        fname = DcmList(i).name;
        try
            inf= dicominfo(strcat(DcmDir, fname));
            InstanceIdx = inf.InstanceNumber;
        catch
            continue
        end
        %outs = split(fname, '.');
        %InstanceIdx = str2num(outs{1});
        fnamelis(InstanceIdx).fname = fname;
    end
    idx = arrayfun(@(x) isempty(fnamelis(x).fname), 1:length(fnamelis));
    fnamelis(idx) = [];
    %% Read images
    %inf = dicominfo(strcat(DcmDir,fnamelis(1).fname));
    dicomImgs = zeros(512, 512, length(fnamelis)); % init to zeros
    dists = zeros([1, length(fnamelis)]);
    for i= 1 : length(fnamelis)
        % Read input file information
        %inf= dicominfo([DcmDir,'\',DcmList(i).name]);
        %if  strcmp (inf.SeriesDescription,'HEAD 5mm STND')
        % If the name of the file in the range
        %%
        fname = fnamelis(i).fname;
        
        % Read input image
        inf = dicominfo(fullfile(DcmDir,fname));
        rawImg=dicomread(strcat(DcmDir,fname));    
        dis = inf.ImagePositionPatient;
        cosines = inf.ImageOrientationPatient;
        normal = zeros([1,3]);
        normal(1) = cosines(2)*cosines(6) - cosines(3)*cosines(5);
        normal(2) = cosines(3)*cosines(4) - cosines(1)*cosines(6);
        normal(3) = cosines(1)*cosines(5) - cosines(2)*cosines(4);
        
        dist = 0;
        for d = 1:3
            dist = dist + normal(d)*dis(d);
        end
        dists(i) = dist;    
        
        
        [h,w] = size(rawImg);
        delta_h = 0;
        delta_w = 0;
        if h<512
            delta_h = ceil((512-h)/2);
        end

        if w<512
            delta_w = ceil((512-w)/2);
        end   

        Im = padarray(rawImg, [delta_h, delta_w], rawImg(1,1));
        Im_sub=ROI(Im, 512);
        
        
        %dicomImgs(:, :, i) = Im_sub;
        % Adjust Raw Image for final output
        dicomImg = inf.RescaleSlope *  double(Im_sub) + inf.RescaleIntercept;
        I_adjust = ContAdj(dicomImg, 0, 140); % Before: (0,160) but 140 in paper
        dicomImgs(:, :, i) = dicomImg;
        adjustImgs = cat(3, adjustImgs, uint8(I_adjust));
    end

    dists = sort(dists);
    sliceThickness = dists(2:end) - dists(1:end-1);
end


function out=ROI(im,Size)
% Crop the image's center region with the given size
% 'Size' is the final size of the output image
  [h,w]=size(im);
  h_start=round((h-Size)/2);w_start=round((w-Size)/2);
  out=im(h_start+1:h_start+Size,w_start+1:w_start+Size);
end
