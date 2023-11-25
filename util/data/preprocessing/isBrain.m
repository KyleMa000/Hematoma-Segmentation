function output = isBrain(path)
% To check whether the input is a valid dicom file.
% Author: Heming Yao
% Platform: Linux/macOS
    fname = dir(path);
    fname = fname(~strncmpi('.dcm', {fname.name},1));
    img = fname(1).name;
    try
        info = dicominfo(fullfile(path, img));
        fprintf(info.SeriesDescription)
    catch
        output=0;
        
    end

end
