function  I_adjust = ContAdj(rawImg, win_min, win_max)
%Contrast adjustment
%Input: 
%  rawImg: An image read form dicom file.
%  win_min: minimal window value
%  win_max: maximal window value
%Output:
%  I_adjust: The image after contrast adjustment
%
% Author: Heming Yao
% Platform: Linux/macOS

    adjustImg = rawImg;
    adjustImg(adjustImg < win_min) = win_min;
    adjustImg(adjustImg > win_max) = win_max;
    I_adjust = double(adjustImg-win_min)*255/(win_max-win_min);
end