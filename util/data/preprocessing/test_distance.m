% To find out one way that can calculate the 'distance'/'similarity' of two
% images from the same dicom file but with different window widths and window
% ceters in contrast adjustment.
% Author: Heming Yao
% Platform: Linux/macOS
%% Draw images
for i = 1:12
    subplot(3,4,i);
    I_adjust = ContAdj(img, i*5, 160);
    imshow(uint8(I_adjust))
end

%% Draw the relationship between the proposed distance calculation and the window width
brains = PatientsData(1).dicomImgs;
img = brains;
%img = brains(:,:,20);

img1 = ContAdj(img, 10, 160);
ccLis = zeros(1,20);
for i = 1:20
    img2 = ContAdj(img,i, 160);
    cc = NormCrossCorrelation(img1, img2);
    %cc = sum((img1(:) - img2(:)).^2);
    %cc = HellingerD(img1, img2);
    ccLis(i) = cc;
end
figure;plot(ccLis)