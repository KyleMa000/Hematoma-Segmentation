function NCC = NormCrossCorrelation(img1, img2)
%The normalized cross correlation is the cross
%correlation (CC) applied after first normalizing the images to
%zero mean and variance one:

img1_norm = zscore(img1(:), 1);
img2_norm = zscore(img2(:), 1);

NCC = sum(img1_norm.*img2_norm)/length(img1_norm);
end