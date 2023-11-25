function HD = HellingerD(img1, img2)
% The Hellinger distance is used to quantify the similarity
% of two probability distributions. Here, the histogram
% of an image is the probability distribution of the image
%
% Author: Heming Yao
% Platform: Linux/macOS

n = 255;
s = 0;

h1 = hist(img1(:), n);
h2 = hist(img2(:), n);

for i = 1:n
    s = s + (sqrt(h1(i)) - sqrt(h2(i)))^2;
end

%% For comparison. Use Euclidean distance.
% for i = 1:n
%     s = s + (h1(i) - h2(i))^2;
% end

HD = -0.5*sqrt(s);

end