function rota = img_rotate(imgs, center, rotate_angle, method)
% Rotate the brain volume (3D) based on the given rotation angle and
% rotation center.
% Input: 
%    imgs: the brain volume in 3D
%    center: rotation center
%    rotate_angle: rotation angle
%    method: the method used for interpolation. 'nearest', 'bilinear', 'bicubic'
% Output:
%    rota: rotated brain volume
%
% Author: Heming Yao
% Platform: Linux/macOS

% The rotation angle is calcualted based on centered image 
% (refer to function getRotatedAngleByApproCenter)
center(1) = floor(center(1));
center(2) = floor(center(2));
[m,n,z]=size(imgs);
Canvas=ones(m*2,n*2,z)*double(imgs(1,1,1));
lcm=floor(m/2); lcn=floor(n/2); % left corner where the image paste 
% paste image to the canvas
Canvas(lcm+1:lcm+m,lcn+1:lcn+n,:)=imgs;
%Canvas = uint8(Canvas);
dx = m/2-center(1);
dy = n/2-center(2);
brain_t = imtranslate(Canvas,[dx, dy],'OutputView','same');
brain_tr = imrotate(brain_t,rotate_angle,method,'crop');
brain_trt = imtranslate(brain_tr,[-dx, -dy],'OutputView','same');
rota = brain_trt(lcm+1:lcm+m,lcn+1:lcn+n,:);

% subplot(2,2,1);imshow(Canvas(:,:,10))
% subplot(2,2,2);imshow(brain_t(:,:,10))
% subplot(2,2,3);imshow(brain_tr(:,:,10))
% subplot(2,2,4);imshow(rota(:,:,10))
end