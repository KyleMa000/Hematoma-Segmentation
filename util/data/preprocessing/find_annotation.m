function [ImAnnot , cond] = find_annotation(img, brain, mode)
% Read manual annotations. 
% Input:
%    img: the annotation image
%    brain: the brain image after contrast adjustment.
%    mode: which dataset
% Output:
%    ImAnnot: A binary mask showing the hematoma regions
%    cond: whether any hematoma region is annotated.
% Author: Heming Yao
% Platform: Linux/macOS

    if nargin == 2
        mode = 1;
    end
    
    % Find the brain regions
    m = brain>0;
    m = imfill(m, 'holes');
    mask_small = ones(size(img,1), size(img,2));
    x = regionprops(m, 'Area', 'PixelIdxList');
    small = find([x.Area]<1000);
    for i = 1:length(small)
        mask_small(x(small(i)).PixelIdxList) = 0;
    end
    
    % Read the annotations
    mask = cat(3, ~m, ~m, ~m);
    % Pixels outside of soft brain tissue should not be hematoma
    img(mask)=0;
    if mode==1
        base = cat(3, img(:,:,1), img(:,:,1), img(:,:,1));
        annot = base~=img;
    elseif mode==2
        base = cat(3, img(:,:,1), img(:,:,1), img(:,:,1));
        annot_temp = base~=img;
        base = cat(3, 255*ones(size(brain)), 242*ones(size(brain)), zeros(size(brain)));
        annot_1 = base==img;
        annot_2 = annot_temp == cat(3, zeros(size(brain)), zeros(size(brain)), ones(size(brain)));
        annot = annot_1 + annot_2;
    end
    
    % If hematoma region is segmented, perform the post-processing step.
    if sum(annot(:))
        cond = 1;
    else
        cond = 0;
    end
    
    if cond==1
        if mode==1
            line = sum(annot,3)>0;
        elseif mode==2
            line_1 = sum(annot_1,3)==3;
            line_2 = sum(annot_2,3)==3;
            line = line_1 + line_2;
        end
        BW = bwlabel(line);
        ImAnnot = imfill(BW, 'holes');
    else
        ImAnnot = zeros(size(brain));
    end
    
    ImAnnot = ImAnnot.*mask_small;
    
%% The following is an old version. Keep it just in case.
%     dim = size(img);
%     dim_2d = dim(1:2);
%     x = dim(1);
%     y = dim(2);
%     indexs = [];
%     cond = 0;
%     
%     img_new = zeros(dim_2d);
%     img_new2 = img;
%     
%     for i = 1:x
%         for j = 1:y
%             value = img(j,i,:);
%             if mode ==2
%                 if (value(1)==value(2)&&value(2)~=value(3))|| ...
%                     (sum(value(:)==[255,242,0]')==3)
%                     index = sub2ind(dim_2d, j, i);
%                     indexs(end+1) = index;
%                 end
%             elseif mode==1
%                 if ~(value(1)==value(2)&&value(2)==value(3))
%                     index = sub2ind(dim_2d, j, i);
%                     indexs(end+1) = index;
%                 end
%             end
%         end
%     end
%     
%     img_new(indexs)= 255;
% 
%     BW = bwlabel(img_new);
%     ImFilled = imfill(BW, 'holes');
% 
%     regions = regionprops(ImFilled, 'PixelList');
% 
%     for i = 1:length(regions)
%         pixellist = regions(i).PixelList;
%         if (length(pixellist)>5)
%             cond = 1;
%             break
% %             for j = 1:length(pixellist)
% %                 x = pixellist(j,1);
% %                 y = pixellist(j,2);
% %                 img_new2(y,x,:) = [255;0;0];
% %             end
%         end
%     end
    
end