function img=cropOrPadding(img,Size)
  % 'Size' is the final size of the output image
  [h,w]=size(img);
  if h<Size
      delta = ceil((Size-h)/2); 
      img = padarray(img, [delta,0]);
      img = img(1:Size,1:w);
  elseif h>Size 
      h_start=round((h-Size)/2);
      img = img(h_start+1:h_start+Size,:);
  end
  [h,w]=size(img);
  if w<Size
      delta = ceil((Size-w)/2); 
      img = padarray(img, [0, delta]);
      img = img(1:h,1:Size);
  elseif w>Size 
      w_start=round((w-Size)/2);
      img = img(:,w_start+1:w_start+Size);
  end
end