function bmap = seg2bmap(segs)
% CV F14 - Provided Code
%
% Get boundaries from a segmentation map with a faster and cleaner way
%
% Input:
%   segs - segmentation output, should be of size H x W where H is the
%          height, and W is the width
% Output:
%   bmap - binary boundary map, with 1 pixel wide boundaries. The boundary 
%          pixels are offset by 1/2 pixel towards the origin from the actual 
%          segment boundary.
% 
% Created by Wen-Sheng Chu (Sep-20-2014)

bmap = rangefilt(segs,[0,0,1;0,1,1;0,1,0])>0;