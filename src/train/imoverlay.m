function Iout = imoverlay(I, bmap, color)
% CV F14 - Provided Code
%
% Create an image overlayed with the boundary map
%
% Input:
%   I     - input image
%   bmap  - binary boundary map denoting segment boundaries
%   color - (optional) color of the boundary
% Output:
%   Iout  - output overlayed image
%
% Created by Wen-Sheng Chu (Sep-20-2014)

if nargin < 3
	color = [255,0,0];
end
Iout = reshape(I,[],3);
Iout(bmap(:)~=0,:) = repmat(color,sum(bmap(:)),1);
Iout = reshape(Iout, size(I));

