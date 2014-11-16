function genMCGVisualization(imgsDir, mcgdir, resdir)
mkdir(resdir);
imgs = dir(mcgdir);
imgs = arrayfun(@(x) x.name, imgs, 'UniformOutput', false);
imgs = imgs(3:end);
for img = imgs(:)'
    img = img{:};
    I = imread(fullfile(imgsDir, [img, '.jpg']));
    segList = dir(fullfile(mcgdir, img, '*.jpg'));
    segList = arrayfun(@(x) x.name, segList, 'UniformOutput', false);
    mkdir(fullfile(resdir, img));
    for seg = segList(:)'
        seg = seg{:};
        S = imread(fullfile(mcgdir, img, seg));
        bmap = seg2bmap(S);
        I2 = imoverlay(I, bmap);
        I2 = uint8(double(I2) .* repmat(double(S < eps) .* 0.25 + double(S >= eps) .* 0.75, [1, 1, 3]));
        imwrite(I2, fullfile(resdir, img, seg));
    end
end

