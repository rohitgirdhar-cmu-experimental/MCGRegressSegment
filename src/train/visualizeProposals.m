function visualizeProposals(resDir, imgsDir)
outputDir = fullfile(resDir, 'visSegs');
outputFile = fullfile(resDir, 'topProposals.mat');
files = getAllFiles(resDir);
top_masks = {};
top_scores = [];
top_imgs = {};
TOPN = 5000;
cnt = 0;
for file = files(:)'
    cnt = cnt + 1;
    file = file{:};
    load(fullfile(resDir, file));
    [ious, order] = sort(IoUs, 'descend');
    top_scores = [top_scores, ious];
    [~, fname, ~] = fileparts(file);
    for i = 1 : length(order)
        id = order(i);
        top_masks{end + 1} = ...
        ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        top_imgs{end + 1} = fname;
    end
    [top_scores, order] = sort(top_scores, 'descend');
    cur_size = min(TOPN, length(top_scores));
    top_scores = top_scores(1 : cur_size);
    top_masks = top_masks(order(1 : cur_size));
    top_imgs = top_imgs(order(1 : cur_size));
    fprintf('Seen %d / %d files, max = %f, >0 = %d \n', cnt, numel(files), ...
            max(top_scores), sum(top_scores > 0));
end

save(outputFile, 'top_scores', 'top_masks', 'top_imgs');
mkdir(outputDir);
scores_fid = fopen(fullfile(resDir, 'scores.txt'), 'w');

for i = 1 : TOPN
    [~, fname, ~] = fileparts(top_imgs{i});
    I = imread(fullfile(imgsDir, [fname, '.jpg']));
    bmap = seg2bmap(top_masks{i});
    I = imoverlay(I, bmap);
    I = uint8(double(I) .* repmat(double(top_masks{i} < eps) .* 0.25 + double(top_masks{i}), [1, 1, 3]));
    imwrite(I, fullfile(outputDir, [num2str(i) '.jpg']));
    fprintf(scores_fid, '%f\n', top_scores(i));
end
fclose(scores_fid);

