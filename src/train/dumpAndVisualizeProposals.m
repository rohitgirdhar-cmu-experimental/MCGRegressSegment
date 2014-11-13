function dumpAndVisualizeProposals(resDir, vocDevKit)
if ~isdeployed
    addpath('../utils');
end
mcgDir = fullfile(resDir, 'mcg_proposals');
imgsDir = fullfile(vocDevKit, 'JPEGImages');
outputDir = fullfile(resDir, 'visSegs');
outputFile = fullfile(resDir, 'topProposals.mat');
proposalMaskDir = fullfile(resDir, 'top_proposed_masks'); mkdir(proposalMaskDir);
files = getAllFiles(fullfile(vocDevKit, 'SegmentationObject'));
top_masks = {};
top_scores = [];
top_imgs = {};
TOPN = 5000;
cnt = 0;
if exist(outputFile, 'file')
    load(outputFile, 'top_scores', 'top_masks', 'top_imgs');
else
    for file = files(:)'
        cnt = cnt + 1;
        file = file{:};
        [~, fname, ~] = fileparts(file);
        try
            load(fullfile(mcgDir, [fname, '.mat']));
        catch
            continue;
        end
        [ious, order] = sort(IoUs, 'descend');
        top_scores = [top_scores, ious];
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
end
mkdir(outputDir);
scores_fid = fopen(fullfile(resDir, 'scores.txt'), 'w');
imgnames_fid = fopen(fullfile(resDir, 'topimgs.txt'), 'w');

for i = 1 : TOPN
    [~, fname, ~] = fileparts(top_imgs{i});
    I = imread(fullfile(imgsDir, [fname, '.jpg']));
    bmap = seg2bmap(top_masks{i});
    imwrite(top_masks{i}, fullfile(proposalMaskDir, [num2str(i), '.jpg']));
    I = imoverlay(I, bmap);
    I = uint8(double(I) .* repmat(double(top_masks{i} < eps) .* 0.25 + double(top_masks{i}), [1, 1, 3]));
    imwrite(I, fullfile(outputDir, [num2str(i) '.jpg']));
    fprintf(scores_fid, '%f\n', top_scores(i));
    fprintf(imgnames_fid, '%s\n', fname);
end
fclose(scores_fid);
fclose(imagenames_fid);

