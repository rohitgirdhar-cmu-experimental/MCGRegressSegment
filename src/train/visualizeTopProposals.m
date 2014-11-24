function visualizeTopProposals(resDir, vocDevKit)
if ~isdeployed
    addpath('../utils');
end
mcgDir = fullfile(resDir, 'mcg_proposals');
imgsDir = fullfile(vocDevKit, 'JPEGImages');
outputDir = fullfile(resDir, 'visBestSegs'); mkdir(outputDir);
segDir = fullfile(vocDevKit, 'SegmentationClass');
files = getAllFiles(fullfile(vocDevKit, 'SegmentationClass'));

cnt = 0;
TOPN = 20;
maxIoU = zeros(numel(files), 1);
for file = files(:)'
    cnt = cnt + 1;
    file = file{:};
    [~, fname, ~] = fileparts(file);
    try
        load(fullfile(mcgDir, [fname, '.mat']), 'candidates_mcg');
    catch e
        warning(getReport(e));
        continue;
    end
    gtMask = imread(fullfile(segDir, file));
    masks = {};
    scores = [];
    for i = 1 : numel(candidates_mcg.scores)
        masks{end + 1} = ...
            ismember(candidates_mcg.superpixels, candidates_mcg.labels{i});
        scores(end + 1) = computeIOU(masks{end}, gtMask == 15);
    end
    [scores, order] = sort(scores, 'descend');
    maxIoU(cnt) = scores(1);
    mkdir(fullfile(outputDir, fname));
    for i = 1 : TOPN
        imwrite(masks{order(i)}, fullfile(outputDir, fname, [num2str(i) '.jpg']));
    end
    dlmwrite(fullfile(outputDir, fname, 'order.txt'), order(1 : TOPN), '\n');
    dlmwrite(fullfile(outputDir, fname, 'scores.txt'), scores(1 : TOPN), '\n');
    fprintf('Read for %s (%d / %d)\n', file, cnt, numel(files));
end
dlmwrite(fullfile(outputDir, 'top_ious.txt'), maxIoU, '\n');

function iou = computeIOU(mask, gtMask)
iou = sum(sum(mask & gtMask)) / sum(sum(mask | gtMask));

