function selectProposalsForTraining(resDir, vocDevKit)
if ~isdeployed
    addpath('../utils');
end
mcgDir = fullfile(resDir, 'mcg_proposals');
imgsDir = fullfile(vocDevKit, 'JPEGImages');
outputDir = fullfile(resDir, 'visSegs');
segDir = fullfile(vocDevKit, 'SegmentationClass');
outputFile = fullfile(resDir, 'selProposals.mat');
files = getAllFiles(fullfile(vocDevKit, 'SegmentationObject'));
masks = {};
scores = [];
imgs = {};
bboxes = zeros(0, 4);

SELN = 10000;
cnt = 0;
if exist(outputFile, 'file')
    load(outputFile, 'scores', 'masks', 'imgs', 'bboxes');
else
    for file = files(:)'
        cnt = cnt + 1;
        file = file{:};
        [~, fname, ~] = fileparts(file);
        try
            load(fullfile(mcgDir, [fname, '.mat']), 'candidates_mcg');
        catch e
            getReport(e);
            continue;
        end
        for i = 1 : 20 % take 500 from each image
            masks{end + 1} = ...
            ismember(candidates_mcg.superpixels, candidates_mcg.labels{i});
            imgs{end + 1} = fname;
            gtMask = imread(fullfile(segDir, file));
            scores(end + 1) = computeIOU(masks{end}, gtMask == 15);
            bboxes(end + 1, :) = candidates_mcg.bboxes(i, :);
        end
        fprintf('Read for %s (%d / %d)\n', file, cnt, numel(files));
    end
    fprintf('Read all\n');
    sel = randperm(length(scores), SELN);
    masks = masks(sel);
    imgs = imgs(sel);
    scores = scores(sel);
    bboxes = bboxes(sel, :);
    save(outputFile, 'scores', 'masks', 'imgs', 'bboxes');
end

function iou = computeIOU(mask, gtMask)
iou = sum(sum(mask & gtMask)) / sum(sum(mask | gtMask));

