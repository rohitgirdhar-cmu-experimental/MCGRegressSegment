function selectProposalsForTesting(resDir, vocDevKit)
if ~isdeployed
    addpath('../utils');
end
mcgDir = fullfile(resDir, 'mcg_proposals');
outputFile = fullfile(resDir, 'selProposals.mat');
files = getAllFiles(fullfile(vocDevKit, 'SegmentationClass'));
masks = {};
imgs = {};
bboxes = zeros(0, 4);

cnt = 0;
if exist(outputFile, 'file')
    load(outputFile, 'masks', 'imgs', 'bboxes');
else
    for file = files(:)'
        cnt = cnt + 1;
        file = file{:};
        [~, fname, ~] = fileparts(file);
        try
            load(fullfile(mcgDir, [fname, '.mat']), 'candidates_mcg');
        catch e
            warning('Unable to do for %s\n', fname);
            continue;
        end
        for i = 1 : 20 % take 20 from each image
            masks{end + 1} = ...
                ismember(candidates_mcg.superpixels, candidates_mcg.labels{i});
            imgs{end + 1} = fname;
            bboxes(end + 1, :) = candidates_mcg.bboxes(i, :);
        end
        fprintf('Read for %s (%d / %d)\n', file, cnt, numel(files));
    end
    fprintf('Read all\n');
    save(outputFile, 'masks', 'imgs', 'bboxes');
end

