function genMCGAndScores()
addpath(genpath('/IUS/vmr105/rohytg/projects/003_SelfieSeg/003_SDS/MCG-PreTrained'));
vocdevkit = '/IUS/vmr105/rohytg/projects/003_SelfieSeg/datasets/VOCdevkit/VOC2012'; 

% Read an input image
imgsDir = fullfile(vocdevkit, 'JPEGImages');
gtDir = fullfile(vocdevkit, 'SegmentationClass');
resDir = 'results/voc2012';
mkdir(resDir);
files = getAllFiles(gtDir);
i = 0;
for file = files(:)'
    i = i + 1;
    file = file{:};
    [~, fname, ~] = fileparts(file);
    out_file = fullfile(resDir, [fname '.mat']);
    if exist(out_file, 'file') || exist([out_file, '.lock'])
        fprintf('Already done for %s\n', fname);
        continue;
    end
    mkdir([out_file '.lock']);
    I = imread(fullfile(imgsDir, [fname, '.jpg']));
%    I = imresize(I, [200, NaN]);

    [candidates_mcg, ~] = im2mcg(I,'accurate');
    IoUs = zeros(1, 100);
    for id = 1 : min(100, numel(candidates_mcg.labels))
        mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        gtMask = imread(fullfile(gtDir, file));
        % only keep the GT for person, person ID = 15
        IoUs(id) = computeIOU(mask, gtMask == 15);
    end
    save(fullfile(resDir, [fname '.mat']), 'IoUs', 'candidates_mcg');
    rmdir([out_file '.lock']);
    fprintf('Done for %s %d / %d\n', fname, i, numel(files));
end

function iou = computeIOU(mask, gtMask)
iou = sum(sum(mask & gtMask)) / sum(sum(mask | gtMask));

