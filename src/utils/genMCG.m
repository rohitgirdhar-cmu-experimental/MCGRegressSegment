function genMCG(vocdevkit, resDir)
addpath(genpath('/IUS/vmr105/rohytg/projects/003_SelfieSeg/003_SDS/MCG-PreTrained'));

% Read an input image
imgsDir = fullfile(vocdevkit, 'JPEGImages');
gtDir = fullfile(vocdevkit, 'SegmentationClass');
resDir = fullfile(resDir, 'mcg_proposals');
mkdir(resDir);
files = getAllFiles(gtDir);
i = 0;
for file = files(:)'
    i = i + 1;
    file = file{:};
    [~, fname, ~] = fileparts(file);
    out_file = fullfile(resDir, [fname '.mat']);
    if exist(out_file, 'file') || exist([out_file, '.lock'], 'dir')
        fprintf('Already done for %s\n', fname);
        continue;
    end
    mkdir([out_file '.lock']);
    I = imread(fullfile(imgsDir, [fname, '.jpg']));
%    I = imresize(I, [200, NaN]);

    [candidates_mcg, ~] = im2mcg(I,'accurate');
    save(fullfile(resDir, [fname '.mat']), 'candidates_mcg');
    rmdir([out_file '.lock']);
    fprintf('Done for %s %d / %d\n', fname, i, numel(files));
end

