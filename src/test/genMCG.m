function genMCG(imgsDir, resDir, nProps)
if ~isdeployed
    addpath(genpath('/IUS/vmr105/rohytg/projects/003_SelfieSeg/003_SDS/MCG-PreTrained'));
end
if isdeployed
    nProps = str2num(nProps);
end

system(['mkdir -p ' resDir]);
files = getAllFiles(imgsDir);

i = 0;
for file = files(:)'
    i = i + 1;
    file = file{:};
    [~, fname, ~] = fileparts(file);
    out_dir = fullfile(resDir, fname);
    if exist(out_dir, 'dir')
        fprintf('Already done for %s\n', fname);
        continue;
    end
    mkdir(out_dir);
    I = imread(fullfile(imgsDir, [fname, '.jpg']));
%    I = imresize(I, [200, NaN]);

    [candidates_mcg, ~] = im2mcg(I,'accurate');
    
    for id = 1 : min(nProps, numel(candidates_mcg.labels))
        mask = ismember(candidates_mcg.superpixels, candidates_mcg.labels{id});
        imwrite(mask, fullfile(out_dir, [num2str(id), '.jpg']));
    end
    fprintf('Done for %s %d / %d\n', fname, i, numel(files));
end

