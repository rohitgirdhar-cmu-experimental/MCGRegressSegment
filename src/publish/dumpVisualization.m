function dumpVisualization(resdir, feature, outputdir)
selmat = fullfile(resdir, 'selProposals.mat');
scores_fpath = fullfile(resdir, 'features', feature, 'scores.txt');

load(selmat, 'bboxes', 'imgs', 'masks');
fid = fopen(scores_fpath);
scores = textscan(fid, '%f\n');
fclose(fid);
scores = scores{1};

[uimgs, ia, ic] = unique(imgs);
for i = 1 : numel(uimgs)
    rel_scores = scores(ic == i);
    rel_masks = masks(ic == i);
    [~, order] = sort(rel_scores, 'descend');
    rel_masks = rel_masks(order);
    out_dir = fullfile(outputdir, uimgs{i});
    system(['mkdir -p ' out_dir]);
    for j = 1 : 10
        imwrite(rel_masks{j}, fullfile(out_dir, [num2str(j) '.jpg']));
    end
end

