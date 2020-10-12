profile on
clear variables
load('quad_matrix.mat')
s_matrix = s_matrix + triu(s_matrix, 1)';
intermediate_solve = s_matrix\rhs;
pt_forces = intermediate_solve ./ repmat(repelem(weights, 3)', 1, 7);
pt_forces = reshape(pt_forces, [], 3, 7);
profsave(profile('info'), 'quad_matrix_results')