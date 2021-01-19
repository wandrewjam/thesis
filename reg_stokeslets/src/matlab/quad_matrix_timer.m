profile on
clear variables
maxNumCompThreads(1)
load('quad_matrix.mat')
s_matrix = expand_matrix(s_matrix);

intermediate_solve1 = direct_solve(s_matrix, rhs);
intermediate_solve2 = cg_solve(s_matrix, rhs);
intermediate_solve3 = pcg_solve(s_matrix, rhs);

norm(intermediate_solve1 - intermediate_solve2) / norm(intermediate_solve1)
norm(intermediate_solve1 - intermediate_solve3) / norm(intermediate_solve1)

pt_forces = reshape_solns(intermediate_solve1, weights);

profsave(profile('info'), 'quad_matrix_results')

function full_matrix = expand_matrix(upper_triangle)
    full_matrix = upper_triangle + triu(upper_triangle, 1)';
end

function soln = direct_solve(matrix, rhs)
    soln = matrix\rhs;
end

function vec = reshape_solns(soln, weights)
    vec = soln ./ repmat(repelem(weights, 3)', 1, 7);
    vec = reshape(vec, [], 3, 7);
end

function soln = cg_solve(matrix, rhs)
    soln = zeros(size(rhs));
    for col = 1:size(rhs, 2)
        soln(:, col) = pcg(matrix, rhs(:, col));
    end
end

function soln = pcg_solve(matrix, rhs)
    soln = zeros(size(rhs));
    
    D = diag(diag(matrix));
    for col = 1:size(rhs, 2)
        soln(:, col) = pcg(matrix, rhs(:, col), 1e-6, 20, D);
    end
end