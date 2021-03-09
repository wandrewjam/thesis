clear variables
close all

expt_num = input('Enter experiment number: ', 's');

loadfile = sprintf('~/thesis/reg_stokeslets/data/bd_run/bd_run%s.mat', expt_num);
load(loadfile)

dt = t(2) - t(1);

for i = 1:(length(t)-1)
    sum1 = sum(bond_array(:, 1, i) >= 0);
    sum2 = sum(bond_array(:, 1, i+1) >= 0);
    if sum1 > 0 && (sum1 * sum2) == 0
        for k = 1:5
            new_i = i + 5*(k-2);
            
%             true_receptors(:, :, k) = [x(kk), y(kk), z(kk)] + receptors * R(:, :, kk)';
            true_receptors = [x(new_i), y(new_i), z(new_i)] + receptors * R(:, :, new_i)';
            
            bond_slice = bond_array(:, 1, new_i) >= 0;
            existing_bond_i = bond_array(bond_slice, 1, new_i) + 1;
            num_bonds = length(existing_bond_i);
            x_bond = [true_receptors(existing_bond_i, 1), zeros(num_bonds, 1)];
            y_bond = [true_receptors(existing_bond_i, 2), bond_array(bond_slice, 2, new_i)];
            z_bond = [true_receptors(existing_bond_i, 3), bond_array(bond_slice, 3, new_i)];
            
            generate_3d_plot(R(:, 1, new_i), x(new_i), z(new_i), t(new_i), x_bond, y_bond, z_bond)
        end
    end
end


function generate_3d_plot(em, height, zcom, ti, x_bond, y_bond, z_bond)
    global plt_num
    [x_wall, xp, y_wall, yp, z_wall, zp] = define_parametric_vars();

    ph = acos(em(1));
    th = atan2(em(3), em(2));

    R = [cos(ph), -sin(ph), 0;
         cos(th)*sin(ph), cos(th)*cos(ph), -sin(th);
         sin(th)*sin(ph), sin(th)*cos(ph), cos(th)];

    p_array = cat(3, xp, yp, zp);
    x = zeros(size(xp)); y = zeros(size(xp)); z = zeros(size(xp));

    for i = 1:size(xp, 1)
        for j = 1:size(xp, 2)
            result = R * reshape(p_array(i, j, :), 3, 1);
            x(i, j) = result(1);
            y(i, j) = result(2);
            z(i, j) = result(3);
        end
    end

    x = x + height;
    z = z + zcom;
    z_wall = z_wall + zcom;

    x_vec = repmat(linspace(0, 2, 5)', [1, 5]);
    x_vec = x_vec(2:end, :);
    y_vec = repmat(linspace(-2, 2, 5), [4, 1]);
    z_vec = -2 * ones(4, 5) + zcom;

    u_vec = zeros(size(x_vec));
    v_vec = zeros(size(x_vec));
    w_vec = x_vec;


    % Define color arrays for the ellipsoid and wall
%     C1 = repmat(reshape([0 0 1], 1, 1, 3), size(x, 1), size(x, 2), 1);
    C1 = acos(xp);
    C2 = repmat(reshape([.5 .5 .5], 1, 1, 3), ...
        size(x_wall, 1), size(x_wall, 2), 1);

    figure()
    surf(y, z, x, C1, 'EdgeAlpha', 0.4)
    hold on
    surf(y_wall, z_wall, x_wall, C2)
    hold on
    quiver3(y_vec, z_vec, x_vec, v_vec, w_vec, u_vec, .5)
    line(y_bond, z_bond, x_bond, 'Color', 'k', 'LineWidth', 3)
    
    axis([-2, 2, -2 + zcom, 2+zcom, 0, 3.5])

    titlestr = sprintf('$t = %.2f, \\quad \\vec{e}_m = (%.2f, %.2f, %.2f)$', ti, em(1), em(2), em(3));
    xlabel('$y$', 'Interpreter', 'latex')
    ylabel('$z$', 'Interpreter', 'latex')
    zlabel('$x$', 'Interpreter', 'latex')
    title(titlestr, 'Interpreter', 'latex')

    view([75, 15])
    set(gca, 'Projection', 'perspective')

    filename = sprintf('plt%u_t%.1f', plt_num, ti);
%     savefig(strcat(filename,'.fig'))
%     saveas(gcf, strcat(filename, '.png'))
end

function [x_wall, xp, y_wall, yp, z_wall, zp] = define_parametric_vars()
    u = linspace(0, 2*pi, 100);
    v = linspace(0, pi, 100);

    x_wall = zeros(2, 2);
    y_wall = [-2, 2; -2, 2];
    z_wall = [-2, -2; 2, 2];

    xp = .5 * ones(size(u')) * cos(v);
    yp = 1.5 * cos(u') * sin(v);
    zp = 1.5 * sin(u') * sin(v);
end
