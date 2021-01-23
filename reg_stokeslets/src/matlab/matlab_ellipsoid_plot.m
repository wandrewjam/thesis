clear variables
close all

global plt_num
plt_num = input('Enter plot number: ');

loadfile = sprintf('data/fine%u.mat', plt_num);
load(loadfile)

em_arr = cat(1, e1, e2, e3);
imax = 280;

i0 = 1;
[~, i1] = min(em_arr(3, 1:imax));
[~, i2] = max(em_arr(1, 1:imax));
[~, i3] = max(em_arr(3, 1:imax));
[~, i4] = min(em_arr(1, 2:imax));
i4 = i4 + 1;

for i = [i0, i1, i2, i3, i4]
    em = em_arr(:, i);
    ti = t(i);
    generate_3d_plot(em, x1(i), ti)
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

function generate_3d_plot(em, height, z_com, ti)
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

    x_vec = repmat(linspace(0, 2, 5)', [1, 5]);
    x_vec = x_vec(2:end, :);
    y_vec = repmat(linspace(-2, 2, 5), [4, 1]);
    z_vec = -2 * ones(4, 5);

    u_vec = zeros(size(x_vec));
    v_vec = zeros(size(x_vec));
    w_vec = x_vec;


    % Define color arrays for the ellipsoid and wall
    C1 = repmat(reshape([0 0 1], 1, 1, 3), size(x, 1), size(x, 2), 1);
    C2 = repmat(reshape([.5 .5 .5], 1, 1, 3), ...
        size(x_wall, 1), size(x_wall, 2), 1);

    figure()
    surf(y, z, x, C1)
    hold on
    surf(y_wall, z_wall, x_wall, C2)
    hold on
    quiver3(y_vec, z_vec, x_vec, v_vec, w_vec, u_vec, .5)
    axis([-2, 2, -2, 2, 0, 2.5])

    titlestr = sprintf('$t = %.2f, \\quad \\vec{e}_m = (%.2f, %.2f, %.2f)$', ti, em(1), em(2), em(3));
    xlabel('$y$', 'Interpreter', 'latex')
    ylabel('$z$', 'Interpreter', 'latex')
    zlabel('$x$', 'Interpreter', 'latex')
    title(titlestr, 'Interpreter', 'latex')

    view([105, 15])

    filename = sprintf('plt%u_t%.1f', plt_num, ti);
    savefig(strcat(filename,'.fig'))
    saveas(gcf, strcat(filename, '.png'))
end