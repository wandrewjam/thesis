function fig = draw_3d_platelet(em, height, zcom)
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
    surf(y, z, x, C1)
    hold on
    surf(y_wall, z_wall + zcom, x_wall, C2)
    hold on
    quiver3(y_vec, z_vec, x_vec, v_vec, w_vec, u_vec, .5)
    axis([-2, 2, -2 + zcom, 2 + zcom, 0, 2.5])
    
    fig = gcf;
end