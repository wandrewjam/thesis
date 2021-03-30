clear variables
close all

global plt_num
plt_num = input('Enter plot number: ');

loadfile = sprintf('data/ub_expt/fine%u.mat', plt_num);
load(loadfile)

[x_wall, xp_ref, y_wall, yp_ref, z_wall, zp_ref] = define_parametric_vars();

t = t / 100;
gap = 5;
slice = 1:gap:length(t);
new_length = length(slice);
if mod(length(t) - 1, gap) ~= 0
    error('gap does not evenly divide length of t')
end

ph = acos(e1(slice));
th = atan2(e3(slice), e2(slice));
R = zeros(3, 3, new_length);
R(1, 1, :) = cos(ph);
R(1, 2, :) = -sin(ph);
R(2, 1, :) = cos(th).*sin(ph);
R(2, 2, :) = cos(th).*cos(ph);
R(2, 3, :) = -sin(th);
R(3, 1, :) = sin(th).*sin(ph);
R(3, 2, :) = sin(th).*cos(ph);
R(3, 3, :) = cos(th);

p_array = cat(3, xp_ref, yp_ref, zp_ref);
x = zeros([size(xp_ref), new_length]); y = zeros([size(yp_ref), new_length]);
z = zeros([size(zp_ref), new_length]);

for k = 1:new_length
    for i = 1:size(xp_ref, 1)
        for j = 1:size(xp_ref, 2)
            result = R(:, :, k) * reshape(p_array(i, j, :), 3, 1);
            x(i, j, k) = result(1);
            y(i, j, k) = result(2);
            z(i, j, k) = result(3);
        end
    end

    if mod(gap * (k-1), 50) == 0
        gap * (k-1) + 1
    end
end

x1r = repmat(reshape(x1(slice), 1, 1, []), size(xp_ref, 1), size(xp_ref, 2), 1);
y1r = repmat(reshape(x2(slice), 1, 1, []), size(yp_ref, 1), size(yp_ref, 2), 1);
z1r = repmat(reshape(x3(slice), 1, 1, []), size(zp_ref, 1), size(zp_ref, 2), 1);
x = x + x1r;
y = y + y1r;
z = z + z1r;

C1 = acos(xp_ref);
C2 = repmat(reshape([.5 .5 .5], 1, 1, 3), ...
    size(x_wall, 1), size(x_wall, 2), 1);

figure('Position', [418 1 764 977])
plt_ax = subplot('Position', [0.13 0.54 0.775 0.42]);
s = surf(y(:, :, 1), z(:, :, 1), x(:, :, 1), C1);
s.EdgeAlpha = 0.4;
hold on
w = surf(y_wall, z_wall, x_wall, C2);
axis([-2, 2, -2, 2, 0, 3.2])
prop_names = {'TickLabelInterpreter', 'FontSize', 'XGrid', 'YGrid'}';
prop_values = {'latex', 14,'on', 'on'};

titlestr = sprintf('$t = %.4f$', t(1));
xlabel('$y$', 'Interpreter', 'latex')
ylabel('$z$', 'Interpreter', 'latex')
zlabel('$x$', 'Interpreter', 'latex')
title_obj = title(titlestr, 'Interpreter', 'latex');
set(gca, prop_names, prop_values);

view([95, 2])
% s.EdgeColor = 'blue';

subplot('Position', [0.13,0.3,0.775,0.17])
l1 = plot(t(1), x1(1), 'LineWidth', 3.);
axis([-.004 .504 0.5 1.6])
xlabel('Time (s)', 'Interpreter', 'latex')
ylabel('Height ($\mu$m)', 'Interpreter', 'latex')
set(gca, prop_names, prop_values);

subplot('Position', [0.13,0.07,0.775,0.17])
l2 = plot(t(1), e3(1), 'LineWidth', 3.);
axis([-.004 .504 -1.1 1.1])
xlabel('Time (s)', 'Interpreter', 'latex')
ylabel('$z$-cmp of minor axis', 'Interpreter', 'latex')
set(gca, prop_names, prop_values);

axes(plt_ax)
set(gca, 'DataAspectRatio', [1 1 1], 'Projection', 'perspective')

v = VideoWriter(sprintf('data/videos/video_%u.avi', plt_num));
v.FrameRate = 40;
open(v)

frame = getframe(gcf);
writeVideo(v, frame)

% while true
    for ii = 2:new_length
        s.XData = y(:, :, ii);
        s.YData = z(:, :, ii);
        s.ZData = x(:, :, ii);
        xcom = x1(slice(ii)); ycom = x2(slice(ii)); zcom = x3(slice(ii));
        
        w.XData = y_wall + ycom;
        w.YData = z_wall + zcom;
        
        l1.XData = t(1:slice(ii));
        l1.YData = x1(1:slice(ii));
        
        l2.XData = t(1:slice(ii));
        l2.YData = e3(1:slice(ii));

        x_mark.XData = ycom; x_mark.YData = zcom - 2; x_mark.ZData = xcom;

        axis([-2 + ycom, 2 + ycom, -2 + zcom, 2 + zcom, 0, 3.2])
        title_obj.String = sprintf('$t = %.4f$', t(slice(ii)));
        set(gca, 'TickLabelInterpreter', 'latex')

        pause(0.01 * gap)

        frame = getframe(gcf);
        writeVideo(v, frame);
    end

    close(v)
    
    % Plot center of mass and orientation
    figure('Position', [600, 40, 764, 500])
    tiledlayout(2, 1)
    
    nexttile
    plot(t, x1, 'LineWidth', 3.)
    axis([-.004 .504 0.5 1.6])
    xlabel('Time (s)', 'Interpreter', 'latex')
    ylabel('Height ($\mu$m)', 'Interpreter', 'latex')
    set(gca, prop_names, prop_values);
    
    nexttile
    plot(t, e3, 'LineWidth', 3.)
    axis([-.004 .504 -1.1 1.1])
    xlabel('Time (s)', 'Interpreter', 'latex')
    ylabel('$z$-cmp of minor axis', 'Interpreter', 'latex')
    set(gca, prop_names, prop_values);

%     pause(1)
% end

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
