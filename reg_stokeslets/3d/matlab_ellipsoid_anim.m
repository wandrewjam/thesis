clear variables
close all

global plt_num
plt_num = input('Enter plot number: ');

loadfile = sprintf('data/fine%u.mat', plt_num);
load(loadfile)

[x_wall, xp_ref, y_wall, yp_ref, z_wall, zp_ref] = define_parametric_vars();

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

figure
s = surf(y(:, :, 1), z(:, :, 1), x(:, :, 1), C1);
axis([-2, 2, -2, 2, 0, 3.2])

titlestr = sprintf('$t = %.2f$', t(1));
xlabel('$y$', 'Interpreter', 'latex')
ylabel('$z$', 'Interpreter', 'latex')
zlabel('$x$', 'Interpreter', 'latex')
title_obj = title(titlestr, 'Interpreter', 'latex');

view([105, 15])
% s.EdgeColor = 'blue';

v = VideoWriter(sprintf('data/video_%u.avi', plt_num));
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

        x_mark.XData = ycom; x_mark.YData = zcom - 2; x_mark.ZData = xcom;

        axis([-2 + ycom, 2 + ycom, -2 + zcom, 2 + zcom, 0, 3.2])
        title_obj.String = sprintf('$t = %.2f$', t(slice(ii)));

        pause(0.01 * gap)

        frame = getframe(gcf);
        writeVideo(v, frame);
    end

    close(v)

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
