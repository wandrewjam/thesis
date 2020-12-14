clear variables
close all

global plt_num
expt_num = input('Enter experiment number: ');

loadfile = sprintf('data/bd_expt%03u.mat', expt_num);
load(loadfile)

[x_wall, xp_ref, y_wall, yp_ref, z_wall, zp_ref] = define_parametric_vars();

gap = 5;
dt = t(2) - t(1);
slice = zeros(round((t(end) - t(1))/dt), 1);
for i = 1:length(t)
    if abs(round(t(i) / dt) - t(i) / dt) < 1e-10
        slice(round(t(i) / dt) + 1) = i;
    end
end

gap_slice = 1:gap:length(slice);
new_length = length(gap_slice);
if mod(length(slice) - 1, gap) ~= 0
    error('gap does not evenly divide length of slice')
end

p_array = cat(3, xp_ref, yp_ref, zp_ref);
xmat = zeros([size(xp_ref), new_length]); 
ymat = zeros([size(yp_ref), new_length]);
zmat = zeros([size(zp_ref), new_length]);

for k = 1:new_length
    for i = 1:size(xp_ref, 1)
        for j = 1:size(xp_ref, 2)
            result = R(:, :, slice(gap_slice(k))) * reshape(p_array(i, j, :), 3, 1);
            xmat(i, j, k) = result(1);
            ymat(i, j, k) = result(2);
            zmat(i, j, k) = result(3);
        end
    end

    if mod(gap * (k-1), 50) == 0
        gap * (k-1) + 1
    end
end

x1r = repmat(reshape(x(slice(gap_slice)), 1, 1, []), size(xp_ref, 1), size(xp_ref, 2), 1);
y1r = repmat(reshape(y(slice(gap_slice)), 1, 1, []), size(yp_ref, 1), size(yp_ref, 2), 1);
z1r = repmat(reshape(z(slice(gap_slice)), 1, 1, []), size(zp_ref, 1), size(zp_ref, 2), 1);
xmat = xmat + x1r;
ymat = ymat + y1r;
zmat = zmat + z1r;

C1 = acos(xp_ref);

figure
s = surf(ymat(:, :, 1), zmat(:, :, 1), xmat(:, :, 1), C1);
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
        s.XData = ymat(:, :, ii);
        s.YData = zmat(:, :, ii);
        s.ZData = xmat(:, :, ii);
        xcom = x(slice(gap_slice(ii)));
        ycom = y(slice(gap_slice(ii)));
        zcom = z(slice(gap_slice(ii)));

        x_mark.XData = ycom; x_mark.YData = zcom - 2; x_mark.ZData = xcom;

        axis([-2 + ycom, 2 + ycom, -2 + zcom, 2 + zcom, 0, 3.2])
        title_obj.String = sprintf('$t = %.2f$', t(slice(gap_slice(ii))));

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
