clear variables
close all

expt_num = input('Enter experiment number: ');

loadfile = sprintf('data/bd_expt%03u.mat', expt_num);
load(loadfile)

[x_wall, xp_ref, y_wall, yp_ref, z_wall, zp_ref] = define_parametric_vars();

gap = 1;
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

true_receptors = zeros(size(receptors, 1), 3, new_length);

for k = 1:new_length
    for i = 1:size(xp_ref, 1)
        for j = 1:size(xp_ref, 2)
            result = R(:, :, slice(gap_slice(k))) * reshape(p_array(i, j, :), 3, 1);
            xmat(i, j, k) = result(1);
            ymat(i, j, k) = result(2);
            zmat(i, j, k) = result(3);
        end
    end
    
    kk = slice(gap_slice(k));
    true_receptors(:, :, k) = [x(kk), y(kk), z(kk)] + receptors * R(:, :, kk)';

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

% Define arrays for plotting bonds
existing_bond_i = bond_array(bond_array(:, 1, 1) >= 0, 1, 1);
num_bonds = length(existing_bond_i);
x_bond = [true_receptors(existing_bond_i, 1, 1), zeros(num_bonds, 1)];
y_bond = [true_receptors(existing_bond_i, 2, 1), bond_array(existing_bond_i, 2, 1)];
z_bond = [true_receptors(existing_bond_i, 3, 1), bond_array(existing_bond_i, 3, 1)];

p = line(x_bond, y_bond, z_bond);
axis([-2, 2, -2, 2, 0, 3.2])

titlestr = sprintf('$t = %.2f$', t(1));
xlabel('$y$', 'Interpreter', 'latex')
ylabel('$z$', 'Interpreter', 'latex')
zlabel('$x$', 'Interpreter', 'latex')
title_obj = title(titlestr, 'Interpreter', 'latex');

view([95, 2])
% s.EdgeColor = 'blue';

v = VideoWriter(sprintf('data/be_video_%03u.avi', expt_num));
v.FrameRate = 40;
open(v)

frame = getframe(gcf);
writeVideo(v, frame)

% while true
    for ii = 2:new_length
        k = slice(gap_slice(ii));
        
        s.XData = ymat(:, :, ii);
        s.YData = zmat(:, :, ii);
        s.ZData = xmat(:, :, ii);
        xcom = x(k);
        ycom = y(k);
        zcom = z(k);

        x_mark.XData = ycom; x_mark.YData = zcom - 2; x_mark.ZData = xcom;
        
        existing_bond_i = bond_array(bond_array(:, 1, k) >= 0, 1, k);
        num_bonds = length(existing_bond_i);
        x_bond = [true_receptors(existing_bond_i, 1, ii), zeros(num_bonds, 1)]';
        y_bond = [true_receptors(existing_bond_i, 2, ii), bond_array(bond_array(:, 1, k) >= 0, 2, k)]';
        z_bond = [true_receptors(existing_bond_i, 3, ii), bond_array(bond_array(:, 1, k) >= 0, 3, k)]';
        
        delete(p)
        p = line(y_bond, z_bond, x_bond);
        for jj = 1:length(p)
            p(jj).Color = 'k';
        end
            

        axis([-2 + ycom, 2 + ycom, -2 + zcom, 2 + zcom, 0, 3.2])
        title_obj.String = sprintf('$t = %.2f$', t(k));

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
