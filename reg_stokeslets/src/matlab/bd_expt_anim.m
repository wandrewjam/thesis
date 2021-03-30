clear variables
close all

expt_num = input('Enter experiment number: ');

loadfile = sprintf('/Users/andrewwork/thesis/reg_stokeslets/data/bd_run/bd_run%03u.mat', expt_num);
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
C2 = repmat(reshape([.5 .5 .5], 1, 1, 3), ...
    size(x_wall, 1), size(x_wall, 2), 1);

% Define arrays for plotting bonds
if size(bond_array, 1) > 0
    existing_bond_i = bond_array(bond_array(:, 1, 1) >= 0, 1, 1);
    num_bonds = length(existing_bond_i);
    x_bond = [true_receptors(existing_bond_i, 1, 1), zeros(num_bonds, 1)];
    y_bond = [true_receptors(existing_bond_i, 2, 1), bond_array(existing_bond_i, 2, 1)];
    z_bond = [true_receptors(existing_bond_i, 3, 1), bond_array(existing_bond_i, 3, 1)];

    p = line(x_bond, y_bond, z_bond);
end

figure('Position', [418 1 764 977])
plt_ax = subplot('Position', [0.13 0.54 0.775 0.42]);
s = surf(ymat(:, :, 1), zmat(:, :, 1), xmat(:, :, 1), C1);
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
l1 = plot(t(1), x(1), 'LineWidth', 3.);
axis([-.01 3.01 0.5 1.6])
xlabel('Time (s)', 'Interpreter', 'latex')
ylabel('Height ($\mu$m)', 'Interpreter', 'latex')
set(gca, prop_names, prop_values);

subplot('Position', [0.13,0.07,0.775,0.17])
e3 = R(3, 1, :);
l2 = plot(t(1), e3(1), 'LineWidth', 3.);
axis([-.01 3.01 -1.1 1.1])
xlabel('Time (s)', 'Interpreter', 'latex')
ylabel('$z$-cmp of minor axis', 'Interpreter', 'latex')
set(gca, prop_names, prop_values);

axes(plt_ax)
set(gca, 'DataAspectRatio', [1 1 1], 'Projection', 'perspective')

v = VideoWriter(sprintf('/Users/andrewwork/thesis/reg_stokeslets/data/videos/be_video_%03u.avi', expt_num));
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
        xcom = x(k); ycom = y(k); zcom = z(k);
        
        w.XData = y_wall + ycom;
        w.YData = z_wall + zcom;
        
        l1.XData = t(1:slice(gap_slice(ii)));
        l1.YData = x(1:slice(gap_slice(ii)));
        
        l2.XData = t(1:slice(gap_slice(ii)));
        l2.YData = e3(1:slice(gap_slice(ii)));

        x_mark.XData = ycom; x_mark.YData = zcom - 2; x_mark.ZData = xcom;
        
        if size(bond_array, 1) > 0
            existing_bond_i = bond_array(bond_array(:, 1, k) >= 0, 1, k) + 1;
            num_bonds = length(existing_bond_i);
            x_bond = [true_receptors(existing_bond_i, 1, ii), zeros(num_bonds, 1)]';
            y_bond = [true_receptors(existing_bond_i, 2, ii), bond_array(bond_array(:, 1, k) >= 0, 2, k)]';
            z_bond = [true_receptors(existing_bond_i, 3, ii), bond_array(bond_array(:, 1, k) >= 0, 3, k)]';

            delete(p)
            p = line(y_bond, z_bond, x_bond);
            for jj = 1:length(p)
                p(jj).Color = 'k';
            end
        end

        axis([-2 + ycom, 2 + ycom, -2 + zcom, 2 + zcom, 0, 3.2])
        title_obj.String = sprintf('$t = %.4f$', t(slice(gap_slice(ii))));
        set(gca, 'TickLabelInterpreter', 'latex')

        pause(0.01 * gap)

        frame = getframe(gcf);
        writeVideo(v, frame);
    end

    close(v)

%     pause(1)
% end
