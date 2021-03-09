clear variables
close all

global plt_num
plt_num = input('Enter plot number: ');

loadfile = sprintf('../../data/ub_expt/fine%u.mat', plt_num);
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

function generate_3d_plot(em, height, z_com, ti)
    global plt_num

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