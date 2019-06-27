function generate_3d_mesh(N)

% Generate a 3D mesh for the 2-receptor problem

L = 1;

xsamp = linspace(0, L, N+1);

[X, Y, Z] = meshgrid(xsamp, xsamp, xsamp);

xvec = reshape(X, [], 1);
yvec = reshape(Y, [], 1);
zvec = reshape(Z, [], 1);

mask = logical((xvec >= yvec) .* (xvec >= zvec));

scatter3(xvec(mask), yvec(mask), zvec(mask))
xlabel('$x$', 'Interpreter', 'latex')
ylabel('$y$', 'Interpreter', 'latex')
zlabel('$z$', 'Interpreter', 'latex')

end
