N = 1e4;
a = rand(N, N);
b = rand(N, N);

a = (((a + b) ./ b).^2) + ((2 * a - 3 * b) / pi).^3;