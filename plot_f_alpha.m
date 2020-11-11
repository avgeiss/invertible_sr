clear all;close all;clc;
P = linspace(-1,1,100);
x = linspace(-1,1,100);
[x,P] = meshgrid(x,P);
a = abs((P-x)./(1-sign(P-x).*x));
contour(x,P,a,9);
hold on;
c = colormap();
plot([-1,1],[-1,1],'-','color',c(1,:));
hcb=colorbar();
ylabel(hcb,'(dimensionless)')
caxis([0 1]);
fs = 18;
xlabel('$$\overline{x}$$','interpreter','latex','fontsize',fs);
ylabel('$$P$$','interpreter','latex','fontsize',fs);
title('$$\alpha$$','interpreter','latex','fontsize',fs);
print('./figures/alpha.png','-dpng','-r300');


clear all;close all;clc;
fs = 16;
x = linspace(-1,1,16);
P = linspace(-1,1,16);
[x,P] = meshgrid(x,P);
s = sign(-P);
f = x + P.*(1+s.*x);
figure();
c = -2:.25:2;
c(c==0) = [];
contourf(x,P,f-x,c,'showtext','on','labelspacing',500);
hold on;
plot([-1,1],[0, 0],'k-');
hcb = colorbar();
ylabel(hcb,'Change in Pixel Intensity (dimensionless)')
title('$$f(x_i)-x_i$$','interpreter','latex','fontsize',fs);
ylabel('$$P$$','interpreter','latex','fontsize',fs);
xlabel('$$x_i$$','interpreter','latex','fontsize',fs);
print('./figures/fx_small.png','-dpng','-r300');