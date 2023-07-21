function dx = Cantilever_mass_fcn(t,in,para)

Cp = para(1);
R = para(2);
vphi = para(3);
zeta = para(4);
omega = para(5);
Nr = para(6);
acc = para(7);
rate = para(8);
freq_s_hz = para(9);
xb1ddot = acc*sin((freq_s_hz + rate*t/2)*2*pi*t);

eta1 = in(1);
eta1dot = in(2);
v2 = in(3);
force = 1; %<----------------------------------------------------------------
% eta1ddot = -2*zeta*omega*eta1dot - omega^2*eta1 - vphi*v2 - Nr*xb1ddot;
eta1ddot = -2*zeta*omega*eta1dot - omega^2*eta1 - vphi*v2 - force; % 
v2dot = -1/(Cp*R)*v2 + vphi/Cp*eta1dot;

dx = [eta1dot; eta1ddot; v2dot;];