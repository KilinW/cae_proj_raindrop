% % % Unit: kg m s
clear all
syms omega
global M1 J1 m1

ccc = datetime('now')
tic

acc_g = 1; % excitation acceleration, unit: G 
M1 = 11*1e-6; % kg <---------------------------------------------------獴簑
J1 = M1*(5e-3)^2; % moment of inertia

Ys = 0.5e9; % substrate young's modulus <-----------------------------------------
rho_s = 1390; % substrate density <----------------------------------------------
Yp = 2.45*1e9; % piezo young's modulus 
rho_p = 1780; % piezo density
d31 = -190e-12; %-270*1e-12; % piezoelectric constant (m/V)
epsilon = 15.93e-9; %3500*8.854*1e-12;  % F/m 

h1s = 0.5e-3; % substrate thickness <-------------------------------------
h1p = 0.028e-3; % piezo thickness 
L1 = 20e-3; % substrate length <-----------------------------------------
Lp1 = 0*L1; % 0 means the pizeo cover from the fixed end
Lp2 = 1*L1; % piezo length
b1s = 13e-3; % substrate width <------------------------------------------
b1p = b1s; % piezo width
b1pp = b1p; % electrode width
% Cp = b1pp*(Lp2-Lp1)*epsilon/h1p;
Cp = 0.5*1e-9; % capacitance (F) <---------------------ノ计筿况秖筿甧

R = 1*1e6; % load resistance
zeta = 0.043; % damping ratio <------------------------------------------- 

mode_find = 1; % how many modes to be found

m1 = rho_s*b1s*h1s + rho_p*b1p*h1p;

n = Ys*b1s/(Yp*b1p);

h1pa = (h1p^2+2*n*h1p*h1s+n*h1s^2)/2/(h1p+n*h1s);
h1sa = (h1p^2+2*h1p*h1s+n*h1s^2)/2/(h1p+n*h1s);
h1pc = n*h1s*(h1p+h1s)/2/(h1p+n*h1s);
h1a = -h1sa;
h1b = h1pa-h1p;
h1c = h1pa;
EI1 = b1s/3*Ys*(h1b^3-h1a^3)+b1p/3*Yp*(h1c^3-h1b^3);

% vtheta = Yp*d31*b1pp*h1pc; %
vtheta = 0.85e-7; % <-------------------------------------------- 溃筿舰玒计

alpha = ((omega^2*m1/EI1))^(1/4);

% % % % XX = [A1 B1 C1 D1]'; AA*XX = CC, CC = 0
%% with Mass
AA = [0 1 0 1; alpha 0 alpha 0; -alpha * (J1 * cos(alpha * L1) * omega ^ 2 + EI1 * sin(alpha * L1) * alpha) -alpha * (-J1 * sin(alpha * L1) * omega ^ 2 + EI1 * cos(alpha * L1) * alpha) -alpha * (J1 * cosh(alpha * L1) * omega ^ 2 - EI1 * sinh(alpha * L1) * alpha) -alpha * (J1 * sinh(alpha * L1) * omega ^ 2 - EI1 * cosh(alpha * L1) * alpha); -EI1 * cos(alpha * L1) * alpha ^ 3 + M1 * sin(alpha * L1) * omega ^ 2 EI1 * sin(alpha * L1) * alpha ^ 3 + M1 * cos(alpha * L1) * omega ^ 2 EI1 * cosh(alpha * L1) * alpha ^ 3 + M1 * sinh(alpha * L1) * omega ^ 2 EI1 * sinh(alpha * L1) * alpha ^ 3 + M1 * cosh(alpha * L1) * omega ^ 2;];

AA_string = char(det(AA));
fid = fopen('Cantilever_mass_det.m','w');
fprintf(fid,'function f = Cantilever_mass_det(omega)\n');
fprintf(fid,'f = ');
fprintf(fid,'%s;\n',AA_string);
fprintf(fid,'end');
st = fclose(fid);

guess1 = 2*2*pi;
incre1 = 1.1;
omega_a = zeros(1,mode_find);
options=optimset('MaxIter',1e4,'TolX',1e-14,'TolFun',1e-14);
for i = 1:mode_find  
    if i == 1
        omega_a(i) = fzero(@Cantilever_mass_det,guess1,options); %rad/s
        while omega_a(i)<0
            guess1 = guess1*incre1;
            omega_a(i) = fzero(@Cantilever_mass_det,guess1,options);
        end
    else
        guess1 = guess1*incre1;
        omega_a(i) = fzero(@Cantilever_mass_det,guess1,options);
        while (omega_a(i)-omega_a(i-1))<0.001
            guess1 = guess1*incre1;
            omega_a(i) = fzero(@Cantilever_mass_det,guess1,options);
        end
    end
end

omega_a/2/pi % resonant frequency 
alpha_a = ((omega_a.^2*m1/EI1)).^(1/4);

for mode = 1:mode_find
    clear Cantilever_mass_FindA1
    clear omega alpha
    syms A1 x1
    
    omega = omega_a(mode);
    alpha = alpha_a(mode);    
    
    AA = [0 1 0 1; alpha 0 alpha 0; -alpha * (J1 * cos(alpha * L1) * omega ^ 2 + EI1 * sin(alpha * L1) * alpha) -alpha * (-J1 * sin(alpha * L1) * omega ^ 2 + EI1 * cos(alpha * L1) * alpha) -alpha * (J1 * cosh(alpha * L1) * omega ^ 2 - EI1 * sinh(alpha * L1) * alpha) -alpha * (J1 * sinh(alpha * L1) * omega ^ 2 - EI1 * cosh(alpha * L1) * alpha); -EI1 * cos(alpha * L1) * alpha ^ 3 + M1 * sin(alpha * L1) * omega ^ 2 EI1 * sin(alpha * L1) * alpha ^ 3 + M1 * cos(alpha * L1) * omega ^ 2 EI1 * cosh(alpha * L1) * alpha ^ 3 + M1 * sinh(alpha * L1) * omega ^ 2 EI1 * sinh(alpha * L1) * alpha ^ 3 + M1 * cosh(alpha * L1) * omega ^ 2;];
%     KK = [A1 * (-(-J1 * cosh(alpha * L1) * omega ^ 2 + EI1 * sinh(alpha * L1) * alpha) / (-J1 * sin(alpha * L1) * omega ^ 2 - J1 * sinh(alpha * L1) * omega ^ 2 + EI1 * cos(alpha * L1) * alpha + EI1 * cosh(alpha * L1) * alpha) - 0.1e1 / (-J1 * sin(alpha * L1) * omega ^ 2 - J1 * sinh(alpha * L1) * omega ^ 2 + EI1 * cos(alpha * L1) * alpha + EI1 * cosh(alpha * L1) * alpha) * (J1 * cos(alpha * L1) * omega ^ 2 + EI1 * sin(alpha * L1) * alpha)); -A1; A1 * ((-J1 * cosh(alpha * L1) * omega ^ 2 + EI1 * sinh(alpha * L1) * alpha) / (-J1 * sin(alpha * L1) * omega ^ 2 - J1 * sinh(alpha * L1) * omega ^ 2 + EI1 * cos(alpha * L1) * alpha + EI1 * cosh(alpha * L1) * alpha) + 0.1e1 / (-J1 * sin(alpha * L1) * omega ^ 2 - J1 * sinh(alpha * L1) * omega ^ 2 + EI1 * cos(alpha * L1) * alpha + EI1 * cosh(alpha * L1) * alpha) * (J1 * cos(alpha * L1) * omega ^ 2 + EI1 * sin(alpha * L1) * alpha));];
    BB = AA(2:4,2:4);
    CC = -AA(2:4,1);  
    KK = inv(BB)*CC*A1;  
    B1 = KK(1);
    C1 = KK(2);
    D1 = KK(3);

    phi1 = A1*sin(alpha*x1) + B1*cos(alpha*x1) + C1*sinh(alpha*x1) + D1*cosh(alpha*x1);
    phi1_L1 = subs(phi1,x1,L1);
    d_phi1 = diff(phi1,x1);
    
    mx =  m1*int(phi1*phi1,x1,0,L1) + M1*phi1_L1*phi1_L1;  
    
    fid = fopen('Cantilever_mass_FindA1.m','w');
    fprintf(fid,'function f = Cantilever_mass_FindA1(A1)\n');
    fprintf(fid,'f = abs(');
    fprintf(fid,'%s);\n',char(-1+mx));
    fprintf(fid,'end');
    st = fclose(fid);
    [A1_temp, FVAL] = fminsearch('Cantilever_mass_FindA1',2);
    A1_i(mode) = A1_temp;
    
    clear A1 KK
    A1 = A1_i(mode);
    A1_r(mode) = A1;
    KK = inv(BB)*CC*A1;
    B1_r(mode) = KK(1);
    C1_r(mode) = KK(2);
    D1_r(mode) = KK(3);
end


for mode = 1:mode_find 
    
    omega = omega_a(mode);  
    alpha = alpha_a(mode);  
    
    A1 = A1_r(mode);
    B1 = B1_r(mode);
    C1 = C1_r(mode);
    D1 = D1_r(mode);

    x11 = 0:0.001:L1;
    phi_1 = A1*sin(alpha*x11) + B1*cos(alpha*x11) + C1*sinh(alpha*x11) + D1*cosh(alpha*x11);
    d_phi_1 = A1*cos(alpha*x11)*alpha - B1*sin(alpha*x11)*alpha + C1*cosh(alpha*x11)*alpha + D1*sinh(alpha*x11)*alpha;
    dd_phi_1 = -A1*sin(alpha*x11)*alpha^2 - B1*cos(alpha*x11)*alpha^2 + C1*sinh(alpha*x11)*alpha^2 + D1*cosh(alpha*x11)*alpha^2;

    % % % mode shape
%     figure
%     subplot(3,1,1)
%     plot(x11/max(x11),phi_1/max(phi_1))
%     subplot(3,1,2)
%     plot(x11/max(x11),d_phi_1/max(d_phi_1))
%     subplot(3,1,3)
%     plot(x11/max(x11),dd_phi_1/max(dd_phi_1))
%     min(dd_phi_1(1:length(x11)/2));
%     plot(x11/max(x11))

    acc = acc_g*9.81;
    for dir = 1:1
        if dir == 1
            dir_sign = 1;
        else
            dir_sign = -1;
        end
        freq_s_hz = omega_a(mode)/2/pi - dir_sign*5;
        freq_e_hz = omega_a(mode)/2/pi + dir_sign*5;
        rate = sign(freq_e_hz-freq_s_hz)*0.1; %Hz/sec
        t_end = abs((freq_s_hz-freq_e_hz)/rate); 

        div = 40;
        i_end = (freq_s_hz+freq_e_hz)/2*div*t_end;
        Duration = zeros(1,ceil(i_end));
        for i = 1:i_end
            freq_temp = 2*pi*(freq_s_hz + (freq_e_hz-freq_s_hz)/t_end*Duration(i));
            t_step = 2*pi/freq_temp/div;
            Duration(i+1) = Duration(i) + t_step;
        end
        ttofreq = freq_s_hz + rate*Duration;
        
        Duration = linspace(0,1,1e5); %<--------------------------------------------------------------------家览丁

        % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
        % % % % % % % % % % % %       Coeff of EOM            % % % % % % % % % % 

        clear mx
        A1 = A1_r(mode); B1 = B1_r(mode); C1 = C1_r(mode); D1 = D1_r(mode);
        omega = omega_a(mode);  
        alpha = alpha_a(mode);  

        phi1 = A1*sin(alpha*x1) + B1*cos(alpha*x1) + C1*sinh(alpha*x1) + D1*cosh(alpha*x1);
        phi1_L1 = subs(phi1,x1,L1);
        d_phi1 = diff(phi1,x1);
        d_phi1_Lp1 = subs(d_phi1,x1,Lp1);
        d_phi1_Lp2 = subs(d_phi1,x1,Lp2);
        dd_phi1 = diff(d_phi1,x1);
        dd_phi1_Lp1 = subs(dd_phi1,x1,Lp1);
        dddd_phi1 = diff(diff(dd_phi1,x1),x1);
  
        mx_i(mode) =  double(m1*int(phi1*phi1,x1,0,L1) + M1*phi1_L1*phi1_L1);
        kx_i(mode) = double(EI1*int(dd_phi1*dd_phi1,x1,0,L1));
        kx_i_2(mode) = double(EI1*int(dddd_phi1*phi1,x1,0,L1));
        vphi(mode) = double(vtheta*(d_phi1_Lp2-d_phi1_Lp1));
        Nr(mode) = double(m1*int(phi1,x1,0,L1)) + double(M1*phi1_L1);
        
        para = [Cp R vphi(mode) zeta omega Nr(mode) acc rate freq_s_hz];        
        [t,x] = ode45(@(t,in)Cantilever_mass_fcn(t,in,para),Duration,[0;0;0]); 

        dd_eta = 

        figure

        subplot(4,1,1)
        plot(t, x(:,1))
        xlabel('time')
        ylabel('eta')

        subplot(4,1,2)
        plot(t, x(:,2))
        xlabel('time')
        ylabel('d_eta')

        subplot(4,1,3)
        xlabel('time')
        ylabel('voltage')

        subplot(4,1,4)

        xlabel('time')
        ylabel('voltage')
%         figure; 
%         subplot(3,1,1);
%         plot(ttofreq,x(:,3))
%         str = sprintf('b1=%0.2f, M1=%0.3f, V_{max}=%2.3f',b1s,M1,max(x(:,3)));
%         title(str)
%         subplot(3,1,2);
%         plot(ttofreq,x(:,1)*double(phi1_L1)*1e3)
%         str = sprintf('b1=%0.2f, M1=%0.3f, d_{max}=%2.3f',b1s,M1,max(x(:,1)*double(phi1_L1)*1e3));
%         title(str)
%         nor_ddphi1 = max(x(:,1))*dd_phi_1/(max(x(:,1))*max(abs(dd_phi_1)));
%         subplot(3,1,3)
%         plot(x11,nor_ddphi1,[x11(1) x11(length(x11))],[nor_ddphi1(1) nor_ddphi1(length(x11))])
%         title('normalized strain')
        
        % % Linear
        omega1 = omega_a(mode);
        clear i omega
        omega = 5*2*pi:1e-3:100*2*pi;
        vphi = vphi(mode);
        N1 = Nr(mode);
        Zeta = zeta;

    end
%     XBddot = acc;
%     vv = (-1.*i) .* N1 .* R .* omega .* vphi ./ ((2 .* Cp .* R .* Zeta .* omega .^ 2 .* omega1) + (i) .* Cp .* R .* (omega .^ 3) + (-1.*i) .* Cp .* R .* omega .* (omega1 .^ 2) + (-1.*i) .* R .* omega .* vphi .^ 2 + (-2.*i) .* Zeta .* omega .* omega1 + (omega .^ 2) - (omega1 .^ 2)) .* XBddot;
%     etaeta = -N1 .* ((i) .* Cp .* R .* omega + 1) ./ (2 .* Cp .* R .* Zeta .* omega .^ 2 .* omega1 + (i) .* Cp .* R .* omega .^ 3 + (-1.*i) .* Cp .* R .* omega .* omega1 .^ 2 + (-1.*i) .* R .* omega .* vphi .^ 2 + (-2.*i) .* Zeta .* omega .* omega1 + omega .^ 2 - omega1 .^ 2) .* XBddot;
%     figure;
%     subplot(2,1,1)
%     plot(omega/2/pi,abs(vv))
%     xlabel('Freqency')
%     ylabel('Voltage')
%     str = sprintf('V_{max}=%2.3f',max(abs(vv)));
%     title(str)
%     subplot(2,1,2)
%     plot(omega/2/pi,abs(etaeta)*double(phi1_L1)*1e3)
%     xlabel('Frequency')
%     ylabel('Displacement (mm)')
%     set(findall(gcf,'-property','FontSize'),'FontSize',12)
    
    % strain
%     max(abs(etaeta))*max(double(dd_phi1_Lp1))*(h1c-h1p/2);
end

toc