import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import sin, sinh, cos, cosh
from math import pi, ceil, floor
from scipy.optimize import root, fmin
from scipy.integrate import solve_ivp
import time

## Unit: kg, m, s

omega = sp.symbols( 'omega' )

acc_g = 1                    ## exicitation accleration, unit: G
M1 = 11 * 1e-6               ####### 26g droplet weight (kg)
J1 = M1 * ( 5e-3 )**2        ## moment of inertia

Ys = 0.5 * 1e9               ####### substrate young's modulus
rho_s = 1390                 ####### substarte density
Yp = 2.45 * 1e9              ## piezo young's moudulus
rho_p = 1780                 ## piezo density
d31 = -190e-12               ## piezoelectric constant(m/V) = -270*1e-12
epsilon = 15.93e-9           ## 3500*8.854*1e-12 # F/m

h1s = 0.5 * 1e-3           ####### substrate thickness(m)
h1p = 0.028 * 1e-3           ## piezo thickness(m)
L1 = 20 * 1e-3               ####### substrate length(m)
Lp1 = 0 * L1                 ## 0 means the piezo cover from the fixed end
Lp2 = 1 * L1                 ## piezo length
b1s = 13 * 1e-3              ####### substrate width
b1p = b1s                    ## piezo width
b1pp = b1p                   ## electrode width

# Cp = b1pp * ( Lp2-Lp1 ) * epsilon / h1p
Cp = 0.5 * 1e-9              ####### capacitance (F)

R = 1 * 1e6                  ## load resistance
zeta = 0.043                  ####### damping ratio

mode_find = 1                ## how many modes to be found

m1 = rho_s*b1s*h1s + rho_p*b1p*h1p

n = Ys * b1s / ( Yp*b1p )

h1pa = ( h1p**2 + 2*n*h1p*h1s + n * h1s**2 ) / 2 / ( h1p + n*h1s )
h1sa = ( h1p**2 + 2*h1p*h1s + n * h1s**2 ) / 2 / ( h1p + n*h1s )
h1pc = n * h1s * ( h1p+h1s ) / 2 / ( h1p + n*h1s )

h1a = -h1sa
h1b = h1pa - h1p
h1c = h1pa
EI1 = b1s / 3 * Ys * ( h1b**3 - h1a**3 ) + b1p / 3 * Yp * ( h1c**3 - h1b**3 )

# vtheta = Yp*d31*b1pp*h1pc
vtheta = 0.85e-7                #######

alpha = ( omega**2 * m1 / EI1 )**( 1 / 4 )

#### XX = [A1 B1 C1 D1]', AA*XX = CC, CC = 0
## with Mass


def Cantilever_mass_fcn( t, in_, para ):
    [ Cp, R, vphi, zeta, omega, Nr, acc, rate, freq_s_hz ] = para
    xb1ddot = acc * sin( ( freq_s_hz + rate*t/2 ) * 2 * pi * t )

    eta1 = in_[ 0 ]
    eta1dot = in_[ 1 ]
    v2 = in_[ 2 ]
    force = 0.08*24
    eta1ddot = -2 * zeta * omega * eta1dot - omega**2 * eta1 - vphi*v2 - force
    v2dot = -1 / ( Cp*R ) * v2 + vphi/Cp*eta1dot

    return [ eta1dot, eta1ddot, v2dot ]


def get_AA( alpha, omega ) -> np.ndarray:
    aa = np.array(
        [
            [ 0, 1, 0, 1 ], [ alpha, 0, alpha, 0 ],
            [
                -alpha * ( J1 * cos( alpha * L1 ) * omega**2 + EI1 * sin( alpha * L1 ) * alpha ),   # A2*sin(aL)
                -alpha * ( -J1 * sin( alpha * L1 ) * omega**2 + EI1 * cos( alpha * L1 ) * alpha ),  # A1*cos(aL)
                -alpha * ( J1 * cosh( alpha * L1 ) * omega**2 - EI1 * sinh( alpha * L1 ) * alpha ), # A4*sinh(aL)
                -alpha * ( J1 * sinh( alpha * L1 ) * omega**2 - EI1 * cosh( alpha * L1 ) * alpha )],# A3*cosh(aL)
            [
                -EI1 * cos( alpha * L1 ) * alpha**3 + M1 * sin( alpha * L1 ) * omega**2,            # A2*sin(aL)
                EI1 * sin( alpha * L1 ) * alpha**3 + M1 * cos( alpha * L1 ) * omega**2,             # A1*cos(aL)
                EI1 * cosh( alpha * L1 ) * alpha**3 + M1 * sinh( alpha * L1 ) * omega**2,           # A4*sinh(aL)
                EI1 * sinh( alpha * L1 ) * alpha**3 + M1 * cosh( alpha * L1 ) * omega**2            # A3*cosh(aL)
                ]
            ]
        )
    return aa


def get_phi( x, alpha, A1, B1, C1, D1 ):
    return A1 * sin( alpha * x ) + B1 * cos( alpha * x ) + C1 * sinh( alpha * x ) + D1 * cosh( alpha * x )


AA = get_AA( alpha, omega )
AA = sp.Matrix( AA ).det()

Cantilever_mass_det = lambda x: float(AA.subs( omega, x[0] ))

guess = 88.54445401441941   ##2 * 2 * pi * 10
incre = 1.1
omega_a = np.zeros( mode_find )        ###################!!!!!!!!!!!!!改動，原為高度為一的二維陣列，改為一維
options = { 'maxiter': 1000, 'xtol': 1e-14, 'ftol': 1e-14 }
get_omega_a = lambda in_guess: root( Cantilever_mass_det, in_guess, method='lm', options=options ).get( 'x' )

for i in range( mode_find ):
    if i == 0:
        omega_a[ i ] = get_omega_a( guess )
        while omega_a[ i ] < 0.001:    #################!!!!!!!!!!!!!!改動，原為<0，改為<0.001，否則找不到root
            guess = guess * incre
            omega_a[ i ] = get_omega_a( guess )
    else:
        guess = guess * incre
        omega_a[ i ] = get_omega_a( guess )
        while ( omega_a[ i ] - omega_a[ i - 1 ] ) < 0.001:
            guess = guess * incre
            omega_a[ i ] = get_omega_a( guess )

print(omega_a / 2 / pi)             #################!!!!!!!!!!!!沒有回傳到任何變數?是不是要改omega_a = omega_a/2/pi
alpha_a = ( omega_a**2 * m1 / EI1 )**( 1 / 4 )
print(alpha_a)

A1_r = np.arange( mode_find, dtype='float' )
B1_r = np.arange( mode_find, dtype='float' )
C1_r = np.arange( mode_find, dtype='float' )
D1_r = np.arange( mode_find, dtype='float' )
mx_i = np.arange( mode_find, dtype=np.float64 )
kx_i = np.arange( mode_find, dtype=np.float64 )
kx_i_2 = np.arange( mode_find, dtype=np.float64 )
vphi = np.arange( mode_find, dtype=np.float64 )
Nr = np.arange( mode_find, dtype=np.float64 )

for mode in range( mode_find ):
    A1 = sp.symbols( 'A1' )
    x1 = sp.symbols( 'x1' )

    omega = omega_a[ mode ]
    alpha = alpha_a[ mode ]

    AA = get_AA( alpha, omega )
    AA = AA.astype( np.float64 )
    BB = AA[ 1:4, 1:4 ]
    CC = -AA[ 1:4, 0 ]

    KK = np.dot( np.linalg.inv( BB ), CC ) * A1
    print(KK)
    B1 = KK[ 0 ]
    C1 = KK[ 1 ]
    D1 = KK[ 2 ]

    phi1 = get_phi( x1, alpha, A1, B1, C1, D1 )
    phi1_L1 = phi1.subs( x1, L1 )
    d_phi1 = sp.diff( phi1, x1 )

    print(time.time())
    mx = m1 * sp.integrate( phi1 * phi1, ( x1, 0, L1 ) ) + M1*phi1_L1*phi1_L1 + d_phi1.subs( x1, L1 )**2*J1
    print(time.time())
    print(phi1)
    print(mx)

    A1 = fmin( lambda x: abs( mx.subs( A1, x[0] ) - 1 ), x0=2 )
    A1_r[ mode ] = A1
    KK = np.dot( np.linalg.inv( BB ), CC ) * A1
    B1_r[ mode ] = KK[ 0 ]
    C1_r[ mode ] = KK[ 1 ]
    D1_r[ mode ] = KK[ 2 ]
    print(A1_r[0], B1_r[0], C1_r[0], D1_r[0])

for mode in range( mode_find ):
    acc = acc_g * 9.81
    for dir in range( 1 ):
        if dir == 0:
            dir_sign = 1
        else:
            dir_sign = -1

        freq_s_hz = omega_a[ mode ] / 2 / pi - dir_sign*5
        freq_e_hz = omega_a[ mode ] / 2 / pi + dir_sign*5
        rate = np.sign( freq_e_hz - freq_s_hz ) * 0.1
        t_end = abs( ( freq_s_hz-freq_e_hz ) / rate )

        div = 40
        i_end = ( freq_s_hz+freq_e_hz ) / 2 * div * t_end
        Duration = np.zeros( ( ceil( i_end ), ), dtype=float )
        for i in range( 0, floor( i_end ) ):
            freq_temp = 2 * pi * ( freq_s_hz + ( freq_e_hz-freq_s_hz ) / t_end * Duration[ i ] )
            t_step = 2 * pi / freq_temp / div
            Duration[ i + 1 ] = Duration[ i ] + t_step

        ttofreq = freq_s_hz + rate*Duration

        Duration = np.linspace( 0, 1, 1000 )
        step = 1e-3
        '''''' '''''' '''''' '''''' 'Coeff of EOM' '''''' '''''' '''''' ''''''

        omega = omega_a[ mode ]
        alpha = alpha_a[ mode ]

        A1 = A1_r[ mode ]
        B1 = B1_r[ mode ]
        C1 = C1_r[ mode ]
        D1 = D1_r[ mode ]
        print(A1, B1, C1, D1)
        x1 = sp.symbols( 'x' )
        phi1 = get_phi( x1, alpha, A1, B1, C1, D1 )
        phi1_L1 = phi1.subs( x1, L1 )
        d_phi1 = sp.diff( phi1, x1 )
        print(d_phi1)
        d_phi1_Lp1 = d_phi1.subs( x1, Lp1 )
        d_phi1_Lp2 = d_phi1.subs( x1, Lp2 )
        #dd_phi1 = sp.diff( phi1, x1 )
        #dd_phi1_Lp1 = dd_phi1.subs( x1, Lp1 )
        #dddd_phi1 = sp.diff( sp.diff( dd_phi1, x1 ), x1 )

        #mx_i[ mode ] = m1 * sp.integrate( phi1**2, ( x1, 0, L1 ) ) + M1*phi1_L1*phi1_L1
        #kx_i[ mode ] = EI1 * sp.integrate( dd_phi1**2, ( x1, 0, L1 ) )
        #kx_i_2[ mode ] = EI1 * sp.integrate( dddd_phi1 * phi1, ( x1, 0, L1 ) )
        print(d_phi1_Lp1, d_phi1_Lp2)
        vphi[ mode ] = vtheta * ( d_phi1_Lp2-d_phi1_Lp1 )       # vtheta is piezoelectronic coupling factor the chi_i in (3.1.51)
                                                                # vtheta is piezoelectronic coupling factor the chi_i in (3.1.51)
        #Nr[ mode ] = m1 * sp.integrate( phi1, ( x1, 0, L1 ) ) + M1*phi1_L1

        para = [ Cp, R, vphi[ mode ], zeta, omega, Nr[ mode ], acc, rate, freq_s_hz ]

        print(para)
        r = solve_ivp( Cantilever_mass_fcn, [ 0, 0.2 ], y0=[ 0, 0, 0 ], args=[ para ], t_eval=np.linspace( 0, 0.2, 1000 ))

        print( r.y[ 0 ] )
        print( r.y[ 2 ] )
        figure, ax = plt.subplots( 2, 1 )
        ax[ 0 ].plot( r.t, r.y[ 0 ] * phi1_L1 * 1e3 )
        ax[ 0 ].set_ylabel( 'Displacement' )
        ax[ 0 ].set_xlabel( 'time' )
        ax[ 1 ].plot( r.t, r.y[ 2 ] )
        ax[ 1 ].set_ylabel( 'Voltage' )
        ax[ 1 ].set_xlabel( 'time' )
        plt.savefig( 'Cantilever_mass_fcn_matlab.png' )
