import numpy as np 


# earth rotation rate
earthRotRate = 7.292115e-5 # rad/s

# Earth grav param 
u = 3.986004418e5 #uses km^3*s^-2  

# Earth Radius 
earthRadius=6378137 # m

def missDistance(lat1,lon1, lat2,lon2):
    
    # spherical Earth miss distance
    # via Haverstein formulation 
    
    # The average radius of the Earth in km.
    R = earthRadius/1e3
    # The start point.
    la1 = lat1; lo1 = lon1
    # The end point.
    la2 = lat2; lo2 = lon2
    # Convertion factor from degree to radian.
    d2r = np.pi / 180
    dla = (la2-la1) * d2r
    dlo = (lo2-lo1) * d2r
    a = np.sin(dla/2) ** 2 + np.cos(la1*d2r) * np.cos(la2*d2r) * np.sin(dlo/2) ** 2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    d1 = R * c
    
    return d1


def dist(la1, lo1, la2, lo2, R=earthRadius/1e3, km=True):
    d2r = np.pi / 180
    dla = (la2-la1) * d2r
    dlo = (lo2-lo1) * d2r
    a = np.sin(dla/2) ** 2 + np.cos(la1*d2r) * np.cos(la2*d2r) * np.sin(dlo/2) ** 2
    c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
    if km:
        return R * c
    return c

def faz(lat1,lon1, lat2,lon2):

    d2r = np.pi / 180
    dla = (lat2-lat1) * d2r
    dlo = (lon2-lon1) * d2r

    yy = np.sin(dlo) * np.cos(lat2*d2r)
    xx = np.cos(lat1*d2r) * np.sin(lat2*d2r) - np.sin(lat1*d2r) * np.cos(lat2*d2r) * np.cos(dlo)
    th = np.arctan2(yy, xx)
    az1 = (th / d2r + 360) % 360
    
    return az1

def dest_point(la1, lo1, az, delta):
    d2r = np.pi / 180
    la1 *= d2r
    lo1 *= d2r
    az *= d2r
    delta *= d2r
    lad = np.arcsin(np.sin(la1)*np.cos(delta)+np.cos(la1)*np.sin(delta)*np.cos(az))
    lod = lo1 + np.arctan2(np.sin(az)*np.sin(delta)*np.cos(la1), np.cos(delta)-np.sin(la1)*np.sin(lad))
    return lad/d2r, lod/d2r

def ellipticalTOF(u,e,a,theta1,theta2,k):
    
    k = int(k) 

    theta1Rad = np.deg2rad(theta1)
    theta2Rad = np.deg2rad(theta2)

    # The following formula is taken from Class Notes but
    # it requires that E and theta be in the same half plane..
    #Eo =  np.arccos((e+np.cos(theta1Rad))/(1+e*np.cos(theta1Rad)))
    #E = np.arccos((e+np.cos(theta2Rad))/(1+e*np.cos(theta2Rad)))
    
    #tof = np.sqrt((a*a*a)/u) * \
    #    (2*np.pi*k + (E-e*np.sin(E)) - (Eo-e*np.sin(Eo)))
    
    
    # Taken from "Orbital Dyanmics" - CURTIS (eq: 3.10b)

    Eo = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(theta1Rad/2))
    E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(theta2Rad/2))
    if E < Eo:
        E = E +2*np.pi

    tof = np.sqrt((a*a*a)/u) * \
            (E - Eo + e*(np.sin(Eo) - np.sin(E)))

    return tof


def LLAtoECXSphere(latitude,longitude,altitude,earthRadius=6378137, degreesIn=True):
    # lat:  interial latitude 
    # lon: geocentric longitude
    #      if lon is inertial, the position vector is in ECI
    #      if lon is Earth Fixed, the position vector is in ECEF
    # earthRadius:  default value: 6378137 m, wgs84 equatorial radius
    
    lat = latitude
    lon = longitude
    if degreesIn:
        lat = np.deg2rad(latitude)
        lon = np.deg2rad(longitude)
    
    alt = altitude + earthRadius
    rx = alt*np.cos(lat)*np.cos(lon)
    ry = alt*np.cos(lat)*np.sin(lon)
    rz = alt*np.sin(lat)
    
    return np.array([rx,ry,rz])
    
def ECI_2_ECF_DCM (time, omega=7.292115e-5):
    # time: s
    # omega: earth rot rate, rad/s
    #        default: wgs84 value: 
    #        7.292115 x 10^-5 rad/s
    
    wt = omega*time
    eci2ecf = np.array([[np.cos(wt), -np.sin(wt), 0],
                        [np.sin(wt),  np.cos(wt), 0],
                        [0         ,  0         , 1]])
    return eci2ecf

    
def ECF_2_ENU_DCM (latitude, longitude, degreesIn=True):

    lat = latitude
    lon = longitude
    if degreesIn:
        lat = np.deg2rad(latitude)
        lon = np.deg2rad(longitude)
        
    ecf2enu = np.array([[-np.sin(lon), np.cos(lon), 0],
                        [-np.sin(lat)*np.cos(lon),  -np.sin(lat)*np.sin(lon), np.cos(lat)],
                        [np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return ecf2enu

def FAVtoENUSpeed(flightPathAngle, azimuth, speed, degreesIn=True):
    
    # flightPathAngle: geodetic fpa (== geocentric for spherical earth)
    # azimuth: clock angle from North in NED, (+) for right hand (D)own rotation
    # speed: positive speed magnitude 
    
    
    fpa = flightPathAngle
    azim = azimuth
    
    if degreesIn:
        fpa = np.deg2rad(fpa)
        azim = np.deg2rad(azim)
        
    v3 = speed*np.sin(fpa)
    v2 = np.sqrt( (speed*speed)-(v3*v3) ) * np.cos(azim)
    v1 = v2*np.tan(azim)
    
    return np.array([v1,v2,v3])


# for odeInt args should be:  inState, t, *args
# for solve_ivp, should be:  t, inState, *args
def prop2Body(t,inState, u):
    from math import sqrt
    #system of first order equations
    rx = inState[0]; ry = inState[1]; rz = inState[2]
    vx = inState[3]; vy = inState[4]; vz = inState[5]
    
    rNorm = sqrt( rx*rx + ry*ry + rz*rz )
    ax = (-u*rx)/(rNorm*rNorm*rNorm)
    ay = (-u*ry)/(rNorm*rNorm*rNorm)
    az = (-u*rz)/(rNorm*rNorm*rNorm)
    
    outState = [vx , vy, vz, ax, ay, az]
    return outState


# --------------------------------------------------

#   Lambert Problem Algorithm and Stumpf Equations

# --------------------------------------------------
def stumpf(Z):
    # Vollado pg 230
    if Z > 1e-6:
        C = (1-np.cos(np.sqrt(Z)))/Z
        S = (np.sqrt(Z) - np.sin( np.sqrt(Z) ))/ np.sqrt(Z*Z*Z)
        
    else:
        if Z < -1e-6:
            C = (1 - np.cosh( np.sqrt(-Z) )) / Z
            S = (np.sinh( np.sqrt(-Z) )  - np.sqrt(-Z)) / np.sqrt((-Z)**3)
        else:
            C = 1/2
            S = 1/6
            
    return C, S  # C = c2 , S = c3


def lambo(r1, r2, tof, tm, u, errTol=1e-6):
    # r1 and r2 should be np arrays
    # tm is +1: Short Way,  -1: Long way

    r1Mag = np.linalg.norm(r1)
    r2Mag = np.linalg.norm(r2)
    
    cos_dnu = np.dot(r1,r2)/(r1Mag*r2Mag)
    sin_dnu = tm*np.sqrt(1-(cos_dnu*cos_dnu))
    
    A = tm * np.sqrt(r1Mag*r2Mag*(1+cos_dnu))
    
    if A == 0:
        print("ERROR: we can't calculate the orbit")
        return

    psi_n = 0
    c2 = 1/2
    c3 = 1/6
    
    psi_up = 4*np.pi*np.pi
    psi_low = -4*np.pi        
    
    
    # trying out expanding the psi_low bound.. per a note from office hours
    #psi_low = -4*np.pi*np.pi
    

    y_n = r1Mag + r2Mag + (A*(psi_n*c3-1))/(np.sqrt(c2))
    
    if A > 0 and y_n < 0 :
        # readjust psi_low until y > 0 
        # make bigger negative value
        print("readjust psi_low until y > 0 ")
        return
    xo = np.sqrt(y_n/c2)
    
    del_t = (xo*xo*xo*c3 + A*np.sqrt(y_n)) / np.sqrt(u)
    
    counter = 0 
    while abs( del_t - tof) >= errTol:
        counter+=1
    
        y_n = r1Mag + r2Mag + (A*(psi_n*c3-1))/(np.sqrt(c2))

        if A > 0 and y_n < 0 :
            # readjust psi_low until y > 0 
            # make bigger negative value
            print("readjust psi_low until y > 0 ")
            return
        xo = np.sqrt(y_n/c2)

        del_t = (xo*xo*xo*c3 + A*np.sqrt(y_n)) / np.sqrt(u)
        
        if del_t <= tof:
            psi_low = psi_n
        else:
            psi_up = psi_n;
        
        #psi_n+1
        psi_n = (psi_up + psi_low)/2;
        
        # find c2,c3 based on the new psi_n+1
        c2,c3 = stumpf(psi_n)
    #print("converged after # iterations: ",counter)
    
    f = 1 - y_n/r1Mag
    g_dot = 1 - y_n/r2Mag
    g = A * np.sqrt(y_n/u)
    
    v1 = (r2 - f*r1)/g
    v2 = (g_dot*r2 - r1)/g
    
    return v1, v2



def keplerProblem(ro,vro,a,t,u, r_gce, v_gce):
    
    #print(">>> Running Kepler Problem Solver <<<")

    # initial guess for chi (X)
    X = t*np.sqrt(u)*np.abs(1/a)
    #print("\tInitial Guess X (chi): ",X)
    
    
    r0 =  ro
    v0 =  vro
    
    
    Z = X*X/a
        
    C,S = stumpf(Z)
    
    rdotv = np.dot(r_gce,v_gce)
    tn = (1/np.sqrt(u)) * ( (X*X*X*S) + (rdotv/np.sqrt(u))*(X*X*C) + r0*X*(1-Z*S) )
    
    counter = 0 
    tolerance = 0.00001

    while ( np.abs(t-tn)>tolerance ):
        counter += 1
        
        dtdx = X*X*C + (rdotv/np.sqrt(u))*X*(1-Z*S) + r0*(1-Z*C)
        
        X = X + (t-tn)/dtdx
        
        Z = X*X/a
        
        C,S = stumpf(Z)

        tn = (1/np.sqrt(u)) * ( (X*X*X*S) + (rdotv/np.sqrt(u))*(X*X*C) + r0*X*(1-Z*S) )

    #print("\tSolver converged, Total Steps: ", counter )
    #print("\tFINAL X (chi) value: ", X)
    
    f = 1 - (X*X/r0)*C
    g = t - (X*X*X/np.sqrt(u))*S
    
    position = f*r_gce + g*v_gce
    posMag = np.linalg.norm(position)
    f_dot = (np.sqrt(u)/(r0*posMag))*X*(Z*S-1)
    g_dot = 1 - (X*X/posMag)*C
    velocity = f_dot*r_gce + g_dot*v_gce
    velMag  = np.linalg.norm(velocity)
    
    #print('\t--- New State after %f (s) Time of FLight --- '%(t))
    
    #print('\tPosition (km): ', position)
    #print('\tPosition Mag (km): ', posMag)
    #print('\tVelocity (km/s): ', velocity)
    #print('\tVelocity Mag (km/s): ', velMag)
    

    # TODO :
    # for the first example.. took thousands of steps..
    # so i may need to try a different dot product 
    # and vector selection,.. not eci .. 

    return position, velocity

# Code Cell #1

def cartesian2OrbitalElements(u, rx, ry, rz, vx, vy, vz):
    '''
    This function converts Cartersian coordinates to orbital elements
    
    Inputs:
    --------
    u: Gravitational parameter (km^3/s^2
    rx, ry, rz : Position in Geocentric Equatorial Frame (km)
    vx, vy, vz : Velocity in Geocentric Equatorial Frame (km/s)    
    
    Outputs:
    --------
    a: Semi-major axis (km)
    e: Eccentricity
    i: Inclination (deg)
    omega: Right ascension of the ascending node (deg)
    w: Argument of periapsis
    theta: True anonmaly (deg)
    '''
    from math import isclose, sqrt, sin, cos, pi
    from numpy import arccos, rad2deg
    import numpy as np 
    
    # -------- position --------
    r = np.array([rx, ry, rz])
    rNorm = np.linalg.norm(r)

    # -------- velocity --------
    v = np.array([vx, vy, vz])
    vNorm = np.linalg.norm(v)
    vr = np.dot(r,v)/rNorm

    # -------- angular momentum --------
    h = np.cross(r,v)
    hNorm = np.linalg.norm(h)
    
    # -------- line of nodes --------
    z_hat = np.array([0,0,1])
    n = np.cross(z_hat, h)
    nNorm = np.linalg.norm(n)

    # -------- eccentricity --------
    e = (1/u)*(((vNorm*vNorm) - (u/rNorm))*r - np.dot(r,v)*v)
    eNorm = np.linalg.norm(e)
    # scalar formulation
    #eNorm = (1/u)*sqrt( (2*u-rNorm*vNorm*vNorm)*rNorm*vr*vr + pow((u-rNorm*vNorm*vNorm),2 ))

    # -------- specific orbital energy --------
    epsilon = ((vNorm*vNorm)/2) - (u/rNorm)
    
    # -------- semimajor axis --------
    # TODO: update eNorm unity check to use an isClose() to one
    a = np.Infinity if eNorm == 1 else -u/(2*epsilon)
    
    # -------- inclination --------
    i = arccos(h[2]/hNorm)
    
    # ------- RAAN ---------
    num = n[0]
    denom = nNorm
    temp1 = num/denom
    if isclose(num/denom, -1, rel_tol=1e-13, abs_tol=1e-13): temp1 = -1
    if isclose(num/denom,  1, rel_tol=1e-13, abs_tol=1e-13): temp1 =  1
    #omega = arccos(n[0]/nNorm) if n[1] >= 0 else (2*pi - arccos(n[0]/nNorm))
    omega = arccos(temp1) if n[1] >= 0 else (2*pi - arccos(temp1))

    
    # -------- argument of periapsis --------
    #w = arccos(np.dot(n,e)/(nNorm*eNorm)) if e[2]>=0 else (2*pi - arccos(np.dot(n,e)/(nNorm*eNorm)))
    num = np.dot(n,e)
    denom = nNorm*eNorm
    temp1 = num/denom
    if isclose(num/denom, -1, rel_tol=1e-13, abs_tol=1e-13): temp1 = -1
    if isclose(num/denom,  1, rel_tol=1e-13, abs_tol=1e-13): temp1 =  1
    w = arccos(temp1) if e[2]>=0 else (2*pi - arccos(temp1))

    # explicit handling of special case: Elliptical Equatorial
    if (0<eNorm<1) and isclose(i, 0, rel_tol=1e-13, abs_tol=1e-13):
        # vernal equinox unit vector in geocentric equatorial frame 
        I = np.array([1,0,0])
        num = np.dot(I,e)
        denom = eNorm
        temp1 = num/denom
        if isclose(num/denom, -1, rel_tol=1e-13, abs_tol=1e-13): temp1 = -1
        if isclose(num/denom,  1, rel_tol=1e-13, abs_tol=1e-13): temp1 =  1
        w = arccos(temp1) if e[1]<0 else (2*pi - arccos(temp1))
    
    
    # -------- true anomaly --------
    #original; implementation: 
    #theta = arccos(np.dot(e,r)/(eNorm*rNorm)) if np.dot(r,v) >= 0  else (2*pi - arccos(np.dot(e,r)/(eNorm*rNorm)))
    # new implementation to protect against precision issues
    num = np.dot(e,r)
    denom = eNorm*rNorm
    temp2 = num/denom
    if isclose(num/denom, -1, rel_tol=1e-15, abs_tol=1e-15): temp2 = -1
    if isclose(num/denom,  1, rel_tol=1e-15, abs_tol=1e-15): temp2 =  1
    theta = arccos(temp2) if np.dot(r,v) >= 0  else (2*pi - arccos(temp2))
    
    # TODO: add handling for special case orbits (see vallado text)
    
    '''
    
    print("====")
    print('e enorm: ',eNorm)
    print('e vec: ',e)
    print('r dot v : ', np.dot(r,v))
    print('radial v : ', vr)
    print('eNorm*rNorm: ',eNorm*rNorm)
    print(' e dot r : ',np.dot(e,r))
    print('arccos(1): ',arccos(1))
    print('arccos(np.dot(e,r)/(eNorm*rNorm)) : ',arccos(np.dot(e,r)/(eNorm*rNorm)) )
    
    from fractions import Fraction as frac
    print("frac(e dot r): ",frac(np.dot(e,r)))
    print("eNorm*rNorm: ",frac(eNorm*rNorm))
    print("(np.dot(e,r)/(eNorm*rNorm)) : ", (np.dot(e,r)/(eNorm*rNorm)))
    print("num: ",num)
    print("denom:", eNorm*rNorm)
    print("====")
    '''
    
    return a, e, rad2deg(i), rad2deg(omega), rad2deg(w), rad2deg(theta)


# Code Cell #2

def orbitalElements2Cartesian(u, a, e, i, omega, w, v):
    '''
    This function converts orbital elements to Cartersian coordinates
    
    Inputs:
    --------
    u: Gravitational parameter (km^3/s^2
    a: Semi-major axis (km)
    e: Eccentricity
    i: Inclination (deg)
    omega: Right ascension of the ascending node (deg)
    w: Argument of periapsis
    v: True anonmaly (deg)
    
    Outputs: (units depend on inputs)
    --------
    r_gce: Position in Geocentric Equatorial Frame (km)
    v_gce: Velocity in Geocentric Equatorial Frame (km/s)
    
    '''
    from math import sqrt, sin, cos
    from numpy import deg2rad
    import numpy as np 
    
    v_rad, i_rad, omega_rad, w_rad = deg2rad(v), deg2rad(i), deg2rad(omega), deg2rad(w)
    
    p = a*(1-(e*e)) # Semilatus rectum  
    h = sqrt(p*u)   # specific angular momentum 
    
    # Perifocal position magnitude (km)
    r = p/(1+e*cos(v_rad))

    # Position in Perifocal Frame (km)
    r_pf = np.array([ r*cos(v_rad),  r*sin(v_rad), 0])
    
    # Velocity in Perifocal Frame (km)
    v_pf =  np.array([ -(u/h)*sin(v_rad) ,  (u/h)*(e+cos(v_rad)) , 0 ])

    # Perifocal to Geocentric Equatorial Frame DCM
    r11 = cos(omega_rad)*cos(w_rad) - sin(omega_rad)*sin(w_rad)*cos(i_rad)
    r12 = -cos(omega_rad)*sin(w_rad) - sin(omega_rad)*cos(i_rad)*cos(w_rad)
    r13 = sin(omega_rad)*sin(i_rad)
    r21 = sin(omega_rad)*cos(w_rad) + cos(omega_rad)*cos(i_rad)*sin(w_rad)
    r22 = -sin(omega_rad)*sin(w_rad) + cos(omega_rad)*cos(i_rad)*cos(w_rad)
    r23 = -cos(omega_rad)*sin(i_rad)
    r31 = sin(i_rad)*sin(w_rad)
    r32 = sin(i_rad)*cos(w_rad)
    r33 = cos(i_rad)
    
    R =  np.array([[r11, r12, r13],
                   [r21, r22, r23],
                   [r31, r32, r33]])

    # Geocentric Equatorial Position and Velocity (units depend on inputs)
    r_gce = np.matmul(R, r_pf) 
    v_gce = np.matmul(R, v_pf) 
    
    return r_gce, v_gce



def prop2Body2Impact(t, inState, u):#, earthRadius):
    from math import sqrt
    #system of first order equations
    rx = inState[0]; ry = inState[1]; rz = inState[2]
    vx = inState[3]; vy = inState[4]; vz = inState[5]
    
    rNorm = sqrt( rx*rx + ry*ry + rz*rz )
    ax = (-u*rx)/(rNorm*rNorm*rNorm)
    ay = (-u*ry)/(rNorm*rNorm*rNorm)
    az = (-u*rz)/(rNorm*rNorm*rNorm)
    
    #if rNorm <= earthRadius:
    #    print("Impact at time t")
    #    return 
    
    outState = [vx , vy, vz, ax, ay, az]
    return outState

def stop_condition(t, inState,u):
    from math import sqrt

    earthRadius_km = earthRadius/1e3  # km
    
    rx = inState[0]; ry = inState[1]; rz = inState[2]
    rNorm = sqrt( rx*rx + ry*ry + rz*rz )

    distToImpact = rNorm - earthRadius_km
    
    return distToImpact

stop_condition.terminal = True
stop_condition.direction = -1
