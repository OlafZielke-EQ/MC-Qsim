# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <float.h>

/* Calculate main dislocation contribution to strains and stresses */
void       TDstressFS_inStrnHS(double StsMS[6], double StrMS[6],  double X, double Y, double Z, double P1[3], double P2[3], double P3[3], double Ss, double Ds, double Ts, double mu, double lambda);
/* Calculate harmonic function contribution to strains and stresses */
void TDstress_HarFunc_inStrnHS(double StsFSC[6],double StrFSC[6], double X, double Y, double Z, double P1[3] ,double P2[3], double P3[3], double Ss, double Ds, double Ts, double mu, double lambda);

void       CoordTrans_inStrnHS(double newVal[3], double x_shift, double y_shift, double z_shift, double RotMat[3][3]); 

void      TriModeFind_inStrnHS(int TrimMode[1], double x,double y,double z,double p1_a,double p1_b, double p2_a,double p2_b,double p3_a,double p3_b);

void         TDSetupS_inStrnHS(double x,double y,double z,double alpha,double bx,double by,double bz,double nu, double TriVertex[3],double SideVec[3],double e[6]);

void     AngDisStrain_inStrnHS(double x, double y, double z, double alpha, double bx, double by, double bz, double nu, double e[6]);

void        TensTrans_inStrnHS(double e_out[6], double e_in[6], double B[3][3]);

void    AngSetupFSC_S_inStrnHS(double Stress1[6],double Strain1[6], double X,double Y,double Z,double bX,double bY,double bZ,double Pt1[3], double Pt2[3], double mu,double lambda);

void  AngDisStrainFSC_inStrnHS(double y1, double y2, double y3, double beta, double b1, double b2, double b3, double nu, double a, double Strain[6]);


void StrainHS_Nikkhoo(float Stress[6], float Strain[6], float fX, float fY, float fZ, float fP1[3], float fP2[3], float fP3[3], float fSs, float fDs, float fTs, const float fmu, const float flambda)

/*
this function is translated by Olaf Zielke from Matlab to C
% TDstressHS 
% Calculates stresses and strains associated with a triangular dislocation 
% in an elastic half-space.
%
% TD: Triangular Dislocation
% EFCS: Earth-Fixed Coordinate System
% TDCS: Triangular Dislocation Coordinate System
% ADCS: Angular Dislocation Coordinate System
% 
% INPUTS
% X, Y and Z: 
% Coordinates of calculation points in EFCS (East, North, Up). X, Y and Z 
% must have the same size.
%
% P1,P2 and P3:
% Coordinates of TD vertices in EFCS.
% 
% Ss, Ds and Ts:
% TD slip vector components (Strike-slip, Dip-slip, Tensile-slip).
%
% mu and lambda:
% Lame constants.
%
% OUTPUTS
% Stress:
% Calculated stress tensor components in EFCS. The six columns of Stress 
% are Sxx, Sxy, Sxz, Syy, Syz and Szz, respectively. The stress components 
% have the same unit as Lame constants.
%
% Strain:
% Calculated strain tensor components in EFCS. The six columns of Strain 
% are Exx, Eyy, Ezz, Exy, Exz and Eyz, respectively. The strain components 
% are dimensionless.
% 
% 
% Example: Calculate and plot the first component of stress tensor on a  
% regular grid.
% 
% [X,Y,Z] = meshgrid(-3:.02:3,-3:.02:3,-5);
% [Stress,Strain] = TDstressHS(X,Y,Z,[-1 0 0],[1 -1 -1],[0 1.5 -2],
% -1,2,3,.33e11,.33e11);
% h = surf(X,Y,reshape(Stress(:,1),size(X)),'edgecolor','none');
% view(2)
% axis equal
% axis tight
% set(gcf,'renderer','painters')

% Reference journal article: 
% Nikkhoo M. and Walter T.R., 2015. Triangular dislocation: An analytical, 
% artefact-free solution. 
% Submitted to Geophysical Journal International 

% Copyright (c) 2014 Mehdi Nikkhoo
% 
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files 
% (the "Software"), to deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, merge, publish, 
% distribute, sublicense, and/or sell copies of the Software, and to permit
% persons to whom the Software is furnished to do so, subject to the 
% following conditions:
% 
% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
% NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
% OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
% USE OR OTHER DEALINGS IN THE SOFTWARE.

% I appreciate any comments or bug reports.

% Mehdi Nikkhoo
% created: 2013.1.28
% Last modified: 2014.7.30
% 
% VolcanoTectonics Research Group
% Section 2.1, Physics of Earthquakes and Volcanoes
% Department 2, Physics of the Earth
% Helmholtz Centre Potsdam
% German Research Centre for Geosciences (GFZ)
% 
% email: 
% mehdi.nikkhoo@gfz-potsdam.de 
% mehdi.nikkhoo@gmail.com
*/
{

    double X  = (double)fX;         double Y = (double)fY;          double Z = (double)fZ; 
    double Ss = (double)fSs;        double Ds= (double)fDs;         double Ts= (double)fTs;
    double mu = (double)fmu;        double lambda= (double)flambda; 
    double P1[3],                   P2[3],                          P3[3];
    P1[0] = (double)fP1[0];         P1[1] = (double)fP1[1];         P1[2] = (double)fP1[2];
    P2[0] = (double)fP2[0];         P2[1] = (double)fP2[1];         P2[2] = (double)fP2[2];
    P3[0] = (double)fP3[0];         P3[1] = (double)fP3[1];         P3[2] = (double)fP3[2];

    int   i;
    double StsMS[6],                 StsFSC[6],                  StsIS[6];
    double StrMS[6],                 StrFSC[6],                  StrIS[6];
    double P1mirr[3],                P2mirr[3],                  P3mirr[3];
    
    P1mirr[0] = P1[0];              P2mirr[0] = P2[0];          P3mirr[0] = P3[0];
    P1mirr[1] = P1[1];              P2mirr[1] = P2[1];          P3mirr[1] = P3[1];
    P1mirr[2] = -1.0*P1[2];         P2mirr[2] = -1.0*P2[2];     P3mirr[2] = -1.0*P3[2];
    
    for (i = 0; i < 6; i++)
    {   StsMS[i]   = 0.0;           StrMS[i]  = 0.0;     
        StsFSC[i]  = 0.0;           StrFSC[i] = 0.0;
        StsIS[i]   = 0.0;           StrIS[i]  = 0.0;
    }
    
    
    if ((Z > 0.0) || (P1[2] > 0.0) || (P2[2] > 0.0) || (P3[2] > 0.0))
    {   fprintf(stdout,"A triangle vertex or the center of testing location have positive z-position (above half-space) => abort\n");           
        fprintf(stdout,"Z: %f    P1[2]: %f    P2[2]: %f      P3[2]: %f   \n",Z, P1[2], P2[2],P3[2]);       
        exit(10);
    }
    if ((P1[2] == 0.0) && (P2[2] == 0.0) && (P3[2] == 0.0))
    {   for (i = 0; i < 6; i++)
        {   Stress[i] = 0.0;
            Strain[i] = 0.0;
    }   }
    else
    {
        /* Calculate main dislocation contribution to strains and stresses */
        TDstressFS_inStrnHS(       StsMS, StrMS,  X, Y, Z, P1,     P2,     P3,     Ss, Ds, Ts, mu, lambda);
        /* Calculate harmonic function contribution to strains and stresses */
        TDstress_HarFunc_inStrnHS(StsFSC, StrFSC, X, Y, Z, P1,     P2,     P3,     Ss, Ds, Ts, mu, lambda);
        /* Calculate image dislocation contribution to strains and stresses */
        TDstressFS_inStrnHS(       StsIS, StrIS,  X, Y, Z, P1mirr, P2mirr, P3mirr, Ss, Ds, Ts, mu, lambda);

        if ((P1mirr[2] == 0.0) && (P2mirr[2] == 0.0) && (P3mirr[2] == 0.0))
        {   StsIS[4] = -1.0*StsIS[4];           StsIS[5] = -1.0*StsIS[5];
            StrIS[4] = -1.0*StrIS[4];           StrIS[5] = -1.0*StrIS[5];
        }
        /* Calculate the complete stress and strain tensor components in EFCS */
        for (i = 0; i < 6; i++)
        {   Stress[i] = (StsMS[i] +StsIS[i] +StsFSC[i]);
            Strain[i] = (StrMS[i] +StrIS[i] +StrFSC[i]);
    }   }
    return;
}

void TDstressFS_inStrnHS(double StsMS[6], double StrMS[6],  double X, double Y, double Z, double P1[3], double P2[3], double P3[3], double by, double bz, double bx, double mu, double lambda)
{
    /* bx = Ts; % Tensile-slip;      by = Ss; % Strike-slip;         bz = Ds; % Dip-slip */
    int     casepLog,               casenLog;
    double   A,      B,      C,  x,  y,      z,      nu,     Tempdouble;
    
    int      TriMode[1];      
    double   tempVect1[3],   tempVect2[3];
    double   p1[3],          p2[3],                p3[3];              
    double   e12[3],         e13[3],         e23[3];                 
    double   e_vals1[6],     e_vals2[6],     e_vals3[6],     e_comb[6];
    double   Amat[3][3];

    nu = lambda/(lambda+mu) /2.0; /* Poisson's ratio */
    p1[0]        = 0.0;                     p1[1]        = 0.0;                     p1[2]        = 0.0;
    p2[0]        = 0.0;                     p2[1]        = 0.0;                     p2[2]        = 0.0;
    p3[0]        = 0.0;                     p3[1]        = 0.0;                     p3[2]        = 0.0;

    /* Calculate unit strike, dip and normal to TD vectors: For a horizontal TD as an exception, if the normal vector points upward, the strike and dip */
    /* vectors point Northward and Westward, whereas if the normal vector points downward, the strike and dip vectors point Southward and Westward, respectively. */
    double eY[3],                         eZ[3];
    double Vnorm[3],                    Vstrike[3],                    Vdip[3];
    
    eY[0]        = 0.0;                     eY[1]        = 1.0;                     eY[2]        = 0.0;
    eZ[0]        = 0.0;                     eZ[1]        = 0.0;                     eZ[2]        = 1.0;    

    tempVect1[0] = P2[0] -P1[0];            tempVect1[1] = P2[1] -P1[1];            tempVect1[2] = P2[2] -P1[2];
    tempVect2[0] = P3[0] -P1[0];            tempVect2[1] = P3[1] -P1[1];            tempVect2[2] = P3[2] -P1[2];
      
    Vnorm[0]     = tempVect1[1]*tempVect2[2] - tempVect1[2]*tempVect2[1];
    Vnorm[1]     = tempVect1[2]*tempVect2[0] - tempVect1[0]*tempVect2[2];
    Vnorm[2]     = tempVect1[0]*tempVect2[1] - tempVect1[1]*tempVect2[0];
    Tempdouble   = sqrt(  Vnorm[0]*Vnorm[0] +Vnorm[1]*Vnorm[1] + Vnorm[2]*Vnorm[2]);

    Vnorm[0]     = Vnorm[0]/Tempdouble;     Vnorm[1]     = Vnorm[1]/Tempdouble;     Vnorm[2]     = Vnorm[2]/Tempdouble;

    Vstrike[0]   = eZ[1]*Vnorm[2] - eZ[2]*Vnorm[1];
    Vstrike[1]   = eZ[2]*Vnorm[0] - eZ[0]*Vnorm[2];
    Vstrike[2]   = eZ[0]*Vnorm[1] - eZ[1]*Vnorm[0];
    /* For horizontal elements ("Vnorm(3)" adjusts for Northward or Southward direction) */
    Tempdouble    = sqrt(  Vstrike[0]*Vstrike[0] +Vstrike[1]*Vstrike[1] + Vstrike[2]*Vstrike[2]);
    if (Tempdouble < DBL_EPSILON)
    {   Vstrike[0] = 0.0 ;                  Vstrike[1] = eY[1]*Vnorm[2];            Vstrike[2]   = 0.0;        
        /* For horizontal elements in case of half-space calculation!!! => Correct the strike vector of image dislocation only */
        if (P1[2] > 0.0)
        {   Vstrike[0] = -1.0*Vstrike[0];   Vstrike[1] = -1.0*Vstrike[1];           Vstrike[2]   = -1.0*Vstrike[2];
    }   }
     Tempdouble    = sqrt(  Vstrike[0]*Vstrike[0] +Vstrike[1]*Vstrike[1] + Vstrike[2]*Vstrike[2]);

    Vstrike[0]   = Vstrike[0]/Tempdouble;  Vstrike[1] = Vstrike[1]/Tempdouble;     Vstrike[2]   = Vstrike[2]/Tempdouble;

    Vdip[0]      = Vnorm[1]*Vstrike[2] - Vnorm[2]*Vstrike[1];
    Vdip[1]      = Vnorm[2]*Vstrike[0] - Vnorm[0]*Vstrike[2];
    Vdip[2]      = Vnorm[0]*Vstrike[1] - Vnorm[1]*Vstrike[0];
    Tempdouble    = sqrt(  Vdip[0]*Vdip[0] +Vdip[1]*Vdip[1] + Vdip[2]*Vdip[2]);

    Vdip[0]      = Vdip[0]/Tempdouble;     Vdip[1] = Vdip[1]/Tempdouble;           Vdip[2]      = Vdip[2]/Tempdouble;
    
    /* Transform coordinates and slip vector components from EFCS into TDCS */
    Amat[0][0]   = Vnorm[0];              Amat[0][1]    = Vnorm[1];              Amat[0][2]    = Vnorm[2];
    Amat[1][0]   = Vstrike[0];            Amat[1][1]    = Vstrike[1];            Amat[1][2]    = Vstrike[2];
    Amat[2][0]   = Vdip[0];               Amat[2][1]    = Vdip[1];               Amat[2][2]    = Vdip[2];    

    CoordTrans_inStrnHS(tempVect1, (X-P2[0]),  (Y-P2[1]), (Z-P2[2]), Amat); 
    x          = tempVect1[0];                  y          = tempVect1[1];                  z          = tempVect1[2];
    CoordTrans_inStrnHS(tempVect1, (P1[0]-P2[0]),  (P1[1]-P2[1]), (P1[2]-P2[2]), Amat); 
    p1[0]      = tempVect1[0];                   p1[1]      = tempVect1[1];                  p1[2]      = tempVect1[2];
    CoordTrans_inStrnHS(tempVect2, (P3[0]-P2[0]),  (P3[1]-P2[1]), (P3[2]-P2[2]), Amat); 
    p3[0]      = tempVect2[0];                   p3[1]      = tempVect2[1];                  p3[2]      = tempVect2[2];
    /* Calculate the unit vectors along TD sides in TDCS */
    Tempdouble  = sqrt(  (p2[0]-p1[0])*(p2[0]-p1[0]) +(p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2]));
    
    e12[0]     = (p2[0]-p1[0])/Tempdouble;       e12[1]      = (p2[1]-p1[1])/Tempdouble;       e12[2]     = (p2[2]-p1[2])/Tempdouble;     
    Tempdouble  = sqrt(  (p3[0]-p1[0])*(p3[0]-p1[0]) +(p3[1]-p1[1])*(p3[1]-p1[1]) + (p3[2]-p1[2])*(p3[2]-p1[2]));
    e13[0]     = (p3[0]-p1[0])/Tempdouble;       e13[1]      = (p3[1]-p1[1])/Tempdouble;       e13[2]     = (p3[2]-p1[2])/Tempdouble;
    Tempdouble  = sqrt(  (p3[0]-p2[0])*(p3[0]-p2[0]) +(p3[1]-p2[1])*(p3[1]-p2[1]) + (p3[2]-p2[2])*(p3[2]-p2[2]));
    e23[0]     = (p3[0]-p2[0])/Tempdouble;       e23[1]      = (p3[1]-p2[1])/Tempdouble;       e23[2]     = (p3[2]-p2[2])/Tempdouble; 
    /* Calculate the TD angles */
    Tempdouble  =      e12[0]*e13[0] +      e12[1]*e13[1] +      e12[2]*e13[2];
    A = acos(Tempdouble) ;
    Tempdouble  = -1.0*e12[0]*e23[0] + -1.0*e12[1]*e23[1] + -1.0*e12[2]*e23[2];
    B = acos(Tempdouble) ;
    Tempdouble  =      e23[0]*e13[0] +      e23[1]*e13[1] +     e23[2]*e13[2];
    C = acos(Tempdouble) ;
     
    /* Determine the best arteact-free configuration for each calculation point */

    TriModeFind_inStrnHS(TriMode,y,z,x,p1[1],p1[2], p2[1], p2[2], p3[1], p3[2]);

    if (TriMode[0] == 1)       {       casepLog = 1;   casenLog = 0;      } 
    if (TriMode[0] ==-1)       {       casepLog = 0;   casenLog = 1;      } 
    if (TriMode[0] == 0)       {       casepLog = 0;   casenLog = 0;      } 

    if (casepLog == 1) /* Configuration I */
    {   /* Calculate first angular dislocation contribution */
        tempVect1[0] = -1.0*e13[0];         tempVect1[1] = -1.0*e13[1];         tempVect1[2] = -1.0*e13[2];              
        TDSetupS_inStrnHS(x, y, z, A, bx, by, bz, nu, p1, tempVect1, e_vals1);
        /* Calculate second angular dislocation contribution */
        TDSetupS_inStrnHS(x, y, z, B, bx, by, bz, nu, p2, e12,       e_vals2);
        /* Calculate third angular dislocation contribution */
        TDSetupS_inStrnHS(x, y, z, C, bx, by, bz, nu, p3, e23,       e_vals3);  
    }
    if (casenLog == 1) /* Configuration II */
    {   /* Calculate first angular dislocation contribution */             
        TDSetupS_inStrnHS(x, y, z, A, bx, by, bz, nu, p1, e13,       e_vals1);
        /* Calculate second angular dislocation contribution */
        tempVect1[0] = -1.0*e12[0];         tempVect1[1] = -1.0*e12[1];         tempVect1[2] = -1.0*e12[2];      
        TDSetupS_inStrnHS(x, y, z, B, bx, by, bz, nu, p2, tempVect1, e_vals2);
        /* Calculate third angular dislocation contribution */
        tempVect1[0] = -1.0*e23[0];         tempVect1[1] = -1.0*e23[1];         tempVect1[2] = -1.0*e23[2]; 
        TDSetupS_inStrnHS(x, y, z, C, bx, by, bz, nu, p3, tempVect1, e_vals3);     
    }
    if ((casenLog == 1) || (casepLog == 1))
    {
        e_comb[0]       = e_vals1[0]+e_vals2[0]+e_vals3[0]; /* exx */
        e_comb[1]       = e_vals1[1]+e_vals2[1]+e_vals3[1]; /* exy */
        e_comb[2]       = e_vals1[2]+e_vals2[2]+e_vals3[2]; /* exz */
        e_comb[3]       = e_vals1[3]+e_vals2[3]+e_vals3[3]; /* eyy */
        e_comb[4]       = e_vals1[4]+e_vals2[4]+e_vals3[4]; /* eyz */
        e_comb[5]       = e_vals1[5]+e_vals2[5]+e_vals3[5]; /* ezz */
    }
    else
    {   
        e_comb[0]       = NAN; /* exx => supposed to be "NaN" => have to check */
        e_comb[1]       = NAN; /* exy */
        e_comb[2]       = NAN; /* exz */
        e_comb[3]       = NAN; /* eyy */
        e_comb[4]       = NAN; /* eyz */
        e_comb[5]       = NAN; /* ezz */
    }

    Amat[0][0] = Vnorm[0];         Amat[0][1] = Vstrike[0];       Amat[0][2] = Vdip[0];
    Amat[1][0] = Vnorm[1];         Amat[1][1] = Vstrike[1];       Amat[1][2] = Vdip[1];
    Amat[2][0] = Vnorm[2];         Amat[2][1] = Vstrike[2];       Amat[2][2] = Vdip[2];
    /* Transform the strain tensor components from TDCS into EFCS */
    TensTrans_inStrnHS(StrMS, e_comb, Amat);

    /* Calculate the stress tensor components in EFCS */
    StsMS[0] = 2.0*mu*StrMS[0]+lambda*(StrMS[0]+StrMS[3]+StrMS[5]);  /* sxx */
    StsMS[3] = 2.0*mu*StrMS[3]+lambda*(StrMS[0]+StrMS[3]+StrMS[5]);  /* syy */
    StsMS[5] = 2.0*mu*StrMS[5]+lambda*(StrMS[0]+StrMS[3]+StrMS[5]);  /* szz */
    StsMS[1] = 2.0*mu*StrMS[1];                            /* sxy */
    StsMS[2] = 2.0*mu*StrMS[2];                            /* sxz */
    StsMS[4] = 2.0*mu*StrMS[4];                            /* syz */
    
    return;
}

void TDstress_HarFunc_inStrnHS(double StsFSC[6], double StrFSC[6], double X, double Y, double Z, double P1[3], double P2[3], double P3[3], double by, double bz, double bx, double mu, double lambda)
{
    /* TDstress_HarFunc calculates the harmonic function contribution to the strains and stresses associated with a triangular dislocation in a */
    /* half-space. The function cancels the surface normal tractions induced by the main and image dislocations. */

    /* Calculate unit strike, dip and normal to TD vectors: For a horizontal TD as an exception, if the normal vector points upward, the strike and dip  */
    /* vectors point Northward and Westward, whereas if the normal vector points downward, the strike and dip vectors point Southward and Westward, respectively. */
    int   i;          
    double tempVect1[3],     tempVect2[3];
    double Stress1[6],       Stress2[6],         Stress3[6];
    double Strain1[6],       Strain2[6],         Strain3[6];
    
    double Amat[3][3];

    /* Calculate unit strike, dip and normal to TD vectors: For a horizontal TD as an exception, if the normal vector points upward, the strike and dip  */
    /* vectors point Northward and Westward, whereas if the normal vector points downward, the strike and dip vectors point Southward and Westward, respectively. */
    double Tempdouble,                    eY[3],                         eZ[3];
    double Vnorm[3],                        Vstrike[3],                    Vdip[3];
    
    eY[0]        = 0.0;                     eY[1]        = 1.0;                     eY[2]        = 0.0;
    eZ[0]        = 0.0;                     eZ[1]        = 0.0;                     eZ[2]        = 1.0;    

    tempVect1[0] = P2[0] -P1[0];            tempVect1[1] = P2[1] -P1[1];            tempVect1[2] = P2[2] -P1[2];
    tempVect2[0] = P3[0] -P1[0];            tempVect2[1] = P3[1] -P1[1];            tempVect2[2] = P3[2] -P1[2];
      
    Vnorm[0]     = tempVect1[1]*tempVect2[2] - tempVect1[2]*tempVect2[1];
    Vnorm[1]     = tempVect1[2]*tempVect2[0] - tempVect1[0]*tempVect2[2];
    Vnorm[2]     = tempVect1[0]*tempVect2[1] - tempVect1[1]*tempVect2[0];
    Tempdouble    = sqrt(  Vnorm[0]*Vnorm[0] +Vnorm[1]*Vnorm[1] + Vnorm[2]*Vnorm[2]);

    Vnorm[0]     = Vnorm[0]/Tempdouble;     Vnorm[1]     = Vnorm[1]/Tempdouble;     Vnorm[2]     = Vnorm[2]/Tempdouble;

    Vstrike[0]   = eZ[1]*Vnorm[2] - eZ[2]*Vnorm[1];
    Vstrike[1]   = eZ[2]*Vnorm[0] - eZ[0]*Vnorm[2];
    Vstrike[2]   = eZ[0]*Vnorm[1] - eZ[1]*Vnorm[0];
    /* For horizontal elements ("Vnorm(3)" adjusts for Northward or Southward direction) */
    Tempdouble    = sqrt(  Vstrike[0]*Vstrike[0] +Vstrike[1]*Vstrike[1] + Vstrike[2]*Vstrike[2]);
    if (Tempdouble < DBL_EPSILON)
    {   Vstrike[0] = 0.0;                  Vstrike[1] = eY[1]*Vnorm[2];          Vstrike[2]   = 0.0;        
    }
     Tempdouble     = sqrt(  Vstrike[0]*Vstrike[0] +Vstrike[1]*Vstrike[1] + Vstrike[2]*Vstrike[2]);

    Vstrike[0]    = Vstrike[0]/Tempdouble;  Vstrike[1] = Vstrike[1]/Tempdouble;     Vstrike[2]   = Vstrike[2]/Tempdouble;

    Vdip[0]       = Vnorm[1]*Vstrike[2] - Vnorm[2]*Vstrike[1];
    Vdip[1]       = Vnorm[2]*Vstrike[0] - Vnorm[0]*Vstrike[2];
    Vdip[2]       = Vnorm[0]*Vstrike[1] - Vnorm[1]*Vstrike[0];
    Tempdouble     = sqrt(  Vdip[0]*Vdip[0] +Vdip[1]*Vdip[1] + Vdip[2]*Vdip[2]);

    Vdip[0]       = Vdip[0]/Tempdouble;     Vdip[1] = Vdip[1]/Tempdouble;           Vdip[2]      = Vdip[2]/Tempdouble;

    /* Transform slip vector components from TDCS into EFCS */
    Amat[0][0] = Vnorm[0];          Amat[0][1] = Vstrike[0];            Amat[0][2] = Vdip[0];
    Amat[1][0] = Vnorm[1];          Amat[1][1] = Vstrike[1];            Amat[1][2] = Vdip[1];
    Amat[2][0] = Vnorm[2];          Amat[2][1] = Vstrike[2];            Amat[2][2] = Vdip[2];

    CoordTrans_inStrnHS(tempVect1, bx,  by, bz, Amat); 

    /* Calculate contribution of angular dislocation pair on each TD side */
    AngSetupFSC_S_inStrnHS(Stress1,Strain1, X,Y,Z,tempVect1[0],tempVect1[1],tempVect1[2],P1,P2,mu,lambda); /* P1P2 */
    AngSetupFSC_S_inStrnHS(Stress2,Strain2, X,Y,Z,tempVect1[0],tempVect1[1],tempVect1[2],P2,P3,mu,lambda); /* P2P3 */
    AngSetupFSC_S_inStrnHS(Stress3,Strain3, X,Y,Z,tempVect1[0],tempVect1[1],tempVect1[2],P3,P1,mu,lambda); /* P3P1 */
    /* Calculate total harmonic function contribution to strains and stresses */
    for (i = 0; i < 6; i++)
    {   StsFSC[i] = Stress1[i] + Stress2[i] + Stress3[i];
        StrFSC[i] = Strain1[i] + Strain2[i] + Strain3[i];
    }
    
    return;
}

void TensTrans_inStrnHS(double e_out[6], double e_in[6], double B[3][3])
{
    /* TensTrans Transforms the coordinates of tensors,from x1y1z1 coordinate system to x2y2z2 coordinate system. "A" is the transformation matrix, */
    /* whose columns e1,e2 and e3 are the unit base vectors of the x1y1z1. The coordinates of e1,e2 and e3 in A must be given in x2y2z2. The transpose  */
    /* of A (i.e., A') does the transformation from x2y2z2 into x1y1z1.  */
    double Txx1,         Txy1,           Txz1,           Tyy1;
    double Tyz1,         Tzz1,           Txx2,           Txy2;
    double Txz2,         Tyy2,           Tyz2,           Tzz2;
    double A[9];

    Txx1 = e_in[0];         Txy1 = e_in[1];         Txz1 = e_in[2];
    Tyy1 = e_in[3];         Tyz1 = e_in[4];         Tzz1 = e_in[5];

    A[0] = B[0][0];         A[1] = B[1][0];         A[2] = B[2][0];
    A[3] = B[0][1];         A[4] = B[1][1];         A[5] = B[2][1];
    A[6] = B[0][2];         A[7] = B[1][2];         A[8] = B[2][2];

    Txx2 = A[0]*A[0]*Txx1 +         2.0*A[0]*A[3] *Txy1 +          2.0*A[0]*A[6] *Txz1 +          2.0*A[3]*A[6] *Tyz1 + A[3]*A[3]*Tyy1 + A[6]*A[6]*Tzz1;
    Tyy2 = A[1]*A[1]*Txx1 +         2.0*A[1]*A[4] *Txy1 +          2.0*A[1]*A[7] *Txz1 +          2.0*A[4]*A[7] *Tyz1 + A[4]*A[4]*Tyy1 + A[7]*A[7]*Tzz1;
    Tzz2 = A[2]*A[2]*Txx1 +         2.0*A[2]*A[5] *Txy1 +          2.0*A[2]*A[8] *Txz1 +          2.0*A[5]*A[8] *Tyz1 + A[5]*A[5]*Tyy1 + A[8]*A[8]*Tzz1;
    Txy2 = A[0]*A[1]*Txx1 + (A[0]*A[4]+ A[1]*A[3])*Txy1 + (A[0]*A[7] + A[1]*A[6])*Txz1 + (A[7]*A[3] + A[6]*A[4])*Tyz1 + A[4]*A[3]*Tyy1 + A[6]*A[7]*Tzz1;
    Txz2 = A[0]*A[2]*Txx1 + (A[0]*A[5]+ A[2]*A[3])*Txy1 + (A[0]*A[8] + A[2]*A[6])*Txz1 + (A[8]*A[3] + A[6]*A[5])*Tyz1 + A[5]*A[3]*Tyy1 + A[6]*A[8]*Tzz1;
    Tyz2 = A[1]*A[2]*Txx1 + (A[2]*A[4]+ A[1]*A[5])*Txy1 + (A[2]*A[7] + A[1]*A[8])*Txz1 + (A[7]*A[5] + A[8]*A[4])*Tyz1 + A[4]*A[5]*Tyy1 + A[7]*A[8]*Tzz1;
   
    e_out[0] = Txx2;        e_out[1] = Txy2;      e_out[2] = Txz2;
    e_out[3] = Tyy2;        e_out[4] = Tyz2;      e_out[5] = Tzz2;
    
    return;
}

void  CoordTrans_inStrnHS(double newVal[3], double x1, double x2, double x3, double A[3][3]) 
{
    /* CoordTrans_inStrnHS transforms the coordinates of the vectors, from x1x2x3 coordinate system to X1X2X3 coordinate system. "A" is the */
    /* transformation matrix, whose columns e1,e2 and e3 are the unit base vectors of the x1x2x3. The coordinates of e1,e2 and e3 in A must be given */ 
    /* in X1X2X3. The transpose of A (i.e., A') will transform the coordinates  from X1X2X3 into x1x2x3. */
    
    newVal[0] = A[0][0]*x1 + A[0][1]*x2 + A[0][2]*x3;     newVal[1] = A[1][0]*x1 + A[1][1]*x2 + A[1][2]*x3;       newVal[2] = A[2][0]*x1 + A[2][1]*x2 + A[2][2]*x3;

    return;
}

void      TriModeFind_inStrnHS(int TriMode[1], double x,double y,double z,double p1_a,double p1_b, double p2_a,double p2_b,double p3_a,double p3_b)
{
    /* trimodefinder calculates the normalized barycentric coordinates of  the points with respect to the TD vertices and specifies the appropriate */
    /* artefact-free configuration of the angular dislocations for the  calculations. The input matrices x, y and z share the same size and  */
    /* correspond to the y, z and x coordinates in the TDCS, respectively. p1, p2 and p3 are two-component matrices representing the y and z coordinates  */
    /* of the TD vertices in the TDCS, respectively. The components of the output (trimode) corresponding to each calculation  */
    /* points, are 1 for the first configuration, -1 for the second configuration and 0 for the calculation point that lie on the TD sides. */
    double a,            b,          c;

    a = ((p2_b-p3_b)*(x-p3_a) +(p3_a-p2_a)*(y-p3_b)) /  ((p2_b-p3_b)*(p1_a-p3_a) +(p3_a-p2_a)*(p1_b-p3_b));
    b = ((p3_b-p1_b)*(x-p3_a) +(p1_a-p3_a)*(y-p3_b)) /  ((p2_b-p3_b)*(p1_a-p3_a) +(p3_a-p2_a)*(p1_b-p3_b));
    c = 1.0 -a -b;

    TriMode[0] = 1;
    if  ((a <= 0.0) && (b > c)    && (c > a))             {   TriMode[0] = -1;                  }
    if  ((a > b)    && (b <= 0.0) && (c > a))             {   TriMode[0] = -1;                  }
    if  ((a > b)    && (b > c)    && (c <= 0.0))          {   TriMode[0] = -1;                  }
    if  ((a == 0.0) && (b >= 0.0) && (c >= 0.0))       {   TriMode[0] = 0;                   }
    if  ((a >= 0.0) && (b == 0.0) && (c >= 0.0))       {   TriMode[0] = 0;                   }
    if  ((a >= 0.0) && (b >= 0.0) && (c == 0.0))       {   TriMode[0] = 0;                   }
    if  ((TriMode[0] == 0) && (z != 0.0))              {   TriMode[0] = 1;                   } 

    return;
}

void TDSetupS_inStrnHS(double x,double y,double z,double alpha,double bx,double by,double bz,double nu, double TriVertex[3],double SideVec[3],double e_out[6]) /* TDSetupS transforms coordinates of the calculation points as well as slip vector components from ADCS into TDCS. It then calculates the strains in ADCS and transforms them into TDCS. */
{
    double A[2][2];
    double B[3][3];
    double r1[2];
    double r2[2];
    double y1;
    double z1;
    double by1;
    double bz1;
    double tempVect1[2];
    double e[6];
    /* Transformation matrix */
    A[0][0] = SideVec[2];                   A[0][1]      = -1.0*SideVec[1];
    A[1][0] = SideVec[1];                   A[1][1]      =      SideVec[2]; 
    /* Transform coordinates of the calculation points from TDCS into ADCS */
    tempVect1[0] = y -TriVertex[1];         tempVect1[1] = z -TriVertex[2];
    r1[0]        = A[0][0]*tempVect1[0] +A[0][1]*tempVect1[1];
    r1[1]        = A[1][0]*tempVect1[0] +A[1][1]*tempVect1[1];
    y1           = r1[0];                   z1           = r1[1];
    /* Transform the in-plane slip vector components from TDCS into ADCS */
    tempVect1[0] = by;                      tempVect1[1] = bz;
    r2[0]        = A[0][0]*tempVect1[0] +A[0][1]*tempVect1[1];
    r2[1]        = A[1][0]*tempVect1[0] +A[1][1]*tempVect1[1];
    by1          = r2[0];                   bz1          = r2[1];
    
    /* Calculate strains associated with an angular dislocation in ADCS */
    AngDisStrain_inStrnHS(x, y1, z1, (-1.0*M_PI+alpha), bx, by1, bz1, nu, e); 
    /* Transform strains from ADCS into TDCS */
    B[0][0] = 1.0;          B[0][1] = 0.0;      B[0][2] = 0.0;
    B[1][0] = 0.0;          B[1][1] = A[0][0];  B[1][2] = A[1][0];
    B[2][0] = 0.0;          B[2][1] = A[0][1];  B[2][2] = A[1][1];/* 3x3 Transformation matrix */
    
    TensTrans_inStrnHS(e_out, e, B); /* the e_out is then send back from the function */

    return;
}

void    AngSetupFSC_S_inStrnHS(double Stress[6],double Strain[6], double X,double Y,double Z,double bX,double bY,double bZ,double PA[3], double PB[3],double mu,double lambda)
{   /* AngSetupFSC_S calculates the Free Surface Correction to strains and  stresses associated with angular dislocation pair on each TD side.  */
    int   i,        I;
    double nu,      beta,   TempVal1,       TempVal2;//      eps;
    
    double SideVec[3],       eZ[3],          ey1[3],         ey2[3],     ey3[3];
    double TempVect1[3],     TempVect2[3],   yA[3],          yB[3];
    double v1A[6],           v1B[6],         v_vals[6];

    double A[3][3],          A_t[3][3];
   // eps = DBL_EPSILON;
    nu  = lambda/(mu+lambda)/2.0; /* Poisson's ratio */
    /* Calculate TD side vector and the angle of the angular dislocation pair */
    SideVec[0] = PB[0]-PA[0];           SideVec[1] = PB[1]-PA[1];       SideVec[2] = PB[2]-PA[2];
    eZ[0]      = 0.0;                   eZ[1]      = 0.0;               eZ[2]      = 1.0;

    TempVal1   = sqrt( SideVec[0]*SideVec[0] +SideVec[1]*SideVec[1] +SideVec[2]*SideVec[2]);
    TempVal2   = -1.0*(SideVec[0]*eZ[0]      +SideVec[1]*eZ[1]      +SideVec[2]*eZ[2]);    
    beta       = acos(TempVal2/TempVal1);
    if (fabs(cos(beta)/sin(beta)) > 5.0e+5*(M_PI/360.0))
   // if ((fabs(beta) < eps) || (fabs(M_PI-beta)< eps))
    {   for (i = 0; i < 6; i++)         {       Stress[i] = 0.0;        Strain[i] = 0.0;            } 
    }
    else
    {
        ey1[0]   = SideVec[0];          ey1[1]   = SideVec[1];           ey1[2]   = 0.0;
        TempVal1 = sqrt( ey1[0]*ey1[0] +ey1[1]*ey1[1] +ey1[2]*ey1[2]);
        
        ey1[0]  /= TempVal1;            ey1[1]  /= TempVal1;             ey1[2]  /= TempVal1;
        ey3[0]   = -1.0*eZ[0];          ey3[1]   = -1.0*eZ[1];           ey3[2]   = -1.0*eZ[2];
        
        ey2[0]   = ey3[1]*ey1[2] -ey3[2]*ey1[1]; 
        ey2[1]   = ey3[2]*ey1[0] -ey3[0]*ey1[2]; 
        ey2[2]   = ey3[0]*ey1[1] -ey3[1]*ey1[0]; 
        TempVal1 = sqrt( ey2[0]*ey2[0] +ey2[1]*ey2[1] +ey2[2]*ey2[2]);
        ey2[0]  /= TempVal1;            ey2[1]  /= TempVal1;             ey2[2]  /= TempVal1;
        /* Transformation matrix */
        A[0][0]   = ey1[0];              A[0][1]   = ey2[0];              A[0][2]   = ey3[0];
        A[1][0]   = ey1[1];              A[1][1]   = ey2[1];              A[1][2]   = ey3[1];
        A[2][0]   = ey1[2];              A[2][1]   = ey2[2];              A[2][2]   = ey3[2];
        
        A_t[0][0] = ey1[0];              A_t[0][1] = ey1[1];              A_t[0][2] = ey1[2];
        A_t[1][0] = ey2[0];              A_t[1][1] = ey2[1];              A_t[1][2] = ey2[2];
        A_t[2][0] = ey3[0];              A_t[2][1] = ey3[1];              A_t[2][2] = ey3[2];
       
        /* Transform coordinates from EFCS to the first ADCS */
        CoordTrans_inStrnHS(yA, (X-PA[0]), (Y-PA[1]), (Z-PA[2]), A); 
        /* Transform coordinates from EFCS to the second ADCS */
        CoordTrans_inStrnHS(TempVect1, SideVec[0], SideVec[1], SideVec[2], A); 
        yB[0] = yA[0] - TempVect1[0];
        yB[1] = yA[1] - TempVect1[1];
        yB[2] = yA[2] - TempVect1[2];
        /* Transform slip vector components from EFCS to ADCS */
        CoordTrans_inStrnHS(TempVect2, bX, bY, bZ, A); 
        /* Determine the best arteact-free configuration for the calculation points near the free surface */
        I = (beta*yA[0]) >=0.0 ? 1 : 0;
        /* For singularities at surface */
        for (i = 0; i < 6; i++)         {       v1A[i] = 0.0;       v1B[i] = 0.0;           }
        /* Configuration I */
        if (I == 1)
        {   AngDisStrainFSC_inStrnHS(yA[0],yA[1], yA[2], (-M_PI+beta),TempVect2[0],TempVect2[1], TempVect2[2], nu,(-1.0*PA[2]), v1A);
            AngDisStrainFSC_inStrnHS(yB[0],yB[1], yB[2], (-M_PI+beta),TempVect2[0],TempVect2[1], TempVect2[2], nu,(-1.0*PB[2]), v1B);
        }
        /*  Configuration II */
        else if (I == 0)
        {   AngDisStrainFSC_inStrnHS(yA[0],yA[1], yA[2], beta,TempVect2[0],TempVect2[1], TempVect2[2], nu,(-1.0*PA[2]), v1A);
            AngDisStrainFSC_inStrnHS(yB[0],yB[1], yB[2], beta,TempVect2[0],TempVect2[1], TempVect2[2], nu,(-1.0*PB[2]), v1B);
        }
        /* Calculate total Free Surface Correction to strains in ADCS */
         for (i = 0; i < 6; i++)         {           v_vals[i] = v1B[i] - v1A[i];            }
          
         /* Transform total Free Surface Correction to strains from ADCS to EFCS */
         TensTrans_inStrnHS(Strain, v_vals, A_t);
               
         Stress[0] = 2.0*mu*Strain[0] +lambda*(Strain[0] +Strain[3] +Strain[5]); /* sig_xx */
              Stress[3] = 2.0*mu*Strain[3] +lambda*(Strain[0] +Strain[3] +Strain[5]); /* sig_yy */
         Stress[5] = 2.0*mu*Strain[5] +lambda*(Strain[0] +Strain[3] +Strain[5]); /* sig_zz */
         Stress[1] = 2.0*mu*Strain[1]; /* sig_xy */
         Stress[2] = 2.0*mu*Strain[2]; /* sig_xz */
          Stress[4] = 2.0*mu*Strain[4]; /* sig_yz */
    }
    return;
}

void AngDisStrain_inStrnHS(double x, double y, double z, double alpha, double bx, double by, double bz, double nu, double e[6]) /* AngDisStrain calculates the strains associated with an angular dislocation in an elastic full-space. */
{   double       sinA,           cosA,           eta,            zeta;
    double       x2,             y2,             z2,             r2;
    double       r,              r3,             rz,             r2z2;
    double       r3z,            W,              W2,             Wr;
    double       W2r,            Wr3,            W2r2,           C;
    double       S,              rFi_rx,         rFi_ry,         rFi_rz;
    double       Exx,            Exy,            Exz,            Eyy;
    double       Eyz,            Ezz;
    double       S2,             y3,             imnu,           tnp1;
    double       C2,             cosA2;

    sinA = sin(alpha);         cosA = cos(alpha);
    eta  = y*cosA - z*sinA;     zeta = y*sinA + z*cosA;
    x2   = x*x;                 y2   = y*y;
    z2   = z*z;                 r2   = x2+y2+z2;
    r    = sqrt(r2);           r3   = r*r*r;
    rz   = r*(r-z);             r2z2 = r2*(r-z)*(r-z);
    r3z  = r3*(r-z);
    
    W    = zeta-r;              W2   = W*W;
    Wr   = W*r;                 W2r  = W2*r;
    Wr3  = W*r3;                W2r2 = W2*r2;
    C    = (r*cosA -z)/Wr;      S    = (r*sinA -y)/Wr;
    
    S2   = S*S;                 y3   = y*y*y;
    C2   = C*C;
    imnu = (1.0-nu);            tnp1 = (2.0*nu+1.0);
    cosA2= cosA*cosA;

    /*Partial derivatives of the Burgers' function */
    rFi_rx = (eta/r/(r-zeta)-y/r/(r-z))/4.0/M_PI;
    rFi_ry = (x/r/(r-z)-cosA*x/r/(r-zeta))/4.0/M_PI;
    rFi_rz = (sinA*x/r/(r-zeta))/4.0/M_PI;

    Exx = bx*(rFi_rx)                     + bx/8.0/M_PI/imnu*(eta/Wr+eta*x2/W2r2-eta*x2/Wr3+y/rz-x2*y/r2z2-x2*y/r3z)                                            - by*x/8.0/M_PI/imnu*((tnp1/Wr+x2/W2r2-x2/Wr3)*cosA+tnp1/rz-x2/r2z2-x2/r3z)                   + bz*x*sinA/8.0/M_PI/imnu*(tnp1/Wr+x2/W2r2-x2/Wr3);

    Eyy = by*(rFi_ry)                     + bx/8.0/M_PI/imnu*((1.0/Wr+S2-y2/Wr3)*eta+tnp1*y/rz-y3/r2z2-y3/r3z-2.0*nu*cosA*S)                                    - by*x/8.0/M_PI/imnu*(1.0/rz-y2/r2z2-y2/r3z+(1.0/Wr+S2-y2/Wr3)*cosA)                          + bz*x*sinA/8.0/M_PI/imnu*(1.0/Wr+S2-y2/Wr3);

    Ezz = bz*(rFi_rz)                     + bx/8.0/M_PI/imnu*(eta/W/r+eta*C2-eta*z2/Wr3+y*z/r3+2.0*nu*sinA*C)                                                   - by*x/8.0/M_PI/imnu*((1.0/Wr+C2-z2/Wr3)*cosA+z/r3)                                           + bz*x*sinA/8.0/M_PI/imnu*(1.0/Wr+C2-z2/Wr3);

    Exy = bx*(rFi_ry)/2.0+by*(rFi_rx)/2.0 - bx/8.0/M_PI/imnu*(x*y2/r2z2-nu*x/rz+x*y2/r3z-nu*x*cosA/Wr+eta*x*S/Wr+eta*x*y/Wr3)                                   + by/8.0/M_PI/imnu*(x2*y/r2z2-nu*y/rz+x2*y/r3z+nu*cosA*S+x2*y*cosA/Wr3+x2*cosA*S/Wr)          - bz*sinA/8.0/M_PI/imnu*(nu*S+x2*S/Wr+x2*y/Wr3);

    Exz = bx*(rFi_rz)/2.0+bz*(rFi_rx)/2.0 - bx/8.0/M_PI/imnu*(-x*y/r3+nu*x*sinA/Wr+eta*x*C/Wr+eta*x*z/Wr3)                                                      + by/8.0/M_PI/imnu*(-x2/r3+nu/r+nu*cosA*C+x2*z*cosA/Wr3+x2*cosA*C/Wr)                         - bz*sinA/8.0/M_PI/imnu*(nu*C+x2*C/Wr+x2*z/Wr3);

    Eyz = by*(rFi_rz)/2.0+bz*(rFi_ry)/2.0 + bx/8.0/M_PI/imnu*(y2/r3-nu/r-nu*cosA*C+nu*sinA*S+eta*sinA*cosA/W2-eta*(y*cosA+z*sinA)/W2r+eta*y*z/W2r2-eta*y*z/Wr3) - by*x/8.0/M_PI/imnu*(y/r3+sinA*cosA2/W2-cosA*(y*cosA+z*sinA)/W2r+y*z*cosA/W2r2-y*z*cosA/Wr3) - bz*x*sinA/8.0/M_PI/imnu*(y*z/Wr3-sinA*cosA/W2+(y*cosA+z*sinA)/W2r-y*z/W2r2);

    e[0] = Exx;             e[1] = Exy;             e[2] = Exz;
    e[3] = Eyy;             e[4] = Eyz;             e[5] = Ezz;
    return;
}

void  AngDisStrainFSC_inStrnHS(double y1, double y2, double y3, double beta, double b1, double b2, double b3, double nu, double a, double Strain[6])
{/* AngDisStrainFSC calculates the harmonic function contribution to the strains associated with an angular dislocation in an elastic half-space. */
    
    double sinB,         cosB,           cotB,           y3b,        z1b;
    double z3b,          rb2,            rb,             W1,         W2;
    double W3,           W4,             W5,             W6,         W7;
    double W8,           W9,             N1,             rFib_ry2,   rFib_ry1;
    double rFib_ry3,     v11,            v12,            v13,        v22;
    double v23,          v33;
    
    double powrb3,       powcotB2,          powW62,               powy12,          powW72;
    double powy22,           powrb22,          powy13,               powrb5,          powy23;
    double powcosB2,     tmtnu,         arbW6,          y1sinB,     rbsinB;
    double mtptnu,       omW5,          W5mo,           y3brb1,     omnu;
     
    sinB = sin(beta);                  cosB = cos(beta);               cotB = cosB/sinB;
    y3b  = y3+a+a;                      z1b  = y1*cosB+y3b*sinB;       z3b  = -y1*sinB+y3b*cosB;
    rb2  = y1*y1 + y2*y2 + y3b*y3b;     rb   = sqrt(rb2);              N1   = 1.0-nu-nu;

    W1   = rb*cosB+y3b;                 W2   = cosB  +a/rb;            W3   = cosB +y3b/rb;
    W4   = nu +a/rb;                    W5   = nu+nu +a/rb;            W6   = rb   +y3b;
    W7   = rb +z3b;                     W8   = y3 +a;                  W9   = 1.0  +a/rb/cosB;

    /* Partial derivatives of the Burgers' function */
    rFib_ry2 =      z1b/rb/(rb+z3b)      -y1/rb/(rb+y3b); /* y2 = x in ADCS */
    rFib_ry1 =       y2/rb/(rb+y3b) -cosB*y2/rb/(rb+z3b); /* y1 = y in ADCS */
    rFib_ry3 = -sinB*y2/rb/(rb+z3b);                      /* y3 = z in ADCS */
 
    powrb3   = rb*rb*rb;                        powcotB2 = cotB*cotB;                            powW62   = W6*W6;
    powy12   = y1*y1;                            powW72   = W7*W7;                                 powy22   = y2*y2;
    powrb22  = rb2*rb2;                           powy13   = y1*y1*y1;                             powrb5   = rb*rb*rb*rb*rb;
    powy23   = y2*y2*y2;                        powcosB2 = cosB*cosB;                       y1sinB   = (y1/rb-sinB);
    
   
    tmtnu   = (2.0-nu-nu);                  arbW6    = (a/rb2+1.0/W6);                  rbsinB   = (rb*sinB-y1);
    mtptnu  = (-2.0+nu+nu);                 omW5     = (1.0-W5);                        W5mo     = (W5-1.0);
    y3brb1  = (y3b/rb+1.0);                 omnu     = (1.0-nu);

    v11 = b1*(0.25*(mtptnu*N1*rFib_ry1*powcotB2-N1*y2/powW62*(omW5*cotB-y1/W6*W4)/rb*y1+N1*y2/W6*(a/powrb3*y1*cotB-1.0/W6*W4+powy12/powW62*W4/rb+powy12/W6*a/powrb3)-N1*y2*cosB*cotB/powW72*W2*y1sinB-N1*y2*cosB*cotB/W7*a/powrb3*y1-3.0*a*y2*W8*cotB/powrb5*y1-y2*W8/powrb3/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1-y2*W8/rb2/powW62*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y1+y2*W8/rb/W6*(1.0/W6*W5-powy12/powW62*W5/rb-powy12/W6*a/powrb3+a/rb2-2.0*a*powy12/powrb22)-y2*W8/powrb3/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)*y1-y2*W8/rb/powW72*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)*y1sinB+y2*W8/rb/W7*(-cosB/powW72*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)*y1sinB+cosB/W7*(1.0/rb*cosB*y1*(N1*cosB-a/rb)*cotB+W1*a/powrb3*y1*cotB+tmtnu*(1.0/rb*sinB*y1-1.0)*cosB)+2.0*a*y3b*cosB*cotB/powrb22*y1))/M_PI/omnu)
        + b2*(0.25*(N1*((tmtnu*powcotB2+nu)/rb*y1/W6-(tmtnu*powcotB2+1.0)*cosB*y1sinB/W7)-N1/powW62*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/rb+powy12/W6*W4)/rb*y1+N1/W6*(-N1*cotB+a*cotB/rb-a*powy12*cotB/powrb3+2.0*y1/W6*W4-powy13/powW62*W4/rb-powy13/W6*a/powrb3)+N1*cotB/powW72*(z1b*cosB-a*rbsinB/rb/cosB)*y1sinB-N1*cotB/W7*(powcosB2-a*(1.0/rb*sinB*y1-1.0)/rb/cosB+a*rbsinB/powrb3/cosB*y1)-a*W8*cotB/powrb3+3.0*a*powy12*W8*cotB/powrb5-W8/powW62*(2.0*nu+1.0/rb*(N1*y1*cotB+a)-powy12/rb/W6*W5-a*powy12/powrb3)/rb*y1+W8/W6*(-1.0/powrb3*(N1*y1*cotB+a)*y1+1.0/rb*N1*cotB-2.0*y1/rb/W6*W5+powy13/powrb3/W6*W5+powy13/rb2/powW62*W5+powy13/powrb22/W6*a-2.0*a/powrb3*y1+3.0*a*powy13/powrb5)-W8*cotB/powW72*(-cosB*sinB+a*y1*y3b/powrb3/cosB+rbsinB/rb*(tmtnu*cosB-W1/W7*W9))*y1sinB+W8*cotB/W7*(a*y3b/powrb3/cosB-3.0*a*powy12*y3b/powrb5/cosB+(1.0/rb*sinB*y1-1.0)/rb*(tmtnu*cosB-W1/W7*W9)-rbsinB/powrb3*(tmtnu*cosB-W1/W7*W9)*y1+rbsinB/rb*(-1.0/rb*cosB*y1/W7*W9+W1/powW72*W9*y1sinB+W1/W7*a/powrb3/cosB*y1)))/M_PI/omnu)
        + b3*(0.25*(N1*(-y2/powW62*(1.0+a/rb)/rb*y1-y2/W6*a/powrb3*y1+y2*cosB/powW72*W2*y1sinB+y2*cosB/W7*a/powrb3*y1)+y2*W8/powrb3*arbW6*y1-y2*W8/rb*(-2.0*a/powrb22*y1-1.0/powW62/rb*y1)-y2*W8*cosB/powrb3/W7*(W1/W7*W2+a*y3b/rb2)*y1-y2*W8*cosB/rb/powW72*(W1/W7*W2+a*y3b/rb2)*y1sinB+y2*W8*cosB/rb/W7*(1.0/rb*cosB*y1/W7*W2-W1/powW72*W2*y1sinB-W1/W7*a/powrb3*y1-2.0*a*y3b/powrb22*y1))/M_PI/omnu);
        
    v22 = b1*(0.25*(N1*((tmtnu*powcotB2-nu)/rb*y2/W6-(tmtnu*powcotB2+1.0-2.0*nu)*cosB/rb*y2/W7)+N1/powW62*(y1*cotB*omW5+nu*y3b-a+powy22/W6*W4)/rb*y2-N1/W6*(a*y1*cotB/powrb3*y2+2.0*y2/W6*W4-powy23/powW62*W4/rb-powy23/W6*a/powrb3)+N1*z1b*cotB/powW72*W2/rb*y2+N1*z1b*cotB/W7*a/powrb3*y2+3.0*a*y2*W8*cotB/powrb5*y1-W8/powW62*(-2.0*nu+1.0/rb*(N1*y1*cotB-a)+powy22/rb/W6*W5+a*powy22/powrb3)/rb*y2+W8/W6*(-1.0/powrb3*(N1*y1*cotB-a)*y2+2.0*y2/rb/W6*W5-powy23/powrb3/W6*W5-powy23/rb2/powW62*W5-powy23/powrb22/W6*a+2.0*a/powrb3*y2-3.0*a*powy23/powrb5)-W8/powW72*(powcosB2-1.0/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/powrb3-1.0/rb/W7*(powy22*powcosB2-a*z1b*cotB/rb*W1))/rb*y2+W8/W7*(1.0/powrb3*(N1*z1b*cotB+a*cosB)*y2-3.0*a*y3b*z1b*cotB/powrb5*y2+1.0/powrb3/W7*(powy22*powcosB2-a*z1b*cotB/rb*W1)*y2+1.0/rb2/powW72*(powy22*powcosB2-a*z1b*cotB/rb*W1)*y2-1.0/rb/W7*(2.0*y2*powcosB2+a*z1b*cotB/powrb3*W1*y2-a*z1b*cotB/rb2*cosB*y2)))/M_PI/omnu)
        + b2*(0.25*(tmtnu*N1*rFib_ry2*powcotB2+N1/W6*(W5mo*cotB+y1/W6*W4)-N1*powy22/powW62*(W5mo*cotB+y1/W6*W4)/rb+N1*y2/W6*(-a/powrb3*y2*cotB-y1/powW62*W4/rb*y2-y2/W6*a/powrb3*y1)-N1*cotB/W7*W9+N1*powy22*cotB/powW72*W9/rb+N1*powy22*cotB/W7*a/powrb3/cosB-a*W8*cotB/powrb3+3.0*a*powy22*W8*cotB/powrb5+W8/rb/W6*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))-powy22*W8/powrb3/W6*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))-powy22*W8/rb2/powW62*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))+y2*W8/rb/W6*(2.0*nu*y1/powW62/rb*y2+a*y1/powrb3*(1.0/rb+1.0/W6)*y2-a*y1/rb*(-1.0/powrb3*y2-1.0/powW62/rb*y2))+W8*cotB/rb/W7*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)-powy22*W8*cotB/powrb3/W7*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)-powy22*W8*cotB/rb2/powW72*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)+y2*W8*cotB/rb/W7*(1.0/rb*cosB*y2/W7*W9-W1/powW72*W9/rb*y2-W1/W7*a/powrb3/cosB*y2-2.0*a*y3b/powrb22/cosB*y2))/M_PI/omnu)
        + b3*(0.25*(N1*(-sinB/rb*y2/W7+y2/powW62*(1.0+a/rb)/rb*y1+y2/W6*a/powrb3*y1-z1b/powW72*W2/rb*y2-z1b/W7*a/powrb3*y2)-y2*W8/powrb3*arbW6*y1+y1*W8/rb*(-2.0*a/powrb22*y2-1.0/powW62/rb*y2)+W8/powW72*(sinB*(cosB-a/rb)+z1b/rb*(1.0+a*y3b/rb2)-1.0/rb/W7*(powy22*cosB*sinB-a*z1b/rb*W1))/rb*y2-W8/W7*(sinB*a/powrb3*y2-z1b/powrb3*(1.0+a*y3b/rb2)*y2-2.0*z1b/powrb5*a*y3b*y2+1.0/powrb3/W7*(powy22*cosB*sinB-a*z1b/rb*W1)*y2+1.0/rb2/powW72*(powy22*cosB*sinB-a*z1b/rb*W1)*y2-1/rb/W7*(2*y2*cosB*sinB+a*z1b/powrb3*W1*y2-a*z1b/rb2*cosB*y2)))/M_PI/omnu);

    v33 = b1*(0.25*(tmtnu*(N1*rFib_ry3*cotB-y2/powW62*W5*y3brb1-0.5*y2/W6*a/powrb3*2.0*y3b+y2*cosB/powW72*W2*W3+0.5*y2*cosB/W7*a/powrb3*2.0*y3b)+y2/rb*(2.0*nu/W6+a/rb2)-0.5*y2*W8/powrb3*(2.0*nu/W6+a/rb2)*2.0*y3b+y2*W8/rb*(-2.0*nu/powW62*y3brb1-a/powrb22*2.0*y3b)+y2*cosB/rb/W7*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)-0.5*y2*W8*cosB/powrb3/W7*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)*2.0*y3b-y2*W8*cosB/rb/powW72*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)*W3+y2*W8*cosB/rb/W7*(-(cosB*y3b/rb+1.0)/W7*W2+W1/powW72*W2*W3+0.5*W1/W7*a/powrb3*2.0*y3b-a/rb2+a*y3b/powrb22*2.0*y3b))/M_PI/omnu)
        + b2*(0.25*(mtptnu*N1*cotB*(y3brb1/W6-cosB*W3/W7)+tmtnu*y1/powW62*W5*y3brb1+0.5*tmtnu*y1/W6*a/powrb3*2.0*y3b+tmtnu*sinB/W7*W2-tmtnu*z1b/powW72*W2*W3-0.5*tmtnu*z1b/W7*a/powrb3*2.0*y3b+1.0/rb*(N1*cotB-2.0*nu*y1/W6-a*y1/rb2)-0.5*W8/powrb3*(N1*cotB-2.0*nu*y1/W6-a*y1/rb2)*2.0*y3b+W8/rb*(2.0*nu*y1/powW62*y3brb1+a*y1/powrb22*2.0*y3b)-1.0/W7*(cosB*sinB+W1*cotB/rb*(tmtnu*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))+W8/powW72*(cosB*sinB+W1*cotB/rb*(tmtnu*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*W3-W8/W7*((cosB*y3b/rb+1.0)*cotB/rb*(tmtnu*cosB-W1/W7)-0.5*W1*cotB/powrb3*(tmtnu*cosB-W1/W7)*2.0*y3b+W1*cotB/rb*(-(cosB*y3b/rb+1.0)/W7+W1/powW72*W3)-0.5*a/powrb3*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7)*2.0*y3b+a/rb*(-z1b/rb2-y3b*sinB/rb2+y3b*z1b/powrb22*2.0*y3b-sinB*W1/rb/W7-z1b*(cosB*y3b/rb+1.0)/rb/W7+0.5*z1b*W1/powrb3/W7*2.0*y3b+z1b*W1/rb/powW72*W3)))/M_PI/omnu)
        + b3*(0.25*(tmtnu*rFib_ry3-tmtnu*y2*sinB/powW72*W2*W3-0.5*tmtnu*y2*sinB/W7*a/powrb3*2.0*y3b+y2*sinB/rb/W7*(1.0+W1/W7*W2+a*y3b/rb2)-0.5*y2*W8*sinB/powrb3/W7*(1.0+W1/W7*W2+a*y3b/rb2)*2.0*y3b-y2*W8*sinB/rb/powW72*(1.0+W1/W7*W2+a*y3b/rb2)*W3+y2*W8*sinB/rb/W7*((cosB*y3b/rb+1.0)/W7*W2-W1/powW72*W2*W3-0.5*W1/W7*a/powrb3*2.0*y3b+a/rb2-a*y3b/powrb22*2.0*y3b))/M_PI/omnu);

    v12 = b1/2.0*(0.25*(mtptnu*N1*rFib_ry2*powcotB2+N1/W6*(omW5*cotB-y1/W6*W4)-N1*powy22/powW62*(omW5*cotB-y1/W6*W4)/rb+N1*y2/W6*(a/powrb3*y2*cotB+y1/powW62*W4/rb*y2+y2/W6*a/powrb3*y1)+N1*cosB*cotB/W7*W2-N1*powy22*cosB*cotB/powW72*W2/rb-N1*powy22*cosB*cotB/W7*a/powrb3+a*W8*cotB/powrb3-3.0*a*powy22*W8*cotB/powrb5+W8/rb/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)-powy22*W8/powrb3/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)-powy22*W8/rb2/powW62*(-N1*cotB+y1/W6*W5+a*y1/rb2)+y2*W8/rb/W6*(-y1/powW62*W5/rb*y2-y2/W6*a/powrb3*y1-2.0*a*y1/powrb22*y2)+W8/rb/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)-powy22*W8/powrb3/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)-powy22*W8/rb2/powW72*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)+y2*W8/rb/W7*(-cosB/powW72*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)/rb*y2+cosB/W7*(1.0/rb*cosB*y2*(N1*cosB-a/rb)*cotB+W1*a/powrb3*y2*cotB+tmtnu/rb*sinB*y2*cosB)+2.0*a*y3b*cosB*cotB/powrb22*y2))/M_PI/omnu)
        + b2/2.0*(0.25*(N1*((tmtnu*powcotB2+nu)/rb*y2/W6-(tmtnu*powcotB2+1.0)*cosB/rb*y2/W7)-N1/powW62*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/rb+powy12/W6*W4)/rb*y2+N1/W6*(-a*y1*cotB/powrb3*y2-powy12/powW62*W4/rb*y2-powy12/W6*a/powrb3*y2)+N1*cotB/powW72*(z1b*cosB-a*rbsinB/rb/cosB)/rb*y2-N1*cotB/W7*(-a/rb2*sinB*y2/cosB+a*rbsinB/powrb3/cosB*y2)+3.0*a*y2*W8*cotB/powrb5*y1-W8/powW62*(2.0*nu+1.0/rb*(N1*y1*cotB+a)-powy12/rb/W6*W5-a*powy12/powrb3)/rb*y2+W8/W6*(-1.0/powrb3*(N1*y1*cotB+a)*y2+powy12/powrb3/W6*W5*y2+powy12/rb2/powW62*W5*y2+powy12/powrb22/W6*a*y2+3.0*a*powy12/powrb5*y2)-W8*cotB/powW72*(-cosB*sinB+a*y1*y3b/powrb3/cosB+rbsinB/rb*(tmtnu*cosB-W1/W7*W9))/rb*y2+W8*cotB/W7*(-3.0*a*y1*y3b/powrb5/cosB*y2+1.0/rb2*sinB*y2*(tmtnu*cosB-W1/W7*W9)-rbsinB/powrb3*(tmtnu*cosB-W1/W7*W9)*y2+rbsinB/rb*(-1.0/rb*cosB*y2/W7*W9+W1/powW72*W9/rb*y2+W1/W7*a/powrb3/cosB*y2)))/M_PI/omnu)
        + b3/2.0*(0.25*(N1*(1.0/W6*(1.0+a/rb)-powy22/powW62*(1.0+a/rb)/rb-powy22/W6*a/powrb3-cosB/W7*W2+powy22*cosB/powW72*W2/rb+powy22*cosB/W7*a/powrb3)-W8/rb*arbW6+powy22*W8/powrb3*arbW6-y2*W8/rb*(-2.0*a/powrb22*y2-1.0/powW62/rb*y2)+W8*cosB/rb/W7*(W1/W7*W2+a*y3b/rb2)-powy22*W8*cosB/powrb3/W7*(W1/W7*W2+a*y3b/rb2)-powy22*W8*cosB/rb2/powW72*(W1/W7*W2+a*y3b/rb2)+y2*W8*cosB/rb/W7*(1.0/rb*cosB*y2/W7*W2-W1/powW72*W2/rb*y2-W1/W7*a/powrb3*y2-2.0*a*y3b/powrb22*y2))/M_PI/omnu)
        + b1/2.0*(0.25*(N1*((tmtnu*powcotB2-nu)/rb*y1/W6-(tmtnu*powcotB2+1.0-2.0*nu)*cosB*y1sinB/W7)+N1/powW62*(y1*cotB*omW5+nu*y3b-a+powy22/W6*W4)/rb*y1-N1/W6*(omW5*cotB+a*powy12*cotB/powrb3-powy22/powW62*W4/rb*y1-powy22/W6*a/powrb3*y1)-N1*cosB*cotB/W7*W2+N1*z1b*cotB/powW72*W2*y1sinB+N1*z1b*cotB/W7*a/powrb3*y1-a*W8*cotB/powrb3+3.0*a*powy12*W8*cotB/powrb5-W8/powW62*(-2.0*nu+1.0/rb*(N1*y1*cotB-a)+powy22/rb/W6*W5+a*powy22/powrb3)/rb*y1+W8/W6*(-1.0/powrb3*(N1*y1*cotB-a)*y1+1.0/rb*N1*cotB-powy22/powrb3/W6*W5*y1-powy22/rb2/powW62*W5*y1-powy22/powrb22/W6*a*y1-3.0*a*powy22/powrb5*y1)-W8/powW72*(powcosB2-1.0/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/powrb3-1.0/rb/W7*(powy22*powcosB2-a*z1b*cotB/rb*W1))*y1sinB+W8/W7*(1.0/powrb3*(N1*z1b*cotB+a*cosB)*y1-1.0/rb*N1*cosB*cotB+a*y3b*cosB*cotB/powrb3-3.0*a*y3b*z1b*cotB/powrb5*y1+1.0/powrb3/W7*(powy22*powcosB2-a*z1b*cotB/rb*W1)*y1+1.0/rb/powW72*(powy22*powcosB2-a*z1b*cotB/rb*W1)*y1sinB-1.0/rb/W7*(-a*cosB*cotB/rb*W1+a*z1b*cotB/powrb3*W1*y1-a*z1b*cotB/rb2*cosB*y1)))/M_PI/omnu)
        + b2/2.0*(0.25*(tmtnu*N1*rFib_ry1*powcotB2-N1*y2/powW62*(W5mo*cotB+y1/W6*W4)/rb*y1+N1*y2/W6*(-a/powrb3*y1*cotB+1.0/W6*W4-powy12/powW62*W4/rb-powy12/W6*a/powrb3)+N1*y2*cotB/powW72*W9*y1sinB+N1*y2*cotB/W7*a/powrb3/cosB*y1+3.0*a*y2*W8*cotB/powrb5*y1-y2*W8/powrb3/W6*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))*y1-y2*W8/rb2/powW62*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))*y1+y2*W8/rb/W6*(-2.0*nu/W6+2.0*nu*powy12/powW62/rb-a/rb*(1.0/rb+1.0/W6)+a*powy12/powrb3*(1.0/rb+1.0/W6)-a*y1/rb*(-1.0/powrb3*y1-1.0/powW62/rb*y1))-y2*W8*cotB/powrb3/W7*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)*y1-y2*W8*cotB/rb/powW72*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)*y1sinB+y2*W8*cotB/rb/W7*(1.0/rb*cosB*y1/W7*W9-W1/powW72*W9*y1sinB-W1/W7*a/powrb3/cosB*y1-2.0*a*y3b/powrb22/cosB*y1))/M_PI/omnu)
        + b3/2.0*(0.25*(N1*(-sinB*y1sinB/W7-1.0/W6*(1.0+a/rb)+powy12/powW62*(1.0+a/rb)/rb+powy12/W6*a/powrb3+cosB/W7*W2-z1b/powW72*W2*y1sinB-z1b/W7*a/powrb3*y1)+W8/rb*arbW6-powy12*W8/powrb3*arbW6+y1*W8/rb*(-2.0*a/powrb22*y1-1.0/powW62/rb*y1)+W8/powW72*(sinB*(cosB-a/rb)+z1b/rb*(1.0+a*y3b/rb2)-1.0/rb/W7*(powy22*cosB*sinB-a*z1b/rb*W1))*y1sinB-W8/W7*(sinB*a/powrb3*y1+cosB/rb*(1.0+a*y3b/rb2)-z1b/powrb3*(1.0+a*y3b/rb2)*y1-2.0*z1b/powrb5*a*y3b*y1+1.0/powrb3/W7*(powy22*cosB*sinB-a*z1b/rb*W1)*y1+1.0/rb/powW72*(powy22*cosB*sinB-a*z1b/rb*W1)*y1sinB-1.0/rb/W7*(-a*cosB/rb*W1+a*z1b/powrb3*W1*y1-a*z1b/rb2*cosB*y1)))/M_PI/omnu);

    v13 = b1/2.0*(0.25*(mtptnu*N1*rFib_ry3*powcotB2-N1*y2/powW62*(omW5*cotB-y1/W6*W4)*y3brb1+N1*y2/W6*(0.5*a/powrb3*2.0*y3b*cotB+y1/powW62*W4*y3brb1+0.5*y1/W6*a/powrb3*2.0*y3b)-N1*y2*cosB*cotB/powW72*W2*W3-0.5*N1*y2*cosB*cotB/W7*a/powrb3*2.0*y3b+a/powrb3*y2*cotB-1.5*a*y2*W8*cotB/powrb5*2.0*y3b+y2/rb/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)-0.5*y2*W8/powrb3/W6*(-N1*cotB+y1/W6*W5+a*y1/rb2)*2.0*y3b-y2*W8/rb/powW62*(-N1*cotB+y1/W6*W5+a*y1/rb2)*y3brb1+y2*W8/rb/W6*(-y1/powW62*W5*y3brb1-0.5*y1/W6*a/powrb3*2.0*y3b-a*y1/powrb22*2.0*y3b)+y2/rb/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)-0.5*y2*W8/powrb3/W7*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)*2.0*y3b-y2*W8/rb/powW72*(cosB/W7*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)-a*y3b*cosB*cotB/rb2)*W3+y2*W8/rb/W7*(-cosB/powW72*(W1*(N1*cosB-a/rb)*cotB+tmtnu*rbsinB*cosB)*W3+cosB/W7*((cosB*y3b/rb+1.0)*(N1*cosB-a/rb)*cotB+0.5*W1*a/powrb3*2.0*y3b*cotB+0.5*tmtnu/rb*sinB*2.0*y3b*cosB)-a*cosB*cotB/rb2+a*y3b*cosB*cotB/powrb22*2.0*y3b))/M_PI/omnu)
        + b2/2.0*(0.25*(N1*((tmtnu*powcotB2+nu)*y3brb1/W6-(tmtnu*powcotB2+1.0)*cosB*W3/W7)-N1/powW62*(-N1*y1*cotB+nu*y3b-a+a*y1*cotB/rb+powy12/W6*W4)*y3brb1+N1/W6*(nu-0.5*a*y1*cotB/powrb3*2.0*y3b-powy12/powW62*W4*y3brb1-0.5*powy12/W6*a/powrb3*2.0*y3b)+N1*cotB/powW72*(z1b*cosB-a*rbsinB/rb/cosB)*W3-N1*cotB/W7*(cosB*sinB-0.5*a/rb2*sinB*2.0*y3b/cosB+0.5*a*rbsinB/powrb3/cosB*2.0*y3b)-a/powrb3*y1*cotB+1.5*a*y1*W8*cotB/powrb5*2.0*y3b+1.0/W6*(2.0*nu+1.0/rb*(N1*y1*cotB+a)-powy12/rb/W6*W5-a*powy12/powrb3)-W8/powW62*(2.0*nu+1.0/rb*(N1*y1*cotB+a)-powy12/rb/W6*W5-a*powy12/powrb3)*y3brb1+W8/W6*(-0.5/powrb3*(N1*y1*cotB+a)*2.0*y3b+0.5*powy12/powrb3/W6*W5*2.0*y3b+powy12/rb/powW62*W5*y3brb1+0.5*powy12/powrb22/W6*a*2.0*y3b+1.5*a*powy12/powrb5*2.0*y3b)+cotB/W7*(-cosB*sinB+a*y1*y3b/powrb3/cosB+rbsinB/rb*(tmtnu*cosB-W1/W7*W9))-W8*cotB/powW72*(-cosB*sinB+a*y1*y3b/powrb3/cosB+rbsinB/rb*(tmtnu*cosB-W1/W7*W9))*W3+W8*cotB/W7*(a/powrb3/cosB*y1-1.5*a*y1*y3b/powrb5/cosB*2.0*y3b+0.5/rb2*sinB*2.0*y3b*(tmtnu*cosB-W1/W7*W9)-0.5*rbsinB/powrb3*(tmtnu*cosB-W1/W7*W9)*2.0*y3b+rbsinB/rb*(-(cosB*y3b/rb+1.0)/W7*W9+W1/powW72*W9*W3+0.5*W1/W7*a/powrb3/cosB*2.0*y3b)))/M_PI/omnu)
        + b3/2.0*(0.25*(N1*(-y2/powW62*(1.0+a/rb)*y3brb1-0.5*y2/W6*a/powrb3*2.0*y3b+y2*cosB/powW72*W2*W3+0.5*y2*cosB/W7*a/powrb3*2.0*y3b)-y2/rb*arbW6+0.5*y2*W8/powrb3*arbW6*2.0*y3b-y2*W8/rb*(-a/powrb22*2.0*y3b-1.0/powW62*y3brb1)+y2*cosB/rb/W7*(W1/W7*W2+a*y3b/rb2)-0.5*y2*W8*cosB/powrb3/W7*(W1/W7*W2+a*y3b/rb2)*2.0*y3b-y2*W8*cosB/rb/powW72*(W1/W7*W2+a*y3b/rb2)*W3+y2*W8*cosB/rb/W7*((cosB*y3b/rb+1.0)/W7*W2-W1/powW72*W2*W3-0.5*W1/W7*a/powrb3*2.0*y3b+a/rb2-a*y3b/powrb22*2.0*y3b))/M_PI/omnu)
        + b1/2.0*(0.25*(tmtnu*(N1*rFib_ry1*cotB-y1/powW62*W5/rb*y2-y2/W6*a/powrb3*y1+y2*cosB/powW72*W2*y1sinB+y2*cosB/W7*a/powrb3*y1)-y2*W8/powrb3*(2.0*nu/W6+a/rb2)*y1+y2*W8/rb*(-2.0*nu/powW62/rb*y1-2.0*a/powrb22*y1)-y2*W8*cosB/powrb3/W7*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)*y1-y2*W8*cosB/rb/powW72*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)*y1sinB+y2*W8*cosB/rb/W7*(-1.0/rb*cosB*y1/W7*W2+W1/powW72*W2*y1sinB+W1/W7*a/powrb3*y1+2.0*a*y3b/powrb22*y1))/M_PI/omnu)
        + b2/2.0*(0.25*(mtptnu*N1*cotB*(1.0/rb*y1/W6-cosB*y1sinB/W7)-tmtnu/W6*W5+tmtnu*powy12/powW62*W5/rb+tmtnu*powy12/W6*a/powrb3+tmtnu*cosB/W7*W2-tmtnu*z1b/powW72*W2*y1sinB-tmtnu*z1b/W7*a/powrb3*y1-W8/powrb3*(N1*cotB-2.0*nu*y1/W6-a*y1/rb2)*y1+W8/rb*(-2.0*nu/W6+2.0*nu*powy12/powW62/rb-a/rb2+2.0*a*powy12/powrb22)+W8/powW72*(cosB*sinB+W1*cotB/rb*(tmtnu*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))*y1sinB-W8/W7*(1.0/rb2*cosB*y1*cotB*(tmtnu*cosB-W1/W7)-W1*cotB/powrb3*(tmtnu*cosB-W1/W7)*y1+W1*cotB/rb*(-1.0/rb*cosB*y1/W7+W1/powW72*y1sinB)-a/powrb3*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7)*y1+a/rb*(-y3b*cosB/rb2+2.0*y3b*z1b/powrb22*y1-cosB*W1/rb/W7-z1b/rb2*cosB*y1/W7+z1b*W1/powrb3/W7*y1+z1b*W1/rb/powW72*y1sinB)))/M_PI/omnu)
        + b3/2.0*(0.25*(tmtnu*rFib_ry1-tmtnu*y2*sinB/powW72*W2*y1sinB-tmtnu*y2*sinB/W7*a/powrb3*y1-y2*W8*sinB/powrb3/W7*(1.0+W1/W7*W2+a*y3b/rb2)*y1-y2*W8*sinB/rb/powW72*(1.0+W1/W7*W2+a*y3b/rb2)*y1sinB+y2*W8*sinB/rb/W7*(1.0/rb*cosB*y1/W7*W2-W1/powW72*W2*y1sinB-W1/W7*a/powrb3*y1-2.0*a*y3b/powrb22*y1))/M_PI/omnu);

    v23 = b1/2.0*(0.25*(N1*((tmtnu*powcotB2-nu)*y3brb1/W6-(tmtnu*powcotB2+1.0-2.0*nu)*cosB*W3/W7)+N1/powW62*(y1*cotB*omW5+nu*y3b-a+powy22/W6*W4)*y3brb1-N1/W6*(0.5*a*y1*cotB/powrb3*2.0*y3b+nu-powy22/powW62*W4*y3brb1-0.5*powy22/W6*a/powrb3*2.0*y3b)-N1*sinB*cotB/W7*W2+N1*z1b*cotB/powW72*W2*W3+0.5*N1*z1b*cotB/W7*a/powrb3*2.0*y3b-a/powrb3*y1*cotB+1.5*a*y1*W8*cotB/powrb5*2.0*y3b+1.0/W6*(-2.0*nu+1.0/rb*(N1*y1*cotB-a)+powy22/rb/W6*W5+a*powy22/powrb3)-W8/powW62*(-2.0*nu+1.0/rb*(N1*y1*cotB-a)+powy22/rb/W6*W5+a*powy22/powrb3)*y3brb1+W8/W6*(-0.5/powrb3*(N1*y1*cotB-a)*2.0*y3b-0.5*powy22/powrb3/W6*W5*2.0*y3b-powy22/rb/powW62*W5*y3brb1-0.5*powy22/powrb22/W6*a*2.0*y3b-1.5*a*powy22/powrb5*2.0*y3b)+1.0/W7*(powcosB2-1.0/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/powrb3-1.0/rb/W7*(powy22*powcosB2-a*z1b*cotB/rb*W1))-W8/powW72*(powcosB2-1.0/rb*(N1*z1b*cotB+a*cosB)+a*y3b*z1b*cotB/powrb3-1.0/rb/W7*(powy22*powcosB2-a*z1b*cotB/rb*W1))*W3+W8/W7*(0.5/powrb3*(N1*z1b*cotB+a*cosB)*2.0*y3b-1.0/rb*N1*sinB*cotB+a*z1b*cotB/powrb3+a*y3b*sinB*cotB/powrb3-1.5*a*y3b*z1b*cotB/powrb5*2.0*y3b+0.5/powrb3/W7*(powy22*powcosB2-a*z1b*cotB/rb*W1)*2.0*y3b+1.0/rb/powW72*(powy22*powcosB2-a*z1b*cotB/rb*W1)*W3-1.0/rb/W7*(-a*sinB*cotB/rb*W1+0.5*a*z1b*cotB/powrb3*W1*2.0*y3b-a*z1b*cotB/rb*(cosB*y3b/rb+1.0))))/M_PI/omnu)
        + b2/2.0*(0.25*(tmtnu*N1*rFib_ry3*powcotB2-N1*y2/powW62*(W5mo*cotB+y1/W6*W4)*y3brb1+N1*y2/W6*(-0.5*a/powrb3*2.0*y3b*cotB-y1/powW62*W4*y3brb1-0.5*y1/W6*a/powrb3*2.0*y3b)+N1*y2*cotB/powW72*W9*W3+0.5*N1*y2*cotB/W7*a/powrb3/cosB*2.0*y3b-a/powrb3*y2*cotB+1.5*a*y2*W8*cotB/powrb5*2.0*y3b+y2/rb/W6*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))-0.5*y2*W8/powrb3/W6*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))*2.0*y3b-y2*W8/rb/powW62*(N1*cotB-2.0*nu*y1/W6-a*y1/rb*(1.0/rb+1.0/W6))*y3brb1+y2*W8/rb/W6*(2.0*nu*y1/powW62*y3brb1+0.5*a*y1/powrb3*(1.0/rb+1.0/W6)*2.0*y3b-a*y1/rb*(-0.5/powrb3*2.0*y3b-1.0/powW62*y3brb1))+y2*cotB/rb/W7*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)-0.5*y2*W8*cotB/powrb3/W7*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)*2.0*y3b-y2*W8*cotB/rb/powW72*(mtptnu*cosB+W1/W7*W9+a*y3b/rb2/cosB)*W3+y2*W8*cotB/rb/W7*((cosB*y3b/rb+1.0)/W7*W9-W1/powW72*W9*W3-0.5*W1/W7*a/powrb3/cosB*2.0*y3b+a/rb2/cosB-a*y3b/powrb22/cosB*2.0*y3b))/M_PI/omnu)
        + b3/2.0*(0.25*(N1*(-sinB*W3/W7+y1/powW62*(1.0+a/rb)*y3brb1+0.5*y1/W6*a/powrb3*2.0*y3b+sinB/W7*W2-z1b/powW72*W2*W3-0.5*z1b/W7*a/powrb3*2.0*y3b)+y1/rb*arbW6-0.5*y1*W8/powrb3*arbW6*2.0*y3b+y1*W8/rb*(-a/powrb22*2.0*y3b-1.0/powW62*y3brb1)-1.0/W7*(sinB*(cosB-a/rb)+z1b/rb*(1.0+a*y3b/rb2)-1.0/rb/W7*(powy22*cosB*sinB-a*z1b/rb*W1))+W8/powW72*(sinB*(cosB-a/rb)+z1b/rb*(1.0+a*y3b/rb2)-1.0/rb/W7*(powy22*cosB*sinB-a*z1b/rb*W1))*W3-W8/W7*(0.5*sinB*a/powrb3*2.0*y3b+sinB/rb*(1.0+a*y3b/rb2)-0.5*z1b/powrb3*(1.0+a*y3b/rb2)*2.0*y3b+z1b/rb*(a/rb2-a*y3b/powrb22*2.0*y3b)+0.5/powrb3/W7*(powy22*cosB*sinB-a*z1b/rb*W1)*2.0*y3b+1.0/rb/powW72*(powy22*cosB*sinB-a*z1b/rb*W1)*W3-1.0/rb/W7*(-a*sinB/rb*W1+0.5*a*z1b/powrb3*W1*2.0*y3b-a*z1b/rb*(cosB*y3b/rb+1.0))))/M_PI/omnu)
        + b1/2.0*(0.25*(tmtnu*(N1*rFib_ry2*cotB+1.0/W6*W5-powy22/powW62*W5/rb-powy22/W6*a/powrb3-cosB/W7*W2+powy22*cosB/powW72*W2/rb+powy22*cosB/W7*a/powrb3)+W8/rb*(2.0*nu/W6+a/rb2)-powy22*W8/powrb3*(2.0*nu/W6+a/rb2)+y2*W8/rb*(-2.0*nu/powW62/rb*y2-2.0*a/powrb22*y2)+W8*cosB/rb/W7*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)-powy22*W8*cosB/powrb3/W7*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)-powy22*W8*cosB/rb2/powW72*(1.0-2.0*nu-W1/W7*W2-a*y3b/rb2)+y2*W8*cosB/rb/W7*(-1.0/rb*cosB*y2/W7*W2+W1/powW72*W2/rb*y2+W1/W7*a/powrb3*y2+2.0*a*y3b/powrb22*y2))/M_PI/omnu)
        + b2/2.0*(0.25*(mtptnu*N1*cotB*(1.0/rb*y2/W6-cosB/rb*y2/W7)+tmtnu*y1/powW62*W5/rb*y2+tmtnu*y1/W6*a/powrb3*y2-tmtnu*z1b/powW72*W2/rb*y2-tmtnu*z1b/W7*a/powrb3*y2-W8/powrb3*(N1*cotB-2.0*nu*y1/W6-a*y1/rb2)*y2+W8/rb*(2.0*nu*y1/powW62/rb*y2+2.0*a*y1/powrb22*y2)+W8/powW72*(cosB*sinB+W1*cotB/rb*(tmtnu*cosB-W1/W7)+a/rb*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7))/rb*y2-W8/W7*(1.0/rb2*cosB*y2*cotB*(tmtnu*cosB-W1/W7)-W1*cotB/powrb3*(tmtnu*cosB-W1/W7)*y2+W1*cotB/rb*(-cosB/rb*y2/W7+W1/powW72/rb*y2)-a/powrb3*(sinB-y3b*z1b/rb2-z1b*W1/rb/W7)*y2+a/rb*(2.0*y3b*z1b/powrb22*y2-z1b/rb2*cosB*y2/W7+z1b*W1/powrb3/W7*y2+z1b*W1/rb2/powW72*y2)))/M_PI/omnu)
        + b3/2.0*(0.25*(tmtnu*rFib_ry2+tmtnu*sinB/W7*W2-tmtnu*powy22*sinB/powW72*W2/rb-tmtnu*powy22*sinB/W7*a/powrb3+W8*sinB/rb/ W7*(1.0+W1/W7*W2+a*y3b/rb2)-powy22*W8*sinB/powrb3/W7*(1.0+W1/W7*W2+a*y3b/rb2)-powy22*W8*sinB/rb2/powW72*(1.0+W1/W7*W2+a* y3b/rb2)+y2*W8*sinB/rb/W7*(1.0/rb*cosB*y2/W7*W2-W1/powW72* W2/rb*y2-W1/W7*a/powrb3*y2-2.0*a*y3b/powrb22*y2))/M_PI/omnu);

    Strain[0] = v11;           Strain[1] = v12;         Strain[2] = v13;
    Strain[3] = v22;           Strain[4] = v23;         Strain[5] = v33;
    return;
}
