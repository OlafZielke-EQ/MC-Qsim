#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
//#include <cblas.h>
#include <gsl/gsl_cblas.h>
#include <mpi.h>
#include <time.h>
#include <limits.h>

#define USEVPVSVELOCITY         1u
#define CUTDISTANCE             0.66f  //K-matrix related, for flagging bad (too close) elements
#define MININTSEISSTRESSCHANGE  50000.0f //this is in Pa; 

#define INTERNALFRICTION        0.75f //a simple yielding implementation; if stresses exceed rock strength, then it breaks => no elastic interaction in that event
#define INTERNALCOHESION        5.0E+6f //I keep everything in Pa (not MPa)!

#define MAXITERATION4BOUNDARY   200u //when determining the stressing rate from loading from below
#define MAXMOMRATEFUNCLENGTH    5000u //the max iterations an individual event may take, the value is arbitrary and should be adjusted for large fault systems (when permitting time propagation of elastic signal)
#define NeighborFactor          1.0f // factor to multiply mean leg length with to determine if elements are neighbors; should be value equal or above 1.0f


#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

#define MAX(A,B) ( ((A) >   (B))  *(A)   +  ((A) <= (B))  *(B)    )
#define MIN(A,B) ( ((A) <   (B))  *(A)   +  ((A) >= (B))  *(B)    )
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
extern void  StrainHS_Nikkhoo(float Stress[6], float Strain[6], float X, float Y, float Z, float P1[3], float P2[3], float P3[3], float SS, float Ds, float Ts, const float mu, const float lambda);
float        ran0(long *idum);
float        FloatPow(float Base, unsigned int Exp);
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
int main(int argc, char **argv)
{   if ( (argc  > 3 ) || (argc < 2) ) {   fprintf(stdout,"Input Error\n Please start the code in the following way:\n\n mpirun -np 4 ./MCQsimV2 RunParaFile.txt -optionally adding more file name here");   exit(10);    }
    //------------------------------------------------------------------
    unsigned int i,   j,   k;
    int iRANK,   iSIZE;
    double time_taken,   time_InterSeis = 0.0,   time_CoSeis = 0.0;
    clock_t     timer0,   timerI,   timerC;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &iRANK);
    MPI_Comm_size(MPI_COMM_WORLD, &iSIZE);
    MPI_Status  STATUS;
    MPI_Offset  OFFSETall;
    MPI_File    fp_KMAT;
    MPI_File    fp_PREVRUN;
    MPI_File    fp_STF;
    //------------------------------------------------------------------------------------
    double dAddedTime = 0.0,   dRecordLength = 0.0,   dMeanRecurTime = 0.0; //dAddedTime is for continued catalogs (i.e., the end-time of continued catalog)
    long lSeed = 1234; //this is back-up seed value for RNG if no other value is provided
    unsigned int uFPNum = 0u,   uFVNum = 0u,   uBPNum = 0u,   uBVNum = 0u,   uEQcntr = 0u;
    unsigned int uRunNum = 0u,   uCatType = 0u,   uUseTimeProp = 0u,   uChgBtwEQs = 0u,   uContPrevRun = 0u,   uLoadPrev_Kmat = 0u,   uPlotCatalog2Screen = 0u,   uMinElemNum4Cat = 0u,   uLoadStep_POW2 = 0u, uStoreSTF4LargeEQs = 0u;
    float fPreStressFract = 0.0f,   fOvershootFract = 0.0f,   fMinCoSeisSlipRate = 0.0f,   fIntSeisLoadStep = 0.0f,   fMinMag4STF = 0.0f,   fMinMag4Prop = 0.0f,   fAfterSlipTime = 0.0f,   fDeepRelaxTime = 0.0f,   fHealFact = 0.0f;
    float fPSeisTime = 0.0f;
    float fFltSide,   fBndSide,   fElemArea,   fBoundArea,  fUnitSlipF,   fUnitSlipB;
    //                        0            1              2         3       4       5       6         7           8          9            10
    float fModPara[11];//density, addednormalstress, shearmod, poisson, lambda, vpvelo, vpvsratio, realDeltT, usedDeltT, Vs-velo,   MinSlipIncrem
    char cInputName[512],   cOutputName[512],   cKmatFileName[512],   cPrevStateName[512],   cNameAppend[512],   cSTFName[512],   cOutputNameCUT[512];
    //------------------------------------------------------------------------------------
    {   FILE        *fp1,                   *fp2,                       *fp3; 
        char        cFileName1[512],        cFileName2[512],            cFileName3[512];
        char        cTempVals[512];
        strcpy(cFileName1,  argv[1]);// opening and reading the "run file" that contains specifics about this model run
        //--------------------------------------------------------------------------------
        if ((fp1 = fopen(cFileName1,"r")) == NULL)      {   fprintf(stdout,"Error -cant open %s file in initializeVariables function \n",cFileName1);      exit(10);     }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %s", cInputName);                 }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uRunNum);                   }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uCatType);                  }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uPlotCatalog2Screen);       }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %ld",&lSeed);                     }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uMinElemNum4Cat);           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uContPrevRun);              }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uLoadPrev_Kmat);            }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %s", cKmatFileName);              }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uStoreSTF4LargeEQs);        }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fMinMag4STF);               }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uUseTimeProp);              }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fMinMag4Prop);              }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fIntSeisLoadStep);          }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %u", &uLoadStep_POW2);            }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fAfterSlipTime);            }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fDeepRelaxTime);            }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fHealFact);                 }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fOvershootFract);           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fPreStressFract);           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %f", &fMinCoSeisSlipRate);        }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {   sscanf(cTempVals,"%*s %lf",&dRecordLength);             }
        if (fgets(cTempVals, 512, fp1)    != NULL)      {                                                           }

        uContPrevRun   = (uContPrevRun == 1u) ? 1u : 0u;
        uLoadPrev_Kmat = (uContPrevRun == 1u) ? 2u : uLoadPrev_Kmat; //if I continue a previous run, then I have! to load the previous Kh_mat as well
        //-----------------------------
        fclose(fp1);
        strcpy(cFileName2, cInputName);     sprintf(cNameAppend, "_%u_Roughn.dat",uRunNum);   strcat(cFileName2,cNameAppend);
        strcpy(cFileName3, cInputName);     strcat(cFileName3,"_BNDtrig.dat"); 
        //--------------------------------------------------------------------------------
        if ((fp2 = fopen(cFileName2,"rb"))     == NULL)     {    printf("Error -cant open *_Roughn.dat file.  in Initialize variables...   %s\n", cFileName2);      exit(10);     }
        if (fread(&uFPNum,     sizeof(unsigned int),   1, fp2) != 1)       {   exit(10);   }// this is currently read PatchNum 
        if (fread(&uFVNum,     sizeof(unsigned int),   1, fp2) != 1)       {   exit(10);   }// this is currently read VertexNum
        if (fread(fModPara,    sizeof(float),          4, fp2) != 4)       {   exit(10);   }// medium density, added normal stress, shearmod, poisson
        if (fread(&uChgBtwEQs, sizeof(unsigned int),   1, fp2) != 1)       {   exit(10);   }// if friction values may be changed (by permitted variation value) between events
        //------------------------------------------------------------------------------------     
        fclose(fp2);
        //--------------------------------------------------------------------------------
        uBPNum   = 0u;               uBVNum   = 0u;
        if ((fp3 = fopen(cFileName3,"rb"))     != NULL)     
        {   if (fread(&uBPNum,   sizeof(unsigned int),   1, fp3) != 1)       {   exit(10);   }// this is currently read PatchNum 
            if (fread(&uBVNum,   sizeof(unsigned int),   1, fp3) != 1)       {   exit(10);   }// this is currently read VertexNum
            fclose(fp3);
        }
    }//this is the first block of loading input data (element number etc...)
    //------------------------------------------------------------------------------------
    int  uFBASEelem = (int)(uFPNum/iSIZE); 
    int  uFADDelem  = (int)(uFPNum%iSIZE);
    int  uBBASEelem = (int)(uBPNum/iSIZE); 
    int  uBADDelem  = (int)(uBPNum%iSIZE);
    //---------------------------
    int   *iFSTART  = calloc(iSIZE, sizeof *iFSTART ); //these starts and offsets relate to how the code accesses "local" and "global" vectors => use the start and offset to 
    int   *iBSTART  = calloc(iSIZE, sizeof *iBSTART );
    int   *iFOFFSET = calloc(iSIZE, sizeof *iFOFFSET );//locate where each rank will put/get data from when accessing a "global" vector
    int   *iBOFFSET = calloc(iSIZE, sizeof *iBOFFSET );
    //---------------------------
    for (i = 0u; i < iSIZE;     i++)      {   iFOFFSET[i]     = uFBASEelem;                            }
    for (i = 0u; i < uFADDelem; i++)      {   iFOFFSET[i]    += 1u;                                    }
    for (i = 1u; i < iSIZE;     i++)      {   iFSTART[i]      = iFSTART[i-1] + iFOFFSET[i-1];          }
    
    for (i = 0u; i < iSIZE;     i++)      {   iBOFFSET[i]     = uBBASEelem;                            }
    for (i = 0u; i < uBADDelem; i++)      {   iBOFFSET[i]    += 1u;                                    }
    for (i = 1u; i < iSIZE;     i++)      {   iBSTART[i]      = iBSTART[i-1] + iBOFFSET[i-1];          }
    //------------------------------------------------------------------------------------
    //                                                                 0   1    2    3      4        5 
    unsigned int *uF_temp   = calloc( 6*uFPNum,   sizeof *uF_temp );//v1, v2, v3, segid, fltid, flagged,
    unsigned int *uB_temp   = calloc( 4*uBPNum,   sizeof *uB_temp );//v1, v2, v3, segid
    //                                                                   0      1      2        3          4        5        6        7           8             9        10       11        12         13       14
    float        *fF_temp   = calloc(15*uFPNum,   sizeof *fF_temp );//CentE, CentN, CentZ, StressRate, SlipRate, Rake,   FltNorm_E, FltNorm_N, FltNorm_Z,  FltStk_E, FltStk_N, FltStk_Z  FltDip_E, FltDip_N, FltDip_Z
    float        *fB_temp   = calloc(15*uBPNum,   sizeof *fB_temp );//CentE, CentN, CentZ, StkSlip,   DipSlip, NormSlip, FltNorm_E, FltNorm_N, FltNorm_Z,  FltStk_E, FltStk_N, FltStk_Z  FltDip_E, FltDip_N, FltDip_Z
    float        *fFV_temp  = calloc( 4*uFVNum,   sizeof *fFV_temp );//VertE, VertN, VertZ, VertHeight
    float        *fBV_temp  = calloc( 3*uBVNum,   sizeof *fBV_temp );//VertE, VertN, VertZ,
    //                                                                           0               1               2            3               4               5              6             7             8             9              10                11            12            13                14                      15
    float        *fFRef     = calloc(14*iFOFFSET[iRANK],  sizeof *fFRef   );//RefArea,       RefNrmStress, MeanSelfStiff, RefStaFric,    RefStaFricVari, RefDynFric  , RefDynFricVari, RefDc,         RefDcVari,   Cent_E           Cent_N            Cent_Z     AsignSlipRate   AllSlipEver
    float        *fFFric    = calloc( 6*iFOFFSET[iRANK],  sizeof *fFFric  );//StatFric,       DynFric,     ArrFric,       TempRefFric,   Pseis_T0_Fric, TempRefFric2
    float        *fFEvent   = calloc(16*iFOFFSET[iRANK],  sizeof *fFEvent );//CurStressStk   CurStressDip  CurStressNrm    curFric       SelfStiffStik   SelfStifDip   SelfStiffNrm    curD_c        StressRateStk  StressRateDip   OvershotStr   curSlipStk/Dip   AccSlipCurMo   AccumSlipStk     AccumSlipDip   maxSlipPerStep(radiation damping)
    unsigned int *uFEvent   = calloc( 5*iFOFFSET[iRANK],  sizeof *uFEvent );//StabType,       Activated,   PtchBroken,    Ptch_t0,       Ptch_dur; 
    unsigned int *uFActivEl = calloc( 1*iFOFFSET[iRANK],  sizeof *uFActivEl ); //contains index of elements that were activated in current turn;
    //                                                                               0              1               2             3             4               5             6             7              8
    float        *fBEvent   = calloc( 9*iBOFFSET[iRANK],  sizeof *fBEvent ); //CurrStressStk, CurStressDip  CurStressNrm    ---------     SelfStiffStik   SelfStifDip   SelfStiffNr   PostSeis_T0_S,   PostSeis_T0_N
    
    float        *fFTempVal = calloc( 5*iFOFFSET[iRANK],  sizeof *fFTempVal );//strikeshear,    dipshear,    normstress,    currFric(b4), slip used from post-seismic
    float        *fBTempVal = calloc( 3*iBOFFSET[iRANK],  sizeof *fBTempVal );//strikeshear,    dipshear,    normstress

    float        *fMRFvals   = calloc( MAXMOMRATEFUNCLENGTH, sizeof *fMRFvals);
    //------------------------------------------------------------------------------------
    {   unsigned int *uTempF = calloc(uFPNum, sizeof *uTempF );
        unsigned int *uTempB = calloc(uBPNum, sizeof *uTempB );
        float *fTempF        = calloc(uFPNum, sizeof *fTempF );
        float *fTempB        = calloc(uBPNum, sizeof *fTempB );
        float *fTempFv       = calloc(uFVNum, sizeof *fTempFv );
        float *fTempBv       = calloc(uBVNum, sizeof *fTempBv );
        float  fTemp,   fMeanHeight = 0.0f;
        float  fP1[3],   fP2[3],   fP3[3],   fP1P2[3],   fP1P3[3],   fP2P3[3],   fArea,   fP1Pc[3],   fP2Pc[3];
        char   cAppend[512];
        char   cFileName1[512],   cFileName2[512],   cFileName3[512];
        FILE   *fp1,   *fp2,   *fp3;
        //--------------------------------------------------------------------------------
        //reading/loading the remaining information from txt and dat files.. this step is done by each rank individually; also, some values are read locally (every rank only reads the information relevant for it) while other values are read globally
        strcpy(cFileName1,cInputName);      sprintf(cAppend, "_%u_Roughn.dat",uRunNum);     strcat(cFileName1,cAppend);
        strcpy(cFileName2,cInputName);      sprintf(cAppend, "_%u_Strgth.dat",uRunNum);     strcat(cFileName2,cAppend);
        strcpy(cFileName3,cInputName);      strcat(cFileName3,"_BNDtrig.dat");
        //--------------------------------------------------------------------------------
        if ((fp1 = fopen(cFileName1,"rb"))     == NULL)     {   printf("Error -cant open %s LoadInputParameter function...\n",cFileName1);      exit(10);     }
        //-----------------------------------------------------
        fseek(fp1, (4*sizeof(float)+3*sizeof(unsigned int)), SEEK_SET); //skip the part that is in front of segId for specific RANK
        //-----------------------------------------------------
        if (fread(uTempF, sizeof(unsigned int), uFPNum, fp1) != uFPNum)       {   exit(10); }//this is V1 address/position in VertList
        for (i = 0u; i < uFPNum; i++)       {   uF_temp[i*6 +0] = uTempF[i] -1u;            }
        if (fread(uTempF, sizeof(unsigned int), uFPNum, fp1) != uFPNum)       {   exit(10); }//this is V2 address/position in VertList
        for (i = 0u; i < uFPNum; i++)       {   uF_temp[i*6 +1] = uTempF[i] -1u;            }
        if (fread(uTempF, sizeof(unsigned int), uFPNum, fp1) != uFPNum)       {   exit(10); }//this is V3 address/position in VertList
        for (i = 0u; i < uFPNum; i++)       {   uF_temp[i*6 +2] = uTempF[i] -1u;            }
        if (fread(uTempF, sizeof(unsigned int), uFPNum, fp1) != uFPNum)       {   exit(10); }//this is seg id
        for (i = 0u; i < uFPNum; i++)       {   uF_temp[i*6 +3] = uTempF[i] -1u;            }
        if (fread(uTempF, sizeof(unsigned int), uFPNum, fp1) != uFPNum)       {   exit(10); }//this is fault id   
        for (i = 0u; i < uFPNum; i++)       {   uF_temp[i*6 +4] = uTempF[i] -1u;            }
        //-----------------------------------------------------
        if (fread(fTempF, sizeof(float), uFPNum,fp1) != uFPNum) {   exit(10);          }//stress rate (still MPa/yr)
        for (i = 0u; i < uFPNum; i++)       {   fF_temp[i*15 +3] = fTempF[i];          }
        if (fread(fTempF, sizeof(float), uFPNum,fp1) != uFPNum) {   exit(10);          }//slip rate (still mm/yr)
        for (i = 0u; i < uFPNum; i++)       {   fF_temp[i*15 +4] = fTempF[i];          }
        if (fread(fTempF, sizeof(float), uFPNum,fp1) != uFPNum) {   exit(10);          }//rake
        for (i = 0u; i < uFPNum; i++)       {   fF_temp[i*15 +5] = fTempF[i];          }
        //-----------------------------------------------------
        if (fread(fTempF, sizeof(float), uFPNum,fp1) != uFPNum)     {   exit(10);       }//TrigCenter East (in meters)
        for (i = 0u; i < uFPNum; i++)       {   fF_temp[i*15 +0] = fTempF[i]*1000.0f;   }
        if (fread(fTempF, sizeof(float), uFPNum,fp1) != uFPNum)     {   exit(10);       }//TrigCenter North (in meters)
        for (i = 0u; i < uFPNum; i++)       {   fF_temp[i*15 +1] = fTempF[i]*1000.0f;   }
        if (fread(fTempF, sizeof(float), uFPNum,fp1) != uFPNum)     {   exit(10);       }//TrigCenter Depth (in meters)
        for (i = 0u; i < uFPNum; i++)       {   fF_temp[i*15 +2] = fTempF[i]*1000.0f;   }
        //-----------------------------------------------------
        if (fread(fTempFv, sizeof(float), uFVNum,fp1) != uFVNum)      {   exit(10);     }//Easting position of vertex (in meters)
        for (i = 0u; i < uFVNum; i++)       {   fFV_temp[i*4 +0] = fTempFv[i]*1000.0f;  }
        if (fread(fTempFv, sizeof(float), uFVNum,fp1) != uFVNum)      {   exit(10);     }//Northing position of vertex  (in meters)
        for (i = 0u; i < uFVNum; i++)       {   fFV_temp[i*4 +1] = fTempFv[i]*1000.0f;  }
        if (fread(fTempFv, sizeof(float), uFVNum,fp1) != uFVNum)      {   exit(10);     }//Depth position of vertex  (in meters)
        for (i = 0u; i < uFVNum; i++)       {   fFV_temp[i*4 +2] = fTempFv[i]*1000.0f;  }
        if (fread(fTempFv, sizeof(float), uFVNum,fp1) != uFVNum)      {   exit(10);     }//Height (above fault plane) position of vertex  (in meters)
        for (i = 0u; i < uFVNum; i++)       {   fFV_temp[i*4 +3] = fTempFv[i]*1000.0f;  }
        //--------------------------------------------------------------------------------    
        fclose(fp1);
        //--------------------------------------------------------------------------------       
        if ((fp2 = fopen(cFileName2,"rb"))     == NULL)     {   printf("Error -cant open %s LoadInputParameter function...\n",cFileName2);      exit(10);     }    
        //--------------------------------------------------------------------------------  
        fseek(fp2, sizeof(unsigned int), SEEK_SET); //skip first entry again -which is number of fault elements (already known)
    
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +3] = fTempF[i];          }//ref static friction
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +5] = fTempF[i];          }//ref dynamic friction 
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +1] = fTempF[i];          }//ref normal stress
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +7] = fTempF[i];          }//ref DcValue
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +4] = fTempF[i]*0.01f*fFRef[(i-iFSTART[iRANK])*14 +3];          }//ref static fric VARI
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +6] = fTempF[i]*0.01f*fFRef[(i-iFSTART[iRANK])*14 +5];          }//ref dynamic fric VARI
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +8] = fTempF[i]*0.01f*fFRef[(i-iFSTART[iRANK])*14 +7];          }//ref Dcvalue VARI
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +3] += fTempF[i];          } //load and directly apply manual stat. friction modification to ref. static friction
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +5] += fTempF[i];          } //load and directly apply manual dyn. friction modification to ref. dyn friction
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +1] += fTempF[i];          } //load and directly apply manual normal stress modification to ref. normal stress
        if (fread(fTempF, sizeof(float), uFPNum,fp2) != uFPNum) {   exit(10);           }
        for (i = iFSTART[iRANK]; i < (iFSTART[iRANK]+iFOFFSET[iRANK]); i++)       {   fFRef[(i-iFSTART[iRANK])*14 +7] += fTempF[i];          } //load and directly apply manual Dc  modification to ref. Dc value
        //--------------------------------------------------------------------------------    
        fclose(fp2);
        //--------------------------------------------------------------------------------
        if ((fp3 = fopen(cFileName3,"rb")) != NULL)
        {   //-------------------------------------
            fseek(fp3, (2*sizeof(unsigned int)), SEEK_SET);//skip over the first 2 entries (element/vertex number); already have them
            //-------------------------------------
            if (fread(uTempB,    sizeof(unsigned int), uBPNum,fp3) != uBPNum)  {   exit(10); }//V1 address/position in vertlist
            for (i = 0u; i < uBPNum; i++)       {   uB_temp[i*4 +0] = uTempB[i] -1u;         }
            if (fread(uTempB,    sizeof(unsigned int), uBPNum,fp3) != uBPNum)  {   exit(10); }//V2 address/position in vertlist
            for (i = 0u; i < uBPNum; i++)       {   uB_temp[i*4 +1] = uTempB[i] -1u;         }
            if (fread(uTempB,    sizeof(unsigned int), uBPNum,fp3) != uBPNum)  {   exit(10); }//V3 address/position in vertlist
            for (i = 0u; i < uBPNum; i++)       {   uB_temp[i*4 +2] = uTempB[i] -1u;         }
            if (fread(uTempB,    sizeof(unsigned int), uBPNum,fp3) != uBPNum)  {   exit(10); }//seg id
            for (i = 0u; i < uBPNum; i++)       {   uB_temp[i*4 +3] = uTempB[i] -1u;         }
            //-------------------------------------
            if (fread(fTempB,    sizeof(float), uBPNum,fp3) != uBPNum)  {   exit(10);       }//TrigCenterE (in meters)
            for (i = 0u; i < uBPNum; i++)       {   fB_temp[i*15 +0] = fTempB[i]*1000.0f;   }
            if (fread(fTempB,    sizeof(float), uBPNum,fp3) != uBPNum)  {   exit(10);       }//TrigCenterN (in meters)
            for (i = 0u; i < uBPNum; i++)       {   fB_temp[i*15 +1] = fTempB[i]*1000.0f;   }
            if (fread(fTempB,    sizeof(float), uBPNum,fp3) != uBPNum)  {   exit(10);       }//TrigCenterZ (in meters)
            for (i = 0u; i < uBPNum; i++)       {   fB_temp[i*15 +2] = fTempB[i]*1000.0f;   }
            //-------------------------------------
            if (fread(fTempBv,    sizeof(float), uBVNum,fp3) != uBVNum)  {   exit(10);      }//VertexLocation_E (in meters)
            for (i = 0u; i < uBVNum; i++)       {   fBV_temp[i*3 +0] = fTempBv[i]*1000.0f;  }
            if (fread(fTempBv,    sizeof(float), uBVNum,fp3) != uBVNum)  {   exit(10);      }//VertexLocation_N (in meters)
            for (i = 0u; i < uBVNum; i++)       {   fBV_temp[i*3 +1] = fTempBv[i]*1000.0f;  }
            if (fread(fTempBv,    sizeof(float), uBVNum,fp3) != uBVNum)  {   exit(10);      }//VertexLocation_Z (in meters)
            for (i = 0u; i < uBVNum; i++)       {   fBV_temp[i*3 +2] = fTempBv[i]*1000.0f;  }
            //-------------------------------------
            if (fread(fTempB,    sizeof(float), uBPNum,fp3) != uBPNum)  {   exit(10);       }//slip-rate in strike (m/yr)
            for (i = 0u; i < uBPNum; i++)       {   fB_temp[i*15 +3] = fTempB[i]*0.001f;    }
            if (fread(fTempB,    sizeof(float), uBPNum,fp3) != uBPNum)  {   exit(10);       }//slip-rate in dip (m/yr)
            for (i = 0u; i < uBPNum; i++)       {   fB_temp[i*15 +4] = fTempB[i]*0.001f;    }
            if (fread(fTempB,    sizeof(float), uBPNum,fp3) != uBPNum)  {   exit(10);       }//slip-rate in normal (m/yr)
            for (i = 0u; i < uBPNum; i++)       {   fB_temp[i*15 +5] = fTempB[i]*0.001f;    }
            //-------------------------------------
            fclose(fp3);
        }
        else    {   fprintf(stdout,"\nWarning -cant open  %s; Continue without boundary surface\n",cFileName3);     }
        //--------------------------------------------------------------------------------
        fFltSide = 0.0f;   fBndSide = 0.0f;   fElemArea = 0.0f,   fBoundArea = 0.0f;
        for (i = 0u; i < uFPNum; i++)
        {   fP1[0] = fFV_temp[uF_temp[i*6+ 0]*4 +0];                    fP1[1] = fFV_temp[uF_temp[i*6+ 0]*4 +1];                fP1[2] = fFV_temp[uF_temp[i*6+ 0]*4 +2];
            fP2[0] = fFV_temp[uF_temp[i*6+ 1]*4 +0];                    fP2[1] = fFV_temp[uF_temp[i*6+ 1]*4 +1];                fP2[2] = fFV_temp[uF_temp[i*6+ 1]*4 +2];
            fP3[0] = fFV_temp[uF_temp[i*6+ 2]*4 +0];                    fP3[1] = fFV_temp[uF_temp[i*6+ 2]*4 +1];                fP3[2] = fFV_temp[uF_temp[i*6+ 2]*4 +2];
            fP1P2[0] = fP2[0] - fP1[0];                                 fP1P2[1] = fP2[1] - fP1[1];                             fP1P2[2] = fP2[2] - fP1[2];  
            fP1P3[0] = fP3[0] - fP1[0];                                 fP1P3[1] = fP3[1] - fP1[1];                             fP1P3[2] = fP3[2] - fP1[2];
            fP2P3[0] = fP3[0] - fP2[0];                                 fP2P3[1] = fP3[1] - fP2[1];                             fP2P3[2] = fP3[2] - fP2[2];
            fP1Pc[0] = fF_temp[i*15 +0] -fP1[0];                        fP1Pc[1] = fF_temp[i*15 +1] -fP1[1];                    fP1Pc[2] = fF_temp[i*15 +2] -fP1[2];
            fP2Pc[0] = fF_temp[i*15 +0] -fP2[0];                        fP2Pc[1] = fF_temp[i*15 +1] -fP2[1];                    fP2Pc[2] = fF_temp[i*15 +2] -fP2[2];

            fArea        = 0.5f*sqrtf( powf(fP1P2[1]*fP1Pc[2]-fP1P2[2]*fP1Pc[1],2.0f) + powf(fP1P2[2]*fP1Pc[0]-fP1P2[0]*fP1Pc[2],2.0f) + powf(fP1P2[0]*fP1Pc[1]-fP1P2[1]*fP1Pc[0],2.0f));
            fTemp        = sqrtf(fP1P2[0]*fP1P2[0] + fP1P2[1]*fP1P2[1] + fP1P2[2]*fP1P2[2]); 
            fMeanHeight  = fMeanHeight + ((2.0f*fArea)/fTemp);

            fArea        = 0.5f*sqrtf( powf(fP1P3[1]*fP1Pc[2]-fP1P3[2]*fP1Pc[1],2.0f) + powf(fP1P3[2]*fP1Pc[0]-fP1P3[0]*fP1Pc[2],2.0f) + powf(fP1P3[0]*fP1Pc[1]-fP1P3[1]*fP1Pc[0],2.0f));
            fTemp        = sqrtf(fP1P3[0]*fP1P3[0] + fP1P3[1]*fP1P3[1] + fP1P3[2]*fP1P3[2]); 
            fMeanHeight  = fMeanHeight + ((2.0f*fArea)/fTemp);
            
            fArea        = 0.5f*sqrtf( powf(fP2P3[1]*fP2Pc[2]-fP2P3[2]*fP2Pc[1],2.0f) + powf(fP2P3[2]*fP2Pc[0]-fP2P3[0]*fP2Pc[2],2.0f) + powf(fP2P3[0]*fP2Pc[1]-fP2P3[1]*fP2Pc[0],2.0f));
            fTemp        = sqrtf(fP2P3[0]*fP2P3[0] + fP2P3[1]*fP2P3[1] + fP2P3[2]*fP2P3[2]); 
            fMeanHeight  = fMeanHeight + ((2.0f*fArea)/fTemp);

            fFltSide  += (sqrtf(fP1P2[0]*fP1P2[0] + fP1P2[1]*fP1P2[1] + fP1P2[2]*fP1P2[2]) + sqrtf(fP1P3[0]*fP1P3[0] + fP1P3[1]*fP1P3[1] + fP1P3[2]*fP1P3[2]) + sqrtf(fP2P3[0]*fP2P3[0] + fP2P3[1]*fP2P3[1] + fP2P3[2]*fP2P3[2]))/3.0f;
            fElemArea +=  0.5f*sqrtf( (fP1P2[1]*fP1P3[2]-fP1P2[2]*fP1P3[1])*(fP1P2[1]*fP1P3[2]-fP1P2[2]*fP1P3[1]) + (fP1P2[2]*fP1P3[0]-fP1P2[0]*fP1P3[2])*(fP1P2[2]*fP1P3[0]-fP1P2[0]*fP1P3[2]) + (fP1P2[0]*fP1P3[1]-fP1P2[1]*fP1P3[0])*(fP1P2[0]*fP1P3[1]-fP1P2[1]*fP1P3[0]) );
        }
        fFltSide    = fFltSide/(float)uFPNum;
        fElemArea   = fElemArea/(float)uFPNum;
        fMeanHeight = fMeanHeight/((float)uFPNum*3.0f); //fMeanHeight refers to height of center point triangle element side => to get average distance from center to sides (for signal travel time etc.)
        //-------------------------------
        for (i = 0u; i < uBPNum; i++)
        {   fP1[0] = fBV_temp[uB_temp[i*4+ 0]*3 +0];                    fP1[1] = fBV_temp[uB_temp[i*4+ 0]*3 +1];                fP1[2] = fBV_temp[uB_temp[i*4+ 0]*3 +2];
            fP2[0] = fBV_temp[uB_temp[i*4+ 1]*3 +0];                    fP2[1] = fBV_temp[uB_temp[i*4+ 1]*3 +1];                fP2[2] = fBV_temp[uB_temp[i*4+ 1]*3 +2];
            fP3[0] = fBV_temp[uB_temp[i*4+ 2]*3 +0];                    fP3[1] = fBV_temp[uB_temp[i*4+ 2]*3 +1];                fP3[2] = fBV_temp[uB_temp[i*4+ 2]*3 +2];
            fP1P2[0] = fP2[0] - fP1[0];                                 fP1P2[1] = fP2[1] - fP1[1];                             fP1P2[2] = fP2[2] - fP1[2];  
            fP1P3[0] = fP3[0] - fP1[0];                                 fP1P3[1] = fP3[1] - fP1[1];                             fP1P3[2] = fP3[2] - fP1[2];
            fP2P3[0] = fP3[0] - fP2[0];                                 fP2P3[1] = fP3[1] - fP2[1];                             fP2P3[2] = fP3[2] - fP2[2];
    
            fBndSide   += (sqrtf(fP1P2[0]*fP1P2[0] + fP1P2[1]*fP1P2[1] + fP1P2[2]*fP1P2[2]) + sqrtf(fP1P3[0]*fP1P3[0] + fP1P3[1]*fP1P3[1] + fP1P3[2]*fP1P3[2]) + sqrtf(fP2P3[0]*fP2P3[0] + fP2P3[1]*fP2P3[1] + fP2P3[2]*fP2P3[2]))/3.0f;
            fBoundArea +=  0.5f*sqrtf( (fP1P2[1]*fP1P3[2]-fP1P2[2]*fP1P3[1])*(fP1P2[1]*fP1P3[2]-fP1P2[2]*fP1P3[1]) + (fP1P2[2]*fP1P3[0]-fP1P2[0]*fP1P3[2])*(fP1P2[2]*fP1P3[0]-fP1P2[0]*fP1P3[2]) + (fP1P2[0]*fP1P3[1]-fP1P2[1]*fP1P3[0])*(fP1P2[0]*fP1P3[1]-fP1P2[1]*fP1P3[0]) );
        }
        fBndSide    = (uBPNum > 0u) ? fBndSide/(float)uBPNum   : fBndSide;
        fBoundArea  = (uBPNum > 0u) ? fBoundArea/(float)uBPNum : fBoundArea;
        //--------------------------------------------------------------------------------
        fUnitSlipF = fFltSide *1.0E-5f; 
        fUnitSlipB = (fBoundArea/fElemArea)*fUnitSlipF;
        //--------------------------------------------------------------------------------
        fModPara[2] = fModPara[2]*1.0E+9f; //this is bringing shear modulus to Pa
        fModPara[4] =(2.0f*fModPara[2]*fModPara[3])/(1.0f-2.0f*fModPara[3]); // this is lambda
        fTemp       = (fModPara[0] > 0.0f)*fModPara[0] + (fModPara[0] <= 0.0f)*2700.0f; //need to define SOME density in order to calculate vp/vs velocities
        fModPara[5] = sqrtf((fModPara[4] +2.0f*fModPara[2])/fTemp); //this is vp-velocity      
        fModPara[6] = (USEVPVSVELOCITY == 1u)*(fModPara[5]/sqrtf(fModPara[2]/fTemp)) + (USEVPVSVELOCITY != 1u)*1.0f; //the vp/vs ratio
        fModPara[7] = (2.0f*fMeanHeight)/fModPara[5];//the "real" deltaT, defined by p-wave travel time from one element center to next (distance is ~0.6 SideLengths)
        fModPara[8] = (uUseTimeProp == 1u)*fModPara[7] + (uUseTimeProp != 1u)*FLT_MAX;//the deltaT that is used for signal prop., if timeprop is not used, this happens instantaneous (i.e., setting distance traveled during one iteration step to FLT_MAX)
        fModPara[9] = sqrtf(fModPara[2]/fTemp); //shear wave velocity
        fModPara[10]= fMinCoSeisSlipRate*fModPara[7];//this is the minium slip increment that an element needs to procduce (in current iteration step) in order to be considered still slipping; is in meters
        //--------------------------------------------------------------------------------
        if (iRANK == 0) 
        {   fprintf(stdout,"----------------------\n");
            fprintf(stdout,"Number of RANKS: %u\n",iSIZE);
            fprintf(stdout,"----------------------\n");
            fprintf(stdout,"System info: Byte Size for FLOAT: %lu     unsigned INT: %lu    \n", sizeof(float), sizeof(unsigned int));
            fprintf(stdout,"----------------------\n");
            fprintf(stdout,"FileName:           %s\n",cInputName);                      fprintf(stdout,"RunNumber:          %u\n",uRunNum);
            fprintf(stdout,"CatalogType:        %u\n",uCatType);                        fprintf(stdout,"PlotCat2Screen:     %u\n",uPlotCatalog2Screen);            
            fprintf(stdout,"SeedLocation:       %ld\n",lSeed);                          fprintf(stdout,"MinElem4Catalog:    %u\n",uMinElemNum4Cat);            
            fprintf(stdout,"ContPreviousRun:    %u\n",uContPrevRun);                    fprintf(stdout,"LoadPrev_Kh_mat:    %u\n",uLoadPrev_Kmat);  
            fprintf(stdout,"K_mat file name:    %s\n",cKmatFileName);                  
            fprintf(stdout,"----------------------\n");  
            fprintf(stdout,"SaveSTF4LargeEQ:    %u\n",uStoreSTF4LargeEQs);              fprintf(stdout,"MinMag2SaveSTF:     %f\n",fMinMag4STF); 
            fprintf(stdout,"UseTime4RuptProp:       THIS IS NOT USED/INCLUDED IN V2p5\n");  fprintf(stdout,"MinMag2UseRuptProp:     THIS IS NOT USED/INCLUDED IN V2p5\n");
            fprintf(stdout,"----------------------\n"); 
            fprintf(stdout,"IntSeisTimeStep:    %2.3f days\n",fIntSeisLoadStep);        fprintf(stdout,"LoadSteps_POW2:     %u\n",uLoadStep_POW2);  
            fprintf(stdout,"----------------------\n"); 
            fprintf(stdout,"ViscAftSlip:        %3.1f\n",fAfterSlipTime);               fprintf(stdout,"ViscDeepRelax:      %3.1f\n",fDeepRelaxTime);
            fprintf(stdout,"----------------------\n"); 
            fprintf(stdout,"CoSeisHealFract:  %5.2f\n",fHealFact);                      fprintf(stdout,"OvershootFract:   %5.2f\n",fOvershootFract);
            fprintf(stdout,"PreStressFract:   %5.2f\n",fPreStressFract);                fprintf(stdout,"MinCoSeisSlipRate (m/s):  %e\n",fMinCoSeisSlipRate);
            fprintf(stdout,"----------------------\n"); 
            fprintf(stdout,"RecLength:          %5.1lf\n",dRecordLength);               
            fprintf(stdout,"----------------------\n"); 
            fprintf(stdout,"Elastic properties\n");
            fprintf(stdout,"Medium density (kg/m^3):   %6.2f\n",fModPara[0]);           fprintf(stdout,"AddedNormalStress (MPa):  %6.2f\n",fModPara[1]);
            fprintf(stdout,"ShearModulus (Pa):         %e\n",fModPara[2]);              fprintf(stdout,"PoissonRatio:             %6.3f\n",fModPara[3]);
            fprintf(stdout,"ChangeFricBtwEQs:          %u\n",uChgBtwEQs);               fprintf(stdout,"Lambda (Pa):               %e\n",fModPara[4]);
            fprintf(stdout,"P-waveVelocity (m/s):      %6.3f\n",fModPara[5]);           fprintf(stdout,"vP/vS ratio:              %6.3f\n",fModPara[6]);
            fprintf(stdout,"unit slip (m, fault):     %6.3f\n",fUnitSlipF);             fprintf(stdout,"unit slip (m, bound):     %6.3f\n",fUnitSlipB);
            fprintf(stdout,"min. slip 2 start EQ (m):  %2.3e\n",fModPara[10]); 
            fprintf(stdout,"coseis. timestep (s):      %2.3e\n",fModPara[7]);           fprintf(stdout,"mean height (m):           %6.3e\n",fMeanHeight); 
            //-----------------------------
            float fTemp;
            fTemp = 100.0f - expf(-1.0f/fAfterSlipTime)*100.0f; //factor/fraction from decay function
            fprintf(stdout,"\nFractional post-seismic change during first year of after-slip: %2.4f percent released \n",fTemp);
            fTemp = 100.0f - expf(-1.0f/fDeepRelaxTime)*100.0f; //factor/fraction from decay function
            fprintf(stdout,"Fractional post-seismic change during first year of 'deep relaxation': %2.4f percent released \n",fTemp);
            //-----------------------------
            fprintf(stdout,"FaultPatchNumber %d     FaultVertexNumber %d\n", uFPNum, uFVNum);
            fprintf(stdout,"BoundPatchNumber %d     BoundVertexNumber %d\n", uBPNum, uBVNum);
            fprintf(stdout,"FaultSideLenth and ElementArea: %5.2fm  and %4.3fkm^2;   BoundarySideLength and ElementArea: %5.2fm  and %4.3fkm^2\n\n",   fFltSide, fElemArea*1.0E-6f, fBndSide,fBoundArea*1.0E-6f);
        } //some print to screen to sure that the data were imported from file correctly
    } //loading input data -the fault/strength info
    //------------------------------------------------------------------------------------
    lSeed += (long)iRANK;
    //------------------------------------------------------------------------------------
    {   unsigned int uGlobPos;
        float fP1[3], fP2[3], fP3[3], fP1P2[3], fP1P3[3], fNrm[3], feZ[3], fStk[3], fDip[3];
        float fVectLgth;
        
        for (i = 0u; i < iFOFFSET[iRANK]; i++) //here the current static/dynamic friction coefficients are determines, also the current Dc value; further, the stressing rates are assigned
        {   uGlobPos   = i + iFSTART[iRANK];
            fFEvent[i*16 +8]  =  cosf(fF_temp[uGlobPos*15 +5]*(M_PI/180.0f)) *fF_temp[uGlobPos*15 +3]*1.0E+6f;//stressing rate in strike direction (from applied stress rate and rake)
            fFEvent[i*16 +9]  =  sinf(fF_temp[uGlobPos*15 +5]*(M_PI/180.0f)) *fF_temp[uGlobPos*15 +3]*1.0E+6f;;//stressing rate in dip direction (from applied stress rate and rake)
            fFEvent[i*16+13]  =  cosf(fF_temp[uGlobPos*15 +5]*(M_PI/180.0f)) *1.0E-3f*fF_temp[uGlobPos*15 +4];//slip rate (now in m/yr) in strike direction (from applied stress rate and rake)
            fFEvent[i*16+14]  =  sinf(fF_temp[uGlobPos*15 +5]*(M_PI/180.0f)) *1.0E-3f*fF_temp[uGlobPos*15 +4];//slip rate (now in m/yr)in dip direction (from applied stress rate and rake)
            //temporarily store the slip rates in position where currentSlip in strike/dip will be held (in position 10 and 11 of fFEevent)
            //----------------------------------------------------------------------------
            fP1[0] = fFV_temp[uF_temp[uGlobPos*6+ 0]*4 +0];             fP1[1] = fFV_temp[uF_temp[uGlobPos*6+ 0]*4 +1];         fP1[2] = fFV_temp[uF_temp[uGlobPos*6+ 0]*4 +2];
            fP2[0] = fFV_temp[uF_temp[uGlobPos*6+ 1]*4 +0];             fP2[1] = fFV_temp[uF_temp[uGlobPos*6+ 1]*4 +1];         fP2[2] = fFV_temp[uF_temp[uGlobPos*6+ 1]*4 +2];
            fP3[0] = fFV_temp[uF_temp[uGlobPos*6+ 2]*4 +0];             fP3[1] = fFV_temp[uF_temp[uGlobPos*6+ 2]*4 +1];         fP3[2] = fFV_temp[uF_temp[uGlobPos*6+ 2]*4 +2];
            fP1P2[0] = fP2[0] - fP1[0];                                 fP1P2[1] = fP2[1] - fP1[1];                             fP1P2[2] = fP2[2] - fP1[2];  
            fP1P3[0] = fP3[0] - fP1[0];                                 fP1P3[1] = fP3[1] - fP1[1];                             fP1P3[2] = fP3[2] - fP1[2];
            fNrm[0]  = fP1P2[1]*fP1P3[2] - fP1P2[2]*fP1P3[1];           fNrm[1]  = fP1P2[2]*fP1P3[0] - fP1P2[0]*fP1P3[2];       fNrm[2]  = fP1P2[0]*fP1P3[1] - fP1P2[1]*fP1P3[0];
            //----------------------------------------------------------------------------
            fVectLgth = sqrtf( fNrm[0]*fNrm[0] +fNrm[1]*fNrm[1] +fNrm[2]*fNrm[2]);
            fFRef[i*14 +0]  = 0.5f *fVectLgth; //is area of this fault element, needed for magnitude calculation
            fFRef[i*14 +1]  = fModPara[0] * 9.81f * fF_temp[uGlobPos*15 +2] +(fModPara[1]*1.0E+6f); //reference normal stress
            fFEvent[i*16 +2]= fFRef[i*14 +1];
            fFRef[i*14 +9]  = fF_temp[uGlobPos*15 +0];
            fFRef[i*14 +10] = fF_temp[uGlobPos*15 +1];
            fFRef[i*14 +11] = fF_temp[uGlobPos*15 +2];
            //----------------------------------------------------------------------------
        }
        for (i = 0u; i < uFPNum; i++)
        {   fP1[0] = fFV_temp[uF_temp[i*6+ 0]*4 +0];                    fP1[1]   = fFV_temp[uF_temp[i*6+ 0]*4 +1];              fP1[2]   = fFV_temp[uF_temp[i*6+ 0]*4 +2];
            fP2[0] = fFV_temp[uF_temp[i*6+ 1]*4 +0];                    fP2[1]   = fFV_temp[uF_temp[i*6+ 1]*4 +1];              fP2[2]   = fFV_temp[uF_temp[i*6+ 1]*4 +2];
            fP3[0] = fFV_temp[uF_temp[i*6+ 2]*4 +0];                    fP3[1]   = fFV_temp[uF_temp[i*6+ 2]*4 +1];              fP3[2]   = fFV_temp[uF_temp[i*6+ 2]*4 +2];
            fP1P2[0] = fP2[0] - fP1[0];                                 fP1P2[1] = fP2[1] - fP1[1];                             fP1P2[2] = fP2[2] - fP1[2];  
            fP1P3[0] = fP3[0] - fP1[0];                                 fP1P3[1] = fP3[1] - fP1[1];                             fP1P3[2] = fP3[2] - fP1[2];
            fNrm[0]  = fP1P2[1]*fP1P3[2] - fP1P2[2]*fP1P3[1];           fNrm[1]  = fP1P2[2]*fP1P3[0] - fP1P2[0]*fP1P3[2];       fNrm[2]  = fP1P2[0]*fP1P3[1] - fP1P2[1]*fP1P3[0];
            //----------------------------------
            fVectLgth = sqrtf( fNrm[0]*fNrm[0] +fNrm[1]*fNrm[1] +fNrm[2]*fNrm[2]);
            fNrm[0]  /= fVectLgth;                                      fNrm[1]  /= fVectLgth;                                  fNrm[2]  /= fVectLgth;
            //----------------------------------------------------------------------------
            feZ[0] = 0.0f;                                              feZ[1] = 0.0f;                                          feZ[2] = 1.0f;
            fStk[0]= feZ[1]*fNrm[2] - feZ[2]*fNrm[1];                   fStk[1]= feZ[2]*fNrm[0] - feZ[0]*fNrm[2];               fStk[2]= feZ[0]*fNrm[1] - feZ[1]*fNrm[0];
            fVectLgth = sqrtf( fStk[0]*fStk[0] +fStk[1]*fStk[1] +fStk[2]*fStk[2]);
            if (fVectLgth < FLT_EPSILON)
            {   fStk[0]    = 0.0f;                                      fStk[1] = 1.0f;                                         fStk[2] = 0.0f;                    }
            else
            {   fStk[0]   /= fVectLgth;                                 fStk[1]/= fVectLgth;                                    fStk[2]/= fVectLgth;               }
            //----------------------------------------------------------------------------
            fDip[0] = fNrm[1]*fStk[2] - fNrm[2]*fStk[1];                fDip[1] = fNrm[2]*fStk[0] - fNrm[0]*fStk[2];            fDip[2] = fNrm[0]*fStk[1] - fNrm[1]*fStk[0];
            fVectLgth  = sqrtf(fDip[0]*fDip[0] +fDip[1]*fDip[1] +fDip[2]*fDip[2]);
            fDip[0]   /= fVectLgth;                                     fDip[1]/= fVectLgth;                                    fDip[2]/= fVectLgth;
            //----------------------------------------------------------------------------
            fF_temp[i*15 + 6] = fNrm[0];                                fF_temp[i*15 + 7] = fNrm[1];                            fF_temp[i*15 + 8] = fNrm[2];   //the element's normal vector
            fF_temp[i*15 + 9] = fStk[0];                                fF_temp[i*15 +10] = fStk[1];                            fF_temp[i*15 +11] = fStk[2];   //the element's strike vector
            fF_temp[i*15 +12] = fDip[0];                                fF_temp[i*15 +13] = fDip[1];                            fF_temp[i*15 +14] = fDip[2];   //the element's dip vector
        }
        for (i = 0u; i < uBPNum; i++)
        {   fP1[0] = fBV_temp[uB_temp[i*4+ 0]*3 +0];                    fP1[1] = fBV_temp[uB_temp[i*4+ 0]*3 +1];                fP1[2] = fBV_temp[uB_temp[i*4+ 0]*3 +2];
            fP2[0] = fBV_temp[uB_temp[i*4+ 1]*3 +0];                    fP2[1] = fBV_temp[uB_temp[i*4+ 1]*3 +1];                fP2[2] = fBV_temp[uB_temp[i*4+ 1]*3 +2];
            fP3[0] = fBV_temp[uB_temp[i*4+ 2]*3 +0];                    fP3[1] = fBV_temp[uB_temp[i*4+ 2]*3 +1];                fP3[2] = fBV_temp[uB_temp[i*4+ 2]*3 +2];
            fP1P2[0] = fP2[0] - fP1[0];                                 fP1P2[1] = fP2[1] - fP1[1];                             fP1P2[2] = fP2[2] - fP1[2];  
            fP1P3[0] = fP3[0] - fP1[0];                                 fP1P3[1] = fP3[1] - fP1[1];                             fP1P3[2] = fP3[2] - fP1[2];
            fNrm[0]  = fP1P2[1]*fP1P3[2] - fP1P2[2]*fP1P3[1];           fNrm[1]  = fP1P2[2]*fP1P3[0] - fP1P2[0]*fP1P3[2];       fNrm[2]  = fP1P2[0]*fP1P3[1] - fP1P2[1]*fP1P3[0];         
            //----------------------------------
            fVectLgth = sqrtf( fNrm[0]*fNrm[0] +fNrm[1]*fNrm[1] +fNrm[2]*fNrm[2]);
            fNrm[0]  /= fVectLgth;                                      fNrm[1]  /= fVectLgth;                                  fNrm[2]  /= fVectLgth;
            //----------------------------------------------------------------------------
            feZ[0] = 0.0f;                                              feZ[1] = 0.0f;                                          feZ[2] = 1.0f;
            fStk[0]= feZ[1]*fNrm[2] - feZ[2]*fNrm[1];                   fStk[1]= feZ[2]*fNrm[0] - feZ[0]*fNrm[2];               fStk[2]= feZ[0]*fNrm[1] - feZ[1]*fNrm[0];
            fVectLgth = sqrtf( fStk[0]*fStk[0] +fStk[1]*fStk[1] +fStk[2]*fStk[2]);
            if (fVectLgth < FLT_EPSILON)
            {   fStk[0]    = 0.0f;                                      fStk[1] = 1.0f;                                         fStk[2] = 0.0f;                    }
            else
            {   fStk[0]   /= fVectLgth;                                 fStk[1]/= fVectLgth;                                    fStk[2]/= fVectLgth;               }
            //----------------------------------------------------------------------------
            fDip[0] = fNrm[1]*fStk[2] - fNrm[2]*fStk[1];                fDip[1] = fNrm[2]*fStk[0] - fNrm[0]*fStk[2];            fDip[2] = fNrm[0]*fStk[1] - fNrm[1]*fStk[0];
            fVectLgth  = sqrtf(fDip[0]*fDip[0] +fDip[1]*fDip[1] +fDip[2]*fDip[2]);
            fDip[0]   /= fVectLgth;                                     fDip[1]/= fVectLgth;                                    fDip[2]/= fVectLgth;
            //----------------------------------------------------------------------------
            fB_temp[i*15 +6]  = fNrm[0];                                fB_temp[i*15 +7]  = fNrm[1];                            fB_temp[i*15 +8]  = fNrm[2];   //the element's normal vector
            fB_temp[i*15 +9]  = fStk[0];                                fB_temp[i*15 +10] = fStk[1];                            fB_temp[i*15 +11] = fStk[2];   //the element's strike vector
            fB_temp[i*15 +12] = fDip[0];                                fB_temp[i*15 +13] = fDip[1];                            fB_temp[i*15 +14] = fDip[2];   //the element's dip vector
        }
    } //assigning reference values (local coordinate system, fault element area, first realization of curent friction values )
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    float **fK_FFvalstk = NULL;
    float **fK_FFvaldip = NULL;
    float **fK_FFvalnrm = NULL;
    fK_FFvalstk = calloc(iFOFFSET[iRANK], sizeof *fK_FFvalstk );
    fK_FFvaldip = calloc(iFOFFSET[iRANK], sizeof *fK_FFvaldip );
    fK_FFvalnrm = calloc(iFOFFSET[iRANK], sizeof *fK_FFvalnrm );
    //---------------------------------------
    float **fK_FBvalstk = NULL;
    float **fK_FBvaldip = NULL;
    fK_FBvalstk = calloc(iFOFFSET[iRANK], sizeof *fK_FBvalstk );
    fK_FBvaldip = calloc(iFOFFSET[iRANK], sizeof *fK_FBvaldip );
    //---------------------------------------
    float **fK_BBvalstk = NULL;
    float **fK_BBvaldip = NULL;
    float **fK_BBvalnrm = NULL;
    fK_BBvalstk = calloc(iBOFFSET[iRANK], sizeof *fK_BBvalstk );
    fK_BBvaldip = calloc(iBOFFSET[iRANK], sizeof *fK_BBvaldip );
    fK_BBvalnrm = calloc(iBOFFSET[iRANK], sizeof *fK_BBvalnrm );
    //---------------------------------------
    float **fK_BFvalstk = NULL;
    float **fK_BFvaldip = NULL;
    float **fK_BFvalnrm = NULL;
    fK_BFvalstk = calloc(iBOFFSET[iRANK], sizeof *fK_BFvalstk );
    fK_BFvaldip = calloc(iBOFFSET[iRANK], sizeof *fK_BFvaldip );
    fK_BFvalnrm = calloc(iBOFFSET[iRANK], sizeof *fK_BFvalnrm );
    ///------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    if ((uLoadPrev_Kmat == 0) || (uLoadPrev_Kmat == 1)) // 0 = make new, 1 = make new and save, 2 = load previous
    {   timer0 = clock();
        unsigned int uGlobPos;
        float fTemp0,   fTemp1,   fTemp2,   fDist;
        float fRcv[3],   fSrc[3],   fP1[3],   fP2[3],   fP3[3],   fStressStk[6],   fStressDip[6],   fStressNrm[6],   fStrain[6];
        float fNrm[3],   fStk[3],   fDip[3];
        //----------------------------------------------------------------------------
        for (i = 0u; i < iFOFFSET[iRANK]; i++) //going through the individual receivers that the rank controls
        {   //----------------------------------------------
            fK_FFvalstk[i] = calloc(   2*uFPNum, sizeof *fK_FFvalstk[i] );
            fK_FFvaldip[i] = calloc(   2*uFPNum, sizeof *fK_FFvaldip[i] );
            fK_FFvalnrm[i] = calloc(   2*uFPNum, sizeof *fK_FFvalnrm[i] );
            //----------------------------------------------
            fK_FBvalstk[i] = calloc(   3*uBPNum, sizeof *fK_FBvalstk[i] );
            fK_FBvaldip[i] = calloc(   3*uBPNum, sizeof *fK_FBvaldip[i] );
            //----------------------------------------------
            uGlobPos = i+ iFSTART[iRANK];
            fRcv[0]  = fF_temp[uGlobPos*15 + 0];        fRcv[1]  = fF_temp[uGlobPos*15 + 1];        fRcv[2]  = fF_temp[uGlobPos*15 + 2];
            //----------------------------------------------
            fNrm[0]  = fF_temp[uGlobPos*15 + 6];        fNrm[1]  = fF_temp[uGlobPos*15 + 7];        fNrm[2]  = fF_temp[uGlobPos*15 + 8];
            fStk[0]  = fF_temp[uGlobPos*15 + 9];        fStk[1]  = fF_temp[uGlobPos*15 +10];        fStk[2]  = fF_temp[uGlobPos*15 +11];
            fDip[0]  = fF_temp[uGlobPos*15 +12];        fDip[1]  = fF_temp[uGlobPos*15 +13];        fDip[2]  = fF_temp[uGlobPos*15 +14];
            //-------------------------------------------------------
            for (j = 0u; j < uFPNum; j++) //going through all sources
            {   //-------------------------------------------------------
                fSrc[0]  = fF_temp[j*15 +0];                               fSrc[1]  = fF_temp[j*15 +1];                           fSrc[2]  = fF_temp[j*15 +2];
                fDist = sqrtf((fSrc[0]-fRcv[0])*(fSrc[0]-fRcv[0]) + (fSrc[1]-fRcv[1])*(fSrc[1]-fRcv[1]) + (fSrc[2]-fRcv[2])*(fSrc[2]-fRcv[2]));
                //-------------------------------------------------------
                fTemp0 = (fDist >= CUTDISTANCE*fFltSide)*1.0f + (fDist < CUTDISTANCE*fFltSide)*0.0f; //if distance between source and receiver centers is less than CUTDISTANCE then set it to zero
                fTemp0 = (uF_temp[uGlobPos*6 +4] == uF_temp[j*6 +4])*1.0f + (uF_temp[uGlobPos*6 +4] != uF_temp[j*6 +4])*fTemp0;//if the source and receiver are from the same FaultID, set to one (regardless of distance)
                uF_temp[uGlobPos*6 +5] = (fTemp0 == 1.0f)*uF_temp[uGlobPos*6 +5] + (fTemp0 != 1.0f)*1u;//interaction is flagged if fTemp0 was set to zero i.e., if distance too small (and not the same fault)
                //-------------------------------------------------------
                fP1[0] = fFV_temp[uF_temp[j*6+ 0]*4 +0];                fP1[1] = fFV_temp[uF_temp[j*6+ 0]*4 +1];            fP1[2] = fFV_temp[uF_temp[j*6+ 0]*4 +2];
                fP2[0] = fFV_temp[uF_temp[j*6+ 1]*4 +0];                fP2[1] = fFV_temp[uF_temp[j*6+ 1]*4 +1];            fP2[2] = fFV_temp[uF_temp[j*6+ 1]*4 +2];
                fP3[0] = fFV_temp[uF_temp[j*6+ 2]*4 +0];                fP3[1] = fFV_temp[uF_temp[j*6+ 2]*4 +1];            fP3[2] = fFV_temp[uF_temp[j*6+ 2]*4 +2];
                //-------------------------------------------------------
                StrainHS_Nikkhoo(fStressStk, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, fUnitSlipF, 0.0f, 0.0f, fModPara[2], fModPara[4]);
                StrainHS_Nikkhoo(fStressDip, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, 0.0f, fUnitSlipF, 0.0f, fModPara[2], fModPara[4]);
                //-------------------------------------------------------
                fStressStk[0] *=(fTemp0/fUnitSlipF);      fStressStk[1] *=(fTemp0/fUnitSlipF);      fStressStk[2] *=(fTemp0/fUnitSlipF);
                fStressStk[3] *=(fTemp0/fUnitSlipF);      fStressStk[4] *=(fTemp0/fUnitSlipF);      fStressStk[5] *=(fTemp0/fUnitSlipF);
                fStressDip[0] *=(fTemp0/fUnitSlipF);      fStressDip[1] *=(fTemp0/fUnitSlipF);      fStressDip[2] *=(fTemp0/fUnitSlipF);
                fStressDip[3] *=(fTemp0/fUnitSlipF);      fStressDip[4] *=(fTemp0/fUnitSlipF);      fStressDip[5] *=(fTemp0/fUnitSlipF);
                //-------------------------------------------------------
                fK_FFvalstk[i][2*j +0] = fNrm[0]*fStk[0]*fStressStk[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressStk[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressStk[4] + fStk[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fStk[2]*fStressStk[5]; //induced strike component
                fK_FFvaldip[i][2*j +0] = fNrm[0]*fDip[0]*fStressStk[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressStk[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressStk[4] + fDip[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fDip[2]*fStressStk[5]; //induced dip component
                fK_FFvalnrm[i][2*j +0] = fNrm[0]*fNrm[0]*fStressStk[0] +              2.0f*fNrm[0]*fNrm[1] *fStressStk[1] +               2.0f*fNrm[0]*fNrm[2] *fStressStk[2] +               2.0f*fNrm[1]*fNrm[2] *fStressStk[4] + fNrm[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fNrm[2]*fStressStk[5]; //induced normal component
                //-------------------------------------------------------
                fK_FFvalstk[i][2*j +1] = fNrm[0]*fStk[0]*fStressDip[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressDip[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressDip[4] + fStk[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fStk[2]*fStressDip[5]; //induced strike component
                fK_FFvaldip[i][2*j +1] = fNrm[0]*fDip[0]*fStressDip[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressDip[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressDip[4] + fDip[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fDip[2]*fStressDip[5]; //induced dip component
                fK_FFvalnrm[i][2*j +1] = fNrm[0]*fNrm[0]*fStressDip[0] +              2.0f*fNrm[0]*fNrm[1] *fStressDip[1] +               2.0f*fNrm[0]*fNrm[2] *fStressDip[2] +               2.0f*fNrm[1]*fNrm[2] *fStressDip[4] + fNrm[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fNrm[2]*fStressDip[5]; //induced normal component
                //-------------------------------------------------------
                if (uGlobPos == j)
                {   //this will be for self-stiffness
                    fFEvent[i*16 +4]  = fK_FFvalstk[i][2*j +0];//self-stiffness in strike direction (induced strike-shear due to strike slip on self)
                    fFEvent[i*16 +5]  = fK_FFvaldip[i][2*j +1];//self-stiffness in dip direction (induced dip-shear due to dip slip on self)
                    fFRef[i*14 +2]    = MAX(fabs(fFEvent[i*16 +4]),fabs(fFEvent[i*16 +5])); //this is reference "self-stiffness"
                    //--------------------------------------------------------------------
                    fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
                    fFEvent[i*16 +7] = fFRef[i*14 +7] + (fFRef[i*14 +8] * fTemp0); //current Dc value; wird gleich wieder ueberschrieben/geupdated
                    
                    fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
                    fFFric[i*6 +0] = fFRef[i*14 +3] + (fFRef[i*14 +4] * fTemp0); //current static friction
                    fFEvent[i*16 +3] = fFFric[i*6 +0]; //use static friction also as current friction
                    
                    fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
                    fFFric[i*6 +1] = fFRef[i*14 +5] + (fFRef[i*14 +6] * fTemp0); // current dynamic friction
                    //--------------------------------------------------------------------
                    fTemp0 =(fFFric[i*6 +0] - fFFric[i*6 +1]) *-1.0f*fFRef[i*14 +1];; //strength difference between static and dynamic strength; positive when weakening
                    fTemp1 = fTemp0/fFRef[i*14 +2];//is slip required to release strength difference
                    uFEvent[i*5 +0] = (fTemp0 >= 0.0f)*2u + (fTemp0 < 0.0f)*3u;//gets a 2 if drop is not negative (in which case it would be strengthening i.e., stable == 3)
                    uFEvent[i*5 +0] = (fTemp1 > fFEvent[i*16 +7])*1u + (fTemp1 <= fFEvent[i*16 +7])*uFEvent[i*5 +0];//give it a 1 if it is unstable (slip associated w/ change from static to dynamic is large enough to over come Dc)
                    //---------------------------------
                    fTemp0 = fFFric[i*6 +1]*-1.0f*fFRef[i*14 +1] + fFRef[i*14 +2]*fFEvent[i*16 +7]; //this is dynFric*normal stress (ie., dyn strength) minus  selfstiff*DcVal; the latter is therefore the stress change needed/associated with slip Dc; adding both to see what fric strength would at lease need to have Dc amount of slip
                    fTemp1 = fFFric[i*6 +0]*-1.0f*fFRef[i*14 +1]; //this is the static strength
                    fTemp2 = MAX(fTemp0, fTemp1)/(-1.0f*fFRef[i*14 +1]); //friction coefficient...
                    fFFric[i*6 +2]  = fTemp2 - fOvershootFract*(fTemp2-fFFric[i*6 +1]); //this is arrest friction
                    //---------------------------------
                    fFEvent[i*16 +7] *= ((fTemp2 - fFFric[i*6 +2])/(fTemp2 - fFFric[i*6 +1])); //updated Dc value => now adjusted for the breakdown all the way to arrest stress
                    fFEvent[i*16 +10] = (fFFric[i*6 +1] - fFFric[i*6 +2])*-1.0f*fFRef[i*14 +1];//this is overshoot stress (difference between arrest (lower) and dynamic (higher) strength)
                    //--------------------------------------------------------------------
                }
            } //take all local sources and determine induced stress on selected receiver
            //------------------------------------------------------------------------
            for (j = 0u; j < uBPNum; j++) //going through all sources
            {   //-------------------------------------------------------
                fSrc[0]  = fB_temp[j*15 +0];                            fSrc[1]  = fB_temp[j*15 +1];                        fSrc[2]  = fB_temp[j*15 +2];
                fDist = sqrtf((fSrc[0]-fRcv[0])*(fSrc[0]-fRcv[0]) + (fSrc[1]-fRcv[1])*(fSrc[1]-fRcv[1]) + (fSrc[2]-fRcv[2])*(fSrc[2]-fRcv[2]));
                //-------------------------------------------------------; 
                fTemp0 = (fDist >= 1.0f*fBndSide)*1.0f + (fDist < 1.0f*fBndSide)*0.0f;//if distance between source and receiver centers is less than CUTDISTANCE then set it to zero
                uF_temp[uGlobPos*6 +5] = (fTemp0 == 1.0f)*uF_temp[uGlobPos*6 +5] + (fTemp0 != 1.0f)*1u;//interaction is flagged if fTemp0 was set to zero i.e., if distance too small (and not the same fault)
                //-------------------------------------------------------
                fP1[0] = fBV_temp[uB_temp[j*4+ 0]*3 +0];                fP1[1] = fBV_temp[uB_temp[j*4+ 0]*3 +1];            fP1[2] = fBV_temp[uB_temp[j*4+ 0]*3 +2];
                fP2[0] = fBV_temp[uB_temp[j*4+ 1]*3 +0];                fP2[1] = fBV_temp[uB_temp[j*4+ 1]*3 +1];            fP2[2] = fBV_temp[uB_temp[j*4+ 1]*3 +2];
                fP3[0] = fBV_temp[uB_temp[j*4+ 2]*3 +0];                fP3[1] = fBV_temp[uB_temp[j*4+ 2]*3 +1];            fP3[2] = fBV_temp[uB_temp[j*4+ 2]*3 +2];
                //-------------------------------------------------------
                StrainHS_Nikkhoo(fStressStk, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, fUnitSlipB, 0.0f, 0.0f, fModPara[2], fModPara[4]);
                StrainHS_Nikkhoo(fStressDip, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, 0.0f, fUnitSlipB, 0.0f, fModPara[2], fModPara[4]);
                StrainHS_Nikkhoo(fStressNrm, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, 0.0f, 0.0f, fUnitSlipB, fModPara[2], fModPara[4]);
                //-------------------------------------------------------
                fStressStk[0] *=(fTemp0/fUnitSlipB);      fStressStk[1] *=(fTemp0/fUnitSlipB);      fStressStk[2] *=(fTemp0/fUnitSlipB);
                fStressStk[3] *=(fTemp0/fUnitSlipB);      fStressStk[4] *=(fTemp0/fUnitSlipB);      fStressStk[5] *=(fTemp0/fUnitSlipB);
                fStressDip[0] *=(fTemp0/fUnitSlipB);      fStressDip[1] *=(fTemp0/fUnitSlipB);      fStressDip[2] *=(fTemp0/fUnitSlipB);
                fStressDip[3] *=(fTemp0/fUnitSlipB);      fStressDip[4] *=(fTemp0/fUnitSlipB);      fStressDip[5] *=(fTemp0/fUnitSlipB);
                fStressNrm[0] *=(fTemp0/fUnitSlipB);      fStressNrm[1] *=(fTemp0/fUnitSlipB);      fStressNrm[2] *=(fTemp0/fUnitSlipB);
                fStressNrm[3] *=(fTemp0/fUnitSlipB);      fStressNrm[4] *=(fTemp0/fUnitSlipB);      fStressNrm[5] *=(fTemp0/fUnitSlipB);
                //-------------------------------------------------------
                fK_FBvalstk[i][3*j +0] = fNrm[0]*fStk[0]*fStressStk[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressStk[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressStk[4] + fStk[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fStk[2]*fStressStk[5]; //induced strike component
                fK_FBvaldip[i][3*j +0] = fNrm[0]*fDip[0]*fStressStk[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressStk[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressStk[4] + fDip[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fDip[2]*fStressStk[5]; //induced dip component
                //-------------------------------------------------------
                fK_FBvalstk[i][3*j +1] = fNrm[0]*fStk[0]*fStressDip[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressDip[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressDip[4] + fStk[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fStk[2]*fStressDip[5]; //induced strike component
                fK_FBvaldip[i][3*j +1] = fNrm[0]*fDip[0]*fStressDip[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressDip[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressDip[4] + fDip[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fDip[2]*fStressDip[5]; //induced dip component
                //-------------------------------------------------------
                fK_FBvalstk[i][3*j +2] = fNrm[0]*fStk[0]*fStressNrm[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressNrm[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressNrm[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressNrm[4] + fStk[1]*fNrm[1]*fStressNrm[3] + fNrm[2]*fStk[2]*fStressNrm[5]; //induced strike component
                fK_FBvaldip[i][3*j +2] = fNrm[0]*fDip[0]*fStressNrm[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressNrm[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressNrm[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressNrm[4] + fDip[1]*fNrm[1]*fStressNrm[3] + fNrm[2]*fDip[2]*fStressNrm[5]; //induced dip component
                //-------------------------------------------------------
            } //take all local sources and determine induced stress on selected receiver
            //------------------------------------------------------------------------
            if ((uPlotCatalog2Screen == 1)&&(iRANK == 0)) {    fprintf(stdout,"\nfault receiver:  %u/%u ",i, iFOFFSET[iRANK]);                  }
        }
        //----------------------------------------------------------------------------
        for (i = 0u; i < iBOFFSET[iRANK]; i++) //going through the individual receivers that the rank controls
        {   //----------------------------------------------
            fK_BBvalstk[i] = calloc(   3*uBPNum, sizeof *fK_BBvalstk[i] );
            fK_BBvaldip[i] = calloc(   3*uBPNum, sizeof *fK_BBvaldip[i] );
            fK_BBvalnrm[i] = calloc(   3*uBPNum, sizeof *fK_BBvalnrm[i] );
            //----------------------------------------------
            fK_BFvalstk[i] = calloc(   2*uFPNum, sizeof *fK_BFvalstk[i] );
            fK_BFvaldip[i] = calloc(   2*uFPNum, sizeof *fK_BFvaldip[i] );
            fK_BFvalnrm[i] = calloc(   2*uFPNum, sizeof *fK_BFvalnrm[i] );
            //----------------------------------------------
            uGlobPos = i + iBSTART[iRANK];
            fRcv[0]  = fB_temp[uGlobPos*15 + 0];        fRcv[1]  = fB_temp[uGlobPos*15 + 1];        fRcv[2]  = fB_temp[uGlobPos*15 + 2];
            //----------------------------------------------
            fNrm[0]  = fB_temp[uGlobPos*15 + 6];        fNrm[1]  = fB_temp[uGlobPos*15 + 7];        fNrm[2]  = fB_temp[uGlobPos*15 + 8];
            fStk[0]  = fB_temp[uGlobPos*15 + 9];        fStk[1]  = fB_temp[uGlobPos*15 +10];        fStk[2]  = fB_temp[uGlobPos*15 +11];
            fDip[0]  = fB_temp[uGlobPos*15 +12];        fDip[1]  = fB_temp[uGlobPos*15 +13];        fDip[2]  = fB_temp[uGlobPos*15 +14];
            //-------------------------------------------------------
            for (j = 0u; j < uBPNum; j++) //going through all sources
            {   //-------------------------------------------------------
                fSrc[0]  = fB_temp[j*15 +0];                               fSrc[1]  = fB_temp[j*15 +1];                           fSrc[2]  = fB_temp[j*15 +2];
                fDist = sqrtf((fSrc[0]-fRcv[0])*(fSrc[0]-fRcv[0]) + (fSrc[1]-fRcv[1])*(fSrc[1]-fRcv[1]) + (fSrc[2]-fRcv[2])*(fSrc[2]-fRcv[2]));
                //-------------------------------------------------------
                fTemp0 = (fDist >= CUTDISTANCE*fBndSide)*1.0f + (fDist < CUTDISTANCE*fBndSide)*0.0f; //if distance between source and receiver centers is less than CUTDISTANCE then set it to zero
                fTemp0 = (uB_temp[uGlobPos*4 +3] == uB_temp[j*4 +3])*1.0f + (uB_temp[uGlobPos*4 +3] != uB_temp[j*4 +3])*fTemp0;//if the source and receiver are from the same SEG/SectionID, set to one (regardless of distance)
                //-------------------------------------------------------
                fP1[0] = fBV_temp[uB_temp[j*4 +0]*3 +0];                fP1[1] = fBV_temp[uB_temp[j*4 +0]*3 +1];            fP1[2] = fBV_temp[uB_temp[j*4 +0]*3 +2];
                fP2[0] = fBV_temp[uB_temp[j*4 +1]*3 +0];                fP2[1] = fBV_temp[uB_temp[j*4 +1]*3 +1];            fP2[2] = fBV_temp[uB_temp[j*4 +1]*3 +2];
                fP3[0] = fBV_temp[uB_temp[j*4 +2]*3 +0];                fP3[1] = fBV_temp[uB_temp[j*4 +2]*3 +1];            fP3[2] = fBV_temp[uB_temp[j*4 +2]*3 +2];
                //-------------------------------------------------------
                StrainHS_Nikkhoo(fStressStk, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, fUnitSlipB, 0.0f, 0.0f, fModPara[2], fModPara[4]);
                StrainHS_Nikkhoo(fStressDip, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, 0.0f, fUnitSlipB, 0.0f, fModPara[2], fModPara[4]);
                StrainHS_Nikkhoo(fStressNrm, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, 0.0f, 0.0f, fUnitSlipB, fModPara[2], fModPara[4]);
                //-------------------------------------------------------
                fStressStk[0] *=(fTemp0/fUnitSlipB);      fStressStk[1] *=(fTemp0/fUnitSlipB);      fStressStk[2] *=(fTemp0/fUnitSlipB);      
                fStressStk[3] *=(fTemp0/fUnitSlipB);      fStressStk[4] *=(fTemp0/fUnitSlipB);      fStressStk[5] *=(fTemp0/fUnitSlipB);
                fStressDip[0] *=(fTemp0/fUnitSlipB);      fStressDip[1] *=(fTemp0/fUnitSlipB);      fStressDip[2] *=(fTemp0/fUnitSlipB);      
                fStressDip[3] *=(fTemp0/fUnitSlipB);      fStressDip[4] *=(fTemp0/fUnitSlipB);      fStressDip[5] *=(fTemp0/fUnitSlipB);
                fStressNrm[0] *=(fTemp0/fUnitSlipB);      fStressNrm[1] *=(fTemp0/fUnitSlipB);      fStressNrm[2] *=(fTemp0/fUnitSlipB);      
                fStressNrm[3] *=(fTemp0/fUnitSlipB);      fStressNrm[4] *=(fTemp0/fUnitSlipB);      fStressNrm[5] *=(fTemp0/fUnitSlipB);
                //-------------------------------------------------------
                fK_BBvalstk[i][3*j +0] = fNrm[0]*fStk[0]*fStressStk[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressStk[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressStk[4] + fStk[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fStk[2]*fStressStk[5]; //induced strike component
                fK_BBvaldip[i][3*j +0] = fNrm[0]*fDip[0]*fStressStk[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressStk[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressStk[4] + fDip[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fDip[2]*fStressStk[5]; //induced dip component
                fK_BBvalnrm[i][3*j +0] = fNrm[0]*fNrm[0]*fStressStk[0] +              2.0f*fNrm[0]*fNrm[1] *fStressStk[1] +               2.0f*fNrm[0]*fNrm[2] *fStressStk[2] +               2.0f*fNrm[1]*fNrm[2] *fStressStk[4] + fNrm[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fNrm[2]*fStressStk[5]; //induced normal component
                //-------------------------------------------------------
                fK_BBvalstk[i][3*j +1] = fNrm[0]*fStk[0]*fStressDip[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressDip[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressDip[4] + fStk[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fStk[2]*fStressDip[5]; //induced strike component
                fK_BBvaldip[i][3*j +1] = fNrm[0]*fDip[0]*fStressDip[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressDip[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressDip[4] + fDip[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fDip[2]*fStressDip[5]; //induced dip component
                fK_BBvalnrm[i][3*j +1] = fNrm[0]*fNrm[0]*fStressDip[0] +              2.0f*fNrm[0]*fNrm[1] *fStressDip[1] +               2.0f*fNrm[0]*fNrm[2] *fStressDip[2] +               2.0f*fNrm[1]*fNrm[2] *fStressDip[4] + fNrm[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fNrm[2]*fStressDip[5]; //induced normal component
                //-------------------------------------------------------
                fK_BBvalstk[i][3*j +2] = fNrm[0]*fStk[0]*fStressNrm[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressNrm[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressNrm[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressNrm[4] + fStk[1]*fNrm[1]*fStressNrm[3] + fNrm[2]*fStk[2]*fStressNrm[5]; //induced strike component
                fK_BBvaldip[i][3*j +2] = fNrm[0]*fDip[0]*fStressNrm[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressNrm[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressNrm[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressNrm[4] + fDip[1]*fNrm[1]*fStressNrm[3] + fNrm[2]*fDip[2]*fStressNrm[5]; //induced dip component
                fK_BBvalnrm[i][3*j +2] = fNrm[0]*fNrm[0]*fStressNrm[0] +              2.0f*fNrm[0]*fNrm[1] *fStressNrm[1] +               2.0f*fNrm[0]*fNrm[2] *fStressNrm[2] +               2.0f*fNrm[1]*fNrm[2] *fStressNrm[4] + fNrm[1]*fNrm[1]*fStressNrm[3] + fNrm[2]*fNrm[2]*fStressNrm[5]; //induced normal component
                //-------------------------------------------------------
                if (uGlobPos == j)
                {   //this will be for self-stiffness
                    fBEvent[i*9 +4]  = fK_BBvalstk[i][3*j +0];//self-stiffness in strike direction (induced strike-shear due to strike slip on self)
                    fBEvent[i*9 +5]  = fK_BBvaldip[i][3*j +1];//self-stiffness in dip direction (induced dip-shear due to dip slip on self)
                    fBEvent[i*9 +6]  = fK_BBvalnrm[i][3*j +2];//self-stiffness in normal direction (induced normal-shear due to normal slip on self)
                }
            } //take all local sources and determine induced stress on selected receiver
            //------------------------------------------------------------------------
            for (j = 0u; j < uFPNum; j++) //going through all sources
            {   //-------------------------------------------------------
                fSrc[0]  = fF_temp[j*15 +0];                               fSrc[1]  = fF_temp[j*15 +1];                           fSrc[2]  = fF_temp[j*15 +2];
                fDist = sqrtf((fSrc[0]-fRcv[0])*(fSrc[0]-fRcv[0]) + (fSrc[1]-fRcv[1])*(fSrc[1]-fRcv[1]) + (fSrc[2]-fRcv[2])*(fSrc[2]-fRcv[2]));
                //-------------------------------------------------------
                fTemp0 = (fDist >= 1.0f*fFltSide)*1.0f + (fDist < 1.0f*fFltSide)*0.0f;//if distance between source and receiver centers is less than then set it to zero
                //-------------------------------------------------------
                fP1[0] = fFV_temp[uF_temp[j*6 +0]*4 +0];                fP1[1] = fFV_temp[uF_temp[j*6 +0]*4 +1];            fP1[2] = fFV_temp[uF_temp[j*6 +0]*4 +2];
                fP2[0] = fFV_temp[uF_temp[j*6 +1]*4 +0];                fP2[1] = fFV_temp[uF_temp[j*6 +1]*4 +1];            fP2[2] = fFV_temp[uF_temp[j*6 +1]*4 +2];
                fP3[0] = fFV_temp[uF_temp[j*6 +2]*4 +0];                fP3[1] = fFV_temp[uF_temp[j*6 +2]*4 +1];            fP3[2] = fFV_temp[uF_temp[j*6 +2]*4 +2];
                //-------------------------------------------------------
                StrainHS_Nikkhoo(fStressStk, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, fUnitSlipF, 0.0f, 0.0f, fModPara[2], fModPara[4]);
                StrainHS_Nikkhoo(fStressDip, fStrain, fRcv[0], fRcv[1], fRcv[2], fP1, fP2, fP3, 0.0f, fUnitSlipF, 0.0f, fModPara[2], fModPara[4]);
                //-------------------------------------------------------
                fStressStk[0] *=(fTemp0/fUnitSlipF);      fStressStk[1] *=(fTemp0/fUnitSlipF);      fStressStk[2] *=(fTemp0/fUnitSlipF);      
                fStressStk[3] *=(fTemp0/fUnitSlipF);      fStressStk[4] *=(fTemp0/fUnitSlipF);      fStressStk[5] *=(fTemp0/fUnitSlipF);
                fStressDip[0] *=(fTemp0/fUnitSlipF);      fStressDip[1] *=(fTemp0/fUnitSlipF);      fStressDip[2] *=(fTemp0/fUnitSlipF);      
                fStressDip[3] *=(fTemp0/fUnitSlipF);      fStressDip[4] *=(fTemp0/fUnitSlipF);      fStressDip[5] *=(fTemp0/fUnitSlipF);
                //-------------------------------------------------------
                fK_BFvalstk[i][2*j +0] = fNrm[0]*fStk[0]*fStressStk[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressStk[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressStk[4] + fStk[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fStk[2]*fStressStk[5]; //induced strike component
                fK_BFvaldip[i][2*j +0] = fNrm[0]*fDip[0]*fStressStk[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressStk[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressStk[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressStk[4] + fDip[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fDip[2]*fStressStk[5]; //induced dip component
                fK_BFvalnrm[i][2*j +0] = fNrm[0]*fNrm[0]*fStressStk[0] +              2.0f*fNrm[0]*fNrm[1] *fStressStk[1] +               2.0f*fNrm[0]*fNrm[2] *fStressStk[2] +               2.0f*fNrm[1]*fNrm[2] *fStressStk[4] + fNrm[1]*fNrm[1]*fStressStk[3] + fNrm[2]*fNrm[2]*fStressStk[5]; //induced normal component
                //-------------------------------------------------------
                fK_BFvalstk[i][2*j +1] = fNrm[0]*fStk[0]*fStressDip[0] + (fNrm[0]*fStk[1]+ fStk[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fStk[2] + fStk[0]*fNrm[2])*fStressDip[2] + (fStk[2]*fNrm[1] + fNrm[2]*fStk[1])*fStressDip[4] + fStk[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fStk[2]*fStressDip[5]; //induced strike component
                fK_BFvaldip[i][2*j +1] = fNrm[0]*fDip[0]*fStressDip[0] + (fNrm[0]*fDip[1]+ fDip[0]*fNrm[1])*fStressDip[1] + (fNrm[0]*fDip[2] + fDip[0]*fNrm[2])*fStressDip[2] + (fDip[2]*fNrm[1] + fNrm[2]*fDip[1])*fStressDip[4] + fDip[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fDip[2]*fStressDip[5]; //induced dip component
                fK_BFvalnrm[i][2*j +1] = fNrm[0]*fNrm[0]*fStressDip[0] +              2.0f*fNrm[0]*fNrm[1] *fStressDip[1] +               2.0f*fNrm[0]*fNrm[2] *fStressDip[2] +               2.0f*fNrm[1]*fNrm[2] *fStressDip[4] + fNrm[1]*fNrm[1]*fStressDip[3] + fNrm[2]*fNrm[2]*fStressDip[5]; //induced normal component
                //-------------------------------------------------------
            } //take all local sources and determine induced stress on selected receiver
            //------------------------------------------------------------------------
            if ((uPlotCatalog2Screen == 1)&&(iRANK == 0))     {    fprintf(stdout,"\nboundary receiver:  %u/%u ",i, iBOFFSET[iRANK]);           }
        }
        //----------------------------------------------------------------------------
        MPI_Allreduce(MPI_IN_PLACE, uF_temp, 6*uFPNum, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
        //---------------------------
        timer0 = clock() - timer0;        time_taken  = ((double)timer0)/CLOCKS_PER_SEC;
        MPI_Allreduce(MPI_IN_PLACE, &time_taken, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
        time_taken /= (double)iSIZE;
        if (iRANK == 0)    {   fprintf(stdout,"\nTotal RunTime in seconds:    %6.4f\n",time_taken);     }
        //----------------------------------------------------------------------------
        if      (uLoadPrev_Kmat == 1)
        {   float *FTempVect = calloc(iFOFFSET[iRANK], sizeof *FTempVect);
            float *BTempVect = calloc(iBOFFSET[iRANK], sizeof *BTempVect);
            unsigned long long *lRankOffset = calloc(iSIZE, sizeof *lRankOffset );
            unsigned long long *lRankStart  = calloc(iSIZE, sizeof *lRankStart );
            unsigned long long  lLoclOffset, lTemp0;
            //----------------------------------------
            MPI_File_delete(cKmatFileName, MPI_INFO_NULL);
            MPI_File_open(MPI_COMM_WORLD, cKmatFileName, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_KMAT);  
            if (iRANK == 0)
            {   MPI_File_write(fp_KMAT,  &uFPNum,     1, MPI_UNSIGNED,  &STATUS);      MPI_File_write(fp_KMAT,  &uFVNum,      1, MPI_UNSIGNED,  &STATUS);
                MPI_File_write(fp_KMAT,  &uBPNum,     1, MPI_UNSIGNED,  &STATUS);      MPI_File_write(fp_KMAT,  &uBVNum,      1, MPI_UNSIGNED,  &STATUS);
            } // first, write the oct-tree; only one element needs to do that -is global info
            OFFSETall = (4 * sizeof(unsigned int));
            //---------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++)  {       FTempVect[i] = fFEvent[i*16 +4];      }
            MPI_File_write_at(fp_KMAT,  (OFFSETall + iFSTART[iRANK]*sizeof(float)),  FTempVect,  iFOFFSET[iRANK], MPI_FLOAT,   &STATUS);                   OFFSETall += (uFPNum * sizeof(float));
            for (i = 0u; i < iFOFFSET[iRANK]; i++)  {       FTempVect[i] = fFEvent[i*16 +5];      }
            MPI_File_write_at(fp_KMAT,  (OFFSETall + iFSTART[iRANK]*sizeof(float)),  FTempVect,  iFOFFSET[iRANK], MPI_FLOAT,   &STATUS);                   OFFSETall += (uFPNum * sizeof(float));
            for (i = 0u; i < iFOFFSET[iRANK]; i++)  {       FTempVect[i] = fFEvent[i*16 +6];      }
            MPI_File_write_at(fp_KMAT,  (OFFSETall + iFSTART[iRANK]*sizeof(float)),  FTempVect,  iFOFFSET[iRANK], MPI_FLOAT,   &STATUS);                   OFFSETall += (uFPNum * sizeof(float));
            
            for (i = 0u; i < iBOFFSET[iRANK]; i++)  {       BTempVect[i] = fBEvent[i*9 +4];      }
            MPI_File_write_at(fp_KMAT,  (OFFSETall + iBSTART[iRANK]*sizeof(float)),  BTempVect,  iBOFFSET[iRANK], MPI_FLOAT,   &STATUS);                   OFFSETall += (uBPNum * sizeof(float));
            for (i = 0u; i < iBOFFSET[iRANK]; i++)  {       BTempVect[i] = fBEvent[i*9 +5];      }
            MPI_File_write_at(fp_KMAT,  (OFFSETall + iBSTART[iRANK]*sizeof(float)),  BTempVect,  iBOFFSET[iRANK], MPI_FLOAT,   &STATUS);                   OFFSETall += (uBPNum * sizeof(float));
            for (i = 0u; i < iBOFFSET[iRANK]; i++)  {       BTempVect[i] = fBEvent[i*9 +6];      }
            MPI_File_write_at(fp_KMAT,  (OFFSETall + iBSTART[iRANK]*sizeof(float)),  BTempVect,  iBOFFSET[iRANK], MPI_FLOAT,   &STATUS);                   OFFSETall += (uBPNum * sizeof(float));
            //----------------------------------------------------------------------------
            lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
            for (i = 0u; i < iFOFFSET[iRANK]; i++)  {   lTemp0 += ( 6*uFPNum*sizeof(float) );                       }
            lRankOffset[iRANK] = lTemp0;
            MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];   }
            //----------------------------------------
            lLoclOffset = 0;
            for (i = 0u; i < iFOFFSET[iRANK]; i++)  
            {   MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FFvalstk[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
                MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FFvaldip[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
                MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FFvalnrm[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
            }
            OFFSETall += lTemp0;
            //----------------------------------------------------------------------------
            lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
            for (i = 0u; i < iFOFFSET[iRANK]; i++)  {   lTemp0 += ( 6*uBPNum*sizeof(float) );                       }
            lRankOffset[iRANK] = lTemp0; 
            MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];    }
            //----------------------------------------
            lLoclOffset = 0;
            for (i = 0u; i < iFOFFSET[iRANK]; i++)  
            {   MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FBvalstk[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
                MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FBvaldip[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
            }
            OFFSETall += lTemp0;
            //----------------------------------------------------------------------------
            lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
            for (i = 0u; i < iBOFFSET[iRANK]; i++)  {   lTemp0 += ( 9*uBPNum*sizeof(float) );                       }
            lRankOffset[iRANK] = lTemp0; 
            MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];    }
            //----------------------------------------
            lLoclOffset = 0;
            for (i = 0u; i < iBOFFSET[iRANK]; i++)  
            {   MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BBvalstk[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
                MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BBvaldip[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
                MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BBvalnrm[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
            }
            OFFSETall += lTemp0;
            //----------------------------------------------------------------------------
            lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
            for (i = 0u; i < iBOFFSET[iRANK]; i++)  {   lTemp0 += ( 6*uFPNum*sizeof(float) );                       }
            lRankOffset[iRANK] = lTemp0; 
            MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];    }
            //----------------------------------------
            lLoclOffset = 0;
            for (i = 0u; i < iBOFFSET[iRANK]; i++)  
            {   MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BFvalstk[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
                MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BFvaldip[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
                MPI_File_write_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BFvalnrm[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
            }
            //----------------------------------------------------------------------------
            MPI_Barrier( MPI_COMM_WORLD );
            MPI_File_close(&fp_KMAT);
        } //storing the K-matrix to file (if corresponding switch was set that way)
    } //the K_matrix is built/populated
    else
    {   float *FTempVect = calloc(iFOFFSET[iRANK], sizeof *FTempVect);
        float *BTempVect = calloc(iBOFFSET[iRANK], sizeof *BTempVect);
        unsigned int uFPNum_t,   uBPNum_t, uFVNum_t, uBVNum_t;
        unsigned long long *lRankOffset = calloc(iSIZE, sizeof *lRankOffset );
        unsigned long long *lRankStart  = calloc(iSIZE, sizeof *lRankStart );
        unsigned long long  lLoclOffset, lTemp0;
        float fTemp0,   fTemp1,   fTemp2;
                    
        MPI_File_open(MPI_COMM_WORLD, cKmatFileName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_KMAT);  
        //--------------------------------------------
        MPI_File_read(fp_KMAT, &uFPNum_t,   1, MPI_UNSIGNED, &STATUS);       MPI_File_read(fp_KMAT, &uFVNum_t,    1, MPI_UNSIGNED, &STATUS);
        MPI_File_read(fp_KMAT, &uBPNum_t,   1, MPI_UNSIGNED, &STATUS);       MPI_File_read(fp_KMAT, &uBVNum_t,    1, MPI_UNSIGNED, &STATUS);
        if ((uFPNum_t != uFPNum) || (uBPNum_t != uBPNum) || (uFVNum_t != uFVNum) || (uBVNum_t != uBVNum))
        {   fprintf(stdout,"K_matrix file was openend, but fault/boundary element numbers not matching => exit\n");        exit(10);    }
        //--------------------------------------------
        OFFSETall = (4 * sizeof(unsigned int));
        //--------------------------------------------
        MPI_File_read_at(fp_KMAT, (OFFSETall + iFSTART[iRANK]*sizeof(float)), FTempVect, iFOFFSET[iRANK], MPI_FLOAT, &STATUS);       OFFSETall += (uFPNum * sizeof(float));
        for (i = 0u; i < iFOFFSET[iRANK]; i++)  {   fFEvent[i*16 +4] = FTempVect[i];     }    
        MPI_File_read_at(fp_KMAT, (OFFSETall + iFSTART[iRANK]*sizeof(float)), FTempVect, iFOFFSET[iRANK], MPI_FLOAT, &STATUS);       OFFSETall += (uFPNum * sizeof(float));
        for (i = 0u; i < iFOFFSET[iRANK]; i++)  {   fFEvent[i*16 +5] = FTempVect[i];     }  
        MPI_File_read_at(fp_KMAT, (OFFSETall + iFSTART[iRANK]*sizeof(float)), FTempVect, iFOFFSET[iRANK], MPI_FLOAT, &STATUS);       OFFSETall += (uFPNum * sizeof(float));
        for (i = 0u; i < iFOFFSET[iRANK]; i++)  {   fFEvent[i*16 +6] = FTempVect[i];     }  
        
        MPI_File_read_at(fp_KMAT, (OFFSETall + iBSTART[iRANK]*sizeof(float)), BTempVect, iBOFFSET[iRANK], MPI_FLOAT, &STATUS);       OFFSETall += (uBPNum * sizeof(float));
        for (i = 0u; i < iBOFFSET[iRANK]; i++)  {   fBEvent[i*9 +4] = BTempVect[i];     }    
        MPI_File_read_at(fp_KMAT, (OFFSETall + iBSTART[iRANK]*sizeof(float)), BTempVect, iBOFFSET[iRANK], MPI_FLOAT, &STATUS);       OFFSETall += (uBPNum * sizeof(float));
        for (i = 0u; i < iBOFFSET[iRANK]; i++)  {   fBEvent[i*9 +5] = BTempVect[i];     }  
        MPI_File_read_at(fp_KMAT, (OFFSETall + iBSTART[iRANK]*sizeof(float)), BTempVect, iBOFFSET[iRANK], MPI_FLOAT, &STATUS);       OFFSETall += (uBPNum * sizeof(float));
        for (i = 0u; i < iBOFFSET[iRANK]; i++)  {   fBEvent[i*9 +6] = BTempVect[i];     }  
        //-----------------------------------------------------------------------
        lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
        //-----------------------------------------------------------------------
        for (i = 0u; i < iFOFFSET[iRANK]; i++) 
        {   fK_FFvalstk[i] = calloc(   3*uFPNum, sizeof *fK_FFvalstk[i] );
            fK_FFvaldip[i] = calloc(   3*uFPNum, sizeof *fK_FFvaldip[i] );
            fK_FFvalnrm[i] = calloc(   3*uFPNum, sizeof *fK_FFvalnrm[i] );
            lTemp0        += ( 6*uFPNum*sizeof(float) );
        }
        lRankOffset[iRANK] = lTemp0;
        MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];    }
        //----------------------------------------
        lLoclOffset = 0;
        for (i = 0u; i < iFOFFSET[iRANK]; i++)  
        {   MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FFvalstk[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
            MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FFvaldip[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
            MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FFvalnrm[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
        }
        OFFSETall += lTemp0;
        //-----------------------------------------------------------------------
        lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
        for (i = 0u; i < iFOFFSET[iRANK]; i++) 
        {   fK_FBvalstk[i] = calloc(   3*uBPNum, sizeof *fK_FBvalstk[i] );
            fK_FBvaldip[i] = calloc(   3*uBPNum, sizeof *fK_FBvaldip[i] );
            lTemp0         += ( 6*uBPNum*sizeof(float) );
        }
        lRankOffset[iRANK] = lTemp0;
        MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];    }
        //----------------------------------------
        lLoclOffset = 0;
        for (i = 0u; i < iFOFFSET[iRANK]; i++)  
        {   MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FBvalstk[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
            MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_FBvaldip[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
        }
        OFFSETall += lTemp0;
        //-----------------------------------------------------------------------
        lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
        for (i = 0u; i < iBOFFSET[iRANK]; i++) 
        {   fK_BBvalstk[i] = calloc(   3*uBPNum, sizeof *fK_BBvalstk[i] );
            fK_BBvaldip[i] = calloc(   3*uBPNum, sizeof *fK_BBvaldip[i] );
            fK_BBvalnrm[i] = calloc(   3*uBPNum, sizeof *fK_BBvalnrm[i] );
            lTemp0        += ( 9*uBPNum*sizeof(float) );
        }
        lRankOffset[iRANK] = lTemp0;
        MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];    }
        //----------------------------------------
        lLoclOffset = 0;
        for (i = 0u; i < iBOFFSET[iRANK]; i++)  
        {   MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BBvalstk[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
            MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BBvaldip[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
            MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BBvalnrm[i], 3*uBPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (3*uBPNum*sizeof(float));
        }
        OFFSETall += lTemp0;
        //-----------------------------------------------------------------------
        lTemp0 = 0;         memset(lRankOffset, 0, iSIZE*sizeof(unsigned long long ));
        for (i = 0u; i < iBOFFSET[iRANK]; i++) 
        {   fK_BFvalstk[i] = calloc(   2*uFPNum, sizeof *fK_BFvalstk[i] );
            fK_BFvaldip[i] = calloc(   2*uFPNum, sizeof *fK_BFvaldip[i] );
            fK_BFvalnrm[i] = calloc(   2*uFPNum, sizeof *fK_BFvalnrm[i] );
            lTemp0        += ( 6*uFPNum*sizeof(float) );
        }
        lRankOffset[iRANK] = lTemp0;
        MPI_Allreduce(MPI_IN_PLACE, &lTemp0,       1,   MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, lRankOffset, iSIZE, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        for (i = 1u; i < iSIZE; i++)            {   lRankStart[i]       = lRankStart[i-1] + lRankOffset[i-1];    }
        //----------------------------------------
        lLoclOffset = 0;
        for (i = 0u; i < iBOFFSET[iRANK]; i++)  
        {   MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BFvalstk[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
            MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BFvaldip[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
            MPI_File_read_at(fp_KMAT,  (OFFSETall + lRankStart[iRANK] + lLoclOffset),  fK_BFvalnrm[i], 2*uFPNum, MPI_FLOAT,      &STATUS);         lLoclOffset += (2*uFPNum*sizeof(float));
        }
        OFFSETall += lTemp0;
        //-----------------------------------------------------------------------
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_File_close(&fp_KMAT);
        //-----------------------------------------------------------------------
        for (i = 0u; i < iFOFFSET[iRANK]; i++)  
        {   //this will be for self-stiffness
            //--------------------------------------------------------------------
            fFRef[i*14 +2]    = MAX(fabs(fFEvent[i*16 +4]),fabs(fFEvent[i*16 +5])); //this is reference "self-stiffness"
            //--------------------------------------------------------------------
            fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
            fFEvent[i*16 +7] = fFRef[i*14 +7] + (fFRef[i*14 +8] * fTemp0); //current Dc value; wird gleich wieder ueberschrieben/geupdated
            
            fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
            fFFric[i*6 +0] = fFRef[i*14 +3] + (fFRef[i*14 +4] * fTemp0); //current static friction
            fFEvent[i*16 +3] = fFFric[i*6 +0]; //use static friction also as current friction
            
            fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
            fFFric[i*6 +1] = fFRef[i*14 +5] + (fFRef[i*14 +6] * fTemp0); // current dynamic friction
            //--------------------------------------------------------------------
            fTemp0 =(fFFric[i*6 +0] - fFFric[i*6 +1]) *-1.0f*fFRef[i*14 +1]; //strength difference between static and dynamic strength; positive when weakening
            fTemp1 = fTemp0/fFRef[i*14 +2];//is slip required to release strength difference
            uFEvent[i*5 +0] = (fTemp0 >= 0.0f)*2u + (fTemp0 < 0.0f)*3u;//gets a 2 if drop is not negative (in which case it would be strengthening i.e., stable == 3)
            uFEvent[i*5 +0] = (fTemp1 > fFEvent[i*16 +7])*1u + (fTemp1 <= fFEvent[i*16 +7])*uFEvent[i*5 +0];//give it a 1 if it is unstable (slip associated w/ change from static to dynamic is large enough to over come Dc)
            //---------------------------------
            fTemp0 = fFFric[i*6 +1]*-1.0f*fFRef[i*14 +1] + fFRef[i*14 +2]*fFEvent[i*16 +7]; //this is dynFric*normal stress (ie., dyn strength) minus  selfstiff*DcVal; the latter is therefore the stress change needed/associated with slip Dc; adding both to see what fric strength would at lease need to have Dc amount of slip
            fTemp1 = fFFric[i*6 +0]*-1.0f*fFRef[i*14 +1]; //this is the static strength
            fTemp2 = MAX(fTemp0, fTemp1)/(-1.0f*fFRef[i*14 +1]); //friction coefficient...
            fFFric[i*6 +2]  = fTemp2 - fOvershootFract*(fTemp2-fFFric[i*6 +1]); //this is arrest friction
            //---------------------------------
            fFEvent[i*16 +7] *= ((fTemp2 - fFFric[i*6 +2])/(fTemp2 - fFFric[i*6 +1])); //updated Dc value => now adjusted for the breakdown all the way to arrest stress
            fFEvent[i*16 +10] = (fFFric[i*6 +1] - fFFric[i*6 +2])*-1.0f*fFRef[i*14 +1];//this is overshoot stress (difference between arrest (lower) and dynamic (higher) strength)
            //--------------------------------------------------------------------

        }
    } // load previously generated K_matrix
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    FILE *fpPrePost,   *fp_CATALOG,   *fp_CATALOGCUT;
    float *fFslip    = calloc(2*uFPNum, sizeof *fFslip); //for slip in strike and dip
    float *fBslip    = calloc(3*uBPNum, sizeof *fBslip); // for slip in strike, dip, and normal
    float *fEQslipSD = calloc(2*uFPNum, sizeof *fEQslipSD); //with stride, alternating strike- and dip-slip; or different orientation
    //-----------------------------------------
    float *fSTFslip  = calloc(2*iFOFFSET[iRANK]*MAXMOMRATEFUNCLENGTH, sizeof *fSTFslip);
    //-------------------------------------------------------
    {   unsigned int uGlobPos,   uTemp1;
        float fTemp0,   fTemp1,   fTemp2,   fCmbSlp1,   fCmbSlp2;
        //---------------------------------------------------
        uTemp1 = 0u;
        for (i = 0u; i < iFOFFSET[iRANK]; i++)
        {   fTemp0   = -1.0f*fFEvent[i*16 +13];         fFEvent[i*16 +13] = 0.0f;
            fTemp1   = -1.0f*fFEvent[i*16 +14];         fFEvent[i*16 +14] = 0.0f;
            //------------------------------
            if ((fabs(fTemp0) + fabs(fTemp1)) > 0.0f)
            {   uGlobPos = i+ iFSTART[iRANK];           uTemp1                = 1u;
                fFslip[uGlobPos*2 +0] = fTemp0;         fFslip[uGlobPos*2 +1] = fTemp1;
        }   }//put slip of fault elements into global/branch slip vector, then set them zero in fFEvent
        //---------------------------------------------------
        for (i = 0u; i < iBOFFSET[iRANK]; i++)
        {   uGlobPos = i+ iBSTART[iRANK];
            fBTempVal[i*3 +0] =  1.0f*fB_temp[uGlobPos*15 +3];
            fBTempVal[i*3 +1] =  1.0f*fB_temp[uGlobPos*15 +4];
            fBTempVal[i*3 +2] =  1.0f*fB_temp[uGlobPos*15 +5];
            if ((fabs(fBTempVal[i*3 +0]) + fabs(fBTempVal[i*3 +1]) + fabs(fBTempVal[i*3 +2])) > 0.0f)              {           uTemp1 = 1u;            }
        } //put slip along boundary fault into temporary vector and set switch to one if any boundary actually has slip
        //---------------------------------------------------
        MPI_Allreduce(MPI_IN_PLACE, &uTemp1, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
        //---------------------------------------------------
        if (uTemp1 == 0u)
        {   //---------------------------------------------------
            if (iRANK == 0)     {       fprintf(stdout,"\nUse assigned stressing rates\n");     }
            //------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++)                  
            {   fFTempVal[i*5 +0] = fFEvent[i*16 +8];           fFTempVal[i*5 +1] = fFEvent[i*16 +9];
                fFTempVal[i*5 +2] = 0.0f;                       fFTempVal[i*5 +3] = 0.0f;
            } //set accum. slip on fault to zero and stressing direction in opposite way
            //------------------------------------
            for (k = 0u; k < MAXITERATION4BOUNDARY; k++)
            {   memset(fFslip, 0, 2*uFPNum*sizeof(float) );
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   uGlobPos = i+ iFSTART[iRANK]; 
                    fTemp0   = -1.0f*fFTempVal[i*5 +0]/fFEvent[i*16 +4];       fFTempVal[i*5 +2] += fTemp0;
                    fTemp1   = -1.0f*fFTempVal[i*5 +1]/fFEvent[i*16 +5];       fFTempVal[i*5 +3] += fTemp1;
                    //------------------------------
                    fFslip[uGlobPos*2 +0] = fTemp0;                            fFslip[uGlobPos*2 +1] = fTemp1; 
                }
                //-------------------------
                MPI_Allreduce(MPI_IN_PLACE, fFslip, 2*uFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); 
                //-------------------------
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fFTempVal[i*5 +0] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvalstk[i], 1);
                    fFTempVal[i*5 +1] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvaldip[i], 1);
            }   }//this gives slip-rate along fault when using stressing-rate
            //------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++) 
            {   fTemp0   = fFTempVal[i*5 +2];           fTemp1   = fFTempVal[i*5 +3];
                fFRef[i*14 +12] = sqrtf(fTemp0*fTemp0 +fTemp1*fTemp1);
        }   }//if I dont have prescribed slip on boundary fault or non-boundary fault, then I use the assigned stressing rate
        else
        {   //---------------------------------------------------
            MPI_Allreduce(MPI_IN_PLACE, fFslip, 2*uFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            //---------------------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++) //this is my receivers
            {   fFEvent[i*16 +8] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvalstk[i], 1);
                fFEvent[i*16 +9] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvaldip[i], 1);
            }
            for (i = 0u; i < iBOFFSET[iRANK]; i++) //this is my receivers
            {   fBEvent[i*9 +0] += cblas_sdot(2*uFPNum, fFslip, 1, fK_BFvalstk[i], 1);
                fBEvent[i*9 +1] += cblas_sdot(2*uFPNum, fFslip, 1, fK_BFvaldip[i], 1);
                fBEvent[i*9 +2] += cblas_sdot(2*uFPNum, fFslip, 1, fK_BFvalnrm[i], 1);
            }
            //-----------------------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++)
            {   fFTempVal[i*5 +0] = fFEvent[i*16 +8];           fFTempVal[i*5 +1] = fFEvent[i*16 +9];
                fFTempVal[i*5 +2] = 0.0f;                       fFTempVal[i*5 +3] = 0.0f;
            } //set accum. slip on fault to zero and stressing direction in opposite way
            for (k = 0u; k < MAXITERATION4BOUNDARY; k++)
            {   memset(fFslip, 0, 2*uFPNum*sizeof(float) );
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   uGlobPos = i+ iFSTART[iRANK]; 
                    fTemp0   = -1.0f*fFTempVal[i*5 +0]/fFEvent[i*16 +4];       fFTempVal[i*5 +2] += fTemp0;
                    fTemp1   = -1.0f*fFTempVal[i*5 +1]/fFEvent[i*16 +5];       fFTempVal[i*5 +3] += fTemp1;
                    //------------------------------
                    fFslip[uGlobPos*2 +0] = fTemp0;                            fFslip[uGlobPos*2 +1] = fTemp1;
                }
                //-------------------------
                MPI_Allreduce(MPI_IN_PLACE, fFslip, 2*uFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); 
                //-------------------------
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fFTempVal[i*5 +0] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvalstk[i], 1);
                    fFTempVal[i*5 +1] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvaldip[i], 1);
            }   } //this gives slip-rate along fault when using back-slip slip-rate to get induced stress-rate and then releasing that stress-rate again to "recreate" the initial slip-rate
            //----------------------------------------------------
            fCmbSlp1 = 0.0f;
            for (i = 0u; i < iFOFFSET[iRANK]; i++) 
            {   fTemp0   = fFTempVal[i*5 +2];           fTemp1   = fFTempVal[i*5 +3];
                fFRef[i*14 +12] = sqrtf(fTemp0*fTemp0 +fTemp1*fTemp1); //that is the element's slip-rate (per year) from back-slip; will be overwritten if boundary faults are used...
                fCmbSlp1       += fFRef[i*14 +12];
            } //determine the combined amount of slip (rate) that is produced when releasing the back-slip-induced stressing (rate); used for scaling
            MPI_Allreduce(MPI_IN_PLACE, &fCmbSlp1, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            //-----------------------------------------------------
            if (uBPNum > 0u)
            {   for (i = 0u; i < iFOFFSET[iRANK]; i++)   {   fFEvent[i*16 +8] = 0.0f;       fFEvent[i*16 +9] = 0.0f;          }
                for (i = 0u; i < iBOFFSET[iRANK]; i++)   {   fBEvent[i*9 +0] *=-1.0f;       fBEvent[i*9 +1] *=-1.0f;         fBEvent[i*9 +2] *=-1.0f;        }
                //-----------------------------------------------------
                for (k = 0u; k < MAXITERATION4BOUNDARY; k++)
                {   memset(fBslip, 0, 3*uBPNum*sizeof(float) );
                    for (i = 0u; i < iBOFFSET[iRANK]; i++)
                    {   uGlobPos = i+ iBSTART[iRANK];
                        fTemp0   = -1.0f*fBEvent[i*9 +0]/fBEvent[i*9 +4];     fBTempVal[i*3 +0] += fTemp0;
                        fTemp1   = -1.0f*fBEvent[i*9 +1]/fBEvent[i*9 +5];     fBTempVal[i*3 +1] += fTemp1;
                        fTemp2   = -1.0f*fBEvent[i*9 +2]/fBEvent[i*9 +6];     fBTempVal[i*3 +2] += fTemp2;
                        //------------------------------
                        fBslip[uGlobPos*3 +0] += fTemp0;          fBslip[uGlobPos*3 +1] += fTemp1;      fBslip[uGlobPos*3 +2] += fTemp2; 
                    }
                    //-------------------------
                    MPI_Allreduce(MPI_IN_PLACE, fBslip, 3*uBPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); 
                    //-------------------------
                    for (i = 0u; i < iBOFFSET[iRANK]; i++)
                    {   fBEvent[i*9 +0] += cblas_sdot(3*uBPNum, fBslip, 1, fK_BBvalstk[i], 1);
                        fBEvent[i*9 +1] += cblas_sdot(3*uBPNum, fBslip, 1, fK_BBvaldip[i], 1);
                        fBEvent[i*9 +2] += cblas_sdot(3*uBPNum, fBslip, 1, fK_BBvalnrm[i], 1);
                }   }//releasing the stress that was induced by slip bound. cond. on seismo-faults
                //-----------------------------------------------------
                memset(fBslip, 0, 3*uBPNum*sizeof(float) );
                for (i = 0u; i < iBOFFSET[iRANK]; i++)
                {   uGlobPos = i+ iBSTART[iRANK];
                    fBslip[uGlobPos*3 +0] += fBTempVal[i*3 +0];          fBslip[uGlobPos*3 +1] += fBTempVal[i*3 +1];      fBslip[uGlobPos*3 +2] += fBTempVal[i*3 +2]; 
                }
                MPI_Allreduce(MPI_IN_PLACE, fBslip, 3*uBPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); 
                //-------------------------
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fFEvent[i*16 +8] += cblas_sdot(3*uBPNum, fBslip, 1, fK_FBvalstk[i], 1);
                    fFEvent[i*16 +9] += cblas_sdot(3*uBPNum, fBslip, 1, fK_FBvaldip[i], 1);
            }   }   //use/release stresses induced on boundary layers and determine corresponding induced stress-rate on simulated faults
            //-----------------------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++)
            {   fFTempVal[i*5 +0] = fFEvent[i*16 +8];           fFTempVal[i*5 +1] = fFEvent[i*16 +9];
                fFTempVal[i*5 +2] = 0.0f;                       fFTempVal[i*5 +3] = 0.0f;
            }
            //-------------------------
            for (k = 0u; k < MAXITERATION4BOUNDARY; k++)
            {   memset(fFslip, 0, 2*uFPNum*sizeof(float) );
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   uGlobPos = i+ iFSTART[iRANK];
                    fTemp0   = -1.0f*fFTempVal[i*5 +0]/fFEvent[i*16 +4];       fFTempVal[i*5 +2] += fTemp0;
                    fTemp1   = -1.0f*fFTempVal[i*5 +1]/fFEvent[i*16 +5];       fFTempVal[i*5 +3] += fTemp1;
                    //------------------------------
                    fFslip[uGlobPos*2 +0] += fTemp0;          fFslip[uGlobPos*2 +1] += fTemp1;
                }
                //-------------------------
                MPI_Allreduce(MPI_IN_PLACE, fFslip, 2*uFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); 
                //-------------------------
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fFTempVal[i*5 +0] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvalstk[i], 1);
                    fFTempVal[i*5 +1] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvaldip[i], 1);
            }   } //release that stress-rate via slip along the simulated faults to get corresponding slip-rate
            //-------------------------
            fCmbSlp2 = 0.0f;
            for (i = 0u; i < iFOFFSET[iRANK]; i++) 
            {   fTemp0   = fFTempVal[i*5 +2];           fTemp1   = fFTempVal[i*5 +3];
                fFRef[i*14 +12] = sqrtf(fTemp0*fTemp0 +fTemp1*fTemp1); //slip-rate (per year) is now overwritten with value from bottom-loading...
                fCmbSlp2       += fFRef[i*14 +12];
            }
            MPI_Allreduce(MPI_IN_PLACE, &fCmbSlp2, 1, MPI_FLOAT,    MPI_SUM, MPI_COMM_WORLD);
            //-------------------------
            if ((fCmbSlp1 > 0.0f) && (fCmbSlp2 > 0.0f))     {   fTemp0 = fCmbSlp1/fCmbSlp2;     }//this is scaling factor from re-created back-slip slip rate and from slip rate due to boundary layers being used => to scale stressing rate
            else                                            {   fTemp0 = 1.0f;                  }
            //-------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++) 
            {   fFEvent[i*16 +8]  *= fTemp0;        fFEvent[i*16 +9]  *= fTemp0;
                fFTempVal[i*5 +2] *= fTemp0;        fFTempVal[i*5 +3] *= fTemp0;
                fFRef[i*14 +12]   *= fTemp0;
            }
            if (iRANK == 0)
            {   if (uBPNum <= 0u)       {   fprintf(stdout,"\nUsing slimple back-slip approach for loading\n");         }
                else                    {   fprintf(stdout,"\nResulting average stressing rate on faults (mm/yr): %5.5f   and  %5.5f,   scalefactor: %5.5f\n",(fCmbSlp1/(float)uFPNum)*1000.0f, (fCmbSlp2/(float)uFPNum)*1000.0f, fTemp0);      }
            }
        } //at least one element has slip boundary condition assigned to it
        //--------------------------------------------------------------------------------
        unsigned int *uTempF0= calloc(uFPNum, sizeof *uTempF0);
        unsigned int *uTempF1= calloc(uFPNum, sizeof *uTempF1);
        float *fTempF0  = calloc(uFPNum, sizeof *fTempF0 );
        float *fTempF1  = calloc(uFPNum, sizeof *fTempF1 );
        float *fTempF2  = calloc(uFPNum, sizeof *fTempF2 );
        float *fTempF3  = calloc(uFPNum, sizeof *fTempF3 );
        float *fTempF4  = calloc(uFPNum, sizeof *fTempF4 );
        float *fTempF5  = calloc(uFPNum, sizeof *fTempF5 );
        float *fTempB1  = calloc(uBPNum, sizeof *fTempB1 );
        float *fTempB2  = calloc(uBPNum, sizeof *fTempB2 );
        float *fTempB3  = calloc(uBPNum, sizeof *fTempB3 );
        unsigned int *uTempFL1 = calloc(iFOFFSET[iRANK], sizeof *uTempFL1 );
        float *fTempFL1 = calloc(iFOFFSET[iRANK], sizeof *fTempFL1 );
        float *fTempFL2 = calloc(iFOFFSET[iRANK], sizeof *fTempFL2 );
        float *fTempFL3 = calloc(iFOFFSET[iRANK], sizeof *fTempFL3 );
        float *fTempFL4 = calloc(iFOFFSET[iRANK], sizeof *fTempFL4 );
        float *fTempFL5 = calloc(iFOFFSET[iRANK], sizeof *fTempFL5 );
        float *fTempBL1 = calloc(iBOFFSET[iRANK], sizeof *fTempBL1 );
        float *fTempBL2 = calloc(iBOFFSET[iRANK], sizeof *fTempBL2 );
        float *fTempBL3 = calloc(iBOFFSET[iRANK], sizeof *fTempBL3 );
        strcpy(cPrevStateName, cInputName);      strcat(cPrevStateName,"_");      sprintf(cNameAppend, "%u",uRunNum);     strcat(cPrevStateName,cNameAppend);     strcat(cPrevStateName,"_PreRunData.dat");
        //following, the pre-run output is written; this is really helpful for debugging etc... and also to check for example that the stressing-rate distribution on the fault is making sense (is possible that grid resolution of boundary fault is low and those patches are too close to EQfaults, negatively affecting the loading function)
        //---------------------------------------------
        for (i = 0u; i < iFOFFSET[iRANK]; i++)      
        {   fTempFL1[i] = fFEvent[i*16 +8];                          fTempFL2[i] = fFEvent[i*16 +9];
            //fTempFL1[i] = fFTempVal[i*5 +2];                         fTempFL2[i] = fFTempVal[i*5 +3];
            fTempFL3[i] = fFFric[i*6 +0] * -1.0*fFRef[i*14 +1];      fTempFL4[i] =(fFFric[i*6 +0] - fFFric[i*6 +1])* -1.0*fFRef[i*14 +1];
            fTempFL5[i] = fFEvent[i*16 +7];
            uTempFL1[i] = uFEvent[i*5 +0];
        }
        for (i = 0u; i < iBOFFSET[iRANK]; i++)
        {   fTempBL1[i] = fBTempVal[i*3 +0];                        fTempBL2[i] = fBTempVal[i*3 +1];                fTempBL3[i] = fBTempVal[i*3 +2];
        }
        MPI_Allgatherv(fTempFL1, iFOFFSET[iRANK], MPI_FLOAT, fTempF1, iFOFFSET, iFSTART, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(fTempFL2, iFOFFSET[iRANK], MPI_FLOAT, fTempF2, iFOFFSET, iFSTART, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(fTempFL3, iFOFFSET[iRANK], MPI_FLOAT, fTempF3, iFOFFSET, iFSTART, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(fTempFL4, iFOFFSET[iRANK], MPI_FLOAT, fTempF4, iFOFFSET, iFSTART, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(fTempFL5, iFOFFSET[iRANK], MPI_FLOAT, fTempF5, iFOFFSET, iFSTART, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(uTempFL1, iFOFFSET[iRANK], MPI_UNSIGNED,uTempF1, iFOFFSET, iFSTART, MPI_UNSIGNED, MPI_COMM_WORLD);
        MPI_Allgatherv(fTempBL1, iBOFFSET[iRANK], MPI_FLOAT, fTempB1, iBOFFSET, iBSTART, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(fTempBL2, iBOFFSET[iRANK], MPI_FLOAT, fTempB2, iBOFFSET, iBSTART, MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(fTempBL3, iBOFFSET[iRANK], MPI_FLOAT, fTempB3, iBOFFSET, iBSTART, MPI_FLOAT, MPI_COMM_WORLD);       
        //---------------------------------------------
        if (iRANK == 0)       
        {   //---------------------------------------------
            if ((fpPrePost = fopen(cPrevStateName,"wb")) == NULL)      {        exit(10);       }
            //---------------------------------------------
            unsigned int *uTempB = calloc(uBPNum, sizeof *uTempB );
            float *fTempFV = calloc(uFVNum, sizeof *fTempFV );
            float *fTempBV = calloc(uBVNum, sizeof *fTempBV );
            //---------------------------------------------
            fwrite( &uFPNum,  sizeof(unsigned int), 1, fpPrePost);
            fwrite( &uFVNum,  sizeof(unsigned int), 1, fpPrePost);
            fwrite( &uBPNum,  sizeof(unsigned int), 1, fpPrePost);
            fwrite( &uBVNum,  sizeof(unsigned int), 1, fpPrePost);
            for (i = 0u; i < uFPNum; i++)   {   uTempF0[i] = uF_temp[i*6 +0];   }     fwrite( uTempF0,     sizeof(unsigned int),  uFPNum, fpPrePost); 
            for (i = 0u; i < uFPNum; i++)   {   uTempF0[i] = uF_temp[i*6 +1];   }     fwrite( uTempF0,     sizeof(unsigned int),  uFPNum, fpPrePost); 
            for (i = 0u; i < uFPNum; i++)   {   uTempF0[i] = uF_temp[i*6 +2];   }     fwrite( uTempF0,     sizeof(unsigned int),  uFPNum, fpPrePost); 
            for (i = 0u; i < uFPNum; i++)   {   uTempF0[i] = uF_temp[i*6 +3];   }     fwrite( uTempF0,     sizeof(unsigned int),  uFPNum, fpPrePost); 
            for (i = 0u; i < uFPNum; i++)   {   uTempF0[i] = uF_temp[i*6 +4];   }     fwrite( uTempF0,     sizeof(unsigned int),  uFPNum, fpPrePost); 
            
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +0];  }      fwrite( fTempFV,     sizeof(float),         uFVNum, fpPrePost); 
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +1];  }      fwrite( fTempFV,     sizeof(float),         uFVNum, fpPrePost);
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +2];  }      fwrite( fTempFV,     sizeof(float),         uFVNum, fpPrePost);
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +3];  }      fwrite( fTempFV,     sizeof(float),         uFVNum, fpPrePost);
            
            for (i = 0u; i < uFPNum; i++)   {   fTempF0[i] = fF_temp[i*15 +0]; }      fwrite( fTempF0,     sizeof(float),         uFPNum, fpPrePost);
            for (i = 0u; i < uFPNum; i++)   {   fTempF0[i] = fF_temp[i*15 +1]; }      fwrite( fTempF0,     sizeof(float),         uFPNum, fpPrePost);
            for (i = 0u; i < uFPNum; i++)   {   fTempF0[i] = fF_temp[i*15 +2]; }      fwrite( fTempF0,     sizeof(float),         uFPNum, fpPrePost);
            
            for (i = 0u; i < uBPNum; i++)   {   uTempB[i] = uB_temp[i*4 +0];   }      fwrite( uTempB,      sizeof(unsigned int),  uBPNum, fpPrePost); 
            for (i = 0u; i < uBPNum; i++)   {   uTempB[i] = uB_temp[i*4 +1];   }      fwrite( uTempB,      sizeof(unsigned int),  uBPNum, fpPrePost); 
            for (i = 0u; i < uBPNum; i++)   {   uTempB[i] = uB_temp[i*4 +2];   }      fwrite( uTempB,      sizeof(unsigned int),  uBPNum, fpPrePost); 
           
            for (i = 0u; i < uBVNum; i++)   {   fTempBV[i]= fBV_temp[i*3 +0];  }      fwrite( fTempBV,     sizeof(float),         uBVNum, fpPrePost); 
            for (i = 0u; i < uBVNum; i++)   {   fTempBV[i]= fBV_temp[i*3 +1];  }      fwrite( fTempBV,     sizeof(float),         uBVNum, fpPrePost);
            for (i = 0u; i < uBVNum; i++)   {   fTempBV[i]= fBV_temp[i*3 +2];  }      fwrite( fTempBV,     sizeof(float),         uBVNum, fpPrePost);
              
            for (i = 0u; i < uFPNum; i++)   {   uTempF0[i] = uF_temp[i*6 +5];   }     fwrite( uTempF0,      sizeof(unsigned int),  uFPNum, fpPrePost); 
            
            fwrite(fTempF1, sizeof(float), uFPNum, fpPrePost);                     
            fwrite(fTempF2, sizeof(float), uFPNum, fpPrePost);
            fwrite(fTempF3, sizeof(float), uFPNum, fpPrePost); //the patch strength
            fwrite(fTempF4, sizeof(float), uFPNum, fpPrePost); //the patch stress drop when having full drop
            fwrite(fTempF5, sizeof(float), uFPNum, fpPrePost); //the Dc value
            fwrite(uTempF1, sizeof(unsigned int), uFPNum, fpPrePost);
            fwrite(fTempB1, sizeof(float), uBPNum, fpPrePost); //the slip/stressing rate in strike direction
            fwrite(fTempB2, sizeof(float), uBPNum, fpPrePost); //the slip/stressing rate in dip direction
            fwrite(fTempB3, sizeof(float), uBPNum, fpPrePost); //the slip/stressing rate in opening direction
            
            fclose(fpPrePost); 
        }
        MPI_Barrier( MPI_COMM_WORLD );
    } //tectonic loading function and writing pre-run data to file
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    MPI_Barrier( MPI_COMM_WORLD );
    strcpy(cOutputName,    cInputName);      sprintf(cNameAppend, "_%u_RAWL3Catalog.dat",uRunNum);        strcat(cOutputName,    cNameAppend);
    strcpy(cPrevStateName, cInputName);      sprintf(cNameAppend, "_%u_PostRunState.dat",uRunNum);        strcat(cPrevStateName, cNameAppend);
    if      (uCatType == 1)     {   sprintf(cNameAppend, "_%u_L1Catalog.dat",uRunNum);                      }
    else if (uCatType == 2)     {   sprintf(cNameAppend, "_%u_L2Catalog.dat",uRunNum);                      }
    else if (uCatType == 3)     {   sprintf(cNameAppend, "_%u_L3Catalog.dat",uRunNum);                      }
    else                        {   fprintf(stdout,"Catalog level incorrect -make it 1/2/3");   exit(111);  }
    strcpy(cOutputNameCUT, cInputName); strcat(cOutputNameCUT, cNameAppend);
    //--------------------------------------------
    if (uContPrevRun == 1u)
    {   //-------------------------------
        unsigned int uDummy;
        unsigned int *uFvTemp   = calloc( iFOFFSET[iRANK],  sizeof *uFvTemp ); 
        float        *fFvTemp   = calloc( iFOFFSET[iRANK],  sizeof *fFvTemp ); 
        float        *fBvTemp   = calloc( iBOFFSET[iRANK],  sizeof *fBvTemp ); 
        //-------------------------------
        MPI_File_open(MPI_COMM_WORLD, cPrevStateName, MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_PREVRUN);
        //-------------------------------
        MPI_File_read(fp_PREVRUN, &dAddedTime, 1, MPI_DOUBLE,             &STATUS);
        MPI_File_read(fp_PREVRUN, &uEQcntr,    1, MPI_UNSIGNED,           &STATUS);
        MPI_File_read(fp_PREVRUN, &uDummy,     1, MPI_UNSIGNED,           &STATUS);
        MPI_File_read(fp_PREVRUN, &fPSeisTime, 1, MPI_FLOAT,              &STATUS);
        //-------------------------------
        OFFSETall = sizeof(double) +2*sizeof(unsigned int) +sizeof(float);
        //--------------------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(unsigned int)), uFvTemp, iFOFFSET[iRANK], MPI_UNSIGNED, &STATUS); //stability type
        OFFSETall += (uFPNum*sizeof(unsigned int));    for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   uFEvent[i*5 +0] = uFvTemp[i];        }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr stress (strike)
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFEvent[i*16 +0] = fFvTemp[i];        }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr stress (dip)
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFEvent[i*16 +1] = fFvTemp[i];        }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr stress (normal)
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFEvent[i*16 +2] = fFvTemp[i];        }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr friction coeff.
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFEvent[i*16 +3] = fFvTemp[i];        }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr Dc_value
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFEvent[i*16 +7] = fFvTemp[i];        }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //overshoot stress (relates to friction and Dc values that can/will change)
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFEvent[i*16 +10] = fFvTemp[i];       }
        //----------------------//----------------------//--------------------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //static friction
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFFric[i*6 +0] = fFvTemp[i];          }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //dynamic friction
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFFric[i*6 +1] = fFvTemp[i];          }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //arrest friction
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFFric[i*6 +2] = fFvTemp[i];          }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //post-seis. fric friction
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFFric[i*6 +4] = fFvTemp[i];          }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //the element's slip-rate
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFRef[i*14 +12] = fFvTemp[i];          }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iFSTART[iRANK]*sizeof(float)), fFvTemp, iFOFFSET[iRANK], MPI_FLOAT, &STATUS); //the element's total accum. slip
        OFFSETall += (uFPNum*sizeof(float));           for (i = 0u; i < iFOFFSET[iRANK]; i++)      {   fFRef[i*14 +13] = fFvTemp[i];          }
        //----------------------//----------------------//--------------------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iBSTART[iRANK]*sizeof(float)), fBvTemp, iBOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr stress (strike)
        OFFSETall += (uBPNum*sizeof(float));           for (i = 0u; i < iBOFFSET[iRANK]; i++)      {   fBEvent[i*9 +0] = fBvTemp[i];         }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iBSTART[iRANK]*sizeof(float)), fBvTemp, iBOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr stress (dip)
        OFFSETall += (uBPNum*sizeof(float));           for (i = 0u; i < iBOFFSET[iRANK]; i++)      {   fBEvent[i*9 +1] = fBvTemp[i];         }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iBSTART[iRANK]*sizeof(float)), fBvTemp, iBOFFSET[iRANK], MPI_FLOAT, &STATUS); //curr stress (normal)
        OFFSETall += (uBPNum*sizeof(float));           for (i = 0u; i < iBOFFSET[iRANK]; i++)      {   fBEvent[i*9 +2] = fBvTemp[i];         }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iBSTART[iRANK]*sizeof(float)), fBvTemp, iBOFFSET[iRANK], MPI_FLOAT, &STATUS); //postseismic stress (shear)
        OFFSETall += (uBPNum*sizeof(float));           for (i = 0u; i < iBOFFSET[iRANK]; i++)      {   fBEvent[i*9 +7] = fBvTemp[i];         }
        //----------------------
        MPI_File_read_at(fp_PREVRUN,   (OFFSETall + iBSTART[iRANK]*sizeof(float)), fBvTemp, iBOFFSET[iRANK], MPI_FLOAT, &STATUS); //postseismic stress (normal)
        OFFSETall += (uBPNum*sizeof(float));           for (i = 0u; i < iBOFFSET[iRANK]; i++)      {   fBEvent[i*9 +8] = fBvTemp[i];         }
        //----------------------//--
        MPI_File_close(&fp_PREVRUN);
        MPI_Barrier( MPI_COMM_WORLD );
        //----------------------//----------------------//--------------------------------
        if (iRANK == 0)     {   if ((fp_CATALOG = fopen(cOutputName,"rb+"))     == NULL)     {   printf("Error -cant open %s RawCatalogFile...\n",cOutputName);      exit(10);     }           }
       //----------------------//----------------------//--------------------------------
    }
    else
    {   //-------------------------------
        if (iRANK == 0)
        {   if ((fp_CATALOG = fopen(cOutputName,"wb+"))     == NULL)     {   printf("Error -cant open %s  RawCatalogFile...\n",cOutputName);      exit(10);     }
            unsigned int uDummy = 3u;
            unsigned int *uTempF = calloc(uFPNum, sizeof *uTempF );
            float *fTempF  = calloc(uFPNum, sizeof *fTempF );
            float *fTempFV = calloc(uFVNum, sizeof *fTempFV );

            fwrite(&uEQcntr,  sizeof(unsigned int), 1, fp_CATALOG);
            fwrite(&uDummy,   sizeof(unsigned int), 1, fp_CATALOG);
            fwrite(&fModPara[2],     sizeof(float), 1, fp_CATALOG);
            fwrite(&fModPara[7],     sizeof(float), 1, fp_CATALOG);
            fwrite(&uFPNum,   sizeof(unsigned int), 1, fp_CATALOG);
            fwrite(&uFVNum,   sizeof(unsigned int), 1, fp_CATALOG);
        
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uF_temp[i*6 +0];   }     fwrite(uTempF, sizeof(unsigned int), uFPNum, fp_CATALOG);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uF_temp[i*6 +1];   }     fwrite(uTempF, sizeof(unsigned int), uFPNum, fp_CATALOG);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uF_temp[i*6 +2];   }     fwrite(uTempF, sizeof(unsigned int), uFPNum, fp_CATALOG);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uF_temp[i*6 +3];   }     fwrite(uTempF, sizeof(unsigned int), uFPNum, fp_CATALOG);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uF_temp[i*6 +4];   }     fwrite(uTempF, sizeof(unsigned int), uFPNum, fp_CATALOG);
            
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +0];  }     fwrite(fTempFV, sizeof(float), uFVNum, fp_CATALOG); 
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +1];  }     fwrite(fTempFV, sizeof(float), uFVNum, fp_CATALOG);
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +2];  }     fwrite(fTempFV, sizeof(float), uFVNum, fp_CATALOG);
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFV_temp[i*4 +3];  }     fwrite(fTempFV, sizeof(float), uFVNum, fp_CATALOG);
            
            for (i = 0u; i < uFPNum; i++)   {   fTempF[i] = fF_temp[i*15 +0]; }      fwrite(fTempF,  sizeof(float), uFPNum, fp_CATALOG);
            for (i = 0u; i < uFPNum; i++)   {   fTempF[i] = fF_temp[i*15 +1]; }      fwrite(fTempF,  sizeof(float), uFPNum, fp_CATALOG);
            for (i = 0u; i < uFPNum; i++)   {   fTempF[i] = fF_temp[i*15 +2]; }      fwrite(fTempF,  sizeof(float), uFPNum, fp_CATALOG);
    }   }//open the existing EQcatalog file AND also set stress state to post-run conditions
    //------------------------------------------------------------------------------------
    free(uF_temp);          free(fFV_temp);         free(fF_temp);
    free(uB_temp);          free(fBV_temp);         free(fB_temp);
    //------------------------------------------------------------------------------------
    if (uContPrevRun == 0u)
    {   float fTemp0,   fTemp1,   fTemp2;
        for (i = 0u; i < iFOFFSET[iRANK]; i++)
        {   fTemp0   = sqrtf(fFEvent[i*16 +8]*fFEvent[i*16 +8] + fFEvent[i*16 +9]*fFEvent[i*16 +9]); //combined stressing rate (still in years)
            fTemp1   = fFEvent[i*16 +3]*-1.0f*fFEvent[i*16 +2]; //the element's current strength
            fTemp2   = (1.0f - fPreStressFract)*ran0(&lSeed); //difference between "loading to failure" and prestressfraction multiplied by random number between 0 and 1 => e.g., if prestress is 0.8; then value is going to be (1.0f - 0.8f)*[0,1] i.e., between 0.0f and 0.2f 
            fFEvent[i*16 +0] =  ((fTemp1/fTemp0)*(fPreStressFract +fTemp2)) * fFEvent[i*16 +8];
            fFEvent[i*16 +1] =  ((fTemp1/fTemp0)*(fPreStressFract +fTemp2)) * fFEvent[i*16 +9];
    }   } // pre-load the fault
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------
    unsigned int uTemp0,   uTemp1,   uTemp2,   uGlobPos,   uEQstillOn,   uTotlRuptT,   uActElmG,   uActElmL,   uMRFlgth;
    float fNxtLoadStep, fNxtHalfStep,   fTemp0,   fTemp1,   fTemp2,   fTemp3,    fTemp4,   fTemp5,   fTemp6,   fTemp7;
    float fMaxLoadStep      = FloatPow(2.0f, uLoadStep_POW2);
    float fLoadingStepInYrs = fIntSeisLoadStep/365.25f;
    float fEQMagn,   fEQMomRel,   fHypoSlip,   fHypoLoc[3],   fEQMeanVals[6];
    double dTemp0;
    //----------------------------
    fMaxLoadStep *= fLoadingStepInYrs;
    //----------------------------
    int *iStartPos = calloc(iSIZE, sizeof *iStartPos);
    int *iOffstPos = calloc(iSIZE, sizeof *iOffstPos);
    long long int *lSTF_offset= calloc(iSIZE, sizeof *lSTF_offset);
    long long int *lSTF_start = calloc(iSIZE, sizeof *lSTF_start);
    unsigned int *uPatchID    = calloc(iFOFFSET[iRANK], sizeof *uPatchID);
    unsigned int *ut0ofPtch   = calloc(iFOFFSET[iRANK], sizeof *ut0ofPtch);
    unsigned int *uStabType   = calloc(iFOFFSET[iRANK], sizeof *uStabType);
    float        *fPtchDTau   = calloc(iFOFFSET[iRANK], sizeof *fPtchDTau);
    float        *fPtchSlpH   = calloc(iFOFFSET[iRANK], sizeof *fPtchSlpH);
    float        *fPtchSlpV   = calloc(iFOFFSET[iRANK], sizeof *fPtchSlpV);
    unsigned int *uTemp2Wrt1  = calloc(uFPNum,   sizeof *uTemp2Wrt1);
    unsigned int *uTemp2Wrt2  = calloc(uFPNum,   sizeof *uTemp2Wrt2);
    unsigned int *uTemp2Wrt3  = calloc(uFPNum,   sizeof *uTemp2Wrt3);
    float        *fTemp2Wrt1  = calloc(uFPNum,   sizeof *fTemp2Wrt1);
    float        *fTemp2Wrt2  = calloc(uFPNum,   sizeof *fTemp2Wrt2);
    float        *fTemp2Wrt3  = calloc(uFPNum,   sizeof *fTemp2Wrt3);
    //----------------------------
    dRecordLength      += dAddedTime;
    double dCurrTime    = dAddedTime;
    double dLastEQtime  = dAddedTime;
    //------------------------
    unsigned int uPSeisSteps = 0u;
    float fMaxMagnitude      = 0.0f;
    //------------------------
    for (j = 0u; j < iBOFFSET[iRANK]; j++)      {   fBEvent[j*9 +0] = 0.0f;        fBEvent[j*9 +1] = 0.0f;         fBEvent[j*9 +2] = 0.0f;          }
    //----------------------------------------------------------------------------
    //----------------------------------------------------------------------------
    if (iRANK == 0)     {   fprintf(stdout,"Starting the catalog...max load step (years) = %f\n",fMaxLoadStep);           }
    timer0 = clock();
    timerI = clock();
    //------------------------
    while (dCurrTime <= dRecordLength)
    {   //--------------------------------------------------------------------------------
        fNxtLoadStep = fMaxLoadStep;
        fNxtHalfStep = fNxtLoadStep;
        //--------------------------------------------------------------------------------
        for (k = 0u; k <= (uLoadStep_POW2+1); k++)
        {   uPSeisSteps += (uEQcntr > 0u)*1u + (uEQcntr <= 0u)*0u;
            //-----------------------------------------------------------------------
            memset(fFslip,0, 2*uFPNum*sizeof(float));
            //-----------------------------------------------------------------------
            uTemp1 = 0u;
            fTemp0 = MAX(0.0f, expf(-1.0f*(fPSeisTime + fNxtLoadStep)/fAfterSlipTime));
            //--------------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++)
            {   dTemp0 = (dCurrTime + (double)fNxtLoadStep)*(double)fFRef[i*14 +12]; //total time that has passed (in years) multiplied w/ annual slip-rate ==> accumulated slip
                fFTempVal[i*5 +0] = (fFRef[i*14 +13] <= dTemp0)*(fFEvent[i*16 +0] +fNxtLoadStep*fFEvent[i*16 +8]) + (fFRef[i*14 +13] > dTemp0)*fFEvent[i*16 +0];
                fFTempVal[i*5 +1] = (fFRef[i*14 +13] <= dTemp0)*(fFEvent[i*16 +1] +fNxtLoadStep*fFEvent[i*16 +9]) + (fFRef[i*14 +13] > dTemp0)*fFEvent[i*16 +1];
                fFTempVal[i*5 +2] = fFEvent[i*16 +2];
                fFTempVal[i*5 +3] = fFFric[i*6 +0] + fTemp0*fFFric[i*6 +4];//Pseis-Fric refers to difference in current and static friction (directly after last event); for unstable elements this is set to zero 
                fFTempVal[i*5 +4] = 0.0f;
                //--------------------------//--------------------------
                fTemp1 = sqrtf( fFTempVal[i*5 +0]*fFTempVal[i*5 +0] + fFTempVal[i*5 +1]*fFTempVal[i*5 +1] );
                fTemp1 = (uFEvent[i*5 +0] == 1u)*0.0f + (uFEvent[i*5 +0] != 1u)*fTemp1; // combined shear stress on element
                fTemp3 = fFTempVal[i*5 +3] *-1.0f*fFTempVal[i*5 +2]; //current frictional strength of element
                //--------------------------
                fTemp5 = MAX(0.0f, (fTemp1 -fTemp3));
                //--------------------------
                if (fTemp5 > MININTSEISSTRESSCHANGE)
                {   fTemp3   = -1.0f*(((fTemp5/fTemp1)*fFTempVal[i*5 +0])/fFEvent[i*16 +4]);
                    fTemp4   = -1.0f*(((fTemp5/fTemp1)*fFTempVal[i*5 +1])/fFEvent[i*16 +5]);
                    fFTempVal[i*5 +4] = sqrtf(fTemp3*fTemp3 +fTemp4*fTemp4);
                    //--------------------------
                    uTemp1                = 1u;             uGlobPos              = i+ iFSTART[iRANK];
                    fFslip[uGlobPos*2 +0] = fTemp3;         fFslip[uGlobPos*2 +1] = fTemp4;
            }   }
            //-----------------------------------------------------------------------
            MPI_Allreduce(MPI_IN_PLACE, &uTemp1, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
            //-----------------------------------------------------------------------
            if (uTemp1 > 0u)
            {   //-----------------------------------------------------------------------
                MPI_Allreduce(MPI_IN_PLACE, fFslip, 2*uFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                //-----------------------------------------------------------------------
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fFTempVal[i*5 +0] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvalstk[i], 1);
                    fFTempVal[i*5 +1] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvaldip[i], 1);
            }   }
            //-----------------------------------------------------------------------
            memset(fBslip,0, 3*uBPNum*sizeof(float));
            //-----------------------------------------------------------------------
            uTemp2 = 0u;
            fTemp0 = MAX(0.0f, expf(-1.0f*(fPSeisTime +fNxtLoadStep)/fDeepRelaxTime));
            //--------------------------------------------
            for (i = 0u; i < iBOFFSET[iRANK]; i++)
            {   fBTempVal[i*3 +0] = fBEvent[i*9 +0];
                fBTempVal[i*3 +1] = fBEvent[i*9 +1];
                fBTempVal[i*3 +2] = fBEvent[i*9 +2];
                //--------------------------//--------------------------
                fTemp1 = sqrtf( fBTempVal[i*3 +0]*fBTempVal[i*3 +0] + fBTempVal[i*3 +1]*fBTempVal[i*3 +1] ); // combined (potentially releasable) shear stress on element
                fTemp2 =   fabs(fBTempVal[i*3 +2]); //potentially releasable normal stress on element
                fTemp3 = fTemp0*fBEvent[i*9 +7]; //permissible shear stress; amount of shear stress on element after last coseismic phase (absolute value) times postseis value => is permitted excess in stress at that point in time
                fTemp4 = fTemp0*fBEvent[i*9 +8]; //permissible normal stress; amount of normal stress on element after last coseismic phase (absolute value)
                //--------------------------
                fTemp5 = MAX(0.0f, (fTemp1 -fTemp3));
                fTemp6 = MAX(0.0f, (fTemp2 -fTemp4));
                fTemp7 = sqrtf(fTemp5*fTemp5 + fTemp6*fTemp6);
                //--------------------------
                if (fTemp7 > MININTSEISSTRESSCHANGE)
                {   fTemp3 = -1.0f*(((fTemp5/fTemp1)*fBTempVal[i*3 +0])/fBEvent[i*9 +4]);
                    fTemp4 = -1.0f*(((fTemp5/fTemp1)*fBTempVal[i*3 +1])/fBEvent[i*9 +5]);
                    fTemp5 = -1.0f*(((fTemp6/fTemp2)*fBTempVal[i*3 +2])/fBEvent[i*9 +6]);
                    //--------------------------
                    uTemp2   = 1u;                          uGlobPos               = i+ iBSTART[iRANK];
                    fBslip[uGlobPos*3 +0] += fTemp3;        fBslip[uGlobPos*3 +1] += fTemp4;      fBslip[uGlobPos*3 +2] += fTemp5; 
            }   }
            //-----------------------------------------------------------------------
            MPI_Allreduce(MPI_IN_PLACE, &uTemp2, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
            //-----------------------------------------------------------------------
            //-----------------------------------------------------------------------
            if (uTemp2 > 0u)
            {   //-----------------------------------------------------------------------
                MPI_Allreduce(MPI_IN_PLACE, fBslip, 3*uBPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                //-----------------------------------------------------------------------
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fFTempVal[i*5 +0] += cblas_sdot(3*uBPNum, fBslip, 1, fK_FBvalstk[i], 1);
                    fFTempVal[i*5 +1] += cblas_sdot(3*uBPNum, fBslip, 1, fK_FBvaldip[i], 1);
            }   }
            //-----------------------------------------------------------------------
            uEQstillOn = 0u;
            fHypoSlip  = 0.0f;      fHypoLoc[0] = 0.0f;       fHypoLoc[1] = 0.0f;      fHypoLoc[2] = 0.0f;
            //------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++)
            {   //----------------------------------------
                fTemp1 = sqrtf( fFTempVal[i*5 +0]*fFTempVal[i*5 +0] + fFTempVal[i*5 +1]*fFTempVal[i*5 +1] );
                fTemp1 = (uFEvent[i*5 +0] == 1u)*fTemp1 + (uFEvent[i*5 +0] != 1u)*0.0f; // combined shear stress on element; shear stress is zero if element is not unstable
                fTemp2 =    fFFric[i*6 +2] *-1.0f*fFTempVal[i*5 +2]; //arrest fault strength
                fTemp3 = fFTempVal[i*5 +3] *-1.0f*fFTempVal[i*5 +2]; //current frictional strength of element
                fTemp5 = MAX(0.0f,(fTemp1 -fTemp3)) /fFRef[i*14 +2]; //slip amount to release excess stress on element that is currently present (the MAX ensures that it is really excess); use this instead of previous "fraction2startrupture" => is more consistent this way
                //---------------------------------
                uTemp0 = (fTemp5 > fModPara[10])*1u + (fTemp5 <= fModPara[10])*0u;
                uTemp0 = ((fTemp1-fTemp2) > fFEvent[i*16 +10])*uTemp0 + ((fTemp1-fTemp2) <= fFEvent[i*16 +10])*0u;
                uEQstillOn += uTemp0;
                fTemp5 = (uTemp0 == 1u)*fTemp5 + (uTemp0 != 1u)*0.0f; 
                //---------------------------------
                if (fTemp5 > fHypoSlip)         {   fHypoSlip  = fTemp5;        fHypoLoc[0] = fFRef[i*14 + 9];       fHypoLoc[1] = fFRef[i*14 +10];      fHypoLoc[2] = fFRef[i*14 +11];         }
                //---------------------------------
            }
            //--------------------------------------------
            MPI_Allreduce(MPI_IN_PLACE, &uEQstillOn, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
            //--------------------------------------------
            fNxtHalfStep /= 2.0f;
            if (uEQstillOn == 0u)
            {   if (k == 0u)                                    {                                               break;   }
                else if ((k > 0u) && (k < uLoadStep_POW2))      {   fNxtLoadStep += fNxtHalfStep;                        }
                else if (k == uLoadStep_POW2)                   {   fNxtLoadStep +=(2.0f*fNxtHalfStep);                  }
            }
            else
            {   if      (k < uLoadStep_POW2)                    {   fNxtLoadStep -= fNxtHalfStep;                        }
                else if (k == uLoadStep_POW2)                   {                                               break;   }
            }
            //--------------------------------------------
        } //apply tectonic loading and postseismic relaxation, also test for failure; do this by first copying stress to tempval and then manipulating tempval, once leaving this loop, copy tempval back to Events
        //--------------------------------------------
        for (i = 0u; i < iBOFFSET[iRANK]; i++)
        {   fBTempVal[i*3 +0] += cblas_sdot(2*uFPNum, fFslip, 1, fK_BFvalstk[i], 1);
            fBTempVal[i*3 +1] += cblas_sdot(2*uFPNum, fFslip, 1, fK_BFvaldip[i], 1);
            fBTempVal[i*3 +2] += cblas_sdot(2*uFPNum, fFslip, 1, fK_BFvalnrm[i], 1);
        
            fBTempVal[i*3 +0] += cblas_sdot(3*uBPNum, fBslip, 1, fK_BBvalstk[i], 1);
            fBTempVal[i*3 +1] += cblas_sdot(3*uBPNum, fBslip, 1, fK_BBvaldip[i], 1);
            fBTempVal[i*3 +2] += cblas_sdot(3*uBPNum, fBslip, 1, fK_BBvalnrm[i], 1);
        }
        //--------------------------------------------
        fPSeisTime += fNxtLoadStep;
        dCurrTime  += (double)fNxtLoadStep;
        //--------------------------------------------
        for (i = 0u; i < iFOFFSET[iRANK]; i++)
        {   fFEvent[i*16 +0] = fFTempVal[i*5 +0];       fFEvent[i*16 +1] = fFTempVal[i*5 +1];       fFEvent[i*16 +2] = fFTempVal[i*5 +2];
            fFEvent[i*16 +3] = fFTempVal[i*5 +3];       fFRef[i*14 +13] += fFTempVal[i*5 +4]; 
        }
        for (i = 0u; i < iBOFFSET[iRANK]; i++)
        {   fBEvent[i*9 +0] = fBTempVal[i*3 +0];        fBEvent[i*9 +1] = fBTempVal[i*3 +1];        fBEvent[i*9 +2] = fBTempVal[i*3 +2];
        }
        //--------------------------------------------------------------------------------
        //--------------------------------------------------------------------------------
        //--------------------------------------------------------------------------------
        if (uEQstillOn > 0u)
        {   timerI = clock() - timerI;        time_InterSeis += (((double)timerI)/CLOCKS_PER_SEC);
            timerC = clock();
            //----------------------------------------
            uMRFlgth  = 0u;             uActElmL  = 0u;         uTotlRuptT = 0u;            fEQMomRel = 0.0f;
            memset(uFActivEl, 0, iFOFFSET[iRANK]*sizeof(unsigned int));
            memset(fEQslipSD, 0, 2*uFPNum*sizeof(float) );
            memset(fMRFvals,  0, MAXMOMRATEFUNCLENGTH*sizeof(float) );
            memset(fSTFslip,  0, 2*iFOFFSET[iRANK]*MAXMOMRATEFUNCLENGTH*sizeof(float));
            //---------------------------
            {   struct { float val;  int rank;  } in[1], out[1];
                in[0].val  = fHypoSlip;         in[0].rank = iRANK;
                //---------------
                MPI_Allreduce(in, out, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
                //---------------
                if (out[0].rank != 0)
                {   if (iRANK == out[0].rank)   {   MPI_Send(fHypoLoc, 3, MPI_FLOAT,    0,        123, MPI_COMM_WORLD);               }
                    else if (iRANK == 0)        {   MPI_Recv(fHypoLoc, 3, MPI_FLOAT, out[0].rank, 123, MPI_COMM_WORLD, &STATUS);      }
                }
            } //determine the hypocenter location
            //----------------------------------------------------------------------------
            for (i = 0u; i < iFOFFSET[iRANK]; i++)
            {   //-----------------------
                fFTempVal[i*5 +0] = fFEvent[i*16 +0];       fFTempVal[i*5 +1] = fFEvent[i*16 +1];       fFTempVal[i*5 +2] = fFEvent[i*16 +2];       fFTempVal[i*5 +3] = fFEvent[i*16 +3];
                //-----------------------
                fTemp0 = (fFFric[i*6 +2] *-1.0f *fFEvent[i*16 +2]) + fFRef[i*14 +2]*fFEvent[i*16 +7]; //this is arrest strength plus the stress/strength needed to achieve D_c slip (use arrest, b/c D_c was adjusted accordingly)
                fFEvent[i*16 +3] = (uFEvent[i*5 +0] == 1)*fFEvent[i*16 +3]  +  (uFEvent[i*5 +0] != 1)*(fTemp0/(-1.0f *fFEvent[i*16 +2]));
                fFFric[i*6 +3]   = fFEvent[i*16 +3]; //the temprefFriction from which I decay from
                fFFric[i*6 +5]   = fFEvent[i*16 +3]; //the temprefFriction to which I will heal
                //-----------------------
            }//determine CurrFric => increased by coseismically induced stresses; and define maxSlip perStep (radiation damping), based on elastic properties and (max.) stress drop
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            while (uEQstillOn > 0u)
            {   uEQstillOn = 0u;
                memset(fFslip,  0, 2*uFPNum*sizeof(float) );
                //----------------------------------------------------
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fTemp2 = fFFric[i*6 +3] - (fFFric[i*6 +3] - fFFric[i*6 +2]) *(fFEvent[i*16 +12]/fFEvent[i*16 +7]);
                    fTemp2 = MAX(fTemp2, fFFric[i*6 +2]);//the weakening part
                    fTemp3 = fFEvent[i*16 +3] + fHealFact*(fFFric[i*6 +5] - fFEvent[i*16 +3]); //healing part, if element is not slipping 
                    //updating friction coefficient for elements that got activated already
                    fFEvent[i*16 +3] = (uFEvent[i*5 +1] != 1u)*fFEvent[i*16 +3] +(uFEvent[i*5 +1] == 1u)*((fFEvent[i*16 +11] == 0.0f)*fTemp3  +(fFEvent[i*16 +11] != 0.0f)*fTemp2           );
                    fFFric[i*6 +3]   = (uFEvent[i*5 +1] != 1u)*fFFric[i*6 +3]   +(uFEvent[i*5 +1] == 1u)*((fFEvent[i*16 +11] == 0.0f)*fTemp3  +(fFEvent[i*16 +11] != 0.0f)*fFFric[i*6 +3]   );
                    fFEvent[i*16 +12]= (fFEvent[i*16 +11] != 0.0f)*fFEvent[i*16 +12]  +  (fFEvent[i*16 +11] == 0.0f)*0.0f;
                    //----------------------------------------------------
                    fTemp0 = sqrtf( fFEvent[i*16 +0]*fFEvent[i*16 +0] + fFEvent[i*16 +1]*fFEvent[i*16 +1] ); //combined shear stress on element
                    fFEvent[i*16 +2] = (fFEvent[i*16 +2] < 0.0f)*fFEvent[i*16 +2] + (fFEvent[i*16 +2] >= 0.0f)*0.0f; //make sure that normal stress does not get higher than zero (positive normal stress == tension) => this is b/c current code does not know how to handle this situation as it is only friction-based
                    //----------------------------------------------------
                    fTemp1 = fFEvent[i*16 +3] *-1.0f*fFEvent[i*16 +2]; //current fault strength
                    fTemp2 =   fFFric[i*6 +2] *-1.0f*fFEvent[i*16 +2]; //arrest fault strength
                    fTemp3 = INTERNALCOHESION + INTERNALFRICTION*-1.0f*fFEvent[i*16 +2]; //this defines the rock strength (as opposed to frictional strength)
                    //----------------------------------------------------
                    uFEvent[i*5 +2] = (fTemp0 < fTemp3)*uFEvent[i*5 +2] + (fTemp0 >= fTemp3)*1u; //currently applied stresses exceed rock strength => yielding -element is "broken" => no elastic interaction (anymore) in the current EQ
                    //----------------------------------------------------
                    fTemp3 = fTemp0 - fTemp1; //amount of excess stress above current friction level
                    fTemp4 = fTemp0 - fTemp2; //amount of excess stress above arrest friction level
                    fTemp5 = MAX(fTemp3, 0.0f) / fFRef[i*14 +2]; //slip amount to release excess stress on element that is currently present (the MAX ensures that it is really excess); use this instead of previous "fraction2startrupture" => is more consistent this way
                    //-----------------------------------------------------
                    uTemp0   = (uFEvent[i*5 +1] == 0u)*1u           + (uFEvent[i*5 +1] != 0u)*0u;
                    uTemp0   = (fTemp5 >= fModPara[10])*uTemp0      + (fTemp5 < fModPara[10])*0u;
                    uTemp0   = (fTemp4 >= fFEvent[i*16 +10])*uTemp0 + (fTemp4 < fFEvent[i*16 +10])*0u;
                    uFActivEl[uActElmL] = (uTemp0 == 1u)*i            + (uTemp0 != 1u)*uFActivEl[uActElmL];
                    uActElmL            = (uTemp0 == 1u)*(uActElmL+1u)+ (uTemp0 != 1u)*uActElmL;
                    uFEvent[i*5 +1]     = (uTemp0 == 1u)*1u           + (uTemp0 != 1u)*uFEvent[i*5 +1];
                    uFEvent[i*5 +3]     = (uTemp0 == 1u)*(uTotlRuptT) + (uTemp0 != 1u)*uFEvent[i*5 +3];
                    //----------------------------------------------------
                    uTemp0   = (uFEvent[i*5 +2] == 0u)*1u           + (uFEvent[i*5 +2] != 0u)*0u;
                    uTemp0   = (uFEvent[i*5 +1] == 1u)*uTemp0       + (uFEvent[i*5 +1] != 1u)*0u;
                    uTemp0   = (fTemp5 >= fModPara[10])*uTemp0      + (fTemp5 < fModPara[10])*0u;
                    uTemp0   = (fTemp4 >= fFEvent[i*16 +10])*uTemp0 + (fTemp4 < fFEvent[i*16 +10])*0u;
                    //------------------------------------------------
                    fTemp5 = 0.0f;
                    fTemp6 = 0.0f;
                    if (uTemp0 == 1u)
                    {   uEQstillOn       += 1u;
                        uMRFlgth          = uTotlRuptT +1u;
                        uFEvent[i*5 +4]   = uTotlRuptT - uFEvent[i*5 +3] +1u;
                        fFEvent[i*16 +15] = fabs( fModPara[9] *(fTemp3/fModPara[2]) *fModPara[7] );
                        //----------------------------
                        fTemp5 = -1.0f*((fTemp3/fTemp0) *fFEvent[i*16 +0]) / fFEvent[i*16 +4];
                        fTemp6 = -1.0f*((fTemp3/fTemp0) *fFEvent[i*16 +1]) / fFEvent[i*16 +5];
                        fTemp7 = sqrtf(fTemp5*fTemp5 +fTemp6*fTemp6);
                        fTemp5 = (fFEvent[i*16 +15] < fTemp7)*(fTemp5*(fFEvent[i*16 +15]/fTemp7))  +  (fFEvent[i*16 +15] >= fTemp7)*fTemp5;
                        fTemp6 = (fFEvent[i*16 +15] < fTemp7)*(fTemp6*(fFEvent[i*16 +15]/fTemp7))  +  (fFEvent[i*16 +15] >= fTemp7)*fTemp6;
                        //------------------------------------------------
                        uGlobPos = i +iFSTART[iRANK];
                        fFslip[   uGlobPos*2 +0]  = fTemp5;                           fFslip[   uGlobPos*2 +1]  = fTemp6;
                        fEQslipSD[uGlobPos*2 +0] += fTemp5;                           fEQslipSD[uGlobPos*2 +1] += fTemp6;
                        //------------------------------------------------
                        uTemp0   = 2u*i*MAXMOMRATEFUNCLENGTH + 2u*(uTotlRuptT - uFEvent[i*5 +3]);
                        fSTFslip[uTemp0 +0]       = fTemp5;                           fSTFslip[uTemp0 +1]       = fTemp6;
                        //------------------------------------------------
                    }
                    //------------------------------------------------
                    fTemp5              = (uFEvent[i*5 +2] == 1u)*(-1.0f*((fTemp4/fTemp0) *fFEvent[i*16 +0])/fFEvent[i*16 +4]) + (uFEvent[i*5 +2] != 1u)*fTemp5;
                    fTemp6              = (uFEvent[i*5 +2] == 1u)*(-1.0f*((fTemp4/fTemp0) *fFEvent[i*16 +1])/fFEvent[i*16 +5]) + (uFEvent[i*5 +2] != 1u)*fTemp6;
                    fFEvent[i*16 +0]    = (uFEvent[i*5 +2] == 1u)*(fFEvent[i*16 +0] + fTemp5*fFEvent[i*16 +4])                 + (uFEvent[i*5 +2] != 1u)*fFEvent[i*16 +0];
                    fFEvent[i*16 +1]    = (uFEvent[i*5 +2] == 1u)*(fFEvent[i*16 +1] + fTemp6*fFEvent[i*16 +5])                 + (uFEvent[i*5 +2] != 1u)*fFEvent[i*16 +1];
                    fFEvent[i*16 +2]    = (uFEvent[i*5 +2] == 1u)*fFRef[i*14 +1]                                               + (uFEvent[i*5 +2] != 1u)*fFEvent[i*16 +2];
                    //----------------------------
                    fFEvent[i*16 +11]   = sqrtf(fTemp5*fTemp5 +fTemp6*fTemp6);
                    fFEvent[i*16 +12]  += fFEvent[i*16 +11];
                    fFEvent[i*16 +13]  += fTemp5;
                    fFEvent[i*16 +14]  += fTemp6;
                    fEQMomRel          += (fFEvent[i*16 +11]*fFRef[i*14 +0]);
                    fMRFvals[uMRFlgth] += (fFEvent[i*16 +11]*fFRef[i*14 +0]);
                    //----------------------------------------------------
                }
                //-----------------------------------------------------------------------
                MPI_Allreduce(MPI_IN_PLACE, &uEQstillOn, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD); 
                //-----------------------------------------------------------------------
                if (uEQstillOn > 0u)
                {   //-------------------------------
                    MPI_Allreduce(MPI_IN_PLACE, fFslip, 2*uFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); 
                    //-------------------------------
                    for (i = 0u; i < iFOFFSET[iRANK]; i++)
                    {   fFEvent[i*16 +0] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvalstk[i], 1);
                        fFEvent[i*16 +1] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvaldip[i], 1);
                        fFEvent[i*16 +2] += cblas_sdot(2*uFPNum, fFslip, 1, fK_FFvalnrm[i], 1);
                }   }
                //-------------------------------
                uEQstillOn = (uTotlRuptT < MAXMOMRATEFUNCLENGTH-1)*uEQstillOn  + (uTotlRuptT >= MAXMOMRATEFUNCLENGTH-1)*0u; //hard stop
                uTotlRuptT += 1u;
            }
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            uActElmG = uActElmL;
            MPI_Allreduce(MPI_IN_PLACE, &uActElmG,   1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &fEQMomRel,  1, MPI_FLOAT,    MPI_SUM, MPI_COMM_WORLD);
            //---------------------------------
            fEQMagn = (log10f(fEQMomRel * fModPara[2]) -9.1f)/1.5f;
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            if (uBPNum > 0)
            {   //------------------------------
                 MPI_Allreduce( MPI_IN_PLACE, fEQslipSD, 2*uFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                //------------------------------
                for (i = 0u; i < iBOFFSET[iRANK]; i++)
                {   fBEvent[i*9 +0] += cblas_sdot(2*uFPNum, fEQslipSD, 1, fK_BFvalstk[i], 1);
                    fBEvent[i*9 +1] += cblas_sdot(2*uFPNum, fEQslipSD, 1, fK_BFvaldip[i], 1);
                    fBEvent[i*9 +2] += cblas_sdot(2*uFPNum, fEQslipSD, 1, fK_BFvalnrm[i], 1);
                    //------------------------------
                    fBEvent[i*9 +7] = sqrtf(fBEvent[i*9 +0]*fBEvent[i*9 +0] + fBEvent[i*9 +1]*fBEvent[i*9 +1]);
                    fBEvent[i*9 +8] = fabs(fBEvent[i*9 +2]);
                }
            } //apply EQ slip to load the boundary surfaces
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            if (uActElmG >= uMinElemNum4Cat)
            {   uEQcntr += 1u;
                for (i = 0u; i < iSIZE; i++)        {   iStartPos[i] = 0;       iOffstPos[i] = 0;       }
                for (i = 0u; i < 6;     i++)        {   fEQMeanVals[i] = 0.0f;                          }
                //-------------
                for (i = 0u; i < uActElmL; i++)
                {   uTemp1   = uFActivEl[i];
                    //-------------------------------
                    fTemp0          = sqrtf(fFEvent[uTemp1*16 +13]*fFEvent[uTemp1*16 +13] +fFEvent[uTemp1*16 +14]*fFEvent[uTemp1*16 +14]);
                    fEQMeanVals[0] += (fFRef[uTemp1*14 + 9]*fTemp0*fFRef[uTemp1*14 +0]); // moment centroid location
                    fEQMeanVals[1] += (fFRef[uTemp1*14 +10]*fTemp0*fFRef[uTemp1*14 +0]);
                    fEQMeanVals[2] += (fFRef[uTemp1*14 +11]*fTemp0*fFRef[uTemp1*14 +0]);
                    fEQMeanVals[3] +=  fFRef[uTemp1*14 +0]; // rupture area
                    fEQMeanVals[4] +=  fTemp0;
                    fEQMeanVals[5] +=  sqrtf(fFTempVal[uTemp1*5 +0]*fFTempVal[uTemp1*5 +0] + fFTempVal[uTemp1*5 +1]*fFTempVal[uTemp1*5 +1]) - sqrtf(fFEvent[uTemp1*16 +0]*fFEvent[uTemp1*16 +0] + fFEvent[uTemp1*16 +1]*fFEvent[uTemp1*16 +1]); //stress drop
                    //-------------------------------------
                    uPatchID[i]  = uFActivEl[i] + iFSTART[iRANK];
                    ut0ofPtch[i] = uFEvent[uTemp1*5 +3];
                    uStabType[i] = uFEvent[uTemp1*5 +0];
                    fPtchDTau[i] = sqrtf(fFTempVal[uTemp1*5 +0]*fFTempVal[uTemp1*5 +0] + fFTempVal[uTemp1*5 +1]*fFTempVal[uTemp1*5 +1]) - sqrtf(fFEvent[uTemp1*16 +0]*fFEvent[uTemp1*16 +0] + fFEvent[uTemp1*16 +1]*fFEvent[uTemp1*16 +1]); //stress drop
                    fPtchSlpH[i] = fFEvent[uTemp1*16 +13];
                    fPtchSlpV[i] = fFEvent[uTemp1*16 +14];
                }
                iOffstPos[iRANK] = (int)uActElmL;
                //----------------------------------------------
                MPI_Allreduce( MPI_IN_PLACE, &uMRFlgth,        1,            MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce( MPI_IN_PLACE, iOffstPos,      iSIZE,          MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                MPI_Allreduce( MPI_IN_PLACE, fMRFvals, MAXMOMRATEFUNCLENGTH, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce( MPI_IN_PLACE, fEQMeanVals,      6,            MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                //---------------------------
                fEQMeanVals[0] /= fEQMomRel;            fEQMeanVals[1] /= fEQMomRel;            fEQMeanVals[2] /= fEQMomRel;
                fEQMeanVals[4] /= (float) uActElmG;     fEQMeanVals[5] /= (float) uActElmG;
                //---------------------------
                for (i = 1; i < iSIZE; i++)        {    iStartPos[i]  = iStartPos[i-1] +iOffstPos[i-1];    } 
                //----------------------------------------------
                if (iRANK == 0)
                {   fMaxMagnitude = (fEQMagn > fMaxMagnitude)*fEQMagn + (fEQMagn <= fMaxMagnitude)*fMaxMagnitude;
                    if (uPlotCatalog2Screen == 1u) { fprintf(stdout,"%5u  EQtime: %6.4lf  (%6.4lf since last)   RuptArea: %8.2f   MeanSlip: %2.3f   MeanDtau: %3.3f   MRF %4u   ActivElems: %6u   EQMagn: %3.2f; Max_Magn: %3.2f\n", uEQcntr, dCurrTime, (dCurrTime-dLastEQtime), (fEQMeanVals[3]*1.0E-6f), fEQMeanVals[4], (fEQMeanVals[5]*1.0E-6f), uMRFlgth, uActElmG, fEQMagn, fMaxMagnitude);       }
                    //----------------------------------------------
                    fseek(fp_CATALOG, 0, SEEK_SET);
                    fwrite(&uEQcntr,    sizeof(unsigned int), 1, fp_CATALOG);
                    fseek(fp_CATALOG, 0, SEEK_END);
                    fwrite(&dCurrTime,  sizeof(double),       1, fp_CATALOG);
                    fwrite(&fEQMagn,    sizeof(float),        1, fp_CATALOG);
                    fwrite(fHypoLoc,    sizeof(float),        3, fp_CATALOG);
                    fwrite(fEQMeanVals, sizeof(float),        6, fp_CATALOG);
                    //----------------------------------------------
                    if (uCatType > 1u)
                    {   fwrite(&uMRFlgth, sizeof(unsigned int), 1,        fp_CATALOG);
                        fwrite(fMRFvals,  sizeof(float),        uMRFlgth, fp_CATALOG);
                    }
                    //----------------------------------------------
                    if (uCatType > 2u)      {       fwrite(&uActElmG, sizeof(unsigned), 1, fp_CATALOG);         }
                }
                //----------------------------------------------
                if (uCatType > 2u)
                {   MPI_Gatherv(uPatchID,  iOffstPos[iRANK], MPI_UNSIGNED, uTemp2Wrt1, iOffstPos, iStartPos, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(ut0ofPtch, iOffstPos[iRANK], MPI_UNSIGNED, uTemp2Wrt2, iOffstPos, iStartPos, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(uStabType, iOffstPos[iRANK], MPI_UNSIGNED, uTemp2Wrt3, iOffstPos, iStartPos, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
                    MPI_Gatherv(fPtchDTau, iOffstPos[iRANK], MPI_FLOAT,    fTemp2Wrt1, iOffstPos, iStartPos, MPI_FLOAT,    0, MPI_COMM_WORLD);
                    MPI_Gatherv(fPtchSlpH, iOffstPos[iRANK], MPI_FLOAT,    fTemp2Wrt2, iOffstPos, iStartPos, MPI_FLOAT,    0, MPI_COMM_WORLD);
                    MPI_Gatherv(fPtchSlpV, iOffstPos[iRANK], MPI_FLOAT,    fTemp2Wrt3, iOffstPos, iStartPos, MPI_FLOAT,    0, MPI_COMM_WORLD);
                    
                    if (iRANK == 0)
                    {   fwrite( uTemp2Wrt1, sizeof(unsigned int), uActElmG, fp_CATALOG);
                        fwrite( uTemp2Wrt2, sizeof(unsigned int), uActElmG, fp_CATALOG);
                        fwrite( fTemp2Wrt1, sizeof(float),        uActElmG, fp_CATALOG);
                        fwrite( fTemp2Wrt2, sizeof(float),        uActElmG, fp_CATALOG);
                        fwrite( fTemp2Wrt3, sizeof(float),        uActElmG, fp_CATALOG);
                        fwrite( uTemp2Wrt3, sizeof(unsigned int), uActElmG, fp_CATALOG);
                }   }
                //----------------------------------------------
                dMeanRecurTime += (uEQcntr > 1u)*(dCurrTime - dLastEQtime) + (uEQcntr <= 1u)*0.0;
                dLastEQtime     = dCurrTime; 
            } //write EQ to catalog file
            MPI_Barrier( MPI_COMM_WORLD );
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            if ((uStoreSTF4LargeEQs == 1u) && (fEQMagn >= fMinMag4STF))
            {   memset(lSTF_offset, 0, iSIZE*sizeof(long long int) );           memset(lSTF_start, 0, iSIZE*sizeof(long long int) );
                long long int lTemp0 = 0;
                float *fSTFvals0 = calloc( MAXMOMRATEFUNCLENGTH, sizeof *fSTFvals0 );
                float *fSTFvals1 = calloc( MAXMOMRATEFUNCLENGTH, sizeof *fSTFvals1 );
                //----------------------------------------------
                for (i = 0; i < iFOFFSET[iRANK]; i++)
                {   lSTF_offset[iRANK] = (uFEvent[i*5 +1] == 0u)*lSTF_offset[iRANK]  +  (uFEvent[i*5 +1] == 1u)*(lSTF_offset[iRANK] + 2*sizeof(float) +4*sizeof(unsigned int) +2*uFEvent[i*5 +4]*sizeof(float) );
                }
                //----------------------------------------------
                MPI_Allreduce( MPI_IN_PLACE, lSTF_offset, iSIZE, MPI_LONG_LONG_INT, MPI_MAX, MPI_COMM_WORLD);
                for (i = 1; i < iSIZE; i++)        {    lSTF_start[i]  = lSTF_start[i-1] + lSTF_offset[i-1];    }
                //----------------------------------------------
                strcpy(cSTFName,cInputName);      strcat(cSTFName,"_M");         sprintf(cNameAppend, "%f",fEQMagn);      strcat(cSTFName,cNameAppend);      strcat(cSTFName,"_t");         sprintf(cNameAppend, "%lf",dCurrTime);      strcat(cSTFName,cNameAppend);      strcat(cSTFName,".srfb");
                //----------------------------------------
                MPI_File_delete(cSTFName, MPI_INFO_NULL);
                MPI_File_open(MPI_COMM_WORLD, cSTFName, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_STF);
                //----------------------------------------
                if (iRANK == 0)
                {   MPI_File_write_at(fp_STF, lTemp0,  &uActElmG,     1, MPI_UNSIGNED,  &STATUS);          lTemp0 += 1*sizeof(unsigned int);//active element #
                    MPI_File_write_at(fp_STF, lTemp0,  &fModPara[7],  1, MPI_FLOAT,     &STATUS);          lTemp0 += 1*sizeof(float);//delta_time
                    fTemp0 = -1.0f;
                    MPI_File_write_at(fp_STF, lTemp0,  &fTemp0,       1, MPI_FLOAT,     &STATUS);          lTemp0 += 1*sizeof(float);//surf-shear velo
                    MPI_File_write_at(fp_STF, lTemp0,  &fModPara[0],  1, MPI_FLOAT,     &STATUS);          lTemp0 += 1*sizeof(float);//density
                }
                //----------------------------------------
                lTemp0 = 1*sizeof(unsigned int) +3*sizeof(float); //this is "point/source number" at the beginning of the file => offset all start positions with it
                //----------------------------------------
                for (i = 0; i < iFOFFSET[iRANK]; i++)
                {   if (uFEvent[i*5 +1] == 1u)
                    {   uTemp0 = i + iFSTART[iRANK];
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0),  &uTemp0,           1, MPI_UNSIGNED,  &STATUS);           lTemp0 += 1*sizeof(unsigned int);//global index of slipping element
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0),  &uFEvent[i*5 +3],  1, MPI_UNSIGNED,  &STATUS);           lTemp0 += 1*sizeof(unsigned int);//RuptStart
                        //----------------------------------------
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0),  &fFEvent[i*16 +13],1, MPI_FLOAT,     &STATUS);           lTemp0 += 1*sizeof(float);//accum slip, strike
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0),  &uFEvent[i*5 +4],  1, MPI_UNSIGNED,  &STATUS);           lTemp0 += 1*sizeof(unsigned int);//duration in strike
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0),  &fFEvent[i*16 +14],1, MPI_FLOAT,     &STATUS);           lTemp0 += 1*sizeof(float);//accum slip, dip
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0),  &uFEvent[i*5 +4],  1, MPI_UNSIGNED,  &STATUS);           lTemp0 += 1*sizeof(unsigned int);//duration in dip
                        //----------------------------------------
                        for (j = 0; j < uFEvent[i*5 +4]; j++)       {   uTemp0 = 2u*i*MAXMOMRATEFUNCLENGTH + 2u*j;          fSTFvals0[j] = fSTFslip[uTemp0 +0];         fSTFvals1[j] = fSTFslip[uTemp0 +1];         }
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0), fSTFvals0, uFEvent[i*5 +4], MPI_FLOAT, &STATUS);          lTemp0 += uFEvent[i*5 +4]*sizeof(float);//STF in strike
                        MPI_File_write_at(fp_STF, (lSTF_start[iRANK]+lTemp0), fSTFvals1, uFEvent[i*5 +4], MPI_FLOAT, &STATUS);          lTemp0 += uFEvent[i*5 +4]*sizeof(float);//STF in dip
            }   }   } //write STF if event large enough
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            fPSeisTime      = 0.0f;
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            {   if (uChgBtwEQs == 1u)
                {   for (i = 0u; i < uActElmL; i++)  
                    {   //--------------------------------------------------------------------
                        uTemp1 = uFActivEl[i]; 
                        fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
                        fFEvent[uTemp1*16 +7] = fFRef[uTemp1*14 +7] + (fFRef[uTemp1*14 +8] * fTemp0); //current Dc value; wird gleich wieder ueberschrieben/geupdated
                            
                        fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
                        fFFric[uTemp1*6 +0] = fFRef[uTemp1*14 +3] + (fFRef[uTemp1*14 +4] * fTemp0); //current static friction
                        fFEvent[uTemp1*16 +3] = fFFric[uTemp1*6 +0]; //use static friction also as current friction
                        
                        fTemp0 = ran0(&lSeed);      fTemp0 = fTemp0*2.0f -1.0f;
                        fFFric[uTemp1*6 +1] = fFRef[uTemp1*14 +5] + (fFRef[uTemp1*14 +6] * fTemp0); // current dynamic friction
                        //--------------------------------------------------------------------
                        fTemp0 =(fFFric[uTemp1*6 +0] - fFFric[uTemp1*6 +1]) *-1.0f*fFRef[uTemp1*14 +1]; //strength difference between static and dynamic strength; positive when weakening
                        fTemp1 = fTemp0/fFRef[uTemp1*14 +2];
                        uFEvent[uTemp1*5 +0] = (fTemp0 >= 0.0f)*2u + (fTemp0 < 0.0f)*3u; //gets a 2 if drop is not negative (in which case it would be strengthening i.e., stable == 3)
                        uFEvent[uTemp1*5 +0] = (fTemp1 > fFEvent[uTemp1*16 +7])*1u + (fTemp1 <= fFEvent[uTemp1*16 +7])*uFEvent[uTemp1*5 +0]; //give it a 1 if it is unstable (slip associated w/ change from static to dynamic is large enough to over come Dc)
                        //---------------------------------
                        fTemp0 = fFFric[uTemp1*6 +1]*-1.0f*fFRef[uTemp1*14 +1] + fFRef[uTemp1*14 +2]*fFEvent[uTemp1*16 +7]; //this is dynFric*normal stress (ie., dyn strength) minus  selfstiff*DcVal; the latter is therefore the stress change needed/associated with slip Dc; adding both to see what fric strength would at lease need to have Dc amount of slip
                        fTemp1 = fFFric[uTemp1*6 +0]*-1.0f*fFRef[uTemp1*14 +1]; //this is the static strength
                        fTemp2 = MAX(fTemp0, fTemp1)/(-1.0f*fFRef[uTemp1*14 +1]); //friction coefficient...
                        
                        fFFric[uTemp1*6 +2]  = fTemp2 - fOvershootFract*(fTemp2-fFFric[uTemp1*6 +1]); //this is arrest friction
                        //---------------------------------
                        fFEvent[uTemp1*16 +7] *= ((fTemp2 - fFFric[uTemp1*6 +2])/(fTemp2 - fFFric[uTemp1*6 +1])); //updated Dc value => now adjusted for the breakdown all the way to arrest stress
                        fFEvent[uTemp1*16 +10] = (fFFric[uTemp1*6 +1] - fFFric[uTemp1*6 +2])*-1.0f*fFRef[uTemp1*14 +1];//this is overshoot stress (difference between arrest (lower) and dynamic (higher) strength)
                        //--------------------------------------------------------------------
                }   }
                for (i = 0u; i < iFOFFSET[iRANK]; i++)
                {   fFRef[i*14 +13]  += sqrtf(fFEvent[i*16 +13]*fFEvent[i*16 +13] +fFEvent[i*16 +14]*fFEvent[i*16 +14]);
                    
                    uFEvent[i*5 +1]   = 0u;         uFEvent[i*5 +2]   = 0u;         uFEvent[i*5 +3]   = 0u;         uFEvent[i*5 +4]   = 0u;
                    fFEvent[i*16 +11] = 0.0f;       fFEvent[i*16 +12] = 0.0f;       fFEvent[i*16 +13] = 0.0f;       fFEvent[i*16 +14] = 0.0f;       fFEvent[i*16 +15] = 0.0f;
                   
                    fFEvent[i*16 +2] = fFRef[i*14 +1];
                    fFEvent[i*16 +3] = fFFric[i*6 +0];
                    
                    fTemp0 = sqrtf(fFEvent[i*16 +0]*fFEvent[i*16 +0] + fFEvent[i*16 +1]*fFEvent[i*16 +1]);
                    fTemp1 = fFEvent[i*16 +3]*-1.0f*fFEvent[i*16 +2];
                    
                    if (fTemp0 >= fTemp1) 
                    {   fFEvent[i*16 +3] = fTemp0/(-1.0f*fFEvent[i*16 +2]); //this gives the friction value, required to "explain" the excess stress
                        fFEvent[i*16 +3] = (fFEvent[i*16 +3] < INTERNALFRICTION)*fFEvent[i*16 +3] + (fFEvent[i*16 +3] >= INTERNALFRICTION)*INTERNALFRICTION;
                        
                        fTemp1 = fFEvent[i*16 +3]*-1.0f*fFEvent[i*16 +2]; //the new (combined) shear stress the element can hold
                        
                        fFEvent[i*16 +0] *=  MIN((fTemp1/fTemp0), 1.0f); //remove stresses exceeding the rock strength
                        fFEvent[i*16 +1] *=  MIN((fTemp1/fTemp0), 1.0f); //otherwise, stresses are NOT modified after leaving EQ
                    } 
                    fFFric[i*6 +4] = fFEvent[i*16 +3] - fFFric[i*6 +0];
                } 
            }// reset friction parameters and prepare for next event
            //----------------------------------------------------------------------------
            //----------------------------------------------------------------------------
            timerC = clock() - timerC;        time_CoSeis += (((double)timerC)/CLOCKS_PER_SEC);
            timerI = clock();
        }
    }
    //----------------------------------------------------------------------------
    MPI_Barrier( MPI_COMM_WORLD );
    //----------------------------------------------------------------------------
    //collect data and store post-event "state"
    unsigned int *uFLTemp   = calloc( iFOFFSET[iRANK],  sizeof *uFLTemp  );
    unsigned int *uFvTemp   = calloc( uFPNum,  sizeof *uFvTemp  );
    float        *fFLTemp0  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp0 );
    float        *fFvTemp0  = calloc( uFPNum,  sizeof *fFvTemp0 );
    float        *fFLTemp1  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp1 );
    float        *fFvTemp1  = calloc( uFPNum,  sizeof *fFvTemp1 );
    float        *fFLTemp2  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp2 );
    float        *fFvTemp2  = calloc( uFPNum,  sizeof *fFvTemp2 );
    float        *fFLTemp3  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp3 );
    float        *fFvTemp3  = calloc( uFPNum,  sizeof *fFvTemp3 );
    float        *fFLTemp4  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp4 );
    float        *fFvTemp4  = calloc( uFPNum,  sizeof *fFvTemp4 );
    float        *fFLTemp5  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp5 );
    float        *fFvTemp5  = calloc( uFPNum,  sizeof *fFvTemp5 );
    float        *fFLTemp6  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp6 );
    float        *fFvTemp6  = calloc( uFPNum,  sizeof *fFvTemp6 );
    float        *fFLTemp7  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp7 );
    float        *fFvTemp7  = calloc( uFPNum,  sizeof *fFvTemp7 );
    float        *fFLTemp8  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp8 );
    float        *fFvTemp8  = calloc( uFPNum,  sizeof *fFvTemp8 );
    float        *fFLTemp9  = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp9 );
    float        *fFvTemp9  = calloc( uFPNum,  sizeof *fFvTemp9 );
    float        *fFLTemp10 = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp10);
    float        *fFvTemp10 = calloc( uFPNum,  sizeof *fFvTemp10 );
    float        *fFLTemp11 = calloc( iFOFFSET[iRANK],  sizeof *fFLTemp11);
    float        *fFvTemp11 = calloc( uFPNum,  sizeof *fFvTemp11 );
    float        *fBLTemp0  = calloc( iBOFFSET[iRANK],  sizeof *fBLTemp0 );
    float        *fBvTemp0  = calloc( uBPNum,  sizeof *fBvTemp0 );
    float        *fBLTemp1  = calloc( iBOFFSET[iRANK],  sizeof *fBLTemp1 );
    float        *fBvTemp1  = calloc( uBPNum,  sizeof *fBvTemp1 );
    float        *fBLTemp2  = calloc( iBOFFSET[iRANK],  sizeof *fBLTemp2 );
    float        *fBvTemp2  = calloc( uBPNum,  sizeof *fBvTemp2 );
    float        *fBLTemp3  = calloc( iBOFFSET[iRANK],  sizeof *fBLTemp3 );
    float        *fBvTemp3  = calloc( uBPNum,  sizeof *fBvTemp3 );
    float        *fBLTemp4  = calloc( iBOFFSET[iRANK],  sizeof *fBLTemp4 );
    float        *fBvTemp4  = calloc( uBPNum,  sizeof *fBvTemp4 );
    //--------------------------------
    for (i = 0u; i < iFOFFSET[iRANK]; i++)      
    {   uFLTemp[i]  = uFEvent[i*5  +0];
        fFLTemp0[i] = fFEvent[i*16 +0];      fFLTemp1[i] = fFEvent[i*16 +1];        fFLTemp2[i] = fFEvent[i*16 +2];      fFLTemp3[i] = fFEvent[i*16 +3];
        fFLTemp4[i] = fFEvent[i*16 +7];      fFLTemp5[i] = fFEvent[i*16 +10];       fFLTemp6[i] = fFFric[i*6 +0];        fFLTemp7[i] = fFFric[i*6 +1];
        fFLTemp8[i] = fFFric[i*6 +2];        fFLTemp9[i] = fFFric[i*6 +4];          fFLTemp10[i]= fFRef[i*14 +12];       fFLTemp11[i]= fFRef[i*14 +13];
    }  
    for (i = 0u; i < iBOFFSET[iRANK]; i++)      
    {   fBLTemp0[i] = fBEvent[i*9 +0];       fBLTemp1[i] = fBEvent[i*9 +1];         fBLTemp2[i] = fBEvent[i*9 +2];       fBLTemp3[i] = fBEvent[i*9 +7];
        fBLTemp4[i] = fBEvent[i*9 +8];
    }
    MPI_Gatherv(uFLTemp,  iFOFFSET[iRANK], MPI_UNSIGNED, uFvTemp,  iFOFFSET, iFSTART, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp0, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp0, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp1, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp1, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp2, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp2, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp3, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp3, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp4, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp4, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp5, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp5, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp6, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp6, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp7, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp7, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp8, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp8, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp9, iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp9, iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp10,iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp10,iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fFLTemp11,iFOFFSET[iRANK], MPI_FLOAT,    fFvTemp11,iFOFFSET, iFSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fBLTemp0, iBOFFSET[iRANK], MPI_FLOAT,    fBvTemp0, iBOFFSET, iBSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fBLTemp1, iBOFFSET[iRANK], MPI_FLOAT,    fBvTemp1, iBOFFSET, iBSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fBLTemp2, iBOFFSET[iRANK], MPI_FLOAT,    fBvTemp2, iBOFFSET, iBSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fBLTemp3, iBOFFSET[iRANK], MPI_FLOAT,    fBvTemp3, iBOFFSET, iBSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(fBLTemp4, iBOFFSET[iRANK], MPI_FLOAT,    fBvTemp4, iBOFFSET, iBSTART, MPI_FLOAT,    0, MPI_COMM_WORLD);
    //--------------------------------
    if (iRANK == 0)
    {   //--------------------------------
        fpPrePost = fopen(cPrevStateName,"wb");
        //--------------------------------
        fwrite(&dCurrTime,  sizeof(double),       1,    fpPrePost);
        fwrite(&uEQcntr,    sizeof(unsigned int), 1,    fpPrePost);
        fwrite(&uCatType,   sizeof(unsigned int), 1,    fpPrePost);
        fwrite(&fPSeisTime, sizeof(float),        1,    fpPrePost);

        fwrite(uFvTemp,  sizeof(unsigned int), uFPNum, fpPrePost);              fwrite(fFvTemp0, sizeof(float),        uFPNum, fpPrePost);
        fwrite(fFvTemp1, sizeof(float),        uFPNum, fpPrePost);              fwrite(fFvTemp2, sizeof(float),        uFPNum, fpPrePost);
        fwrite(fFvTemp3, sizeof(float),        uFPNum, fpPrePost);              fwrite(fFvTemp4, sizeof(float),        uFPNum, fpPrePost);
        fwrite(fFvTemp5, sizeof(float),        uFPNum, fpPrePost);              fwrite(fFvTemp6, sizeof(float),        uFPNum, fpPrePost);
        fwrite(fFvTemp7, sizeof(float),        uFPNum, fpPrePost);              fwrite(fFvTemp8, sizeof(float),        uFPNum, fpPrePost);
        fwrite(fFvTemp9, sizeof(float),        uFPNum, fpPrePost);              fwrite(fFvTemp10,sizeof(float),        uFPNum, fpPrePost);
        fwrite(fFvTemp11,sizeof(float),        uFPNum, fpPrePost);
        fwrite(fBvTemp0, sizeof(float),        uBPNum, fpPrePost);              fwrite(fBvTemp1, sizeof(float),        uBPNum, fpPrePost);
        fwrite(fBvTemp2, sizeof(float),        uBPNum, fpPrePost);              fwrite(fBvTemp3, sizeof(float),        uBPNum, fpPrePost);
        fwrite(fBvTemp4, sizeof(float),        uBPNum, fpPrePost);
        //--------------------------------
        fclose(fpPrePost);
    }
    MPI_Barrier( MPI_COMM_WORLD );
    //----------------------------------------------------------------------------
    //----------------------------------------------------------------------------
    timer0 = clock() - timer0;        time_taken  = ((double)timer0)/CLOCKS_PER_SEC;
    MPI_Allreduce(MPI_IN_PLACE, &time_taken, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
    time_taken /=(double)iSIZE;
    //----------------------------------------------------------------------------
    free(fK_FFvalstk); free(fK_FFvaldip); free(fK_FFvalnrm);
    free(fK_FBvalstk); free(fK_FBvaldip);
    free(fK_BBvalstk); free(fK_BBvaldip); free(fK_BBvalnrm);
    free(fK_BFvalstk); free(fK_BFvaldip); free(fK_BFvalnrm);
    //----------------------------------------------------------------------------
    if (iRANK == 0)
    {   //-----------------------------------------------------------------
        dMeanRecurTime /= (double)uEQcntr;
        fprintf(stdout,"\nTotal RunTime for EQcycle (minutes): %6.4f       and mean inter-event time (days): %3.2lf earthquakes per hour: %3.2lf\n",time_taken/60.0f, dMeanRecurTime*365.25, (double)uEQcntr/((double)(time_taken/3600.0f)) );
        fprintf(stdout,"Time spent in interseismic: %4.2lf   Time spent in coseismic: %4.2lf\n",time_InterSeis/60.0f, time_CoSeis/60.0f);
        fprintf(stdout,"PostSeis iterations per event %u\n", (uPSeisSteps/uEQcntr) );
        //-----------------------------------------------------------------
        float fSecsPerYear = 31622400.0f,   fNeighborDist = 0.0f,   fP2P1[3],   fP1P3[3];
        //---------------------------------
        unsigned int *uTempF  = calloc(uFPNum,   sizeof *uTempF  );
        unsigned int *uFtemp  = calloc(5*uFPNum, sizeof *uFtemp  );
        float *fTempF         = calloc(uFPNum,   sizeof *fTempF  );
        float *fTempFV        = calloc(uFVNum,   sizeof *fTempFV );
        float *fFtemp         = calloc(3*uFPNum, sizeof *fFtemp  );
        float *fFVtemp        = calloc(4*uFVNum, sizeof *fFVtemp );
        float *fF_Area        = calloc(uFPNum,   sizeof *fF_Area );
        //---------------------------------
        fseek(fp_CATALOG, (4*sizeof(unsigned int)+2*sizeof(float)), SEEK_SET);
        //------------------------------------------------------------------
        if (fread(uTempF,    sizeof(unsigned int),    uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   uFtemp[i*5 +0]  = uTempF[i];   }
        if (fread(uTempF,    sizeof(unsigned int),    uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   uFtemp[i*5 +1]  = uTempF[i];   }
        if (fread(uTempF,    sizeof(unsigned int),    uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   uFtemp[i*5 +2]  = uTempF[i];   }
        if (fread(uTempF,    sizeof(unsigned int),    uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   uFtemp[i*5 +3]  = uTempF[i];   }
        if (fread(uTempF,    sizeof(unsigned int),    uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   uFtemp[i*5 +4]  = uTempF[i];   }
        if (fread(fTempFV,   sizeof(float),           uFVNum, fp_CATALOG) != uFVNum)       {   exit(10);   }           for (i = 0u; i < uFVNum; i++)   {   fFVtemp[i*4 +0] = fTempFV[i];  }
        if (fread(fTempFV,   sizeof(float),           uFVNum, fp_CATALOG) != uFVNum)       {   exit(10);   }           for (i = 0u; i < uFVNum; i++)   {   fFVtemp[i*4 +1] = fTempFV[i];  }
        if (fread(fTempFV,   sizeof(float),           uFVNum, fp_CATALOG) != uFVNum)       {   exit(10);   }           for (i = 0u; i < uFVNum; i++)   {   fFVtemp[i*4 +2] = fTempFV[i];  }
        if (fread(fTempFV,   sizeof(float),           uFVNum, fp_CATALOG) != uFVNum)       {   exit(10);   }           for (i = 0u; i < uFVNum; i++)   {   fFVtemp[i*4 +3] = fTempFV[i];  }
        if (fread(fTempF,    sizeof(float),           uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   fFtemp[i*3 +0]  = fTempF[i];   }
        if (fread(fTempF,    sizeof(float),           uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   fFtemp[i*3 +1]  = fTempF[i];   }
        if (fread(fTempF,    sizeof(float),           uFPNum, fp_CATALOG) != uFPNum)       {   exit(10);   }           for (i = 0u; i < uFPNum; i++)   {   fFtemp[i*3 +2]  = fTempF[i];   }
        //------------------------------------------------------------------
        if ((fp_CATALOGCUT = fopen(cOutputNameCUT,"wb")) == NULL)         {   printf("Error -cant open %s  ParsedCatalogFile...\n",cOutputNameCUT);      exit(110);     }
        //-------------------------------------
        fwrite(&uEQcntr,     sizeof(unsigned int), 1, fp_CATALOGCUT);
        fwrite(&uCatType,    sizeof(unsigned int), 1, fp_CATALOGCUT);
        fwrite(&fModPara[2], sizeof(float),        1, fp_CATALOGCUT);
        fwrite(&fModPara[7], sizeof(float),        1, fp_CATALOGCUT);
        //-------------------------------------
        if (uCatType == 3u)
        {   //-------------------------------------
            fwrite(&uFPNum,      sizeof(unsigned int), 1, fp_CATALOGCUT);
            fwrite(&uFVNum,      sizeof(unsigned int), 1, fp_CATALOGCUT);
            //-------------------------------------
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uFtemp[i*5 +0];   }     fwrite(uTempF,  sizeof(unsigned int), uFPNum, fp_CATALOGCUT);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uFtemp[i*5 +1];   }     fwrite(uTempF,  sizeof(unsigned int), uFPNum, fp_CATALOGCUT);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uFtemp[i*5 +2];   }     fwrite(uTempF,  sizeof(unsigned int), uFPNum, fp_CATALOGCUT);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uFtemp[i*5 +3];   }     fwrite(uTempF,  sizeof(unsigned int), uFPNum, fp_CATALOGCUT);
            for (i = 0u; i < uFPNum; i++)   {   uTempF[i] = uFtemp[i*5 +4];   }     fwrite(uTempF,  sizeof(unsigned int), uFPNum, fp_CATALOGCUT);

            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFVtemp[i*4 +0];  }     fwrite(fTempFV, sizeof(float),        uFVNum, fp_CATALOGCUT);
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFVtemp[i*4 +1];  }     fwrite(fTempFV, sizeof(float),        uFVNum, fp_CATALOGCUT);
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFVtemp[i*4 +2];  }     fwrite(fTempFV, sizeof(float),        uFVNum, fp_CATALOGCUT);
            for (i = 0u; i < uFVNum; i++)   {   fTempFV[i]= fFVtemp[i*4 +3];  }     fwrite(fTempFV, sizeof(float),        uFVNum, fp_CATALOGCUT);

            for (i = 0u; i < uFPNum; i++)   {   fTempF[i] = fFtemp[i*3 +0];   }     fwrite(fTempF,  sizeof(float),        uFPNum, fp_CATALOGCUT);
            for (i = 0u; i < uFPNum; i++)   {   fTempF[i] = fFtemp[i*3 +1];   }     fwrite(fTempF,  sizeof(float),        uFPNum, fp_CATALOGCUT);
            for (i = 0u; i < uFPNum; i++)   {   fTempF[i] = fFtemp[i*3 +2];   }     fwrite(fTempF,  sizeof(float),        uFPNum, fp_CATALOGCUT);
        }
        //------------------------------------------------------------------
        {   unsigned int uVert1,   uVert2,   uVert3;
            float fVert1[3],   fVert2[3],   fVert3[3],   fDist[3];
            //-------------------------------------
            for (i = 0u; i < uFPNum; i++)
            {   uVert1    = uFtemp[i*5 +0];            uVert2    = uFtemp[i*5 +1];            uVert3    = uFtemp[i*5 +2];
                fVert1[0] = fFVtemp[uVert1*4 +0];      fVert1[1] = fFVtemp[uVert1*4 +1];      fVert1[2] = fFVtemp[uVert1*4 +2];
                fVert2[0] = fFVtemp[uVert2*4 +0];      fVert2[1] = fFVtemp[uVert2*4 +1];      fVert2[2] = fFVtemp[uVert2*4 +2];
                fVert3[0] = fFVtemp[uVert3*4 +0];      fVert3[1] = fFVtemp[uVert3*4 +1];      fVert3[2] = fFVtemp[uVert3*4 +2];
                fDist[0]  = sqrtf((fVert2[0] -fVert1[0])*(fVert2[0] -fVert1[0]) + (fVert2[1] -fVert1[1])*(fVert2[1] -fVert1[1]) + (fVert2[2] -fVert1[2])*(fVert2[2] -fVert1[2]));
                fDist[1]  = sqrtf((fVert3[0] -fVert2[0])*(fVert3[0] -fVert2[0]) + (fVert3[1] -fVert2[1])*(fVert3[1] -fVert2[1]) + (fVert3[2] -fVert2[2])*(fVert3[2] -fVert2[2]));
                fDist[2]  = sqrtf((fVert1[0] -fVert3[0])*(fVert1[0] -fVert3[0]) + (fVert1[1] -fVert3[1])*(fVert1[1] -fVert3[1]) + (fVert1[2] -fVert3[2])*(fVert1[2] -fVert3[2]));
                //-------------------
                fNeighborDist += ((fDist[0] + fDist[1] +fDist[2])/3.0f); 
                //-------------------
                fP2P1[0]   = fVert2[0] - fVert1[0];      fP2P1[1]  = fVert2[1] - fVert1[1];      fP2P1[2]  = fVert2[2] - fVert1[2];
                fP1P3[0]   = fVert1[0] - fVert3[0];      fP1P3[1]  = fVert1[1] - fVert3[1];      fP1P3[2]  = fVert1[2] - fVert3[2];
                fF_Area[i] =  0.5f*sqrtf( (fP2P1[1]*fP1P3[2]-fP2P1[2]*fP1P3[1])*(fP2P1[1]*fP1P3[2]-fP2P1[2]*fP1P3[1]) + (fP2P1[2]*fP1P3[0]-fP2P1[0]*fP1P3[2])*(fP2P1[2]*fP1P3[0]-fP2P1[0]*fP1P3[2]) + (fP2P1[0]*fP1P3[1]-fP2P1[1]*fP1P3[0])*(fP2P1[0]*fP1P3[1]-fP2P1[1]*fP1P3[0]) );
        }   }
        fNeighborDist *= (NeighborFactor/(float)uFPNum);
        fprintf(stdout,"max. neighbor distance: %f\n",fNeighborDist);
        //------------------------------------------------------------------
        unsigned int **uElNeighbors = NULL;
        //-------------------------------------
        uElNeighbors  = calloc(uFPNum, sizeof *uElNeighbors );
        //-------------------------------------
        for (i = 0; i < uFPNum; i++)
        {   uElNeighbors[i] = calloc( uFPNum, sizeof *uElNeighbors[i] );
            for (j = 0; j < uFPNum; j++)
            {   //-------------------------------------
                fTemp0 = (i == j)*(2.0f*fNeighborDist)  +  (i != j)*sqrtf((fFtemp[i*3 +0]-fFtemp[j*3 +0])*(fFtemp[i*3 +0]-fFtemp[j*3 +0]) + (fFtemp[i*3 +1]-fFtemp[j*3 +1])*(fFtemp[i*3 +1]-fFtemp[j*3 +1]) + (fFtemp[i*3 +2]-fFtemp[j*3 +2])*(fFtemp[i*3 +2]-fFtemp[j*3 +2]));
                //-------------------------------------
                uElNeighbors[i][0]                    = (fTemp0 >= fNeighborDist)*uElNeighbors[i][0]                     +  (fTemp0 < fNeighborDist)*(uElNeighbors[i][0] +1u);
                uElNeighbors[i][uElNeighbors[i][0]] = (fTemp0 >= fNeighborDist)*uElNeighbors[i][uElNeighbors[i][0]]  +  (fTemp0 < fNeighborDist)*j;
            }
            uElNeighbors[i]  = realloc(uElNeighbors[i], (uElNeighbors[i][0]+1) *sizeof *uElNeighbors[i] );
        }
        //------------------------------------------------------------------
        //------------------------------------------------------------------
        double dEQtime, dEQtimeMod;
        float fMagn,   fHypoLoc[3],   fMeanVals[6];
        unsigned int uMRFlgth,   uElNum;
        //-------------------------------
        float *fMRF           = calloc(MAXMOMRATEFUNCLENGTH, sizeof *fMRF );
        unsigned int *uElID   = calloc(uFPNum,       sizeof *uElID );
        //-----------------------
        unsigned int *uStrtT  = calloc(uFPNum,       sizeof *uStrtT );
        unsigned int *uStabT  = calloc(uFPNum,       sizeof *uStabT );
        float *fDtau          = calloc(uFPNum,       sizeof *fDtau );
        float *fSlip_H        = calloc(uFPNum,       sizeof *fSlip_H );
        float *fSlip_V        = calloc(uFPNum,       sizeof *fSlip_V );
        //-----------------------
        unsigned int *ugStrtT = calloc(uFPNum,       sizeof *ugStrtT );
        unsigned int *ugStabT = calloc(uFPNum,       sizeof *ugStabT );
        float *fgDtau         = calloc(uFPNum,       sizeof *fgDtau );
        float *fgSlip_H       = calloc(uFPNum,       sizeof *fgSlip_H );
        float *fgSlip_V       = calloc(uFPNum,       sizeof *fgSlip_V );
        //-----------------------
        unsigned int *ug2Write = calloc(uFPNum,      sizeof *ug2Write );
        float *fg2Write        = calloc(uFPNum,      sizeof *fg2Write );
        //-----------------------------------------------
        unsigned int uSelElem,   uTstElem,   uEl2TestCnt,   uNx2TestCnt,   uSubEQElCnt,   uEQcntrNEW = 0u;
        unsigned int uMinStartTime,   uMinStartTPos,   uElLeft;
        unsigned int *uElVisited   = calloc(uFPNum, sizeof *uElVisited );
        unsigned int *uElSlipped   = calloc(uFPNum, sizeof *uElSlipped );
        unsigned int *uEl2Test     = calloc(uFPNum, sizeof *uEl2Test );
        unsigned int *uNx2Test     = calloc(uFPNum, sizeof *uNx2Test );
        unsigned int *uSubEQEl     = calloc(uFPNum, sizeof *uSubEQEl );
        //------------------------------------------------------------------
        for (i = 0; i < uEQcntr; i++)
        {   if (fread(&dEQtime,    sizeof(double),          1,     fp_CATALOG) != 1)              {   exit(10);   }
            if (fread(&fMagn,      sizeof(float),           1,     fp_CATALOG) != 1)              {   exit(10);   }
            if (fread(fHypoLoc,    sizeof(float),           3,     fp_CATALOG) != 3)              {   exit(10);   }
            if (fread(fMeanVals,   sizeof(float),           6,     fp_CATALOG) != 6)              {   exit(10);   }
            if (fread(&uMRFlgth,   sizeof(unsigned int),    1,     fp_CATALOG) != 1)              {   exit(10);   }
            if (fread(fMRF,        sizeof(float),        uMRFlgth, fp_CATALOG) != uMRFlgth)       {   exit(10);   }
            if (fread(&uElNum,     sizeof(unsigned int),    1,     fp_CATALOG) != 1)              {   exit(10);   }
            if (fread(uElID,       sizeof(unsigned int), uElNum,   fp_CATALOG) != uElNum)         {   exit(10);   }
            if (fread(uStrtT,      sizeof(unsigned int), uElNum,   fp_CATALOG) != uElNum)         {   exit(10);   }
            if (fread(fDtau,       sizeof(float),        uElNum,   fp_CATALOG) != uElNum)         {   exit(10);   }
            if (fread(fSlip_H,     sizeof(float),        uElNum,   fp_CATALOG) != uElNum)         {   exit(10);   }
            if (fread(fSlip_V,     sizeof(float),        uElNum,   fp_CATALOG) != uElNum)         {   exit(10);   }
            if (fread(uStabT,      sizeof(unsigned int), uElNum,   fp_CATALOG) != uElNum)         {   exit(10);   }
            //------------------------------------------------------------------
            //split the event
            memset(uElSlipped, 0, uFPNum*sizeof(unsigned int) );
            memset(uElVisited, 0, uFPNum*sizeof(unsigned int) );
            //-------------------------------
            for (j = 0; j < uElNum; j++)
            {   uElSlipped[uElID[j]] = 1u;                  ugStrtT[uElID[j]]    = uStrtT[j];
                ugStabT[uElID[j]]    = uStabT[j];           fgDtau[uElID[j]]     = fDtau[j];
                fgSlip_H[uElID[j]]   = fSlip_H[j];          fgSlip_V[uElID[j]]   = fSlip_V[j];
            }
            //-----------------
            uElLeft = uElNum;
            //-----------------
            while (uElLeft > 0u)
            {   uMinStartTime = UINT_MAX;
                uMinStartTPos = 0;
                uSubEQElCnt   = 0;
                uEQcntrNEW   += 1u;
                //-----------------
                for (j = 0; j < uElNum; j++)      {   if ((uStrtT[j] < uMinStartTime) && (uElVisited[uElID[j]]) == 0u)        {       uMinStartTime = uStrtT[j];        uMinStartTPos = uElID[j];      }        }
                //-----------------
                dEQtimeMod   = dEQtime + (double)(((float)uMinStartTime *fModPara[7])/fSecsPerYear);
                uEl2Test[0]  = uMinStartTPos;                     uEl2TestCnt    = 1u;
                uSubEQEl[0]  = uMinStartTPos;                     uSubEQElCnt    = 1u;
                uElLeft     -= 1u;
                uElVisited[uMinStartTPos] = uEQcntrNEW;
                //---------------------------------------------
                while (uEl2TestCnt > 0u)
                {   //-------------------------------
                    uNx2TestCnt = 0u;
                    for (j = 0; j < uEl2TestCnt; j++)
                    {   //-------------------------------
                        uSelElem = uEl2Test[j];
                        for (k = 1; k <= uElNeighbors[uSelElem][0]; k++)
                        {   uTstElem = uElNeighbors[uSelElem][k];
                            if ((uElSlipped[uTstElem] == 1u) && (uElVisited[uTstElem] == 0u))
                            {   uNx2Test[uNx2TestCnt] = uTstElem;           uNx2TestCnt += 1u;
                                uSubEQEl[uSubEQElCnt] = uTstElem;           uSubEQElCnt += 1u;
                                uElVisited[uTstElem]  = uEQcntrNEW;
                                uElLeft              -= 1u;
                    }   }   }
                    //-------------------------------
                    if (uNx2TestCnt > 0u)
                    {   memcpy(uEl2Test, uNx2Test,     uNx2TestCnt* sizeof(unsigned int) );
                        uEl2TestCnt = uNx2TestCnt;
                    }
                    else    {   break;      }
                }
                //---------------------------------------------
                memset(fMeanVals, 0, 6*sizeof(float) );
                fTemp2      = 0.0f;
                fHypoLoc[0] = fFtemp[uSubEQEl[0]*3 +0];            fHypoLoc[1] = fFtemp[uSubEQEl[0]*3 +1];            fHypoLoc[2] = fFtemp[uSubEQEl[0]*3 +2];
                //-----------------------
                for (j = 0; j < uSubEQElCnt; j++)
                {   uTemp0        = uSubEQEl[j]; //global position of next element to process
                    fTemp0        = sqrtf( fgSlip_H[uTemp0]*fgSlip_H[uTemp0] + fgSlip_V[uTemp0]*fgSlip_V[uTemp0] );
                    fTemp1        = fTemp0*fF_Area[uTemp0];
                    fTemp2       += fTemp1;
                    //-----------------------
                    fMeanVals[0] += (fFtemp[uTemp0*3 +0] *fTemp1);
                    fMeanVals[1] += (fFtemp[uTemp0*3 +1] *fTemp1);
                    fMeanVals[2] += (fFtemp[uTemp0*3 +2] *fTemp1);
                    fMeanVals[3] += fF_Area[uTemp0];
                    fMeanVals[4] += fTemp0;
                    fMeanVals[5] += fgDtau[uTemp0];
                }
                //-----------------------
                fMeanVals[0] /= fTemp2;                 fMeanVals[1] /= fTemp2;             fMeanVals[2] /= fTemp2;
                fMeanVals[4] /= (float)uSubEQElCnt;     fMeanVals[5] /= (float)uSubEQElCnt;
                fMagn = (log10f(fTemp2*fModPara[2]) -9.1f)/1.5f;
                //-----------------------
                fwrite(&dEQtimeMod, sizeof(double),       1,   fp_CATALOGCUT);
                fwrite(&fMagn,      sizeof(float),        1,   fp_CATALOGCUT);
                fwrite(&fHypoLoc,   sizeof(float),        3,   fp_CATALOGCUT);
                fwrite(&fMeanVals,  sizeof(float),        6,   fp_CATALOGCUT);
                if (uCatType > 1u)
                {   fwrite(&uMRFlgth,   sizeof(unsigned int), 1,   fp_CATALOGCUT);
                    fwrite(fMRF,        sizeof(float),   uMRFlgth, fp_CATALOGCUT);
                }
                if (uCatType > 2u)
                {   fwrite(&uSubEQElCnt,sizeof(unsigned int), 1,   fp_CATALOGCUT);
                    for (j = 0; j < uSubEQElCnt; j++)   {   ug2Write[j] = uSubEQEl[j];            }      fwrite(ug2Write,  sizeof(unsigned int), uSubEQElCnt, fp_CATALOGCUT);
                    for (j = 0; j < uSubEQElCnt; j++)   {   ug2Write[j] = ugStrtT[uSubEQEl[j]];   }      fwrite(ug2Write,  sizeof(unsigned int), uSubEQElCnt, fp_CATALOGCUT);
                    for (j = 0; j < uSubEQElCnt; j++)   {   fg2Write[j] = fgDtau[uSubEQEl[j]];    }      fwrite(fg2Write,  sizeof(float),        uSubEQElCnt, fp_CATALOGCUT);
                    for (j = 0; j < uSubEQElCnt; j++)   {   fg2Write[j] = fgSlip_H[uSubEQEl[j]];  }      fwrite(fg2Write,  sizeof(float),        uSubEQElCnt, fp_CATALOGCUT);
                    for (j = 0; j < uSubEQElCnt; j++)   {   fg2Write[j] = fgSlip_V[uSubEQEl[j]];  }      fwrite(fg2Write,  sizeof(float),        uSubEQElCnt, fp_CATALOGCUT);
                    for (j = 0; j < uSubEQElCnt; j++)   {   ug2Write[j] = ugStabT[uSubEQEl[j]];   }      fwrite(ug2Write,  sizeof(unsigned int), uSubEQElCnt, fp_CATALOGCUT);
        }   }   }
        //------------------------------------------------------------------
        fseek(fp_CATALOGCUT, 0, SEEK_SET);
        fwrite(&uEQcntrNEW, sizeof(unsigned int), 1, fp_CATALOGCUT);
        fprintf(stdout,"EQ number after parsing:  %u\n",uEQcntrNEW);
        //-------------------------------------------------
        fclose(fp_CATALOG);
        fclose(fp_CATALOGCUT);
        //-------------------------------------------------
    }
    //----------------------------------------------------------------------------
    //----------------------------------------------------------------------------
    MPI_Barrier( MPI_COMM_WORLD );
    //MPI_Finalize();
    //MPI_Finalize();
    return 0;
 }
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
 float ran0(long *idum)
{   long k;
    float ans;
    
    *idum ^= MASK;
    k = (*idum)/IQ;
    *idum = IA*(*idum-k*IQ) -IR*k;
    if (*idum < 0) *idum += IM;
    ans = AM*(*idum);
    *idum ^= MASK;

    return ans;
}
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
float FloatPow(float Base, unsigned int Exp)
{   unsigned int i;
    float p = 1.0f;
    for (i = 1u; i <= Exp; i++)       {   p *= Base;   }
    return p;
}
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------