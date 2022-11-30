#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <time.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_statistics_double.h>
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
struct MDstruct
{   int         iRANK,              		iSIZE;
    int       	iF_BASEelem,        		iF_ADDelem,				iB_BASEelem,       			iB_ADDelem;  
    int       	*ivF_OFFSET,        		*ivF_START,				*ivB_OFFSET,        		*ivB_START;
	int         *ivF_ModOFFs,				*ivB_ModOFFs;

    int       	iRunNum,            		iSeedStart,				iUseProp,           		iUsePSeis;
    int       	iFPNum,             		iBPNum,					iFVNum,             		iBVNum; 
    int       	iFSegNum,          		 	iBSegNum,				iAlsoBoundForPostSeis;
    int       	iMaxIterat,      			iEQcntr,      			iMaxMRFlgth,				iMaxSTFlgth;				
    int      	iChgBtwEQs,					iGlobTTmax,				iWritePos,					iUseVpVs;
   
    float     	fLegLgth,          			fUnitSlip,				fDeltT;
	float 		fFltLegs,					fBndLegs,               fMinMag4Prop,				fCutStFrac;

	float       fAftrSlipTime,				fISeisStep,         	fRecLgth,					fHealFact;
	float       fDeepRelaxTime,				fPSeis_Step;

    float       fMedDense,     				fAddNrmStrs,			fVp,						fVs;
    float       fPoisson,           		fLambda,        		fShearMod,					fMeanStiffness;					
    float       fISeisTStp,         		fVpVsRatio,				fDcfrac4Threshold;		
    long long   lTimeYears,					lRecLgth;			
    
    char        cInputName[512];
};
//------------------------------------------------------------------
struct TRstruct
{	int     	*ivFL_StabT,				*ivFL_Activated,		*ivFL_Ptch_t0,				*ivFL_FricLaw;	

    float    	*fvFL_RefNrmStrs,			*fvFL_Area;
	float		*fvFL_SelfStiffStk,			*fvFL_SelfStiffDip,		*fvFL_MeanSelfStiff;
	float		*fvBL_SelfStiffStk,			*fvBL_SelfStiffDip,		*fvBL_SelfStiffOpn;

    float    	*fvFL_RefStaFric,           *fvFL_RefDynFric,    	*fvFL_RefDcVal;
    float    	*fvFL_RefStaFric_vari,      *fvFL_RefDynFric_vari,	*fvFL_RefDcVal_vari;
    float    	*fvFL_StaFric,              *fvFL_DynFric,			*fvFL_CurFric;              
	float       *fvFL_B4_Fric,			    *fvFL_TempRefFric,		*fvFL_CurDcVal;

    float    	*fvFL_PSeis_T0_F,          	*fvFL_AccumSlp,			*fvFL_CutStress;
	float       *fvBL_PSeis_T0_S,			*fvBL_PSeis_T0_N;
	float    	*fvFL_SlipRate_temp,      	*fvFL_SlipRake_temp;
	 
	int      	*ivFG_SegID_temp,			*ivFG_FltID_temp,		*ivFG_Flagged_temp;
    int    		*ivFG_V1_temp,              *ivFG_V2_temp,        	*ivFG_V3_temp;
    float       *fvFG_StressRatetemp,		*fvFG_SlipRatetemp,     *fvFG_Raketemp;
    float       *fvFG_MaxTransient;
	
    int    		*ivBG_V1_temp,              *ivBG_V2_temp,       	*ivBG_V3_temp,				 *ivBG_SegID_temp;
    float    	*fvFG_CentE_temp,          	*fvFG_CentN_temp,    	*fvFG_CentZ_temp;
    float    	*fvBG_CentE_temp,           *fvBG_CentN_temp,    	*fvBG_CentZ_temp;
    float       *fvFL_StaFricMod_temp, 		*fvFL_DynFricMod_temp,	*fvFL_NrmStrsMod_temp,	     *fvFL_DcMod_temp;
    
	gsl_matrix_int	    *imFGL_TTP,         *imFGL_NextP;
	gsl_matrix_int    	*imFGL_TTS,			*imFGL_NextS; 
    gsl_matrix_float    *fmFGL_SrcRcvH,    	*fmFGL_SrcRcvV,      	*fmFGL_SrcRcvN;    

	gsl_vector_float	*fvFL_StrsRateStk, 	*fvFL_StrsRateDip;
	gsl_vector_float	*fvFL_CurStrsH,		*fvFL_CurStrsV,			*fvFL_CurStrsN;
	gsl_vector_float    *fvFL_B4_StrsH,		*fvFL_B4_StrsV,		    *fvFL_B4_StrsN;    
	gsl_vector_float	*fvBL_CurStrsH,		*fvBL_CurStrsV,			*fvBL_CurStrsN;
};
//------------------------------------------------------------------
struct VTstruct
{	float   	*fvFG_VlX_temp,             *fvFG_VlY_temp;
	float  		*fvFG_PosE_temp,          	*fvFG_PosN_temp,      	*fvFG_PosZ_temp, 			*fvFG_Hght_temp;
	float   	*fvBG_PosE_temp,           	*fvBG_PosN_temp,      	*fvBG_PosZ_temp;
};
//------------------------------------------------------------------
struct Kstruct
{	gsl_matrix_float 	*FFr_SS,			*FFr_SD,		*FFr_SO; //first index is "receiver"; second is "source"; first is long; second is short
	gsl_matrix_float 	*FFr_DS,			*FFr_DD,		*FFr_DO; //this means: FB_DS => Fault is receiver, Boundary the source; Dipslip causing strike_stress
	gsl_matrix_float 	*FFr_OS,			*FFr_OD,		*FFr_OO;

	gsl_matrix_float 	*FFs_SS,			*FFs_SD,		*FFs_SO; //first index is "source"; second is "receiver"; first is long; second is short
	gsl_matrix_float 	*FFs_DS,			*FFs_DD,		*FFs_DO; //this means: FB_DS => Fault is receiver, Boundary the source; Dipslip causing strike_stress

	gsl_matrix_float 	*FB_SS,				*FB_SD,			*FB_SO;
	gsl_matrix_float 	*FB_DS,				*FB_DD,			*FB_DO;

	gsl_matrix_float 	*BF_SS,				*BF_SD;		
	gsl_matrix_float 	*BF_DS,				*BF_DD;			
	gsl_matrix_float 	*BF_OS,				*BF_OD;

	gsl_matrix_float 	*BB_SS,				*BB_SD,			*BB_SO;
	gsl_matrix_float 	*BB_DS,				*BB_DD,			*BB_DO;
	gsl_matrix_float 	*BB_OS,				*BB_OD,			*BB_OO;
};
//------------------------------------------------------------------
struct EQstruct
{	int 	iStillOn,			iEndCntr,			iActFPNum;
	int 	iCmbFPNum,			iMRFLgth,			iTotlRuptT;
	
	float	fMaxSlip,			fMaxDTau;
	
	int 	*ivR_WrtStrtPos,	*ivL_ActPtchID,		*ivL_t0ofPtch,		*ivL_StabType;
	float	*fvL_PtchSlpH,		*fvL_PtchSlpV,		*fvL_PtchDTau,		*fvM_MRFvals;

	gsl_vector_float			*fvL_EQslipH,		*fvL_EQslipV;
	gsl_matrix_float			*fmFGL_STF_H,		*fmFGL_STF_V;
};
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
void InitializeVariables(struct MDstruct *MD, struct TRstruct *TR, struct VTstruct *VT, struct EQstruct *EQ,  struct Kstruct *K, int iPlot2Screen, char **argv);
void           LoadInput(struct MDstruct *MD, struct TRstruct *TR, struct VTstruct *VT);
void     DefineMoreParas(struct MDstruct *MD, struct TRstruct *TR, struct VTstruct *VT, int iPlot2Screen);
void      Build_K_Matrix(struct MDstruct *MD, struct TRstruct *TR, struct VTstruct *VT, struct  Kstruct *K);
float GetUpdatedFriction(int StabType, float B4Fric, float RefFric, float CurFric, float DynFric, int FricLaw, float CurD_c, float AccumSlip, float PrevSlip, float HealingFact);
float          fMaxofTwo(float fTemp0, float fTemp1);
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
int main(int argc, char **argv)
{   if ( (argc  > 3 ) || (argc < 2) )			{   			fprintf(stdout,"Input Error\n Please start the code in the following way:\n\n mpirun -np 4 ./MCQsim RunParaFile.txt -optionally adding more file name here");					}
    //------------------------------------------------------------------
    struct 	MDstruct MD; //initializing the structures
    struct 	TRstruct TR;
    struct 	VTstruct VT;
    struct 	Kstruct  K;
	struct 	EQstruct EQ;
    //------------------------------------------------------------------
    MPI_Init(&argc, &argv); //initializing MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &MD.iRANK);
    MPI_Comm_size(MPI_COMM_WORLD, &MD.iSIZE);
    MPI_Status  STATUS;
    MPI_Offset  OFFSETall; //this offset is for writing to file, which is done by all ranks
    MPI_File    fp_MPIOUT; //file pointer for writing the catalog
    //------------------------------------------------------------------
    int     i,        		j,            	k,				iGlobPos,		iTemp0,     	iTemp1,			iTemp2,			STFcnt;			
	int     iPlot2Screen,	iMinPtch4Cat;
        
    float  	fTemp0,   		fTemp1,      	fTemp2,			fTemp3,  		fTemp4,			fTemp5,			fTemp6;	 	
	float   fTemp7,			fTemp9,			fTemp10,		fTemp11,		fTemp12,		fUsedMemory;
	double  dEQtime,        dEQtimeDiff;
	long long lPrevEQtime;
    char 	cFile1_Out[512],    cFile2_Out[512],		cAppend[512];

	FILE 	*fpPre;

	gsl_vector_int      *ivFL_Temp0,    *ivFG_Temp0;
	gsl_vector_float 	*fvFG_Temp0,	*fvFG_Temp1,	*fvFG_Temp2, 	*fvFG_Temp3;
	gsl_vector          *dvFl_Temp0;
	gsl_vector_float 	*fvFL_Temp0,	*fvFL_Temp1,	*fvFL_Temp2;
	gsl_vector_float 	*fvBG_Temp0,	*fvBG_Temp1,	*fvBG_Temp2,	*fvBG_Temp3,	*fvBG_Temp4,	*fvBG_Temp5;
	gsl_vector_float 	*fvBL_Temp0,	*fvBL_Temp1,	*fvBL_Temp2;
	//----------------------------------------

	iPlot2Screen             = 1; //for testing and to ensure that code runs ok (as the name indicates, it is a switch that tells if output should be plotted to screen)
	iMinPtch4Cat             = 5; //to not include those small events from saved catalog; idea here is to keep file size and catalog size managable
    MD.iUseVpVs              = 1; //this is for case that rupture propagation is used (when time component of propagation is also considered), question here is whether I allow to have DIFFERENT velocities for Mode II and Mode III deformation signal; if this is set to zero, then the same velocity is used for both modes
	MD.iAlsoBoundForPostSeis = 1; //whether the boundary fault elements (if they are used/defined) are used in the post-seismic steps; if set to zero, then they would only be used to determine the stressing rate on faults (if slip boundary condition was used)
	fUsedMemory              = 0.0;//to have an idea how much memory each rank needs/uses in the computation
   	//------------------------------------------------------------------
    clock_t timer;
    srand(time(0)); 
    //------------------------------------------------------------------
	//------------------------------------------------------------------
	
	InitializeVariables(&MD, &TR, &VT, &EQ, &K, iPlot2Screen, argv); //some pre-processing steps such as initializing variables/vectors etc...

	//------------------------------------------------------------------	
	//------------------------------------------------------------------
	strcpy(cFile1_Out, MD.cInputName);                  strcat(cFile1_Out,"_");    	    sprintf(cAppend, "%d",MD.iRunNum); 	strcat(cFile1_Out,cAppend);    strcat(cFile1_Out,"_Catalog.dat");
    strcpy(cFile2_Out, MD.cInputName);                  strcat(cFile2_Out,"_");    	    sprintf(cAppend, "%d",MD.iRunNum);	strcat(cFile2_Out,cAppend);    strcat(cFile2_Out,"_PreRunData.dat");
	//------------------------------------------------------------------
	ivFG_Temp0    = gsl_vector_int_calloc(  MD.iFPNum);
 	fvFG_Temp0    = gsl_vector_float_calloc(MD.iFPNum);									fvFG_Temp1    = gsl_vector_float_calloc(MD.iFPNum);											
 	fvFG_Temp2    = gsl_vector_float_calloc(MD.iFPNum);									fvFG_Temp3    = gsl_vector_float_calloc(MD.iFPNum);	
    fvBG_Temp0    = gsl_vector_float_calloc(MD.iBPNum);									fvBG_Temp1    = gsl_vector_float_calloc(MD.iBPNum);											fvBG_Temp2    = gsl_vector_float_calloc(MD.iBPNum);		
	fvBG_Temp3    = gsl_vector_float_calloc(MD.iBPNum);									fvBG_Temp4    = gsl_vector_float_calloc(MD.iBPNum);											fvBG_Temp5    = gsl_vector_float_calloc(MD.iBPNum);
    ivFL_Temp0    = gsl_vector_int_calloc(  MD.ivF_OFFSET[MD.iRANK]);
    fvFL_Temp0    = gsl_vector_float_calloc(MD.ivF_OFFSET[MD.iRANK]);					fvFL_Temp1    = gsl_vector_float_calloc(MD.ivF_OFFSET[MD.iRANK]);							fvFL_Temp2    = gsl_vector_float_calloc(MD.ivF_OFFSET[MD.iRANK]);
    fvBL_Temp0    = gsl_vector_float_calloc(MD.ivB_OFFSET[MD.iRANK]);					fvBL_Temp1    = gsl_vector_float_calloc(MD.ivB_OFFSET[MD.iRANK]);							fvBL_Temp2    = gsl_vector_float_calloc(MD.ivB_OFFSET[MD.iRANK]);
	
	fUsedMemory += (float)(sizeof(float)*( 17*MD.iFPNum+ 15*MD.iBPNum +37*MD.ivF_OFFSET[MD.iRANK] + 11*MD.ivB_OFFSET[MD.iRANK] +7*MD.iFPNum*MD.ivF_OFFSET[MD.iRANK]) );
	fUsedMemory += (float)(sizeof(float)*( 6*MD.iFPNum*MD.ivF_OFFSET[MD.iRANK] + 6*MD.iBPNum*MD.ivF_OFFSET[MD.iRANK] + 6*MD.iFPNum*MD.ivB_OFFSET[MD.iRANK] + 9*MD.iBPNum*MD.ivB_OFFSET[MD.iRANK]));
	
	if (MD.iUseProp == 1)			{				fUsedMemory += (float)(sizeof(float)*(  9*MD.ivF_OFFSET[MD.iRANK]*MD.iFPNum));					}
	//------------------------------------------------------------------
	//------------------------------------------------------------------  
	gsl_rng *fRandN; // this is pretty much straight from the GSL reference, the default RNG has good performance, so no need to change
    const gsl_rng_type *RandT;
	gsl_rng_env_setup(); 
	RandT   = gsl_rng_default;
    fRandN  = gsl_rng_alloc(RandT);
	unsigned long RSeed =(unsigned long)(MD.iRANK + MD.iSeedStart);   //so, every rank has its own random number sequence ==> that sequence differs from the sequences of the other ranks
    gsl_rng_set(fRandN, RSeed);
	//-------------------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------
	
	LoadInput(&MD, &TR, &VT);

	//----------------------------------
	//----------------------------------

	DefineMoreParas(&MD, &TR, &VT, iPlot2Screen);

	//-------------------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------
	MPI_Allreduce(MPI_IN_PLACE, &MD.iGlobTTmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); 
    MD.iMaxSTFlgth = MD.iGlobTTmax +1;//use the +1 here to ensure I didn't mess this up and to make sure that the overwriting of the STF does not happen too fast (b/c I'm looping over the STF to save memory)
	if ((iPlot2Screen == 1) && (MD.iRANK == 0))	{			fprintf(stdout,"Other stuff after loading input and defining more parameters\nMaxSTFlgth: %d      and  %d\nDelta time: %4.4f   and Unit slip: %4.4f \n\n", MD.iMaxSTFlgth, MD.iGlobTTmax,MD.fDeltT, MD.fUnitSlip);				}
	//----------------------------------
	EQ.fmFGL_STF_H = gsl_matrix_float_calloc(MD.ivF_OFFSET[MD.iRANK], MD.iMaxSTFlgth);						EQ.fmFGL_STF_V = gsl_matrix_float_calloc(MD.ivF_OFFSET[MD.iRANK], MD.iMaxSTFlgth);
	fUsedMemory += (float)(sizeof(float)*(2*MD.ivF_OFFSET[MD.iRANK]*MD.iMaxSTFlgth)); //this is the size of the STF matrix
	fUsedMemory *= 1.0e-9;
 	//-------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------	
   	if ((iPlot2Screen == 1) && (MD.iRANK == 0))	{	fprintf(stdout,"APPROXIMATE MEMORY USAGE PER RANK: %2.5f Gb\nNow Building K-matrix....\n",fUsedMemory);			}
    //-------------------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------

	Build_K_Matrix(&MD, &TR, &VT, &K); //later on, if helpful, this could also be stored to file and the loaded here...
	
	//-------------------------------------------------------------------------------------	
	//-------------------------------------------------------------------------------------
	MPI_Allreduce(MPI_IN_PLACE, TR.ivFG_Flagged_temp, MD.iFPNum, MPI_INT, MPI_MAX, MPI_COMM_WORLD); //for output, to write out which fault patches have been flagged
	MPI_Allreduce(MPI_IN_PLACE, &MD.fMeanStiffness, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	MD.fMeanStiffness /= (float)MD.iSIZE;
	
    //-------------------------------------------------------------------------------------	
	//-------------------------------------------------------------------------------------	
	iTemp0 = 0;
	for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)     {			if (fabs(TR.fvFL_SlipRate_temp[i]) > 0.0)    {   iTemp0 = 1;    }      			}
	MPI_Allreduce(MPI_IN_PLACE, &iTemp0, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

	//-----------------------------
    if (iTemp0 == 1) //have a slip boundary condition on at least one patch
    {  
		//--------------------------------------------------
        for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
        {   fTemp1   =  -1.0*cosf(TR.fvFL_SlipRake_temp[i])*TR.fvFL_SlipRate_temp[i]; //the -1.0 is here b/c this is back-slip...
            fTemp2   =  -1.0*sinf(TR.fvFL_SlipRake_temp[i])*TR.fvFL_SlipRate_temp[i];
			gsl_vector_float_set(fvFL_Temp0, i, fTemp1);  //the strike slip component            
			gsl_vector_float_set(fvFL_Temp1, i, fTemp2);  //the dip slip component    
		}
		//--------------------------------------------------
		MPI_Allgatherv(fvFL_Temp0->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
		MPI_Allgatherv(fvFL_Temp1->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
 		
		//this here is the normal/classic back-slip method (if no boundary faults are used but a slip-boundary condition is defined)
        gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SS, fvFG_Temp0, 0.0, fvFL_Temp0);					gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SD, fvFG_Temp0, 0.0, fvFL_Temp1);
        gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
		gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DS, fvFG_Temp1, 0.0, fvFL_Temp0);					gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DD, fvFG_Temp1, 0.0, fvFL_Temp1);
		gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
		//--------------------------------------------------
		if (MD.iBPNum > 0) //I'm using boundary box faults => use the slip boundary condition on the EQ faults to load the boundary faults; at the same time, also "load/release" stress on the EQfaults
        {   //----------------------------------------------------------------------------

        	for (k = 0; k < MD.iMaxIterat; k++)		//release the induced stress on fault elements iteratively to get corresponding/resulting average "slip-rate" on fault as defined back-slip approach; need this value for normalization
			{	//-------------------------------------	
				for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)  // right now, I know the stressing amount on each fault element -as caused by back-slip; release this value via slip to get corresponding slip 
            	{	fTemp0   = -1.0*gsl_vector_float_get(TR.fvFL_CurStrsH,i) / TR.fvFL_SelfStiffStk[i]; //this is slip amount (in Stk-direction) to release the applied stress
					fTemp1   = -1.0*gsl_vector_float_get(TR.fvFL_CurStrsV,i) / TR.fvFL_SelfStiffDip[i]; //same for dip component
					gsl_vector_float_set(fvFL_Temp0, i, fTemp0); 
					gsl_vector_float_set(fvFL_Temp1, i, fTemp1);
				}	
				MPI_Allgatherv(fvFL_Temp0->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START ,MPI_FLOAT, MPI_COMM_WORLD);
				MPI_Allgatherv(fvFL_Temp1->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
				//-------------------------------------
				gsl_vector_float_add(fvFG_Temp2, fvFG_Temp0); 
				gsl_vector_float_add(fvFG_Temp3, fvFG_Temp1);
				//----------------------
				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SS, fvFG_Temp0, 0.0, fvFL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SD, fvFG_Temp0, 0.0, fvFL_Temp1);				
				gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);									
				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DS, fvFG_Temp1, 0.0, fvFL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DD, fvFG_Temp1, 0.0, fvFL_Temp1);				
				gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);														
        	}
			//--------------------------------------------------
			//--------------------------------------------------
			MPI_Allgatherv(TR.ivFL_StabT, MD.ivF_OFFSET[MD.iRANK], MPI_INT, ivFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START, MPI_INT, MPI_COMM_WORLD);
			iTemp1 = 0;
			for (i = 0; i <  MD.iFPNum; i++) //take the prescribed fault slip rates again and use to load the boundary fault elements
			{   if (gsl_vector_int_get(ivFG_Temp0, i) < 3)		{		iTemp1++;		}
			}
			dvFl_Temp0 = gsl_vector_calloc(iTemp1);
			//---------------------------
			iTemp1 = 0;
			for (i = 0; i <  MD.iFPNum; i++) //take the prescribed fault slip rates again and use to load the boundary fault elements
			{   if (gsl_vector_int_get(ivFG_Temp0, i) < 3)		
				{	fTemp4 = sqrtf(gsl_vector_float_get(fvFG_Temp2,i)*gsl_vector_float_get(fvFG_Temp2,i) + gsl_vector_float_get(fvFG_Temp3,i)*gsl_vector_float_get(fvFG_Temp3,i));
					gsl_vector_set(dvFl_Temp0 , iTemp1, (double)fTemp4);
					iTemp1++;	
			}	}
			fTemp4 = (float)gsl_stats_median(dvFl_Temp0->data, 1, iTemp1);
			//-------------------------------------------------- 
			//--------------------------------------------------
        	
        	//----------------------------------------------------------------------------
		  	for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) //take the prescribed fault slip rates again and use to load the boundary fault elements
        	{   fTemp1   =  -1.0*cosf(TR.fvFL_SlipRake_temp[i])*TR.fvFL_SlipRate_temp[i]; //the -1.0 is here b/c this is back-slip...
            	fTemp2   =  -1.0*sinf(TR.fvFL_SlipRake_temp[i])*TR.fvFL_SlipRate_temp[i];
				gsl_vector_float_set(fvFL_Temp0, i, fTemp1);  //the strike slip component            
				gsl_vector_float_set(fvFL_Temp1, i, fTemp2);  //the dip slip component    
			}
			//--------------------------------------------------
			MPI_Allgatherv(fvFL_Temp0->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
			MPI_Allgatherv(fvFL_Temp1->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
			//--------------------------------------------------
			gsl_vector_float_scale(fvFG_Temp0, -1.0); //it appears that I need to use the actual slip direction here...
			gsl_vector_float_scale(fvFG_Temp1, -1.0); //instead of "backslip" i mean
			//--------------------------------------------------
			//apply the slip amount on fault elements to "stress" the boundary elements
			gsl_blas_sgemv(CblasTrans, 1.0, K.FB_SS, fvFG_Temp0, 0.0, fvBL_Temp0); 				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_SD, fvFG_Temp0, 0.0, fvBL_Temp1); 				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_SO, fvFG_Temp0, 0.0, fvBL_Temp2); 
			gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);									gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);
			gsl_blas_sgemv(CblasTrans, 1.0, K.FB_DS, fvFG_Temp1, 0.0, fvBL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_DD, fvFG_Temp1, 0.0, fvBL_Temp1);				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_DO, fvFG_Temp1, 0.0, fvBL_Temp2); 
			gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);									gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);
			//--------------------------------------------------
			for (k = 0; k < MD.iMaxIterat; k++) // release the stress on boundary elements iteratively to determine corresponding slip-rate on boundary fault elements
			{	//-------------------------------------	
				for (i = 0; i < MD.ivB_OFFSET[MD.iRANK]; i++)  //going through the boundary faults to iteratively release the stress that was put onto them by the EQfaults => the corresponding slip to achieve that is stored and defines the boundary fault loading 
            	{	fTemp0   = -1.0*gsl_vector_float_get(TR.fvBL_CurStrsH,i) / TR.fvBL_SelfStiffStk[i]; //this is slip amount (in Stk-direction) to release the applied stress
					fTemp1   = -1.0*gsl_vector_float_get(TR.fvBL_CurStrsV,i) / TR.fvBL_SelfStiffDip[i]; //same for dip component
					fTemp2   = -1.0*gsl_vector_float_get(TR.fvBL_CurStrsN,i) / TR.fvBL_SelfStiffOpn[i]; //and normal component
						
					gsl_vector_float_set(fvBL_Temp0, i, fTemp0); 
					gsl_vector_float_set(fvBL_Temp1, i, fTemp1);
					gsl_vector_float_set(fvBL_Temp2, i, fTemp2);
				}	
				MPI_Allgatherv(fvBL_Temp0->data, MD.ivB_OFFSET[MD.iRANK], MPI_FLOAT, fvBG_Temp0->data, MD.ivB_OFFSET, MD.ivB_START ,MPI_FLOAT, MPI_COMM_WORLD);
				MPI_Allgatherv(fvBL_Temp1->data, MD.ivB_OFFSET[MD.iRANK], MPI_FLOAT, fvBG_Temp1->data, MD.ivB_OFFSET, MD.ivB_START, MPI_FLOAT, MPI_COMM_WORLD);
				MPI_Allgatherv(fvBL_Temp2->data, MD.ivB_OFFSET[MD.iRANK], MPI_FLOAT, fvBG_Temp2->data, MD.ivB_OFFSET, MD.ivB_START, MPI_FLOAT, MPI_COMM_WORLD);
				//-------------------------------------
				gsl_vector_float_add(fvBG_Temp3, fvBG_Temp0); //collecting the total slip in strike/dip/normal => used later for loading of fault patches
				gsl_vector_float_add(fvBG_Temp4, fvBG_Temp1);
				gsl_vector_float_add(fvBG_Temp5, fvBG_Temp2);
				//----------------------
				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_SS, fvBG_Temp0, 0.0, fvBL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_SD, fvBG_Temp0, 0.0, fvBL_Temp1);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_SO, fvBG_Temp0, 0.0, fvBL_Temp2); 
				gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);									gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);
				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_DS, fvBG_Temp1, 0.0, fvBL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_DD, fvBG_Temp1, 0.0, fvBL_Temp1);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_DO, fvBG_Temp1, 0.0, fvBL_Temp2); 
				gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);									gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);
				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_OS, fvBG_Temp2, 0.0, fvBL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_OD, fvBG_Temp2, 0.0, fvBL_Temp1);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_OO, fvBG_Temp2, 0.0, fvBL_Temp2); 
				gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);									gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);
				//--------------------------------------------------
			}
			//set CurrStress on fault elements back to zero, b/c they will be used to determine how much stress was induced by slip along boundary fault elements
			gsl_vector_float_set_zero(TR.fvFL_CurStrsH);											gsl_vector_float_set_zero(TR.fvFL_CurStrsV);
			//--------------------------------------------------
			gsl_blas_sgemv(CblasTrans, 1.0, K.BF_SS, fvBG_Temp3, 0.0, fvFL_Temp0); 					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_SD, fvBG_Temp3, 0.0, fvFL_Temp1); 				
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
			gsl_blas_sgemv(CblasTrans, 1.0, K.BF_DS, fvBG_Temp4, 0.0, fvFL_Temp0);					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_DD, fvBG_Temp4, 0.0, fvFL_Temp1); 
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
			gsl_blas_sgemv(CblasTrans, 1.0, K.BF_OS, fvBG_Temp5, 0.0, fvFL_Temp0);					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_OD, fvBG_Temp5, 0.0, fvFL_Temp1);
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
			//----------------------------------------------------------------------------

			gsl_vector_float_set_zero(fvFG_Temp2);													gsl_vector_float_set_zero(fvFG_Temp3);
			for (k = 0; k < MD.iMaxIterat; k++)		// now I release the stress that was induced on fault elements by slip on boundary elements; goal is to determine the corresponding average "slip rate"
			{	//-------------------------------------	
				for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)  // right now, I know the stressing amount on each fault element -as caused by back-slip; release this value via slip to get corresponding slip 
            	{	fTemp0   = -1.0*gsl_vector_float_get(TR.fvFL_CurStrsH,i) / TR.fvFL_SelfStiffStk[i]; //this is slip amount (in Stk-direction) to release the applied stress
					fTemp1   = -1.0*gsl_vector_float_get(TR.fvFL_CurStrsV,i) / TR.fvFL_SelfStiffDip[i]; //same for dip component
					gsl_vector_float_set(fvFL_Temp0, i, fTemp0); 
					gsl_vector_float_set(fvFL_Temp1, i, fTemp1);
				}	
				MPI_Allgatherv(fvFL_Temp0->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START ,MPI_FLOAT, MPI_COMM_WORLD);
				MPI_Allgatherv(fvFL_Temp1->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
				//-------------------------------------
				gsl_vector_float_add(fvFG_Temp2, fvFG_Temp0); 
				gsl_vector_float_add(fvFG_Temp3, fvFG_Temp1);
				//----------------------
				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SS, fvFG_Temp0, 0.0, fvFL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SD, fvFG_Temp0, 0.0, fvFL_Temp1);				
				gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);									
				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DS, fvFG_Temp1, 0.0, fvFL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DD, fvFG_Temp1, 0.0, fvFL_Temp1);				
				gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);														
        	}
			//--------------------------------------------------
			//--------------------------------------------------
			iTemp1 = 0;
			for (i = 0; i <  MD.iFPNum; i++) //take the prescribed fault slip rates again and use to load the boundary fault elements
			{   if (gsl_vector_int_get(ivFG_Temp0, i) < 3)	
				{	fTemp5 = sqrtf(gsl_vector_float_get(fvFG_Temp2,i)*gsl_vector_float_get(fvFG_Temp2,i) + gsl_vector_float_get(fvFG_Temp3,i)*gsl_vector_float_get(fvFG_Temp3,i));
					gsl_vector_set(dvFl_Temp0 , iTemp1, (double)fTemp5);
					iTemp1++;	
			}	}
			fTemp5 = (float)gsl_stats_median(dvFl_Temp0->data, 1, iTemp1);
			
			gsl_vector_int_set_zero(ivFG_Temp0);	
			gsl_vector_float_set_zero(fvFG_Temp2);	
			gsl_vector_float_set_zero(fvFG_Temp3);	
			//--------------------------------------------------
			//--------------------------------------------------
        	//----------------------------------------------------------------------------
			fTemp6 = fTemp4/fTemp5; //this is the actual scaling factor to be applied...
			//--------------------------------------------------
			if ((iPlot2Screen == 1) &&(MD.iRANK == 0))		{			fprintf(stdout,"Test Slip Rate (from Boundary): %5.5f    Prescribed Slip Rate (from Back-Slip): %5.5f    Scale Factor: %5.5f\n",fTemp5, fTemp4, fTemp6);			}
			//--------------------------------------------------
			//this part is a bit redundant -but conceptually simpler and I don't need to define additional variabls
			//so, here I take again the slip along the bounddary alemeents to see how much stress is induced; this stress amount is then scaled
			//with fTemp6 in order to ensure that resulting slip rates are (on average) the same for back-slip and modified back-slip method
			gsl_vector_float_set_zero(TR.fvFL_CurStrsH);											gsl_vector_float_set_zero(TR.fvFL_CurStrsV);
			//--------------------------------------------------
			gsl_blas_sgemv(CblasTrans, 1.0, K.BF_SS, fvBG_Temp3, 0.0, fvFL_Temp0); 					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_SD, fvBG_Temp3, 0.0, fvFL_Temp1); 				
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
			gsl_blas_sgemv(CblasTrans, 1.0, K.BF_DS, fvBG_Temp4, 0.0, fvFL_Temp0);					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_DD, fvBG_Temp4, 0.0, fvFL_Temp1); 
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
			gsl_blas_sgemv(CblasTrans, 1.0, K.BF_OS, fvBG_Temp5, 0.0, fvFL_Temp0);					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_OD, fvBG_Temp5, 0.0, fvFL_Temp1);
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);										gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);
			//----------------------------------------------------------------------------
			gsl_vector_float_scale(TR.fvFL_CurStrsH, fTemp6); //these are just temporary place holders (not currrent stress but stressing rate put in here) at the moment...
			gsl_vector_float_scale(TR.fvFL_CurStrsV, fTemp6); //the values are actually stressing rates, will be assigned within the next few lines...
		}
        //now, the values from the temporary file are copied/added to Stressing Rate; the "adding" part allows to have mixed boundary conditions -not that this is a good idea... but it is possible to define one fault with stressing rate and another with slip-rate and then get combined stressing rate for each
		gsl_vector_float_add(TR.fvFL_StrsRateStk, TR.fvFL_CurStrsH);			gsl_vector_float_add(TR.fvFL_StrsRateDip, TR.fvFL_CurStrsV);
		
		//------------------------------------------------------------------
		//------------------------------------------------------------------
		gsl_vector_float_memcpy(TR.fvFL_StrsRateStk, TR.fvFL_CurStrsH);			gsl_vector_float_memcpy(TR.fvFL_StrsRateDip, TR.fvFL_CurStrsV); 
		//------------------------------------------------------------------
		//------------------------------------------------------------------
	}	
	//------------------------------------------------------------------------------------------
	fTemp5 = 0.0;
	for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)
	{	fTemp5 += sqrtf(gsl_vector_float_get(TR.fvFL_StrsRateStk, i)*gsl_vector_float_get(TR.fvFL_StrsRateStk, i) +gsl_vector_float_get(TR.fvFL_StrsRateDip, i)*gsl_vector_float_get(TR.fvFL_StrsRateDip, i)); //the amount of applied stressing rate (per interseismic time step)
    }
	//--------------------------------------------------
	MPI_Allreduce(MPI_IN_PLACE, &fTemp5, 1 , MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); //total applied stressing rate
    fTemp5 /= (float)MD.iFPNum;			
    if ((iPlot2Screen == 1) && (MD.iRANK == 0))				{	    fprintf(stdout,"Resulting average stressing-rate on faults (MPa/yr): %5.5f and per time step: %5.6f\n",fTemp5, (fTemp5*MD.fISeisTStp));  			}
	gsl_vector_float_scale(TR.fvFL_StrsRateStk, MD.fISeisTStp ); //reference stressing rate is now including the time step => only need to add that amount
	gsl_vector_float_scale(TR.fvFL_StrsRateDip, MD.fISeisTStp ); //when applicable (when stepping forward by one interseis. time step)
	//------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------

	//following, the pre-run output is written; this is really helpful for debugging etc... and also to check for example that the stressing-rate distribution on the fault is making sense (is possible that grid resolution of boundary fault is low and those patches are too close to EQfaults, negatively affecting the loading function)
	if (MD.iRANK == 0)           
    {	if ((fpPre = fopen(cFile2_Out,"wb")) == NULL)      {   		exit(10);       }

		fwrite( &MD.iFPNum,         sizeof(int),           1, fpPre);			fwrite( &MD.iFVNum,         sizeof(int),           1, fpPre); 
		fwrite( &MD.iBPNum,         sizeof(int),           1, fpPre); 			fwrite( &MD.iBVNum,         sizeof(int),           1, fpPre); 
		      
 		fwrite( TR.ivFG_V1_temp,    sizeof(int),   MD.iFPNum, fpPre); 			fwrite( TR.ivFG_V2_temp,    sizeof(int),   MD.iFPNum, fpPre);			fwrite( TR.ivFG_V3_temp,    sizeof(int),   MD.iFPNum, fpPre);
		fwrite( TR.ivFG_SegID_temp, sizeof(int),   MD.iFPNum, fpPre);			fwrite( TR.ivFG_FltID_temp, sizeof(int),   MD.iFPNum, fpPre);

		fwrite( VT.fvFG_PosE_temp, 	sizeof(float), MD.iFVNum, fpPre);			fwrite( VT.fvFG_PosN_temp, 	sizeof(float), MD.iFVNum, fpPre);
		fwrite( VT.fvFG_PosZ_temp, 	sizeof(float), MD.iFVNum, fpPre);			fwrite( VT.fvFG_Hght_temp, 	sizeof(float), MD.iFVNum, fpPre);
		
		fwrite( TR.fvFG_CentE_temp, sizeof(int),   MD.iFPNum, fpPre); 			fwrite( TR.fvFG_CentN_temp, sizeof(int),   MD.iFPNum, fpPre);			fwrite( TR.fvFG_CentZ_temp, sizeof(int),   MD.iFPNum, fpPre);
		
		fwrite( TR.ivBG_V1_temp,    sizeof(int),   MD.iBPNum, fpPre); 			fwrite( TR.ivBG_V2_temp,    sizeof(int),   MD.iBPNum, fpPre);			fwrite( TR.ivBG_V3_temp,    sizeof(int),   MD.iBPNum, fpPre);

		fwrite( VT.fvBG_PosE_temp, 	sizeof(float), MD.iBVNum, fpPre);			fwrite( VT.fvBG_PosN_temp, 	sizeof(float), MD.iBVNum, fpPre);			fwrite( VT.fvBG_PosZ_temp, 	sizeof(float), MD.iBVNum, fpPre);	
	}
	MPI_Allgatherv(TR.fvFL_StrsRateStk->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgatherv(TR.fvFL_StrsRateDip->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
    if (MD.iRANK == 0)           
    {	fwrite(TR.ivFG_Flagged_temp, sizeof(int),   MD.iFPNum, fpPre); //the "flagged" status for interaction => if removed/overridden etc...
    	fwrite(fvFG_Temp0->data,     sizeof(float), MD.iFPNum, fpPre); //the stressing rate in strike direction
		fwrite(fvFG_Temp1->data,     sizeof(float), MD.iFPNum, fpPre); //the stressing rate in dip direction
	}
	for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
    {   fTemp0   = TR.fvFL_StaFric[i]                        *-1.0*TR.fvFL_RefNrmStrs[i]; //this is the static strength of the patch
		fTemp1   = (TR.fvFL_StaFric[i] - TR.fvFL_DynFric[i]) *-1.0*TR.fvFL_RefNrmStrs[i]; //this is the stress drop of the patch
		fTemp2   = TR.fvFL_CurDcVal[i];
		
		gsl_vector_int_set(  ivFL_Temp0, i, TR.ivFL_StabT[i]);
		gsl_vector_float_set(fvFL_Temp0, i, fTemp0);          
		gsl_vector_float_set(fvFL_Temp1, i, fTemp1);   
		gsl_vector_float_set(fvFL_Temp2, i, fTemp2);
		
	}	
    MPI_Allgatherv(ivFL_Temp0->data, MD.ivF_OFFSET[MD.iRANK], MPI_INT,   ivFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START, MPI_INT,   MPI_COMM_WORLD);
    MPI_Allgatherv(fvFL_Temp0->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgatherv(fvFL_Temp1->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
	MPI_Allgatherv(fvFL_Temp2->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp2->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
	
	MPI_Allreduce(MPI_IN_PLACE, TR.fvFG_MaxTransient, MD.iFPNum , MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    if (MD.iRANK == 0)           
    {	fwrite(fvFG_Temp0->data,     sizeof(float), MD.iFPNum, fpPre); //the patch strength
		fwrite(fvFG_Temp1->data,     sizeof(float), MD.iFPNum, fpPre); //the patch stress drop when having full drop
		fwrite(fvFG_Temp2->data,     sizeof(float), MD.iFPNum, fpPre); //the Dc value
		fwrite(ivFG_Temp0->data,     sizeof(int),   MD.iFPNum, fpPre); //the patch stability type (1 = unstable, 2 = cond. stable, 3 = stable)
		fwrite(TR.fvFG_MaxTransient, sizeof(float), MD.iFPNum, fpPre); //the patch stability type (1 = unstable, 2 = cond. stable, 3 = stable)	
	}
	//-----------------------------------------------------------
	if (MD.iRANK == 0)           
    {	fwrite(fvBG_Temp3->data, sizeof(float), MD.iBPNum, fpPre); //the slip/stressing rate in strike direction
		fwrite(fvBG_Temp4->data, sizeof(float), MD.iBPNum, fpPre); //the slip/stressing rate in dip direction
		fwrite(fvBG_Temp5->data, sizeof(float), MD.iBPNum, fpPre); //the slip/stressing rate in opening direction
		fclose(fpPre);
	}
	//------------------------------------------------------------------------------------------	
	//------------------------------------------------------------------------------------------
    MPI_Barrier( MPI_COMM_WORLD );
	//------------------------------------------------------------------------------------------	
	//------------------------------------------------------------------------------------------
	//open the EQ catalog file here once and then keep open (if not kept open, the continuous opening and closing slows things down and also might cause code to crash
	MPI_File_delete(cFile1_Out, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, cFile1_Out, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_MPIOUT);
    if (MD.iRANK == 0)           
    {   MPI_File_write(fp_MPIOUT, &MD.iEQcntr,   1, MPI_INT,      &STATUS);
		MPI_File_write(fp_MPIOUT, &MD.iFPNum,    1, MPI_INT,      &STATUS);
		MPI_File_write(fp_MPIOUT, &MD.iFVNum,    1, MPI_INT,      &STATUS);        
        MPI_File_write(fp_MPIOUT, &MD.fShearMod, 1, MPI_FLOAT,    &STATUS);    
        MPI_File_write(fp_MPIOUT, &MD.fDeltT,    1, MPI_FLOAT,    &STATUS);      

 		MPI_File_write(fp_MPIOUT, TR.ivFG_V1_temp,    	MD.iFPNum, MPI_INT,    &STATUS);   
		MPI_File_write(fp_MPIOUT, TR.ivFG_V2_temp,    	MD.iFPNum, MPI_INT,    &STATUS);   
		MPI_File_write(fp_MPIOUT, TR.ivFG_V3_temp,   	MD.iFPNum, MPI_INT,    &STATUS);   
		MPI_File_write(fp_MPIOUT, TR.ivFG_SegID_temp, 	MD.iFPNum, MPI_INT,    &STATUS);   
		MPI_File_write(fp_MPIOUT, TR.ivFG_FltID_temp, 	MD.iFPNum, MPI_INT,    &STATUS);

		MPI_File_write(fp_MPIOUT, VT.fvFG_VlX_temp, 	MD.iFVNum, MPI_FLOAT,    &STATUS);
		MPI_File_write(fp_MPIOUT, VT.fvFG_VlY_temp, 	MD.iFVNum, MPI_FLOAT,    &STATUS);

		MPI_File_write(fp_MPIOUT, VT.fvFG_PosE_temp, 	MD.iFVNum, MPI_FLOAT,    &STATUS);
		MPI_File_write(fp_MPIOUT, VT.fvFG_PosN_temp, 	MD.iFVNum, MPI_FLOAT,    &STATUS);
		MPI_File_write(fp_MPIOUT, VT.fvFG_PosZ_temp, 	MD.iFVNum, MPI_FLOAT,    &STATUS);
		MPI_File_write(fp_MPIOUT, VT.fvFG_Hght_temp, 	MD.iFVNum, MPI_FLOAT,    &STATUS);

		MPI_File_write(fp_MPIOUT, TR.fvFG_CentE_temp, 	MD.iFPNum, MPI_FLOAT,    &STATUS);
		MPI_File_write(fp_MPIOUT, TR.fvFG_CentN_temp, 	MD.iFPNum, MPI_FLOAT,    &STATUS);
		MPI_File_write(fp_MPIOUT, TR.fvFG_CentZ_temp, 	MD.iFPNum, MPI_FLOAT,    &STATUS);
    }
    MPI_Barrier( MPI_COMM_WORLD );
    OFFSETall = 3*sizeof(int) + 2*sizeof(float) +5*MD.iFPNum*sizeof(int) +6*MD.iFVNum*sizeof(float) +3*MD.iFPNum*sizeof(float);
	//------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------		
	free(TR.fvFL_SlipRate_temp);	free(TR.fvFL_SlipRake_temp);	free(TR.ivFG_Flagged_temp);	    free(VT.fvFG_VlX_temp);			free(VT.fvFG_VlY_temp);			free(VT.fvFG_Hght_temp);
	free(TR.ivFG_V1_temp);			free(TR.ivFG_V2_temp);			free(TR.ivFG_V3_temp);			free(TR.ivFG_FltID_temp);		free(TR.ivFG_SegID_temp);		free(TR.ivBG_SegID_temp);	
	free(TR.fvFG_StressRatetemp);	free(TR.fvFG_SlipRatetemp);     free(TR.fvFG_Raketemp);
	free(TR.ivBG_V1_temp);			free(TR.ivBG_V2_temp);			free(TR.ivBG_V3_temp);			
	free(TR.fvFG_CentE_temp);		free(TR.fvFG_CentN_temp);		free(TR.fvFG_CentZ_temp);		free(TR.fvBG_CentE_temp);		free(TR.fvBG_CentN_temp);		free(TR.fvBG_CentZ_temp);		
	free(TR.fvFL_StaFricMod_temp);	free(TR.fvFL_DynFricMod_temp);	free(TR.fvFL_NrmStrsMod_temp);	free(TR.fvFL_DcMod_temp);
	free(VT.fvFG_PosE_temp);		free(VT.fvFG_PosN_temp);		free(VT.fvFG_PosZ_temp);		free(VT.fvBG_PosE_temp);		free(VT.fvBG_PosN_temp);		free(VT.fvBG_PosZ_temp);	
	//------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------
	//load all fault patches to be fully loaded -then subtracting some fraction to make it not too artificial in the beginning of the catalog
    for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
    {	fTemp3            = sqrtf(gsl_vector_float_get(TR.fvFL_StrsRateStk, i)*gsl_vector_float_get(TR.fvFL_StrsRateStk, i) + gsl_vector_float_get(TR.fvFL_StrsRateDip, i)*gsl_vector_float_get(TR.fvFL_StrsRateDip, i)); //this is combined loading per time step
        fTemp4            = (TR.fvFL_CurFric[i]*-1.0*TR.fvFL_RefNrmStrs[i]);//this is strength of element; the "-1" is here to make normal stress compression positive again (to get positive strength value)
        fTemp5            = fTemp4/fTemp3; //this is number of loading steps I'd need to reach the strength level
      	fTemp6            = 0.75;//(float)(gsl_rng_uniform(fRandN) *0.09 +0.90); //is a random number between 0.90 and 0.99 => so many loading steps to use (fraction of temp2)
		gsl_vector_float_set(TR.fvFL_CurStrsH,  i, gsl_vector_float_get(TR.fvFL_StrsRateStk, i)*fTemp5*fTemp6); //stressing-rate * number of stressing/loading steps * random number between 0.9 and 0.99
		gsl_vector_float_set(TR.fvFL_CurStrsV,  i, gsl_vector_float_get(TR.fvFL_StrsRateDip, i)*fTemp5*fTemp6);
		gsl_vector_float_set(TR.fvFL_CurStrsN,  i, TR.fvFL_RefNrmStrs[i]);
    }  
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	timer = clock();
	MD.lRecLgth   = (long int)(MD.fRecLgth/MD.fISeisTStp);

    MD.lTimeYears = 0;
    lPrevEQtime   = 0;
    while (MD.lTimeYears <= MD.lRecLgth)
    { 	MD.lTimeYears++;
		//--------------------------------------------------------------
		//first, the regular loading step, is first applied to ALL fault patches, using the respective reference stressing rate
		//this is done to all! (non-boundary) patches => if a patch is aseismic, then this stress will be further redistributed in a following step
     	gsl_vector_float_add(TR.fvFL_CurStrsH, TR.fvFL_StrsRateStk);   
		gsl_vector_float_add(TR.fvFL_CurStrsV, TR.fvFL_StrsRateDip);  	
		//--------------------------------------------------------------
		if (MD.iUsePSeis == 1) //  https://en.wikipedia.org/wiki/Stress_relaxation    http://web.mit.edu/course/3/3.11/www/modules/visco.pdf  => page 9ff; using Maxwell spring-dashpot model    
      	{	MD.fPSeis_Step += 1.0;
		  	//-------------------------------------------------------------	
		   	fTemp0          = expf(-1.0*(MD.fISeisTStp*MD.fPSeis_Step)/MD.fAftrSlipTime); //factor/fraction from decay function
			for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)			
			{	TR.fvFL_CurFric[i] =  TR.fvFL_StaFric[i] + fTemp0*TR.fvFL_PSeis_T0_F[i];					
			}	
			//-------------------------------------------------------------
			fTemp0          = expf(-1.0*(MD.fISeisTStp*MD.fPSeis_Step)/MD.fDeepRelaxTime); //factor/fraction from decay function
			//---------------------------
			if ( (MD.iAlsoBoundForPostSeis == 1) &&  (MD.iBPNum > 0) )
			{	
				for (k = 0; k < MD.iSIZE; k++)	{			MD.ivB_ModOFFs[k] = 0;								}
				
				iTemp0 = 0;
				for (i = 0; i < MD.ivB_OFFSET[MD.iRANK]; i++)	
				{	fTemp1 = fTemp0*TR.fvBL_PSeis_T0_S[i]; 
					fTemp2 = fTemp0*TR.fvBL_PSeis_T0_N[i]; 
					fTemp3 = sqrtf(gsl_vector_float_get(TR.fvBL_CurStrsH, i)*gsl_vector_float_get(TR.fvBL_CurStrsH, i) + gsl_vector_float_get(TR.fvBL_CurStrsV, i)*gsl_vector_float_get(TR.fvBL_CurStrsV, i) );
					fTemp4 = fabs(gsl_vector_float_get(TR.fvBL_CurStrsN, i));

					if (fTemp3 - fTemp1 > MD.fCutStFrac)
					{	fTemp5 = -1.0*((fTemp3-fTemp1)/fTemp3 *gsl_vector_float_get(TR.fvBL_CurStrsH, i)) / TR.fvBL_SelfStiffStk[i];				gsl_vector_float_set(fvBL_Temp0, i, fTemp5); 
						fTemp6 = -1.0*((fTemp3-fTemp1)/fTemp3 *gsl_vector_float_get(TR.fvBL_CurStrsV, i)) / TR.fvBL_SelfStiffDip[i];				gsl_vector_float_set(fvBL_Temp1, i, fTemp6); 
						iTemp0 = 1;
					}
					else		{	gsl_vector_float_set(fvBL_Temp0, i, 0.0); 		gsl_vector_float_set(fvBL_Temp1, i, 0.0); 					}
					
					if (fTemp4 - fTemp2 > MD.fCutStFrac)
					{	fTemp7 = -1.0*((fTemp4-fTemp2)/fTemp4 *gsl_vector_float_get(TR.fvBL_CurStrsN, i)) / TR.fvBL_SelfStiffOpn[i];				gsl_vector_float_set(fvBL_Temp2, i, fTemp7); 
						iTemp0 = 1;
					}
					else		{	gsl_vector_float_set(fvBL_Temp2, i, 0.0); 																	}
				}
				//--------------------------------------------------------------
				if (iTemp0 == 1)		{			MD.ivB_ModOFFs[MD.iRANK] = MD.ivB_OFFSET[MD.iRANK];											}
				MPI_Allreduce(MPI_IN_PLACE, &iTemp0, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
				//--------------------------------------------------------------
				if (iTemp0 == 1)
				{	MPI_Allreduce(MPI_IN_PLACE, MD.ivB_ModOFFs, MD.iSIZE, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

					gsl_vector_float_set_zero(fvBG_Temp0);       										MPI_Allgatherv(fvBL_Temp0->data, MD.ivB_ModOFFs[MD.iRANK], MPI_FLOAT, fvBG_Temp0->data, MD.ivB_ModOFFs, MD.ivB_START, MPI_FLOAT, MPI_COMM_WORLD);
					gsl_vector_float_set_zero(fvBG_Temp1);      	       								MPI_Allgatherv(fvBL_Temp1->data, MD.ivB_ModOFFs[MD.iRANK], MPI_FLOAT, fvBG_Temp1->data, MD.ivB_ModOFFs, MD.ivB_START, MPI_FLOAT, MPI_COMM_WORLD);
					gsl_vector_float_set_zero(fvBG_Temp2);      	       								MPI_Allgatherv(fvBL_Temp2->data, MD.ivB_ModOFFs[MD.iRANK], MPI_FLOAT, fvBG_Temp2->data, MD.ivB_ModOFFs, MD.ivB_START, MPI_FLOAT, MPI_COMM_WORLD);

					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_SS, fvBG_Temp0, 0.0, fvFL_Temp0); 				gsl_blas_sgemv(CblasTrans, 1.0, K.BF_SD, fvBG_Temp0, 0.0, fvFL_Temp1); 						
					gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);									
					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_DS, fvBG_Temp1, 0.0, fvFL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.BF_DD, fvBG_Temp1, 0.0, fvFL_Temp1);				
					gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);	
					gsl_blas_sgemv(CblasTrans, 1.0, K.BF_OS, fvBG_Temp2, 0.0, fvFL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.BF_OD, fvBG_Temp2, 0.0, fvFL_Temp1);				
					gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);	

					gsl_blas_sgemv(CblasTrans, 1.0, K.BB_SS, fvBG_Temp0, 0.0, fvBL_Temp0); 				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_SD, fvBG_Temp0, 0.0, fvBL_Temp1); 					gsl_blas_sgemv(CblasTrans, 1.0, K.BB_SO, fvBG_Temp0, 0.0, fvBL_Temp2); 						
					gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);										gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);									
					gsl_blas_sgemv(CblasTrans, 1.0, K.BB_DS, fvBG_Temp1, 0.0, fvBL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_DD, fvBG_Temp1, 0.0, fvBL_Temp1);					gsl_blas_sgemv(CblasTrans, 1.0, K.BB_DO, fvBG_Temp1, 0.0, fvBL_Temp2); 	
					gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);										gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);	
					gsl_blas_sgemv(CblasTrans, 1.0, K.BB_OS, fvBG_Temp2, 0.0, fvBL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.BB_OD, fvBG_Temp2, 0.0, fvBL_Temp1);					gsl_blas_sgemv(CblasTrans, 1.0, K.BB_OO, fvBG_Temp2, 0.0, fvBL_Temp2); 
					gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);									gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);										gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);	
		}	}	}
		//---------------------------
		for (k = 0; k < MD.iSIZE; k++)	{			MD.ivF_ModOFFs[k] = 0;								}
		iTemp0 = 0;
		for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)
		{	if (TR.ivFL_StabT[i] != 1) 
			{	fTemp3   = sqrtf(gsl_vector_float_get(TR.fvFL_CurStrsH, i)*gsl_vector_float_get(TR.fvFL_CurStrsH, i) + gsl_vector_float_get(TR.fvFL_CurStrsV, i)*gsl_vector_float_get(TR.fvFL_CurStrsV, i));
				fTemp4   = fTemp3 - TR.fvFL_CurFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i); //this is the excess stress (is excess if value > 0) I currently have with respect to curr strength

				if (fTemp4 > MD.fCutStFrac) //if I have excess shear stress above current strength and I am not looking at an unstable patch
				{   fTemp5   = (-1.0*((fTemp4/fTemp3)*gsl_vector_float_get(TR.fvFL_CurStrsH, i)) )/ TR.fvFL_SelfStiffStk[i]; // slip amount to release excess horizontal shear stress
					fTemp6   = (-1.0*((fTemp4/fTemp3)*gsl_vector_float_get(TR.fvFL_CurStrsV, i)) )/ TR.fvFL_SelfStiffDip[i]; // slip amount to release excess vertical shear stress 
					gsl_vector_float_set(fvFL_Temp0, i, fTemp5);  //the strike slip component            
					gsl_vector_float_set(fvFL_Temp1, i, fTemp6);  //the dip slip component  
					iTemp0   = 1;
				}				 
				else
				{	gsl_vector_float_set(fvFL_Temp0, i,    0.0);  //the strike slip component            
					gsl_vector_float_set(fvFL_Temp1, i,    0.0);  //the dip slip component  
			}	}
			else
			{	gsl_vector_float_set(fvFL_Temp0, i, 0.0);  //the strike slip component            
				gsl_vector_float_set(fvFL_Temp1, i, 0.0);  //the dip slip component  
		}	}   	    
		if (iTemp0 == 1)		{			MD.ivF_ModOFFs[MD.iRANK] = MD.ivF_OFFSET[MD.iRANK];					}

		MPI_Allreduce(MPI_IN_PLACE, &iTemp0, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		//--------------------------------------------------------------
		if (iTemp0 == 1)
		{	MPI_Allreduce(MPI_IN_PLACE, MD.ivF_ModOFFs, MD.iSIZE, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
			gsl_vector_float_set_zero(fvFG_Temp0);       										MPI_Allgatherv(fvFL_Temp0->data, MD.ivF_ModOFFs[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_ModOFFs, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
			gsl_vector_float_set_zero(fvFG_Temp1);      	       								MPI_Allgatherv(fvFL_Temp1->data, MD.ivF_ModOFFs[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_ModOFFs, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
		
			gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SS, fvFG_Temp0, 0.0, fvFL_Temp0); 			gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SD, fvFG_Temp0, 0.0, fvFL_Temp1); 			
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);									
			gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DS, fvFG_Temp1, 0.0, fvFL_Temp0);				gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DD, fvFG_Temp1, 0.0, fvFL_Temp1);				
			gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);									gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);									
		}	
		//--------------------------------------------------------------
		EQ.iStillOn = 0;
        for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)
        { 	if (TR.ivFL_StabT[i] == 1 )
           	{  	fTemp3   = sqrtf(gsl_vector_float_get(TR.fvFL_CurStrsH, i)*gsl_vector_float_get(TR.fvFL_CurStrsH, i) + gsl_vector_float_get(TR.fvFL_CurStrsV, i)*gsl_vector_float_get(TR.fvFL_CurStrsV, i));
				fTemp4   = fTemp3 - (TR.fvFL_CurFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i)); //this is the excess stress (is excess if value > 0) I currently have with respect to curr strength
        
        		if (fTemp4  > TR.fvFL_CutStress[i]) 				{			EQ.iStillOn  = 1;      			} 	
        }   }	  
        MPI_Allreduce(MPI_IN_PLACE, &EQ.iStillOn, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);	 
		//------------------------------------------------------------------------------------------
		//------------------------------------------------------------------------------------------
		if (EQ.iStillOn == 1)
        {	//-----------------------------------------
			gsl_vector_float_memcpy(TR.fvFL_B4_StrsH, TR.fvFL_CurStrsH); 			gsl_vector_float_memcpy(TR.fvFL_B4_StrsV, TR.fvFL_CurStrsV);  			gsl_vector_float_memcpy(TR.fvFL_B4_StrsN, TR.fvFL_CurStrsN);
			gsl_vector_float_set_zero(fvFG_Temp0);       							gsl_vector_float_set_zero(fvFG_Temp1); 
			gsl_vector_float_set_zero(fvFL_Temp0);       							gsl_vector_float_set_zero(fvFL_Temp1); 
			gsl_vector_float_set_zero(EQ.fvL_EQslipH);								gsl_vector_float_set_zero(EQ.fvL_EQslipV);	

			for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 				{  		TR.fvFL_B4_Fric[i]  = TR.fvFL_CurFric[i];			TR.fvFL_AccumSlp[i] = 0.0; 						}     	       
			for (i = 0; i < MD.iMaxMRFlgth; i++)						{		EQ.fvM_MRFvals[i]   = 0.0;							}		
			EQ.iActFPNum = 0;		          EQ.iMRFLgth = -1;					EQ.iTotlRuptT =-1;		
        	//------------------------------------------------------------------------------------------
			//------------------------------------------------------------------------------------------  
    		// EARTHQUAKE ITERATION LOOP STARTS
        	while (EQ.iStillOn == 1)
        	{  	
				EQ.iTotlRuptT++;    //continuously count time since initiation, set "ongoing" to FALSE => only if more slip on patches is added in the coming iteration, it gets to be reset            
                EQ.iStillOn = 0; 
				//----------------------------------------------------
				for (k = 0; k < MD.iSIZE; k++)	{			MD.ivF_ModOFFs[k] = 0;								}

            	for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
            	{  	iGlobPos = i + MD.ivF_START[MD.iRANK]; 	
					fTemp3   = sqrtf(gsl_vector_float_get(TR.fvFL_CurStrsH, i)*gsl_vector_float_get(TR.fvFL_CurStrsH, i) + gsl_vector_float_get(TR.fvFL_CurStrsV, i)*gsl_vector_float_get(TR.fvFL_CurStrsV, i));
					fTemp9   = (gsl_vector_float_get(TR.fvFL_CurStrsN, i) < 0.0 ) ? gsl_vector_float_get(TR.fvFL_CurStrsN, i) : 0.0 ;							gsl_vector_float_set(TR.fvFL_CurStrsN, i, fTemp9);
					//------------------------------------------------
					if (TR.ivFL_Activated[i] == 0)
                    {	fTemp0   = TR.fvFL_DynFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i)  -1.0*TR.fvFL_MeanSelfStiff[i] *TR.fvFL_CurDcVal[i];
						fTemp1   = TR.fvFL_StaFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i);
						fTemp0   = fMaxofTwo(fTemp0, fTemp1);
						TR.fvFL_CurFric[i]     = fTemp0/(-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i));
						TR.fvFL_TempRefFric[i] = TR.fvFL_CurFric[i];						
                    }
					else
					{	fTemp2             = sqrtf(gsl_vector_float_get(fvFG_Temp0, iGlobPos) *gsl_vector_float_get(fvFG_Temp0, iGlobPos)  +  gsl_vector_float_get(fvFG_Temp1, iGlobPos) *gsl_vector_float_get(fvFG_Temp1, iGlobPos));
						TR.fvFL_CurFric[i] = GetUpdatedFriction(TR.ivFL_StabT[i] , TR.fvFL_B4_Fric[i], TR.fvFL_TempRefFric[i], TR.fvFL_CurFric[i], TR.fvFL_DynFric[i], TR.ivFL_FricLaw[i], TR.fvFL_CurDcVal[i], TR.fvFL_AccumSlp[i], fTemp2, MD.fHealFact);
					}
					fTemp4 = fTemp3 - (TR.fvFL_CurFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i));
					//------------------------------------------------
                	if (  ( (TR.ivFL_Activated[i] == 0) && (fTemp4 > TR.fvFL_CutStress[i]))   || (TR.ivFL_Activated[i] == 1)  )
                	{  	
						if (TR.ivFL_Activated[i] == 0)                  {    TR.ivFL_Activated[i] = 1;           TR.ivFL_Ptch_t0[i] = EQ.iTotlRuptT;           			EQ.iActFPNum++;					}
		
                    	if (fTemp4 > TR.fvFL_CutStress[i])                      
                    	{  	EQ.iStillOn = 1;                       
                        	EQ.iMRFLgth = EQ.iTotlRuptT;   						
							fTemp5      = -1.0*(fTemp4/fTemp3 *gsl_vector_float_get(TR.fvFL_CurStrsH, i)) / TR.fvFL_SelfStiffStk[i]; // slip amount to release excess horizontal shear stress
                			fTemp6      = -1.0*(fTemp4/fTemp3 *gsl_vector_float_get(TR.fvFL_CurStrsV, i)) / TR.fvFL_SelfStiffDip[i]; // slip amount to release excess vertical shear stress 
							fTemp7      = sqrtf(fTemp5*fTemp5 +fTemp6*fTemp6); //the slip amount needed to release the excess...  
							
							if (TR.fvFL_AccumSlp[i] == 0.0)				{	TR.fvFL_TempRefFric[i] = TR.fvFL_CurFric[i];					}
							TR.fvFL_AccumSlp[i] += fTemp7;

							gsl_vector_float_set(fvFL_Temp0,   i, fTemp5); //also put that slip into "total slip at source patch" list => for event slip
							gsl_vector_float_set(fvFL_Temp1,   i, fTemp6); //so, this will go into EVENTslip...

							if (EQ.iMRFLgth < MD.iMaxMRFlgth)	//write out the moment-rate-function, but only to a maximum of maxMRFlength => can still do the earthquake but cutoff the MRF...
                        	{   EQ.fvM_MRFvals[EQ.iMRFLgth]    += fTemp7 *TR.fvFL_Area[i] *MD.fShearMod; //this is done locally here, will be synchronized once the whole EQ is over!
                    	}   }
                    	else //this is necessary here b/c I can get into that if-statement b/c I got activated but might not have any amount of slip left => set value to zero...
                    	{ 	if (TR.ivFL_StabT[i] != 3)					{	TR.fvFL_AccumSlp[i] = 0.0;			}
							gsl_vector_float_set(fvFL_Temp0,   i, 0.0);
                            gsl_vector_float_set(fvFL_Temp1,   i, 0.0);
				    }   }
                    else
                    {  	if (TR.ivFL_StabT[i] != 3)						{	TR.fvFL_AccumSlp[i] = 0.0;			}
						gsl_vector_float_set(fvFL_Temp0,   i, 0.0); //all this is necessary b/c I cannot reset these TempVectors to zero in a single statement; reason is that it would
				        gsl_vector_float_set(fvFL_Temp1,   i, 0.0); //be needed/contains info for case that velocity weakening is used...; have to make sure that both tempvectors are fully defined/rewritten "by hand"
				}	}
				if (EQ.iStillOn == 1)			{			MD.ivF_ModOFFs[MD.iRANK] = MD.ivF_OFFSET[MD.iRANK];					gsl_vector_float_add(EQ.fvL_EQslipH, fvFL_Temp0);			gsl_vector_float_add(EQ.fvL_EQslipV, fvFL_Temp1);							}
				//-----------------------------------------------------------------------
				MPI_Allreduce( MPI_IN_PLACE, &EQ.iStillOn, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
				
				if (EQ.iStillOn == 1)
				{	MPI_Allreduce(MPI_IN_PLACE, MD.ivF_ModOFFs, MD.iSIZE, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
					gsl_vector_float_set_zero(fvFG_Temp0);       									MPI_Allgatherv(fvFL_Temp0->data, MD.ivF_ModOFFs[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_ModOFFs, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);	
					gsl_vector_float_set_zero(fvFG_Temp1);   										MPI_Allgatherv(fvFL_Temp1->data, MD.ivF_ModOFFs[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_ModOFFs, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
            		//-----------------------------------------------------------------------
					gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SS, fvFG_Temp0, 0.0, fvFL_Temp0); 		gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SD, fvFG_Temp0, 0.0, fvFL_Temp1); 		gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_SO, fvFG_Temp0, 0.0, fvFL_Temp2); 
					gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);								gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);								gsl_vector_float_add(TR.fvFL_CurStrsN, fvFL_Temp2);
					gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DS, fvFG_Temp1, 0.0, fvFL_Temp0);			gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DD, fvFG_Temp1, 0.0, fvFL_Temp1);			gsl_blas_sgemv(CblasTrans, 1.0, K.FFs_DO, fvFG_Temp1, 0.0, fvFL_Temp2);
					gsl_vector_float_add(TR.fvFL_CurStrsH, fvFL_Temp0);								gsl_vector_float_add(TR.fvFL_CurStrsV, fvFL_Temp1);								gsl_vector_float_add(TR.fvFL_CurStrsN, fvFL_Temp2);
				}
				if (EQ.iTotlRuptT >= MD.iMaxMRFlgth)		{  		EQ.iStillOn     = 0;         } 			//a hard stop	
			}
			//-----------------------------------------------------------------------
			fTemp0       = 0.0; //this is temp/test magnitude (i.e., combined seismic potential)
			//----------------------
    		for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
        	{ 	if (TR.ivFL_Activated[i] == 1)     
            	{ 	fTemp3  = sqrtf(gsl_vector_float_get(EQ.fvL_EQslipH, i)*gsl_vector_float_get(EQ.fvL_EQslipH, i) + gsl_vector_float_get(EQ.fvL_EQslipV, i)*gsl_vector_float_get(EQ.fvL_EQslipV, i));
					fTemp0 += fTemp3*TR.fvFL_Area[i]; //this is seis. potential of that patch (added to other patches from same ranke)
        	}   } 
			//-----------------------------------------------------------------------
			MPI_Allreduce(MPI_IN_PLACE,		&fTemp0, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); //combined seismic potential that is released in this event
			//------------------------------------	
            fTemp1 = (log10f(fTemp0*MD.fShearMod)-9.1)/1.5; 	
           	//-----------------------------------------------------------------------
			//-----------------------------------------------------------------------
			if (  (MD.iUseProp == 1) && (fTemp1 >= MD.fMinMag4Prop) )
			{	//------------------------------------		 
				//------------------------------------
                gsl_vector_float_memcpy(TR.fvFL_CurStrsH, TR.fvFL_B4_StrsH); //reset stress to pre-EQ kind...and also all the other things to zero i.e., pre-rupture conditions.
			    gsl_vector_float_memcpy(TR.fvFL_CurStrsV, TR.fvFL_B4_StrsV);
                gsl_vector_float_memcpy(TR.fvFL_CurStrsN, TR.fvFL_B4_StrsN);
                //------------------------------------		
        	    gsl_matrix_float_set_zero(EQ.fmFGL_STF_H);			gsl_matrix_float_set_zero(EQ.fmFGL_STF_V);
                gsl_matrix_int_set_zero(TR.imFGL_NextP);			gsl_matrix_int_set_zero(TR.imFGL_NextS);
				gsl_vector_float_set_zero(EQ.fvL_EQslipH);			gsl_vector_float_set_zero(EQ.fvL_EQslipV);
				gsl_vector_float_set_zero(fvFL_Temp0);              gsl_vector_float_set_zero(fvFL_Temp1);            
                //------------------------------------		
                for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 			{   TR.fvFL_CurFric[i]  = TR.fvFL_B4_Fric[i];					TR.fvFL_AccumSlp[i] = 0.0;				TR.ivFL_Ptch_t0[i]   = 0;			TR.ivFL_Activated[i] = 0;				}   
			    for (i = 0; i < MD.iMaxMRFlgth; i++)					{	EQ.fvM_MRFvals[i]   = 0.0;						        }		
                EQ.iStillOn = 1;			EQ.iActFPNum = 0;				EQ.iMRFLgth = -1;				EQ.iTotlRuptT =-1;			EQ.iEndCntr   = 0;							
			    //--------------------------------------------------------------    
				while (EQ.iStillOn == 1)
        		{	
					EQ.iTotlRuptT++;       
                	EQ.iStillOn = 0; //continuously count time since initiation, set "ongoing" to FALSE => only if more slip on patches is added, it gets to be reset
					gsl_vector_float_set_zero(fvFG_Temp0);              		gsl_vector_float_set_zero(fvFG_Temp1);    					gsl_vector_float_set_zero(fvFG_Temp2);   
					//----------------------------------------------------
					for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
            		{  	
						fTemp3 = sqrtf( gsl_vector_float_get(TR.fvFL_CurStrsH, i)* gsl_vector_float_get(TR.fvFL_CurStrsH, i) + gsl_vector_float_get(TR.fvFL_CurStrsV, i)*gsl_vector_float_get(TR.fvFL_CurStrsV, i));
						fTemp9 = (gsl_vector_float_get(TR.fvFL_CurStrsN, i) < 0.0 ) ? gsl_vector_float_get(TR.fvFL_CurStrsN, i) : 0.0 ;							gsl_vector_float_set(TR.fvFL_CurStrsN, i, fTemp9);
						//------------------------------------------------
						if (TR.ivFL_Activated[i] == 0)
                    	{	fTemp0   = TR.fvFL_DynFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i)  -1.0*TR.fvFL_MeanSelfStiff[i] *TR.fvFL_CurDcVal[i];
							fTemp1   = TR.fvFL_StaFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i);
							fTemp0   = fMaxofTwo(fTemp0, fTemp1);
							TR.fvFL_CurFric[i]     = fTemp0/(-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i));
							TR.fvFL_TempRefFric[i] = TR.fvFL_CurFric[i];
                    	}
						else
						{	fTemp2             = sqrtf(gsl_vector_float_get(fvFL_Temp0, i) *gsl_vector_float_get(fvFL_Temp0, i)  +  gsl_vector_float_get(fvFL_Temp1, i) *gsl_vector_float_get(fvFL_Temp1, i));
							TR.fvFL_CurFric[i] = GetUpdatedFriction(TR.ivFL_StabT[i] , TR.fvFL_B4_Fric[i], TR.fvFL_TempRefFric[i], TR.fvFL_CurFric[i], TR.fvFL_DynFric[i], TR.ivFL_FricLaw[i], TR.fvFL_CurDcVal[i], TR.fvFL_AccumSlp[i], fTemp2, MD.fHealFact);
						}
						fTemp4 = fTemp3 - TR.fvFL_CurFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN, i);

						//------------------------------------------------
                		if (  ( (TR.ivFL_Activated[i] == 0) && (fTemp4 > TR.fvFL_CutStress[i]))   || (TR.ivFL_Activated[i] == 1)  )
                		{  
							STFcnt = EQ.iTotlRuptT -TR.ivFL_Ptch_t0[i];
							if (TR.ivFL_Activated[i] == 0)                  {    STFcnt = 0;     	TR.ivFL_Activated[i] = 1;           TR.ivFL_Ptch_t0[i] = EQ.iTotlRuptT;            EQ.iActFPNum++;			}                                                        

                        	iTemp0 = STFcnt%MD.iMaxSTFlgth;   //gives me remainder of wPos/MaxSTFlength => allows me to loop/overwrite the STF vectors 
		
                    		if (fTemp4 > TR.fvFL_CutStress[i])                       
                    		{  	EQ.iStillOn = 1;                       
                        		EQ.iMRFLgth = EQ.iTotlRuptT;   						
								fTemp5      = -1.0*((fTemp4/fTemp3)*gsl_vector_float_get(TR.fvFL_CurStrsH, i)) /TR.fvFL_SelfStiffStk[i]; // slip amount to release excess horizontal shear stress
                				fTemp6      = -1.0*((fTemp4/fTemp3)*gsl_vector_float_get(TR.fvFL_CurStrsV, i)) /TR.fvFL_SelfStiffDip[i]; // slip amount to release excess vertical shear stress 
								fTemp7      = sqrtf(fTemp5*fTemp5 +fTemp6*fTemp6); //the slip amount needed to release the excess...  
							
								if (TR.fvFL_AccumSlp[i] == 0.0) 		{			TR.fvFL_TempRefFric[i] = TR.fvFL_CurFric[i];					}
							
								TR.fvFL_AccumSlp[i] += fTemp7;
								gsl_matrix_float_set(EQ.fmFGL_STF_H, i, iTemp0, fTemp5); //put into the STFmatrix
								gsl_matrix_float_set(EQ.fmFGL_STF_V, i, iTemp0, fTemp6);
								gsl_vector_float_set(fvFL_Temp0,  i, fTemp5); //also put that slip into "total slip at source patch" list => for event slip
								gsl_vector_float_set(fvFL_Temp1,  i, fTemp6); //so, this will go into EVENTslip...

								if (EQ.iMRFLgth < MD.iMaxMRFlgth)	//write out the moment-rate-function, but only to a maximum of maxMRFlength => can still do the earthquake but cutoff the MRF...
                        		{   EQ.fvM_MRFvals[EQ.iMRFLgth]    += fTemp7 *TR.fvFL_Area[i] *MD.fShearMod; //this is done locally here, will be synchronized once the whole EQ is over!
                    		}   }
                    		else //this is necessary here b/c I can get into that if-statement b/c I got activated but might not have any amount of slip left => set value to zero...
                    		{ 	if (TR.ivFL_StabT[i] !=3)					{	TR.fvFL_AccumSlp[i] = 0.0;			}
							    gsl_matrix_float_set(EQ.fmFGL_STF_H, i, iTemp0, 0.0);
						        gsl_matrix_float_set(EQ.fmFGL_STF_V, i, iTemp0, 0.0);// it is necessary b/c that position would remain its previous value, and I am looping over this stuff => would reuse previous slip amount   
								gsl_vector_float_set(fvFL_Temp0,     i, 0.0);
                            	gsl_vector_float_set(fvFL_Temp1,     i, 0.0);
				   	 		}   
							//-----------------------------------------------------------------------
							//-----------------------------------------------------------------------
							for (j = 0; j  < MD.iFPNum; j++) 
							{	//-----------------------------------------------------------------------
								//first the "P-wave" signal (PROjection of slip vector onto SrcRcvVect)
								iTemp0   = gsl_matrix_int_get(TR.imFGL_NextP, i, j);
								iTemp1   = gsl_matrix_int_get(TR.imFGL_TTP,   i, j); 		//for stress change calculation
								
								for (k = iTemp0; k <= (STFcnt - iTemp1); k++)				//2nd part is negative if signal has not arrived, then the loop is skipped
								{	iTemp2 = k%MD.iMaxSTFlgth; 								//this is the actual "read position" => where to look in the STFmatrix
									fTemp0 = gsl_matrix_float_get(EQ.fmFGL_STF_H, i, iTemp2); //this is the next horizontal slip
									fTemp1 = gsl_matrix_float_get(EQ.fmFGL_STF_V, i, iTemp2); //this is the next vertical slip

									if ((i + MD.ivF_START[MD.iRANK]) == j)
                            		{   fTemp10 = fTemp0*gsl_matrix_float_get(K.FFr_SS, i, j) + fTemp1*gsl_matrix_float_get(K.FFr_DS, i, j); //this is on "self", => use actual slip from STF to determine new stress state i.e., the stress change due to slip
										fTemp11 = fTemp0*gsl_matrix_float_get(K.FFr_SD, i, j) + fTemp1*gsl_matrix_float_get(K.FFr_DD, i, j);
										fTemp12 = 0.0;
									}
									else    
						    		{	//http://sites.science.oregonstate.edu/math/home/programs/undergrad/CalculusQuestStudyGuides/vcalc/dotprod/dotprod.html
										//https://en.wikipedia.org/wiki/Vector_projection
										fTemp9 = fTemp0*gsl_matrix_float_get(TR.fmFGL_SrcRcvH, i, j) + fTemp1*gsl_matrix_float_get(TR.fmFGL_SrcRcvV, i, j); //"normally", this should also include the normal component => but that one is always zero! => don't need it herethis is basically a vector projection, using scalar projection using dot product
										//fTemp9 is the length/projection of the slip vector (fTemp0 and fTemp1 being the horizontal and vertical component) onto the SrcRcv vector (which sits at source and points to receiver, is normalized; ALSO: that vector is in local coordinates of the source patch! -done in "AddMoreParas")	
										fTemp5 = fTemp9*gsl_matrix_float_get(TR.fmFGL_SrcRcvH, i, j); //this part here is actually a really cool step; fTemp9 is length of slip amount in direction of source-receiver vector (can be negative)
										fTemp6 = fTemp9*gsl_matrix_float_get(TR.fmFGL_SrcRcvV, i, j); //this "length" is multiplied with local orientation of source-receiver vector to get the transient slip values to be used...
										fTemp7 = fTemp9*gsl_matrix_float_get(TR.fmFGL_SrcRcvN, i, j); //these are currently slip values... => now need to convert to stress change

										fTemp10= fTemp5*gsl_matrix_float_get(K.FFr_SS, i, j) + fTemp6*gsl_matrix_float_get(K.FFr_DS, i, j) + fTemp7*gsl_matrix_float_get(K.FFr_OS, i, j);
										fTemp11= fTemp5*gsl_matrix_float_get(K.FFr_SD, i, j) + fTemp6*gsl_matrix_float_get(K.FFr_DD, i, j) + fTemp7*gsl_matrix_float_get(K.FFr_OD, i, j);
										fTemp12= fTemp5*gsl_matrix_float_get(K.FFr_SO, i, j) + fTemp6*gsl_matrix_float_get(K.FFr_DO, i, j) + fTemp7*gsl_matrix_float_get(K.FFr_OO, i, j);
									}	
									//remember, there is no actual (static) motion in normal direction, but a transient slip signal when vP/vS is used...
									fTemp2 = gsl_vector_float_get(fvFG_Temp0, j); 				gsl_vector_float_set(fvFG_Temp0, j, (fTemp2 + fTemp10));
									fTemp3 = gsl_vector_float_get(fvFG_Temp1, j);				gsl_vector_float_set(fvFG_Temp1, j, (fTemp3 + fTemp11));
									fTemp4 = gsl_vector_float_get(fvFG_Temp2, j); 				gsl_vector_float_set(fvFG_Temp2, j, (fTemp4 + fTemp12));
								}
								iTemp0 = (STFcnt - iTemp1) >= 0 ? (STFcnt - iTemp1 +1) : 0; 
								gsl_matrix_int_set(TR.imFGL_NextP, i, j, iTemp0);
								//-----------------------------------------------------------------------
								//second the "S-wave" signal (REjection of slip vector onto SrcRcvVect)
								iTemp0   = gsl_matrix_int_get(TR.imFGL_NextS, i, j); 
								iTemp1   = gsl_matrix_int_get(TR.imFGL_TTS,   i, j);
								for (k = iTemp0; k <= (STFcnt - iTemp1); k++)
								{	iTemp2 = k%MD.iMaxSTFlgth; 								//this is the actual "read position" => where to look in the STFmatrix
    								fTemp0 = gsl_matrix_float_get(EQ.fmFGL_STF_H, i, iTemp2); //this is the next horizontal slip
									fTemp1 = gsl_matrix_float_get(EQ.fmFGL_STF_V, i, iTemp2); //this is the next vertical slip

									if (i + MD.ivF_START[MD.iRANK] == j)
                            		{	fTemp10= 0.0;
										fTemp11= 0.0;
										fTemp12= 0.0;	
									}
									else
									{	fTemp9 = fTemp0*gsl_matrix_float_get(TR.fmFGL_SrcRcvH, i, j) + fTemp1*gsl_matrix_float_get(TR.fmFGL_SrcRcvV,  i, j);
								
										fTemp5 = fTemp0 - fTemp9*gsl_matrix_float_get(TR.fmFGL_SrcRcvH, i, j);
										fTemp6 = fTemp1 - fTemp9*gsl_matrix_float_get(TR.fmFGL_SrcRcvV, i, j);
										fTemp7 = 0.0    - fTemp9*gsl_matrix_float_get(TR.fmFGL_SrcRcvN, i, j);

										fTemp10= fTemp5*gsl_matrix_float_get(K.FFr_SS, i, j) + fTemp6*gsl_matrix_float_get(K.FFr_DS, i, j) + fTemp7*gsl_matrix_float_get(K.FFr_OS, i, j);
										fTemp11= fTemp5*gsl_matrix_float_get(K.FFr_SD, i, j) + fTemp6*gsl_matrix_float_get(K.FFr_DD, i, j) + fTemp7*gsl_matrix_float_get(K.FFr_OD, i, j);
										fTemp12= fTemp5*gsl_matrix_float_get(K.FFr_SO, i, j) + fTemp6*gsl_matrix_float_get(K.FFr_DO, i, j) + fTemp7*gsl_matrix_float_get(K.FFr_OO, i, j);
									}
									fTemp2 = gsl_vector_float_get(fvFG_Temp0, j); 				gsl_vector_float_set(fvFG_Temp0, j, (fTemp2 + fTemp10));
									fTemp3 = gsl_vector_float_get(fvFG_Temp1, j);				gsl_vector_float_set(fvFG_Temp1, j, (fTemp3 + fTemp11));
									fTemp4 = gsl_vector_float_get(fvFG_Temp2, j); 				gsl_vector_float_set(fvFG_Temp2, j, (fTemp4 + fTemp12));	
								}
                     			iTemp0 = (STFcnt - iTemp1) >= 0 ? (STFcnt - iTemp1 +1) : 0;   
								gsl_matrix_int_set(TR.imFGL_NextS, i, j, iTemp0);
								//-----------------------------------------------------------------------
						}	}
                    	else
                    	{  	if (TR.ivFL_StabT[i] !=3)					{	TR.fvFL_AccumSlp[i] = 0.0;			}
						    gsl_vector_float_set(fvFL_Temp0,   i, 0.0); //all this is necessary b/c I cannot reset these TempVectors to zero in a single statement; reason is that it would
				        	gsl_vector_float_set(fvFL_Temp1,   i, 0.0); //be needed/contains info for case that velocity weakening is used...; have to make sure that both tempvectors are fully defined/rewritten "by hand"
					}	}
					//-----------------------------------------------------------------------
					gsl_vector_float_add(EQ.fvL_EQslipH, fvFL_Temp0);			gsl_vector_float_add(EQ.fvL_EQslipV, fvFL_Temp1);
					MPI_Allreduce( MPI_IN_PLACE, fvFG_Temp0->data, MD.iFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce( MPI_IN_PLACE, fvFG_Temp1->data, MD.iFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce( MPI_IN_PLACE, fvFG_Temp2->data, MD.iFPNum, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
					MPI_Allreduce( MPI_IN_PLACE, &EQ.iStillOn, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
					for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++)				
					{	iGlobPos = i + MD.ivF_START[MD.iRANK]; 
						fTemp0 = gsl_vector_float_get(TR.fvFL_CurStrsH, i) + gsl_vector_float_get(fvFG_Temp0, iGlobPos);				gsl_vector_float_set(TR.fvFL_CurStrsH, i, fTemp0);
						fTemp1 = gsl_vector_float_get(TR.fvFL_CurStrsV, i) + gsl_vector_float_get(fvFG_Temp1, iGlobPos);				gsl_vector_float_set(TR.fvFL_CurStrsV, i, fTemp1);
						fTemp2 = gsl_vector_float_get(TR.fvFL_CurStrsN, i) + gsl_vector_float_get(fvFG_Temp2, iGlobPos);				gsl_vector_float_set(TR.fvFL_CurStrsN, i, fTemp2);
					}

					if (EQ.iStillOn == 0) // make sure that all the signal is out of the system => even if no slip occurred in last step on any patch; there may still be stress in the system that has not reached a receiver => wait until they all got their share 
            		{ 	EQ.iEndCntr += 1;
                	   	if (EQ.iEndCntr < MD.iMaxSTFlgth)    	{  		EQ.iStillOn     = 1;         } 		
               		}     
            		else 
					{  	EQ.iEndCntr  = 0;              
					}	
					if (EQ.iTotlRuptT >= MD.iMaxMRFlgth)		{  		EQ.iStillOn     = 0;         } 			//a hard stop
				}
			}
			//-----------------------------------------------------------------------
			//-----------------------------------------------------------------------	
			EQ.iCmbFPNum = 0; 
			MPI_Allreduce(&EQ.iActFPNum, 	&EQ.iCmbFPNum,    1      , MPI_INT,   MPI_SUM, MPI_COMM_WORLD);				    
            //-----------------------------------------------------------------------	
		    if (EQ.iCmbFPNum >= iMinPtch4Cat)
			{	
                MD.iEQcntr++;  			EQ.iActFPNum = 0;					EQ.fMaxSlip  = 0.0;		   		 	EQ.fMaxDTau = 0.0;
				fTemp0       = 0.0;	
				EQ.iMRFLgth  = (EQ.iMRFLgth < MD.iMaxMRFlgth) ? EQ.iMRFLgth : MD.iMaxMRFlgth;

				for (i = 0; i < MD.iSIZE; i++)          		{      	EQ.ivR_WrtStrtPos[i] = 0;   						}     
				//----------------------
    			for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
        		{  	if (TR.ivFL_Activated[i] == 1)     
            		{ 	fTemp1  = gsl_vector_float_get(EQ.fvL_EQslipH,   i); //horizontal slip of patch
						fTemp2  = gsl_vector_float_get(EQ.fvL_EQslipV,   i); //vertical slip of patch
						fTemp3  = gsl_vector_float_get(TR.fvFL_B4_StrsH, i);				fTemp4  = gsl_vector_float_get(TR.fvFL_B4_StrsV, i);
						fTemp5  = gsl_vector_float_get(TR.fvFL_CurStrsH, i);				fTemp6  = gsl_vector_float_get(TR.fvFL_CurStrsV, i);
						fTemp7  = sqrtf(fTemp1*fTemp1 +fTemp2*fTemp2);
						fTemp0                        += fTemp7*TR.fvFL_Area[i]; //this is seis. potential of that patch (added to other patches from same ranke)
						
						EQ.ivL_ActPtchID[EQ.iActFPNum] =  i + MD.ivF_START[MD.iRANK]; 
						EQ.ivL_t0ofPtch[EQ.iActFPNum]  = TR.ivFL_Ptch_t0[i];
						EQ.fvL_PtchSlpH[EQ.iActFPNum]  = fTemp1;
						EQ.fvL_PtchSlpV[EQ.iActFPNum]  = fTemp2;
						EQ.fvL_PtchDTau[EQ.iActFPNum]  = sqrtf(fTemp3*fTemp3 +fTemp4*fTemp4) - sqrtf(fTemp5*fTemp5 +fTemp6*fTemp6); //stress DROP is therefore POSITIVE
						EQ.ivL_StabType[EQ.iActFPNum]  = TR.ivFL_StabT[i];
						EQ.fMaxDTau                    =      (EQ.fvL_PtchDTau[EQ.iActFPNum] > EQ.fMaxDTau) ?      EQ.fvL_PtchDTau[EQ.iActFPNum] : EQ.fMaxDTau;
						EQ.fMaxSlip                    = (sqrt(fTemp1*fTemp1 +fTemp2*fTemp2) > EQ.fMaxSlip) ? sqrt(fTemp1*fTemp1 +fTemp2*fTemp2) : EQ.fMaxSlip;  
						EQ.iActFPNum++;


				}   } 	
				//-----------------------------------------------------------------------
				MPI_Allreduce(MPI_IN_PLACE,		&fTemp0,          1      , MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD); //combined seismic potential that is released in this event
				MPI_Allreduce(MPI_IN_PLACE,		&EQ.iMRFLgth,     1      , MPI_INT,   MPI_MAX, MPI_COMM_WORLD); //the length of the MRF
				MPI_Allreduce(MPI_IN_PLACE,		&EQ.fMaxSlip,     1      , MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD); //this is maximum slip
				MPI_Allreduce(MPI_IN_PLACE,		&EQ.fMaxDTau,     1      , MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD); //this is maximum stress drop
				MPI_Allreduce(MPI_IN_PLACE, EQ.fvM_MRFvals,MD.iMaxMRFlgth, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
				//------------------------------------			
				//------------------------------------		
				fTemp1 = (log10f(fTemp0*MD.fShearMod)-9.1)/1.5; 
				//------------------------------------		
				//------------------------------------		
                MPI_Allgather(&EQ.iActFPNum, 1, MPI_INT, EQ.ivR_WrtStrtPos, 1, MPI_INT,  MPI_COMM_WORLD);   
				for (i = 1;     i < MD.iSIZE; i++)           {    EQ.ivR_WrtStrtPos[i] += EQ.ivR_WrtStrtPos[i-1];             }    
        		for (i = (MD.iSIZE-1); i > 0; i--)           {    EQ.ivR_WrtStrtPos[i]  = EQ.ivR_WrtStrtPos[i-1];             }    
        		EQ.ivR_WrtStrtPos[0]  = 0;
				//-----------------------------------------------------------------------
				if ((MD.iRANK == 0)&&(iPlot2Screen == 1) && (fTemp1 > 5.0))
				{   dEQtime     = (double)(MD.lTimeYears*MD.fISeisTStp);
					dEQtimeDiff = (double)((MD.lTimeYears - lPrevEQtime)*MD.fISeisTStp);
				
        			lPrevEQtime = MD.lTimeYears; 
					iTemp0 = (int)fTemp1 -4; //just for nicer plotting...
					fprintf(stdout,"%6d  Earthquake time %5.2f    act patches: %5d   MRF length: %4d   MaxSlip: %4.2f     MaxStressDrop: %4.2f     TimeDiff %4.3f",MD.iEQcntr, dEQtime,  EQ.iCmbFPNum, EQ.iMRFLgth, EQ.fMaxSlip, EQ.fMaxDTau, dEQtimeDiff);
					for (i = 0; i < iTemp0; i++) 	{		fprintf(stdout,"    ");			}
					fprintf(stdout,"Magn: %3.2f\n",fTemp1);              
				}
      			//-----------------------------------------------------------------------
        		if (MD.iRANK == 0)
        		{	MPI_File_write_at(fp_MPIOUT,        0,                                                   &MD.iEQcntr,        1,       MPI_INT,    &STATUS);
				    MPI_File_write_at(fp_MPIOUT,   OFFSETall,                                                &dEQtime,           1,       MPI_DOUBLE, &STATUS);  //Earthquake time
            	    MPI_File_write_at(fp_MPIOUT,   OFFSETall + sizeof(double),                               &fTemp1,            1,       MPI_FLOAT,  &STATUS);  //Earthquake magnitude
            	    MPI_File_write_at(fp_MPIOUT,   OFFSETall + sizeof(double) +sizeof(float),                &EQ.iCmbFPNum,      1,       MPI_INT,    &STATUS);  //#of fault patches participating in EQ
            	    MPI_File_write_at(fp_MPIOUT,   OFFSETall + sizeof(double) +sizeof(float) +1*sizeof(int), &EQ.iMRFLgth,       1,       MPI_INT,    &STATUS);  //length of moment rate function "time steps"
            	    MPI_File_write_at(fp_MPIOUT,   OFFSETall + sizeof(double) +sizeof(float) +2*sizeof(int), EQ.fvM_MRFvals, EQ.iMRFLgth, MPI_FLOAT,  &STATUS);  //moment rate function values..
				}
				OFFSETall += 2*sizeof(int) +(1 +EQ.iMRFLgth)*sizeof(float) +sizeof(double);
				//--------------------------------------------
				MPI_File_write_at(fp_MPIOUT,   OFFSETall +EQ.ivR_WrtStrtPos[MD.iRANK]*sizeof(int),  EQ.ivL_ActPtchID,EQ.iActFPNum, MPI_INT,   &STATUS);
        	    OFFSETall += EQ.iCmbFPNum*sizeof(int);
        	    MPI_File_write_at(fp_MPIOUT,   OFFSETall +EQ.ivR_WrtStrtPos[MD.iRANK]*sizeof(int),  EQ.ivL_t0ofPtch, EQ.iActFPNum, MPI_INT,   &STATUS);
        	    OFFSETall += EQ.iCmbFPNum*sizeof(int);
        	    MPI_File_write_at(fp_MPIOUT,   OFFSETall +EQ.ivR_WrtStrtPos[MD.iRANK]*sizeof(float),EQ.fvL_PtchDTau, EQ.iActFPNum, MPI_FLOAT, &STATUS);
        	    OFFSETall += EQ.iCmbFPNum*sizeof(float);
        	    MPI_File_write_at(fp_MPIOUT,   OFFSETall +EQ.ivR_WrtStrtPos[MD.iRANK]*sizeof(float),EQ.fvL_PtchSlpH, EQ.iActFPNum, MPI_FLOAT, &STATUS);
        	    OFFSETall += EQ.iCmbFPNum*sizeof(float);
        	    MPI_File_write_at(fp_MPIOUT,   OFFSETall +EQ.ivR_WrtStrtPos[MD.iRANK]*sizeof(float),EQ.fvL_PtchSlpV, EQ.iActFPNum, MPI_FLOAT, &STATUS);
        	    OFFSETall += EQ.iCmbFPNum*sizeof(float);
        	    MPI_File_write_at(fp_MPIOUT,   OFFSETall +EQ.ivR_WrtStrtPos[MD.iRANK]*sizeof(int),  EQ.ivL_StabType, EQ.iActFPNum, MPI_INT,   &STATUS);
			    OFFSETall += EQ.iCmbFPNum*sizeof(int);

			    MPI_Barrier( MPI_COMM_WORLD );
		    }
            //-----------------------------------------------------------------------------------------------------------------------
		    //-----------------------------------------------------------------------------------------------------------------------
            //-----------------------------------------------------------------------------------------------------------------------
		    //-----------------------------------------------------------------------------------------------------------------------
		    for (i = 0; i < MD.ivF_OFFSET[MD.iRANK]; i++) 
            {  	//-----------------------------------------
			    if (TR.ivFL_Activated[i] == 1)     
           	    {	if (MD.iChgBtwEQs == 1)
				    {	fTemp0               = (float)(gsl_rng_uniform(fRandN) *2.0 -1.0); //is a random number between -1 and 1
					    TR.fvFL_CurDcVal[i]  = TR.fvFL_RefDcVal[i]  *(1.0 + TR.fvFL_RefDcVal_vari[i]*fTemp0);
					    fTemp0               = (float)(gsl_rng_uniform(fRandN) *2.0 -1.0); //is a random number between -1 and 1
        			    TR.fvFL_StaFric[i]   = TR.fvFL_RefStaFric[i]*(1.0 + TR.fvFL_RefStaFric_vari[i]*fTemp0); 
					    fTemp0               = (float)(gsl_rng_uniform(fRandN) *2.0 -1.0); //is a random number between -1 and 1
					    fTemp1               = (TR.fvFL_RefStaFric[i] - TR.fvFL_RefDynFric[i])/TR.fvFL_RefStaFric[i]; //reference friction change as fraction of static coefficient
					    fTemp2               = fTemp1*(1.0 + TR.fvFL_RefDynFric_vari[i]*fTemp0);
        			    TR.fvFL_DynFric[i]   = TR.fvFL_StaFric[i]*(1.0 - fTemp2);
        			    if (TR.ivFL_FricLaw[i] == 1)		{	TR.fvFL_CutStress[i]     = fabs(TR.fvFL_MeanSelfStiff[i] * MD.fUnitSlip *1.0    *MD.fDcfrac4Threshold);			}
   						else                                {	TR.fvFL_CutStress[i]     = fabs(TR.fvFL_MeanSelfStiff[i] * TR.fvFL_CurDcVal[i]  *MD.fDcfrac4Threshold);			}   
        			}	
					TR.ivFL_Ptch_t0[i]   = 0;			
					TR.ivFL_Activated[i] = 0;	
				}
				TR.fvFL_CurFric[i]   = TR.fvFL_StaFric[i];	
				//-----------------------------------------
				gsl_vector_float_set(TR.fvFL_CurStrsN, i, TR.fvFL_RefNrmStrs[i]);	
				//-----------------------------------------
				fTemp2 = (TR.fvFL_DynFric[i] - TR.fvFL_StaFric[i]) *-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN,i); //gives a negative shear stress b/c of (dynF - staF)
        		fTemp3 = fTemp2/TR.fvFL_MeanSelfStiff[i]; // b/c "self" has negative sign, the two negatives give a positive slip amount... (for weakening case, when dyn < stat fric) 
 				if (fTemp3 > TR.fvFL_CurDcVal[i]) 	{ 	TR.ivFL_StabT[i] = 1; 		}
        		else	
				{   if (fTemp2 < 0.0)  				{ 	TR.ivFL_StabT[i] = 2; 		}
           			else 							{   TR.ivFL_StabT[i] = 3; 		}      
				}
				//-----------------------------------------
				fTemp0 = sqrtf(gsl_vector_float_get(TR.fvFL_CurStrsH,i)*gsl_vector_float_get(TR.fvFL_CurStrsH,i) + gsl_vector_float_get(TR.fvFL_CurStrsV,i)*gsl_vector_float_get(TR.fvFL_CurStrsV,i) );
				fTemp1 = TR.fvFL_CurFric[i]*-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN,i);
				//-----------------------------------------
			    if (MD.iUsePSeis == 1)
				{	if (fTemp0 > fTemp1)
					{	if (TR.ivFL_StabT[i] == 1)		{ 			TR.ivFL_StabT[i] = 2;				}
						TR.fvFL_CurFric[i]    = fTemp0/(-1.0*gsl_vector_float_get(TR.fvFL_CurStrsN,i));
						TR.fvFL_PSeis_T0_F[i] = TR.fvFL_CurFric[i] - TR.fvFL_StaFric[i];
				}	}
				else
				{	if (fTemp0 > fTemp1)
					{	if (TR.ivFL_StabT[i] == 1)		{ 			TR.ivFL_StabT[i] = 2;				}
						TR.fvFL_PSeis_T0_F[i] = 0.0;
				}	}
			}
			//-----------------------------------------
			//-----------------------------------------
			MD.fPSeis_Step = 0.0;
		    //--------------------------------------------------------------    
			if ((MD.iAlsoBoundForPostSeis == 1) && (MD.iBPNum > 0) && (MD.iUsePSeis == 1))
			{	
				
				MPI_Allgatherv(EQ.fvL_EQslipH->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp0->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);
				MPI_Allgatherv(EQ.fvL_EQslipV->data, MD.ivF_OFFSET[MD.iRANK], MPI_FLOAT, fvFG_Temp1->data, MD.ivF_OFFSET, MD.ivF_START, MPI_FLOAT, MPI_COMM_WORLD);

				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_SS, fvFG_Temp0, 0.0, fvBL_Temp0); 					gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);
				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_SD, fvFG_Temp0, 0.0, fvBL_Temp1);					gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);
				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_SO, fvFG_Temp0, 0.0, fvBL_Temp2);					gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);
				
				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_DS, fvFG_Temp1, 0.0, fvBL_Temp0);					gsl_vector_float_add(TR.fvBL_CurStrsH, fvBL_Temp0);			
				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_DD, fvFG_Temp1, 0.0, fvBL_Temp1);					gsl_vector_float_add(TR.fvBL_CurStrsV, fvBL_Temp1);
				gsl_blas_sgemv(CblasTrans, 1.0, K.FB_DO, fvFG_Temp1, 0.0, fvBL_Temp2);					gsl_vector_float_add(TR.fvBL_CurStrsN, fvBL_Temp2);

				for (i = 0; i < MD.ivB_OFFSET[MD.iRANK]; i++) 
				{	TR.fvBL_PSeis_T0_S[i] = sqrtf(gsl_vector_float_get(TR.fvBL_CurStrsH, i)*gsl_vector_float_get(TR.fvBL_CurStrsH, i) + gsl_vector_float_get(TR.fvBL_CurStrsV, i)*gsl_vector_float_get(TR.fvBL_CurStrsV, i) );
					TR.fvBL_PSeis_T0_N[i] = fabs(gsl_vector_float_get(TR.fvBL_CurStrsN, i));
			}	}
			//--------------------------------------------------------------    
        	MPI_Barrier( MPI_COMM_WORLD );  
        }
	}
    //-------------------------------------------------------------------------------------
    //-------------------------------------------------------------------------------------
	if (MD.iRANK == 0)		{			fclose(fpPre);																				}
	if (MD.iRANK == 0)		{			MPI_File_write_at(fp_MPIOUT, 0, &MD.iEQcntr,   1, MPI_INT,      &STATUS);					}

 	MPI_File_close(&fp_MPIOUT);
	//-------------------------------------------------------------------------------------
    if (MD.iRANK == 0)             
    {   timer = clock() - timer;     
        double time_taken;
        time_taken  = ((double)timer)/CLOCKS_PER_SEC;
        time_taken /= 60.0;
        fprintf(stdout,"Total RunTime in minutes: %6.2f\n",time_taken);
		fprintf(stdout,"Times iSize =>  total of %6.2f CPU hours\n",(time_taken*(float)MD.iSIZE)/60.0);
    }
   	//-------------------------------------------------------------------------------------
	MPI_Finalize();
    return 0;
}
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
float GetUpdatedFriction(int StabType, float B4Fric, float RefFric, float CurFric, float DynFric, int FricLaw, float CurD_c, float AccumSlip, float PrevSlip, float HealingFact)
{	float NextFric = CurFric,		fTemp;
	//-----------------------------------
	if (PrevSlip > 0.0)	
	{
		if      (FricLaw == 1)		{				NextFric = DynFric;																																}
		else if (FricLaw == 2)		{				fTemp    = AccumSlip/CurD_c < 1.0 ? AccumSlip/CurD_c : 1.0;						NextFric = DynFric + (1.0 - fTemp)*(RefFric - DynFric);					}
		else if (FricLaw == 3)		{				fTemp    = PrevSlip/(CurD_c/10.0)  < 1.0 ? PrevSlip/(CurD_c/10.0)  : 1.0;		NextFric = DynFric + (1.0 - fTemp)*(RefFric - DynFric);					}
    }
	else
	{	if (StabType == 3)			{				NextFric = CurFric;												}
		else						{				NextFric = CurFric + HealingFact*(B4Fric -CurFric);				}
	}

	return NextFric;
}
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
float fMaxofTwo(float fTemp0, float fTemp1)
{	float fMaxVal;
	fMaxVal = (fTemp0 >= fTemp1) ? fTemp0 : fTemp1;	
	return fMaxVal;
}
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
void InitializeVariables(struct MDstruct *MD, struct TRstruct *TR, struct VTstruct *VT, struct EQstruct *EQ, struct Kstruct *K, int iPlot2Screen, char **argv)
{	int     i;
	float 	fTemp0;
	char 	ctempVals[512],     cFile1_In[512], 		cFile2_In[512],			cAppend[512];
 	FILE 	*fpIn,  *fp1,  *fp2;			
	//------------------------------------------------------------------
    strcpy(cFile1_In,  argv[1]);// opening and reading the "run file" that contains specifics about this model run
    if ((fpIn = fopen(cFile1_In,"r")) == NULL)      {   fprintf(stdout,"Error -cant open %s file in initializeVariables function \n",cFile1_In);      exit(10);     }

    if (fgets(ctempVals, 512, fpIn) != NULL)        {                       	                             }                            
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %s", MD->cInputName);          }             
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %d", &MD->iRunNum);            }
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %d", &MD->iUsePSeis);          } 
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %d", &MD->iUseProp);           }           
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %f", &MD->fMinMag4Prop);       }
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %f", &MD->fISeisStep);         }           
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %f", &MD->fAftrSlipTime);      }  
	if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %f", &MD->fDeepRelaxTime);     }            
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %f", &MD->fRecLgth);           }
	if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %f", &MD->fHealFact);          }
    if (fgets(ctempVals, 512, fpIn) != NULL)        {   sscanf(ctempVals,"%*s %d", &MD->iSeedStart);         }
    if (MD->iSeedStart <= 0)				        {   MD->iSeedStart = rand();                             }
	MD->fHealFact = (MD->fHealFact < 0.0)	? 0.0 : MD->fHealFact;
	MD->fHealFact = (MD->fHealFact > 1.0)	? 1.0 : MD->fHealFact;
	fclose(fpIn);
	//------------------------------------------------------------------
    //-----------------------------------------------------------------
	strcpy(cFile1_In,MD->cInputName);      strcat(cFile1_In,"_");               	sprintf(cAppend, "%d",MD->iRunNum); 	strcat(cFile1_In,cAppend);     	strcat(cFile1_In,"_Roughn.dat");   
    if ((fp1 = fopen(cFile1_In,"rb"))     == NULL)     { 	printf("Error -cant open *_Roughn.dat file.  in Initialize variables...   %s\n", cFile1_In);      exit(10);     }
	if (fread(&MD->iFPNum,   sizeof(int),   1, fp1) != 1)  		{	exit(10);	}// this is currently read PatchNum 
    if (fread(&MD->iFVNum,   sizeof(int),   1, fp1) != 1) 		{	exit(10);	}// this is currently read VertexNum
    if (fread(&MD->fFltLegs, sizeof(float), 1, fp1) != 1)  		{	exit(10);	}// this is currently read PatchNum 
    if (fread(&fTemp0,       sizeof(float), 1, fp1) != 1)  		{	exit(10);	}// this is currently read PatchNum 
    fclose(fp1);
    //-----------------------------------------------------------------
	strcpy(cFile2_In,MD->cInputName);      strcat(cFile2_In,"_BNDtrig.dat");
    if ((fp2 = fopen(cFile2_In,"rb"))     == NULL)     
    {    printf("Warning -cant open *_BNDtrig.dat file. in Initialize variables...\n");      
   	 	MD->iBPNum   = 0;
     	MD->iBVNum   = 0;
     	MD->fBndLegs = 1.0e+9;
    }
	else
	{ 	if (fread(&MD->iBPNum,   sizeof(int),   1, fp2) != 1)  		{	exit(10);	}// this is currently read PatchNum 
    	if (fread(&MD->iBVNum,   sizeof(int),   1, fp2) != 1) 		{	exit(10);	}// this is currently read VertexNum
		if (fread(&MD->fBndLegs, sizeof(float), 1, fp2) != 1)  		{	exit(10);	}// this is currently read PatchNum 
    	if (fread(&fTemp0,       sizeof(float), 1, fp2) != 1)  		{	exit(10);	}// this is currently read PatchNum   
		fclose(fp2);
	}
	//------------------------------------------------------------------
	MD->fFltLegs    *= 1.0E+3; // now it is in meters 
    MD->fBndLegs    *= 1.0E+3; // now it is in meters 
    //------------------------------------------------------------------
    MD->fLegLgth     = (MD->fFltLegs <= MD->fBndLegs) ? MD->fFltLegs : MD->fBndLegs;
 
 	MD->fUnitSlip    = 1.0E-4*MD->fLegLgth; //for a 1000m fault patch (leg length) the 1e-4 means a test slip of 0.10m
	MD->fCutStFrac   = 0.1; //this is in MPa, only if the excess stress for postseismic is exceeding that value it will be released => helps to speed up
    MD->fDcfrac4Threshold = 0.1; //in order to start slipping and continue to do so, the excess stress must be larger than this fraction of Dc
    MD->iGlobTTmax   = 0; //max. global travel time => is used when "use rupture propagation" is used
    MD->fPSeis_Step  = 0.0;
	MD->iEQcntr      = 0; //counting the number of events that were written to file
	MD->iMaxIterat   = 500; //this is for case where I want to use boundary faults for loading (using them to get stressing rate on faults) => look at corresponding portion of code for additional description
	MD->iMaxMRFlgth  = 10000; //some "randomly" high number to ensure that the entire MRF will fit into this vector..
	MD->fISeisTStp   = MD->fISeisStep/365.25; //fraction of a year of the interseismic time step
    
	EQ->iEndCntr     = 0; //this counter is used to check if an event is really over (at the end of EQ while loop)
	EQ->fMaxDTau     = 0.0; //max change in shear stress during the event
	EQ->fMaxSlip     = 0.0; //max in-plane slip of the event
 	//-----------------------------------------------------------------
	MD->ivF_START    = (int *) calloc(MD->iSIZE, sizeof(int)); //these starts and offsets relate to how the code accesses "local" and "global" vectors => use the start and offset to 
    MD->ivB_START    = (int *) calloc(MD->iSIZE, sizeof(int));
    MD->ivF_OFFSET   = (int *) calloc(MD->iSIZE, sizeof(int));//locate where each rank will put/get data from when accessing a "global" vector
    MD->ivB_OFFSET   = (int *) calloc(MD->iSIZE, sizeof(int));
	MD->ivF_ModOFFs  = (int *) calloc(MD->iSIZE, sizeof(int));
	MD->ivB_ModOFFs  = (int *) calloc(MD->iSIZE, sizeof(int));

    MD->iF_BASEelem  = (int)(MD->iFPNum/MD->iSIZE); 
    MD->iF_ADDelem   = (int)(MD->iFPNum%MD->iSIZE);
    MD->iB_BASEelem  = (int)(MD->iBPNum/MD->iSIZE); 
    MD->iB_ADDelem   = (int)(MD->iBPNum%MD->iSIZE);
    //---------------------------
    for (i = 0; i < MD->iSIZE;     i++)      {   MD->ivF_OFFSET[i]     = MD->iF_BASEelem;         							}
    for (i = 0; i < MD->iF_ADDelem;i++)      {   MD->ivF_OFFSET[i]    += 1;          		    	 						}
    for (i = 1; i < MD->iSIZE;     i++)      {   MD->ivF_START[i]      = MD->ivF_START[i-1] + MD->ivF_OFFSET[i-1];      	}
    
    for (i = 0; i < MD->iSIZE;     i++)      {   MD->ivB_OFFSET[i]     = MD->iB_BASEelem;                              		}
    for (i = 0; i < MD->iB_ADDelem;i++)      {   MD->ivB_OFFSET[i]    += 1;                                                	}
    for (i = 1; i < MD->iSIZE;     i++)      {   MD->ivB_START[i]      = MD->ivB_START[i-1] + MD->ivB_OFFSET[i-1];          }
	//------------------------------------------------------------------
	TR->ivFL_Activated       = ( int *)  calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof( int));			TR->ivFL_Ptch_t0         = ( int *)  calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof( int));
    TR->ivFL_FricLaw         = ( int *)  calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof( int));			TR->ivFL_StabT           = ( int *)  calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof( int));	
	
	TR->fvFL_SelfStiffStk	 = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_SelfStiffDip    = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));
	TR->fvFL_MeanSelfStiff   = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));

	TR->fvFL_Area            = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_RefNrmStrs      = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));
	TR->fvFL_RefStaFric      = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_RefDynFric      = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));
	TR->fvFL_RefDcVal        = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_RefStaFric_vari = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));
	TR->fvFL_RefDynFric_vari = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_RefDcVal_vari   = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));
	
	TR->fvFL_StaFric         = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_DynFric         = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));
	TR->fvFL_CurFric         = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_CurDcVal        = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));
	TR->fvFL_B4_Fric         = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_TempRefFric     = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));	
	TR->fvFL_AccumSlp        = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_CutStress       = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));	
	//----------------------------------	
	TR->fvFL_StrsRateStk     = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);					TR->fvFL_StrsRateDip     = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);
	TR->fvFL_B4_StrsH        = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);					TR->fvFL_B4_StrsV        = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);                	TR->fvFL_B4_StrsN        = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);
	TR->fvFL_CurStrsH        = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);					TR->fvFL_CurStrsV        = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);					TR->fvFL_CurStrsN        = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);
	
	TR->fvFL_PSeis_T0_F      = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		
	TR->fvBL_PSeis_T0_S      = (float *) calloc(MD->ivB_OFFSET[MD->iRANK],   sizeof(float));		TR->fvBL_PSeis_T0_N      = (float *) calloc(MD->ivB_OFFSET[MD->iRANK],   sizeof(float));
	//----------------------------------
	TR->fvBL_CurStrsH        = gsl_vector_float_calloc(MD->ivB_OFFSET[MD->iRANK]);					TR->fvBL_CurStrsV        = gsl_vector_float_calloc(MD->ivB_OFFSET[MD->iRANK]);					TR->fvBL_CurStrsN        = gsl_vector_float_calloc(MD->ivB_OFFSET[MD->iRANK]);
	TR->fvBL_SelfStiffStk	 = (float *) calloc(MD->ivB_OFFSET[MD->iRANK],   sizeof(float));		TR->fvBL_SelfStiffDip    = (float *) calloc(MD->ivB_OFFSET[MD->iRANK],   sizeof(float));		TR->fvBL_SelfStiffOpn    = (float *) calloc(MD->ivB_OFFSET[MD->iRANK],   sizeof(float));
	TR->fvFL_SlipRate_temp   = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		TR->fvFL_SlipRake_temp   = (float *) calloc(MD->ivF_OFFSET[MD->iRANK],   sizeof(float));		
	//--------------------------------------
	
	TR->ivFG_Flagged_temp    = ( int *)  calloc(MD->iFPNum,   sizeof( int));
	TR->ivFG_V1_temp         = ( int *)  calloc(MD->iFPNum,   sizeof( int));						TR->ivFG_V2_temp         = ( int *)  calloc(MD->iFPNum,   sizeof( int));						TR->ivFG_V3_temp         = ( int *)  calloc(MD->iFPNum,   sizeof( int));
	TR->ivFG_SegID_temp      = ( int *)  calloc(MD->iFPNum,   sizeof( int));						TR->ivFG_FltID_temp      = ( int *)  calloc(MD->iFPNum,   sizeof( int));
	TR->fvFG_StressRatetemp  = (float*)  calloc(MD->iFPNum,   sizeof(float));						TR->fvFG_SlipRatetemp    = (float*)  calloc(MD->iFPNum,   sizeof(float));
	TR->fvFG_Raketemp        = (float*)  calloc(MD->iFPNum,   sizeof(float));						TR->fvFG_MaxTransient    = (float*)  calloc(MD->iFPNum,   sizeof(float));

	TR->ivBG_V1_temp         = ( int *)  calloc(MD->iBPNum,   sizeof( int));						TR->ivBG_V2_temp         = ( int *)  calloc(MD->iBPNum,   sizeof( int));						TR->ivBG_V3_temp         = ( int *)  calloc(MD->iBPNum,   sizeof( int));
	TR->ivBG_SegID_temp      = ( int *)  calloc(MD->iBPNum,   sizeof( int));
	TR->fvFG_CentE_temp  	 = (float *) calloc(MD->iFPNum,   sizeof(float));						TR->fvFG_CentN_temp      = (float *) calloc(MD->iFPNum,   sizeof(float));						TR->fvFG_CentZ_temp      = (float *) calloc(MD->iFPNum,   sizeof(float));	
	TR->fvBG_CentE_temp      = (float *) calloc(MD->iBPNum,   sizeof(float));						TR->fvBG_CentN_temp      = (float *) calloc(MD->iBPNum,   sizeof(float));						TR->fvBG_CentZ_temp      = (float *) calloc(MD->iBPNum,   sizeof(float));

    TR->fvFL_StaFricMod_temp = (float *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof(float)); 			TR->fvFL_DynFricMod_temp = (float *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof(float)); 
    TR->fvFL_NrmStrsMod_temp = (float *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof(float)); 			TR->fvFL_DcMod_temp      = (float *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof(float));
	//--------------------------------------
	TR->imFGL_TTP            = gsl_matrix_int_calloc(MD->ivF_OFFSET[MD->iRANK], MD->iFPNum);  		TR->imFGL_TTS            = gsl_matrix_int_calloc(MD->ivF_OFFSET[MD->iRANK], MD->iFPNum);  
	TR->imFGL_NextP          = gsl_matrix_int_calloc(MD->ivF_OFFSET[MD->iRANK], MD->iFPNum);  		TR->imFGL_NextS          = gsl_matrix_int_calloc(MD->ivF_OFFSET[MD->iRANK], MD->iFPNum); 
    //these two tell me the index for source-receiver pair => how much of STF at source has been "seen" at receiver already... takes more memory but is worth it b/c it allows to make the STF shorter and ensures that it will not "blow up" when events are too long...
	TR->fmFGL_SrcRcvH        = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK], MD->iFPNum); 		TR->fmFGL_SrcRcvV        = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK], MD->iFPNum); 		TR->fmFGL_SrcRcvN        = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK], MD->iFPNum); 
	//------------------------------------------------------------------
    VT->fvFG_PosE_temp       = (float *) calloc(MD->iFVNum,   sizeof(float));						VT->fvFG_PosN_temp       = (float *) calloc(MD->iFVNum,   sizeof(float));						VT->fvFG_PosZ_temp       = (float *) calloc(MD->iFVNum,   sizeof(float));	
	VT->fvFG_VlX_temp     	 = (float *) calloc(MD->iFVNum,   sizeof(float));						VT->fvFG_VlY_temp        = (float *) calloc(MD->iFVNum,   sizeof(float));						VT->fvFG_Hght_temp       = (float *) calloc(MD->iFVNum,   sizeof(float));	
	VT->fvBG_PosE_temp       = (float *) calloc(MD->iBVNum,   sizeof(float));						VT->fvBG_PosN_temp       = (float *) calloc(MD->iBVNum,   sizeof(float));						VT->fvBG_PosZ_temp       = (float *) calloc(MD->iBVNum,   sizeof(float));
	//-------------------------------------------------------------------------------------
	//first dimension is number of rows (slow direction) => is the SOURCES; second dimension is number of columns (fast direction) => is the RECEIVERS
	K->FFs_SS  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivF_OFFSET[MD->iRANK]); 					K->FFs_SD  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivF_OFFSET[MD->iRANK]);					K->FFs_SO  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivF_OFFSET[MD->iRANK]);
    K->FFs_DS  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivF_OFFSET[MD->iRANK]);					K->FFs_DD  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivF_OFFSET[MD->iRANK]);					K->FFs_DO  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivF_OFFSET[MD->iRANK]);
   	if (MD->iUseProp == 1)
	{	K->FFr_SS  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum); 				K->FFr_SD  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);						K->FFr_SO  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);//vorne stehen die sources, hinten die receiver
    	K->FFr_DS  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);					K->FFr_DD  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);						K->FFr_DO  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);
		K->FFr_OS  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);					K->FFr_OD  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);						K->FFr_OO  = gsl_matrix_float_calloc(MD->ivF_OFFSET[MD->iRANK],MD->iFPNum);
	}

   	K->FB_SS  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivB_OFFSET[MD->iRANK]); 					K->FB_SD  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivB_OFFSET[MD->iRANK]);						K->FB_SO  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivB_OFFSET[MD->iRANK]);
    K->FB_DS  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivB_OFFSET[MD->iRANK]);						K->FB_DD  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivB_OFFSET[MD->iRANK]);						K->FB_DO  = gsl_matrix_float_calloc(MD->iFPNum, MD->ivB_OFFSET[MD->iRANK]);
 
 	K->BF_SS  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivF_OFFSET[MD->iRANK]);						K->BF_SD  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivF_OFFSET[MD->iRANK]);						
    K->BF_DS  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivF_OFFSET[MD->iRANK]);						K->BF_DD  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivF_OFFSET[MD->iRANK]);						
    K->BF_OS  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivF_OFFSET[MD->iRANK]);						K->BF_OD  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivF_OFFSET[MD->iRANK]);						
   	
 	K->BB_SS  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);						K->BB_SD  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);						K->BB_SO  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);
    K->BB_DS  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);						K->BB_DD  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);						K->BB_DO  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);
    K->BB_OS  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);						K->BB_OD  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);						K->BB_OO  = gsl_matrix_float_calloc(MD->iBPNum, MD->ivB_OFFSET[MD->iRANK]);
 	//-------------------------------------------------------------------------------------
	EQ->ivR_WrtStrtPos = (  int *) calloc(MD->iSIZE,                sizeof( int));					EQ->ivL_ActPtchID  = (  int *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof( int));
	EQ->ivL_t0ofPtch   = (  int *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof( int));					EQ->ivL_StabType   = (  int *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof( int));
	EQ->fvL_PtchSlpH   = (float *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof(float));  				EQ->fvL_PtchSlpV   = (float *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof(float));  
  	EQ->fvL_PtchDTau   = (float *) calloc(MD->ivF_OFFSET[MD->iRANK], sizeof(float));  				EQ->fvM_MRFvals    = (float *) calloc(MD->iMaxMRFlgth,          sizeof(float));
	EQ->fvL_EQslipH    = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);						EQ->fvL_EQslipV    = gsl_vector_float_calloc(MD->ivF_OFFSET[MD->iRANK]);
	//-------------------------------------------------------------------------------------
    if ((iPlot2Screen == 1) && (MD->iRANK == 0)) //making sure that the data were imported from file correctly
    {	fprintf(stdout,"Number of RANKS: %d\n",MD->iSIZE);					
		fprintf(stdout,"System info: Byte Size for FLOAT: %lu     INT: %lu    \n\n", sizeof(float), sizeof(int));
		fprintf(stdout,"FileName:           %s\n",MD->cInputName);		 		    fprintf(stdout,"RunNumber:          %d\n",MD->iRunNum);	
		fprintf(stdout,"UsePostSeis:        %d\n",MD->iUsePSeis);
        fprintf(stdout,"UseRuptProp:        %d\n",MD->iUseProp);	                fprintf(stdout,"MinMag2UseRuptProp: %f\n",MD->fMinMag4Prop);
    	fprintf(stdout,"IntSeisTStep:       %3.1f\n",MD->fISeisStep);					
		fprintf(stdout,"ViscAftSlip:        %3.1f\n",MD->fAftrSlipTime);			
		fprintf(stdout,"ViscDeepRelax:      %3.1f\n",MD->fDeepRelaxTime);
		fprintf(stdout,"RecLength:          %5.1f\n",MD->fRecLgth);				
		fprintf(stdout,"CoSeisHealFraction: %5.1f\n",MD->fHealFact);				fprintf(stdout,"SeedLocation:       %5d\n",MD->iSeedStart);
		fTemp0  = 100 - expf(-1.0*1/MD->fAftrSlipTime)*100.0; //factor/fraction from decay function
		fprintf(stdout,"Fractional post-seismic change during first year: %2.4f percent released \n\n",fTemp0);

    	fprintf(stdout,"FaultPatchNumber %d     FaultVertexNumber %d\n", MD->iFPNum, MD->iFVNum);
    	fprintf(stdout,"BoundPatchNumber %d     BoundVertexNumber %d\n", MD->iBPNum, MD->iBVNum);
    	fprintf(stdout,"MeanLegLenth     %f        %f   %f\n",MD->fLegLgth, MD->fBndLegs, MD->fFltLegs);   
	}	
	return;
}
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//