from UserInput import model


SPheno_path = "SPheno-4.0.5"
HB_path = "higgsbounds-5.10.2/build"
HS_path = "higgssignals-2.6.2/build"
CT_path = "DRalgo-1.0.2-beta/examples"
CT_infile_name = "LS_TColor_DRPython" #Remove .py
CT_class_name = "LS_TColor"

#=============================================================== PATHS ========================================================
LesHouches_path = "{}/LesHouches.in.{}".format(SPheno_path, model)     # Create seperate variables for package names
SPheno_spc_path = "{}/SPheno.spc.{}".format(SPheno_path, model)        

#HB_script_path = "higgsbounds-5.10.2/build/RunHiggsBounds.sh"
HB_output_path = "{}/HiggsBounds_results.dat".format(SPheno_path)

#HS_script_path = "higgssignals-2.6.2/build/RunHiggsSignals.sh"
HS_output_path = "{}/HiggsSignals_results.dat".format(SPheno_path)

#CT_directory_path = "DRalgo-1.0.2-beta/examples"
#CT_infile_name = "LS_TColor_DRPython" #Remove .py
#CT_class_name = "LS_TColor"
#============================================================================================================================
