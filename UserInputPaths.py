from UserInput import BSM_model


SPheno_path = "SPheno-4.0.5"
HB_path = "higgsbounds-5.10.2/build"    # Fix paths (remove build) and fix Run HBS shell script paths
HS_path = "higgssignals-2.6.2/build"
CT_path = "DRalgo-1.0.2-beta/examples"
#CT_infile_name = "LS_TColor_DRPython" #Remove .py
#CT_class_name = "LS_TColor"     # Not being used
CT_infile_name = "THDM_DRPython" #Remove .py
CT_class_name = "THDM" 

#=============================================================== PATHS ========================================================
LesHouches_filename = "LesHouches.in.{}".format(BSM_model)
SPheno_spc_filename = "SPheno.spc.{}".format(BSM_model) 

SPheno_path_S = "../../{}".format(SPheno_path)
HB_path_S = "../../{}".format(HB_path)
HS_path_S = "../../{}".format(HS_path)


#LesHouches_path = "{}/LesHouches.in.{}".format(SPheno_path, model)     # Create seperate variables for package names
#SPheno_spc_path = "{}/SPheno.spc.{}".format(SPheno_path, model)        

#HB_script_path = "higgsbounds-5.10.2/build/RunHiggsBounds.sh"
#HB_output_path = "{}/HiggsBounds_results.dat".format(SPheno_path)

#HS_script_path = "higgssignals-2.6.2/build/RunHiggsSignals.sh"
#HS_output_path = "{}/HiggsSignals_results.dat".format(SPheno_path)


HB_output_filename = "HiggsBounds_results.dat"
HS_output_filename = "HiggsSignals_results.dat"

#CT_directory_path = "DRalgo-1.0.2-beta/examples"
#CT_infile_name = "LS_TColor_DRPython" #Remove .py
#CT_class_name = "LS_TColor"
#============================================================================================================================
