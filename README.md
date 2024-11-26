# BSMscanner
Beyond the Standard Model parameter space scanner assisted by machine learning. Scanner applies theoretical, collider and cosmic constraints.

## Dependecies
Following libraries are required for applying the theoretical, collider and cosmic constraints using BSMscanner
1. `SPheno` for computation of collider observables and other theoretical computations (https://spheno.hepforge.org/)
2. `HiggsBounds` and `HiggsSignals` for Higgs-related collider observables (https://gitlab.com/higgsbounds/higgsbounds, https://gitlab.com/higgsbounds/higgssignals)
3. `CosmoTransitions` for computation of gravitational-wave related observables (https://github.com/clwainwright/CosmoTransitions)
4. gwFuns.py script written by António Morais for computation of gravitational-wave related observables (e.g. energy budget alpha or inverse phase transition duration beta) (script not publicly available)

For the BSM theory toy models available in the code, the following were also used

5. `SARAH` for model implementation (https://sarah.hepforge.org/)
6. `DRalgo` for construction of finite-temperature potantial (https://github.com/DR-algo/DRalgo)
7. export code from `DRalgo` to `CosmoTransitions` written by Mårten Bertenstam (scirpt not publicly available)

## Pre-defined toy models
Three toy models (THDM, SM+Singlet, SM+Singlet+Triplet) are already defined in the code. The relevant files to these models using the HEP packages can be found in the HEPfiles directory. The THDM is implemented for the collider constraints, SM+Singlet for cosmic constraints and SM+Singlet+Triplet for collider+cosmic constraints. 

## Manual
1. For collider constraints, install and compile collider (HEP) packages
   - Install `SPheno`, `HiggsBounds` and `HiggsSignals`. The packages are available in the links provided above, but specific versions can also be found in the HEPfiles directory.
   - Implement and compile your BSM model in SPheno (e.g. using `SARAH`).
   - If you want to use one of the pre-defined toy models, the LesHouches input files and `SPheno` executable files for each model are found in the HEPfiles/{ModelName} directory. Place the LesHouches file in the SPheno directory, and the executable SPheno{ModelName} in the SPheno/bin directory.
   - Compile `HiggsBounds` and `HiggsSignals`.
   - Make sure that each package runs using your BSM theory for a specific parameter space point.

2. For cosmic constraints, install and compile cosmic (HEP) packages
   - Install `CosmoTransitions`. The package is available in the links provided above, but a specific version can also be found in the HEPfiles directory.
   - Implement the finite-temperature potential into `CosmoTransitions` (e.g. using `DRalgo` and Mårtens code).
   - If you want to use one of the pre-defined toy models, you can find the `CosmoTransition` files for each model is found in the HEPfiles/{ModelName} directory (the THDM CosmoTrnasition file is a bit buggy, but still runs).
   - Make sure CosmoTransitions runs for a specific parameter space point.

3. Place the `CosmoTransitions` Python file in the example directoy of `DRalgo`.

4. Place the gwFuns.py (found in HEPfiles) written by Antonio in the `DRalgo` examples directory.

5. In the UserInput.py file, add the paths to the HEP packages and files.



#### BSM theory input:
1. Insert BSM model parameter details into BSMscanner via the UserInput.py file. Note that here you must supply the `SPheno` input parameters, and can additionally include other parameters such as particle masses, mixing angles etc. These extra parameters must be  connected to the SPheno parameters by some known relation. The following information is required.
   - Parameter names: names must work as Python variable names. Note that the `SPheno` input parameter's names do NOT have to match names given in `SARAH` or `SPheno`.
   - LesHouches numbers: Assign the `SPheno` input parameters their LesHouches numbers, make sure that they match the ones given in the SPheno.m file in `SARAH`. Parameters that are not SPheno input parameters and therefore do not have LesHouches numbers (e.g. masses) are given None as LesHouches number.
   - Range start and Range end: Supply the intervals for which each (free) parameter should be scanned. Parameters that are fixed by others (e.g. by a parameter space inversion) and therefore are not free, are given None in Range start and Range end. If you have a parameter that should be set to a fixed value (e.g. SM Higgs mass), set an identical value for its Range start and Range end. This parameter will then not be treated as free in the code.
   - Dependence: Supply the expressions to compute the parameters that are not free and are fixed by the others (i.e have None as Range start and Range end). The expressions must be written so that Python can read them. If you use special functions such as arctan or sqrt, use np.arctan(...) and cmath.sqrt(...) respectively. The code evaluates these functions by looking inside the dict_const_param dictionary defined in DerivedUserInput.py. If the code does not recognize a function you have used, it might be due to that function not existing in the dictionary. In that case you need to add it via the DerivedUserInput.py file (e.g. dict_const_param["np.sin"] = np.sin) . When supplying dependencies, make sure that you use the given names for the parameters as defined in the Parameter names field. The free (and constant) parameters are given a None value in this field.
   - Solve order: If you perform multiple (seperate) parameter space inversions (e.g. tadpole equations, and then parameter space inversion using particle masses), the expressions in Dependence must be evauluated in the correct order (in the given example, the parameter space inversion must be performed first. The expressions for the Lagrangian parameters in terms of the particle masses can then be plugged into the tadpole equations). Solve order is assigned by integers (1,2,3, ...). Parameters that have no Dependence, are given a None value in this field. 

2. (Optional) Implement additional constraints on your parameter space, e.g. boundedness from below, perturbitivity. This is currently not done in the UserInput file, but must be directly coded into the function EvalFcn in Hub.py. You will find examples for other BSM theories inside the function. 

#### Constructing training data via random scan
1. Set construct_training_data=True. The training data is saved to DataFiles/TDataFiles and a summary of the number of points satisfying each constraint is printed. If you have old data in the TDataFiles (belonging to the same BSM theory!), which you want to save, set keep_old_data=True.
2. Intersections of the n-dimensional parameter space containing these points will automatically be plotted and saved in the Figures directory, where n is the number of free parameters for the BSM theory. One can choose to plot either negative/positive points, or points satisfying each constraint seperately. This is controlled by the plot_seperate_constr input.
3. The number of points scanned is controlled by the exp_num_training_points input. 
4. For multiprocessing, number of concurrent processes to run is chosen by number_of_processes. Used when analyzing randomly sampled points.

#### Applying deep learning
1. To train a neural network using the data stored in DataFiles/TDataFiles, set train_ANN=True. Most hyperparameters (ANN parameters) can be found under the ANN Settings section. To change number of hidden layers and nodes, activation functions, optimizer etc., one must go the the Network.py file directly where the ANN is defined, under the ConstructModel function. The trained ANN will be saved if save_ANN=True, in the SavedANNs/BSM_model-ANN directory, which means one can save one ANN model per BSM theory. 
2. If a previously trained and saved ANN exists, it can be loaded by load_ANN=True. The ANN belonging to the given BSM theory will be loaded. 
3. If ANN_predicts=True, a (second preferably larger) parameter space sampling is performed. The number of points is controlled by exp_num_pred_points. The ANN will then predict which points it believes to be positve (i.e satisfy the constraints it has been trained on). 
4. ANN_controls=True will analyze the positively precited points using the HEP packages. The true positive points are are saved to DataFiles/FDataFiles and a summary of the number of points is printed. If you have old data in the FDataFiles (belonging to the same BSM theory!), which you want to save, set keep_old_data=True. 
5. For multiprocessing, number of concurrent processes to run is chosen by number_of_processes. Used when analyzing positively predicted points.



#### When you run the scanner (construct_training_data=True or ANN_controls=True) with collider constraints
1. `HiggsSignals` p-value controlled by pvalue_threshold. Currently set to 0.05.
2. Accepted STU parameter intervals controlled by S-, T- and U_threshold variables. Currently set according to Particle Data Group, at 68% C.L.
3. automatic_cs = True is recommended (only if cosmic constraints not being evaluated!). This gives a chunk of points (more than one point) to a process at a time. Reduces overhead when doing multiprocessing. Currently set to False (may increase overhead!).

#### When you run the scanner (construct_training_data=True or ANN_controls=True) with cosmic constraints
1. Make sure that the params_4D_ref in CT_InputFcn is UserInput is in the right form, that is matches the return statement of the calculateParams4DRef function in the CosmoTransitions Python file. For the toy models, simply uncomment the correct line(s).
2. Since cosmic constraints can sometimes take too long to evaluate for some points, a cut-off time is added, controlled by CT_wait_time. The scanner will wait CT_wait_time hours before aborting the computation of the cosmic observables of a point. Currently, CT_wait_time is set to 3 (hours).
3. The GW detectability constrant is currently in the form of GW peak amplitude, given by Omega. This can be changed using the omega_exp variable. Currently omega_exp is set to -18. 
4. automatic_cs = False is recommended. This gives one point at a time to each process. Since cosmic constraints can take a long time to run, this avoids the evens out the run time for each process. 


## Known bugs/issues:
1. Training data construction and neural network training needs to be done in two seperate sessions, i.e. code needs to be run twice (which practically is what one usually does). 
2. Input for CosmoTransitions (in CT_InputFcn in UserInput.py) must be set manually at the moment. 
3. Previously, `SPheno` parameters had to be given LesHouches numbers in increasing order, starting at 1. This is now fixed, needs to be tested. 
