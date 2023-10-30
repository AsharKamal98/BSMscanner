(* ::Package:: *)

(* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ *)

(* :Title: DRPythonExport                                                         	*)

(*
       This software is covered by the GNU General Public License 3.
       Copyright (C) 2021-2022 Andreas Ekstedt
       Copyright (C) 2021-2022 Philipp Schicho
       Copyright (C) 2021-2022 Tuomas V.I. Tenkanen

*)

(* :Summary: Export the relevant expressions to DRPython *)	

(* ------------------------------------------------------------------------ *)


ExportModelToDRPython[gaugeCouplingNames_,auxParams_,\[Phi]VeV3D_,modelName_,indent_,printTemplate_,fileName_] := Module[{},

gaugeSub = Table[gaugeCouplingNames[[i]]-> Sqrt[Symbol[ToString[gaugeCouplingNames[[i]]]<>"sq" ]],{i,Length[gaugeCouplingNames]}];

vevs3D = {};
For[i=1,i<=Length[\[Phi]VeV3D],i++,If[ToString[\[Phi]VeV3D[[i]]]!="0",AppendTo[vevs3D,\[Phi]VeV3D[[i]]],]];
nVevs3D = Length[vevs3D];
 
 

fileHeaderExpr={
"from generic_potential_DR_class_based import generic_potential_DR",
"from numpy import array, sqrt, log, exp, pi",
"from numpy import asarray, zeros, ones, concatenate, empty",
"from numpy.linalg import eigh, eigvalsh",
"from modelDR_class_based import Lb as LbFunc",
"from modelDR_class_based import Lf as LfFunc",
"from modelDR_class_based import EulerGamma",
"from modelDR_class_based import Glaisher",
"",
"#This is the number of field dimensions (the Ndim parameter):",
"nVevs = "<>ToString[nVevs3D],
"",
If[Length[auxParams]==0,"#No auxParams for this model.","#Specify the auxParams "<>ToString[auxParams]<>" as a tuple in this order."],
""};
(*
Add in later. To make it clean, a reorganization of the code is likely needed. The Module should take the tensors as input,
run the import model inside the module, and get the variable names from calling variables on the various tensors.
"\[Currency]This choice of params4DMin imposes positivity of the squared gauge ",
"\[Currency]couplings as well as the lower perturbativity bound of all other couplings:",
"\[Currency]This choice of params4DMax imposes the upper perturbativity bound ",
"\[Currency]of all couplings:"
*)






classHeaderExpr={
"class "<>modelName<>"(generic_potential_DR):",
indent<>"\"\"\"",
indent<>"Insert class description here.",
indent<>"\"\"\""};



(* Build the string for RGFuncs4D. *)
PerformDRhard[];
BetaFunctions4DSaved = BetaFunctions4D[]/.gaugeSub;
nParams4D = Length[BetaFunctions4DSaved];
params4D = BetaFunctions4DSaved[[All,1]];
betaFuncs4D = BetaFunctions4DSaved[[All,2]];

RGFuncs4DExpr={
indent<>"def RGFuncs4D(self, mu4D, params4D, *auxParams):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the RHS of the RG-equation dp/d\[Mu]4D = \[Beta](p)/\[Mu]4D.",
indent<>indent<>"",
indent<>indent<>"This function returns an array_like object with the \[Beta]-functions for", 
indent<>indent<>"the 4D-parameters divided by the RG-scale \[Mu]4D, i.e. the right-hand",
indent<>indent<>"side in the RG-equation dp/d\[Mu]4D = \[Beta](p)/\[Mu]4D, where p denotes the array",
indent<>indent<>"of 4D-parameters.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"mu4D : float",
indent<>indent<>indent<>"The 4D RG-scale parameter (i.e. \[Mu]4D) ",
indent<>indent<>"params4D : array",
indent<>indent<>indent<>"Array of the 4D-parameters at scale \[Mu]4D",
indent<>indent<>"auxParams : tuple",
indent<>indent<>indent<>"Tuple of auxiliary parameters",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"RHS of the RG-equation dp/d\[Mu]4D = \[Beta](p)/\[Mu]4D as an array",
indent<>indent<>"\"\"\"",
indent<>indent<>"\[Mu]4D = mu4D",
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[params4D],Thread[{"{","}"}->""]]<>" = params4D",
indent<>indent<>If[Length[auxParams]>0,If[Length[auxParams]>1,StringReplace[ToString[auxParams],Thread[{"{","}"}->""]]<>" = auxParams",ToString[auxParams[[1]]]<>", = auxParams"],""],
indent<>indent<>"",
Table[indent<>indent<>"\[Beta]"<>ToString[params4D[[i]]]<>" = "<>ToString[betaFuncs4D[[i]],FortranForm],{i,nParams4D}],
indent<>indent<>"",
indent<>indent<>StringReplace[ToString["\[Beta]"<>ToString[#]&/@params4D],{"{"->"return array([","}"->"])/\[Mu]4D"}],
indent<>indent<>""}//Flatten;

Print["RGFuncs4D done!"];

(* Build the string for DebyeMassSq. *)
DebyeMassLOSaved = PrintDebyeMass["LO"]/.gaugeSub;
DebyeMassNames = DebyeMassLOSaved[[All,1]];
nDebyeMasses = Length[DebyeMassLOSaved];
DebyeMassesLO = DebyeMassLOSaved[[All,2]];
DebyeMassesNLOCombined = DebyeMassesLO + PrintDebyeMass["NLO"][[All,2]]/.gaugeSub; 

DebyeMassSqExpr={
indent<>"def DebyeMassSq(self, T, mu4DH, params4D, order, *auxParams):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the squared Debye masses.",
indent<>indent<>"",
indent<>indent<>"This function is used to calculate the squared Debye masses as a", 
indent<>indent<>"function of the temperature T, the hard matching scale mu4DH (\[Mu]4DH)",
indent<>indent<>"and the values of the 4D-parameters at scale \[Mu]4DH. The masses can be",
indent<>indent<>"calculated at LO or NLO (order = 0 and 1, respectively).",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"T : float",
indent<>indent<>indent<>"The temperature",
indent<>indent<>"mu4DH : float",
indent<>indent<>indent<>"The hard matching scale (i.e. \[Mu]4DH)",
indent<>indent<>"params4D : array",
indent<>indent<>indent<>"Array of the 4D-parameters at scale \[Mu]4DH",
indent<>indent<>"order : int",
indent<>indent<>indent<>"The order at which the Debye masses are calculated (0 or 1)",
indent<>indent<>"auxParams : tuple",
indent<>indent<>indent<>"Tuple of auxiliary parameters",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The squared Debye masses as an array",
indent<>indent<>"\"\"\"",
indent<>indent<>"Lb = LbFunc(mu4DH,T)",
indent<>indent<>"Lf = LfFunc(mu4DH,T)",
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[params4D],Thread[{"{","}"}->""]]<>" = params4D",
indent<>indent<>If[Length[auxParams]>0,If[Length[auxParams]>1,StringReplace[ToString[auxParams],Thread[{"{","}"}->""]]<>" = auxParams",ToString[auxParams[[1]]]<>", = auxParams"],""],
indent<>indent<>"",
indent<>indent<>"if order == 0:",
Table[indent<>indent<>indent<>ToString[DebyeMassNames[[i]]]<>" = "<>ToString[DebyeMassesLO[[i]],FortranForm],{i,nDebyeMasses}],
indent<>indent<>"elif order == 1:",
Table[indent<>indent<>indent<>ToString[DebyeMassNames[[i]]]<>" = "<>ToString[DebyeMassesNLOCombined[[i]],FortranForm],{i,nDebyeMasses}],
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[ToString[#]&/@DebyeMassNames],{"{"->"return array([","}"->"])"}],
indent<>indent<>""}//Flatten;

Print["DebyeMassSq done!"];

(* Build the string for DRStep. *)
(* Soft limit: *)
couplingsSaved = PrintCouplings[]/.gaugeSub;
couplingNamesS = couplingsSaved[[All,1]]/.Table[Symbol[ToString[gaugeCouplingNames[[i]]]<>"3d"] -> Sqrt[Symbol[ToString[gaugeCouplingNames[[i]]]<>"sq3d" ]],{i,Length[gaugeCouplingNames]}];
couplingNamesSOld = couplingNamesS/.Table[Symbol[ToString[gaugeCouplingNames[[i]]]<>"sq3d"] -> Symbol[ToString[gaugeCouplingNames[[i]]]<>"3d"],{i,Length[gaugeCouplingNames]}];
nCouplings = Length[couplingNamesS]; 
couplingNames3D = couplingNamesS/.Table[couplingNamesS[[i]]->Symbol[StringReplace[ToString[couplingNamesS[[i]]],"3d"->"\[LetterSpace]3D"]],{i,nCouplings}];
couplingNamesS = couplingNamesS/.Table[couplingNamesS[[i]]->Symbol[StringReplace[ToString[couplingNamesS[[i]]],"3d"->"\[LetterSpace]3D\[LetterSpace]S"]],{i,nCouplings}];
couplingsSSub = Table[couplingNamesSOld[[i]]->couplingNamesS[[i]],{i,nCouplings}]/.Table[Symbol[ToString[gaugeCouplingNames[[i]]]<>"sq\[LetterSpace]3D\[LetterSpace]S"] -> Sqrt[Symbol[ToString[gaugeCouplingNames[[i]]]<>"sq\[LetterSpace]3D\[LetterSpace]S" ]],{i,Length[gaugeCouplingNames]}];
couplingsS = couplingsSaved[[All,2]];
Print[couplingNamesS]
Print["Soft couplings done!"];

temporalScalarCouplingsSaved = PrintTemporalScalarCouplings[]/.gaugeSub;
temporalScalarCouplingNamesOld = temporalScalarCouplingsSaved[[All,1]];
nTemporalScalarCouplings = Length[temporalScalarCouplingNamesOld];
temporalScalarCouplingNames = temporalScalarCouplingNamesOld/.Table[temporalScalarCouplingNamesOld[[i]]->Symbol[StringReplace[ToString[temporalScalarCouplingNamesOld[[i]]],{"["->"","]"->""}]],{i,nTemporalScalarCouplings}];
temporalScalarCouplingsSub = Table[temporalScalarCouplingNamesOld[[i]]->temporalScalarCouplingNames[[i]],{i,nTemporalScalarCouplings}];
temporalScalarCouplings = temporalScalarCouplingsSaved[[All,2]];

Print["Temporal scalar couplings done!"];

scalarMassLOSaved = PrintScalarMass["LO"]/.gaugeSub/.couplingsSSub/.temporalScalarCouplingsSub;
scalarMassNamesS = scalarMassLOSaved[[All,1]];
nScalarMasses = Length[scalarMassNamesS];
scalarMassNames3D = scalarMassNamesS/.Table[scalarMassNamesS[[i]] -> Symbol[StringReplace[ToString[scalarMassNamesS[[i]]],"3d"->"\[LetterSpace]3D"]],{i,nScalarMasses}]; 
scalarMassNamesS = scalarMassNamesS/.Table[scalarMassNamesS[[i]] -> Symbol[StringReplace[ToString[scalarMassNamesS[[i]]],"3d"->"\[LetterSpace]3D\[LetterSpace]S"]],{i,nScalarMasses}]; 
scalarMassNamesSOld = scalarMassLOSaved[[All,1]];
scalarMassesSSub = Table[scalarMassNamesSOld[[i]]->scalarMassNamesS[[i]],{i,nScalarMasses}];
scalarMassesSLO = scalarMassLOSaved[[All,2]];
scalarMassesSNLOCombined = scalarMassesSLO + PrintScalarMass["NLO"][[All,2]]/.gaugeSub/.couplingsSSub/.temporalScalarCouplingsSub;

Print["Soft masses done!"];

params3D = Join[couplingNames3D,scalarMassNames3D];
nParams3D = Length[params3D];

(* Ultrasoft limit: *)
PerformDRsoft[{}];
couplingsUSSaved = PrintCouplingsUS[]/.couplingsSSub/.scalarMassesSSub/.temporalScalarCouplingsSub;
couplingNamesUS = couplingsUSSaved[[All,1]]/.Table[Symbol[ToString[gaugeCouplingNames[[i]]]<>"3dUS"] -> Sqrt[Symbol[ToString[gaugeCouplingNames[[i]]]<>"sq3dUS" ]],{i,Length[gaugeCouplingNames]}];
Print[couplingNamesUS];
Print[nCouplings];
couplingNamesUS = couplingNamesUS/.Table[couplingNamesUS[[i]]->Symbol[StringReplace[ToString[couplingNamesUS[[i]]],"3dUS"->"\[LetterSpace]3D\[LetterSpace]US"]],{i,nCouplings}];
couplingsUS = couplingsUSSaved[[All,2]];

Print["US couplings done!"];

scalarMassUSLOSaved = PrintScalarMassUS["LO"]/.couplingsSSub/.scalarMassesSSub/.temporalScalarCouplingsSub;
scalarMassNamesUS = scalarMassUSLOSaved[[All,1]];
scalarMassNamesUS = scalarMassNamesUS/.Table[scalarMassNamesUS[[i]] -> Symbol[StringReplace[ToString[scalarMassNamesUS[[i]]],"3dUS"->"\[LetterSpace]3D\[LetterSpace]US"]],{i,nScalarMasses}]; 
scalarMassesUSLO = scalarMassUSLOSaved[[All,2]];
scalarMassesUSNLOCombined = scalarMassesUSLO + PrintScalarMassUS["NLO"][[All,2]]/.couplingsSSub/.scalarMassesSSub/.temporalScalarCouplingsSub;

params3DNamesUS = Join[couplingNamesUS,scalarMassNamesUS];
nParams3DUS = Length[params3DNamesUS];
print[nParams3DUS];

Print["US masses done!"];

DRStepExpr={
indent<>"def DRStep(self, T, mu4DH, mu3DS, params4D, order, *auxParams):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 3D-parameters in the ultrasoft limit.",
indent<>indent<>"",
indent<>indent<>"This function is used to perform the dimensional reduction to the", 
indent<>indent<>"ultrasoft limit. Thus, it calculates the values of the 3D parameters",
indent<>indent<>"in the ultrasoft limit as a function of the temperature T, the hard",
indent<>indent<>"matching scale mu4DH (\[Mu]4DH), the hard-to-soft matching scale mu3DS",
indent<>indent<>"(\[Mu]3DS) and the values of the 4D-parameters at scale \[Mu]4DH.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"T : float",
indent<>indent<>indent<>"The temperature",
indent<>indent<>"mu4DH : float",
indent<>indent<>indent<>"The hard matching scale (i.e. \[Mu]4DH)",
indent<>indent<>"mu3DS : float",
indent<>indent<>indent<>"The hard-to-soft matching scale (i.e. \[Mu]3DS)",
indent<>indent<>"params4D : array",
indent<>indent<>indent<>"Array of the 4D-parameters at scale \[Mu]4DH",
indent<>indent<>"order : int",
indent<>indent<>indent<>"The order at which the dimensional reduction is performed (0 or 1)",
indent<>indent<>"auxParams : tuple",
indent<>indent<>indent<>"Tuple of auxiliary parameters",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"Array of the 3D parameters in the ultrasoft limit",
indent<>indent<>"\"\"\"",
indent<>indent<>"\[Mu] = mu4DH",
indent<>indent<>"\[Mu]3 = mu3DS",
indent<>indent<>"\[Mu]3US = mu3DS"<>" #Temporary fix due to error in DRalgo notation", (*Note! This can be removed after the DRalgo update. *)
indent<>indent<>"Lb = LbFunc(mu4DH,T)",
indent<>indent<>"Lf = LfFunc(mu4DH,T)",
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[params4D],Thread[{"{","}"}->""]]<>" = params4D",
indent<>indent<>If[Length[auxParams]>0,If[Length[auxParams]>1,StringReplace[ToString[auxParams],Thread[{"{","}"}->""]]<>" = auxParams",ToString[auxParams[[1]]]<>", = auxParams"],""],
indent<>indent<>"",
indent<>indent<>"#The couplings in the soft limit:",
Table[indent<>indent<>ToString[couplingNamesS[[i]]]<>" = "<>ToString[couplingsS[[i]],FortranForm],{i,nCouplings}],
indent<>indent<>"",
indent<>indent<>"#The temporal scalar couplings:",
Table[indent<>indent<>ToString[temporalScalarCouplingNames[[i]]]<>" = "<>ToString[temporalScalarCouplings[[i]],FortranForm],{i,nTemporalScalarCouplings}],
indent<>indent<>"",
indent<>indent<>"#The Debye masses:",
indent<>indent<>"if order == 0:",
Table[indent<>indent<>indent<>ToString[DebyeMassNames[[i]]]<>" = "<>ToString[DebyeMassesLO[[i]],FortranForm],{i,nDebyeMasses}],
indent<>indent<>"elif order == 1:",
Table[indent<>indent<>indent<>ToString[DebyeMassNames[[i]]]<>" = "<>ToString[DebyeMassesNLOCombined[[i]],FortranForm],{i,nDebyeMasses}],
indent<>indent<>"",
indent<>indent<>"#The scalar masses in the soft limit:",
indent<>indent<>"if order == 0:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesS[[i]]]<>" = "<>ToString[scalarMassesSLO[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"elif order == 1:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesS[[i]]]<>" = "<>ToString[scalarMassesSNLOCombined[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"",
indent<>indent<>"#The couplings in the ultrasoft limit:",
Table[indent<>indent<>ToString[couplingNamesUS[[i]]]<>" = "<>ToString[couplingsUS[[i]],FortranForm],{i,nCouplings}],
indent<>indent<>"",
indent<>indent<>"#The scalar masses in the ultrasoft limit:",
indent<>indent<>"if order == 0:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesUS[[i]]]<>" = "<>ToString[scalarMassesUSLO[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"elif order == 1:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesUS[[i]]]<>" = "<>ToString[scalarMassesUSNLOCombined[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[ToString[#]&/@params3DNamesUS],{"{"->"return array([","}"->"])"}],
indent<>indent<>""}//Flatten;

Print["DRStep done!"];

(* Build the string for VEff3DLO. *)
params3DUSSub = Table[Symbol[StringReplace[ToString[params3DNamesUS[[i]]],"\[LetterSpace]3D\[LetterSpace]US"->""]]->params3DNamesUS[[i]],{i,nParams3DUS}];
DefineNewTensorsUS[\[Mu]ij,\[Lambda]4,\[Lambda]3,gvss,gvvv];
DefineVEVS[SparseArray[\[Phi]VeV3D]];
tensorsVEVSaved = PrintTensorsVEV[]/.gaugeSub/.params3DUSSub;
CalculatePotentialUS[];
(* The "Simplify" seems to make the expression unsymmetric sometimes. However, it appears to be needed to remove spurious imaginary terms. *)
VEff3DLO = PrintEffectivePotential["LO"]/.gaugeSub/.params3DUSSub//Simplify;

VEff3DLOExpr={
indent<>"def VEff3DLO(self, X3D, params3DUS, *auxParams):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 3D effective potential at LO (tree-level).",
indent<>indent<>"",
indent<>indent<>"This function calculates the 3D effective potential at LO in terms of", 
indent<>indent<>"the vevs X3D and the 3D parameters in the ultrasoft limit. Note that",
indent<>indent<>"the vevs X3D are assumed to live in three-dimensional space, so that",
indent<>indent<>"they have mass dimension 1/2. The relation between the three-",
indent<>indent<>"dimensional vevs X3D and the four-dimensional vevs X4D is given by",
indent<>indent<>"X3D = X4D/\[Sqrt]T, where T denotes the temperature.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"X3D : array_like",
indent<>indent<>indent<>"The 3D vevs as either a single point or an array of points",
indent<>indent<>"params3DUS : array",
indent<>indent<>indent<>"Array of the 3D-parameters in the ultrasoft limit",
indent<>indent<>"auxParams : tuple",
indent<>indent<>indent<>"Tuple of auxiliary parameters",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The 3D effective potential at LO",
indent<>indent<>"\"\"\"",
indent<>indent<>"X3D = asarray(X3D)"}//Flatten;
If[nVevs3D==1,VEff3DLOExpr = Join[VEff3DLOExpr,{indent<>indent<>"if X3D.shape != ():",indent<>indent<>indent<>ToString[vevs3D[[1]]]<>" = X3D[...,0]",
											   indent<>indent<>"else:",indent<>indent<>indent<>ToString[vevs3D[[1]]]<>" = X3D"}], Null];
If[nVevs3D>1,
   AppendTo[VEff3DLOExpr,
            indent<>indent<>StringReplace[ToString[vevs3D],Thread[{"{","}"}->""]]<>" = "<>StringReplace[ToString[Table["X3D[...,"<>ToString[i-1]<>"]",{i,nVevs3D}]],{"{","}"}->""]],Null];
VEff3DLOExpr = Join[VEff3DLOExpr,Flatten[{
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[params3DNamesUS],Thread[{"{","}"}->""]]<>" = params3DUS",
indent<>indent<>If[Length[auxParams]>0,If[Length[auxParams]>1,StringReplace[ToString[auxParams],Thread[{"{","}"}->""]]<>" = auxParams",ToString[auxParams[[1]]]<>", = auxParams"],""],
indent<>indent<>"",
indent<>indent<>"return "<>ToString[VEff3DLO,FortranForm],
indent<>indent<>""}]];

Print["Veff3DLO done!"];

(* Build the string for vectMassSq3DUSLO. *)
vMSq3D = tensorsVEVSaved[[2]]//Normal//Simplify;
vMSq3DIdx = ConnectedComponents@AdjacencyGraph[Unitize[vMSq3D] /. Unitize[x_] -> 1];
vMSq3DSinglesElems = {};
vMSq3DBlocksElems = {};
For[i=1,i<=Length[vMSq3DIdx],i++,If[Length[vMSq3DIdx[[i]]]==1 && ToString[vMSq3D[[vMSq3DIdx[[i]][[1]],vMSq3DIdx[[i]][[1]]]]]!="0",
	AppendTo[vMSq3DSinglesElems,vMSq3D[[vMSq3DIdx[[i]][[1]],vMSq3DIdx[[i]][[1]]]]],Null]];
For[i=1,i<=Length[vMSq3DIdx],i++,If[Length[vMSq3DIdx[[i]]]>1,AppendTo[vMSq3DBlocksElems,vMSq3D[[vMSq3DIdx[[i]],vMSq3DIdx[[i]]]]],Null]];
vMSq3DSinglesElems = vMSq3DSinglesElems/.gaugeSub/.params3DUSSub;
vMSq3DBlocksElems = vMSq3DBlocksElems/.gaugeSub/.params3DUSSub;
vMSq3DSinglesNames = Table[Symbol["mVSq"<>ToString[i]],{i,Length[vMSq3DSinglesElems]}];
nSingles = Length[vMSq3DSinglesElems];
nBlocks = Length[vMSq3DBlocksElems];
nBArr = Table[Length[vMSq3DBlocksElems[[i]]],{i,Length[vMSq3DBlocksElems]}];
nBArrAcc = Accumulate[nBArr]; 
vMSq3DBlocksElemsString = vMSq3DBlocksElems;
For[i=1,i<=nBlocks,i++,
	For[j=1,j<=nBArr[[i]],j++,
		For[k=1,k<=nBArr[[i]],k++,
			vMSq3DBlocksElemsString[[i]][[j]][[k]] = ToString[vMSq3DBlocksElems[[i]][[j]][[k]],FortranForm]]]];
vMSq3DBlocksNames = Table[Table[Symbol["mVSq"<>ToString[j]],{j,1+nSingles+nBArrAcc[[i]]-nBArr[[i]],nSingles+nBArrAcc[[i]]}],{i,nBlocks}];
vMSq3DBlocksMatrixNames = Table["A"<>ToString[i],{i,nBlocks}];
vMSq3DBlocksMatrixElems = Table[Table[Table["A"<>ToString[i]<>"[...,"<>ToString[j-1]<>","<>ToString[k-1]<>"]",{k,nBArr[[i]]}],{j,nBArr[[i]]}],{i,nBlocks}];
vMSq3DBlocksMatrixNamesEig = Table[Table["A"<>ToString[i]<>"_eig[...,"<>ToString[j-1]<>"]",{j,nBArr[[i]]}],{i,nBlocks}];

vMSq3DNames = Join[vMSq3DSinglesNames,Flatten[vMSq3DBlocksNames]];

vectMassSq3DUSLOExpr={
indent<>"def vectMassSq3DUSLO(self, X3D, params3DUS, *auxParams):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 3D field dependent vector boson masses.",
indent<>indent<>"",
indent<>indent<>"This function is used to calculate the 3D field dependent vector boson", 
indent<>indent<>"squared masses in the ultrasoft limit in terms of the vevs X3D and",
indent<>indent<>"the 3D parameters in the ultrasoft limit. The masses are calculated at",
indent<>indent<>"LO, i.e. from mass matrix derived from the LO potential VEff3DLO.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"X3D : array_like",
indent<>indent<>indent<>"The 3D vevs as either a single point or an array of points",
indent<>indent<>"params3DUS : array",
indent<>indent<>indent<>"Array of the 3D-parameters in the ultrasoft limit",
indent<>indent<>"auxParams : tuple",
indent<>indent<>indent<>"Tuple of auxiliary parameters",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The 3D field dependent vector boson masses as an array",
indent<>indent<>"\"\"\"",
indent<>indent<>"X3D = asarray(X3D)"}//Flatten;
If[nVevs3D==1,vectMassSq3DUSLOExpr = Join[vectMassSq3DUSLOExpr,{indent<>indent<>"if X3D.shape != ():",indent<>indent<>indent<>ToString[vevs3D[[1]]]<>" = X3D[...,0]",
											                   indent<>indent<>"else:",indent<>indent<>indent<>ToString[vevs3D[[1]]]<>" = X3D"}],Null];
If[nVevs3D>1,
   AppendTo[vectMassSq3DUSLOExpr,
            indent<>indent<>StringReplace[ToString[vevs3D],Thread[{"{","}"}->""]]<>" = "<>StringReplace[ToString[Table["X3D[...,"<>ToString[i-1]<>"]",{i,nVevs3D}]],{"{","}"}->""]],Null];											                   
vectMassSq3DUSLOExpr = Join[vectMassSq3DUSLOExpr,Flatten[{											                   
indent<>indent<>"_shape = "<>ToString[vevs3D[[1]]]<>".shape",
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[params3DNamesUS],Thread[{"{","}"}->""]]<>" = params3DUS",
indent<>indent<>If[Length[auxParams]>0,If[Length[auxParams]>1,StringReplace[ToString[auxParams],Thread[{"{","}"}->""]]<>" = auxParams",ToString[auxParams[[1]]]<>", = auxParams"],""],
indent<>indent<>"_type = "<>ToString[params3DNamesUS[[1]]]<>".dtype",
indent<>indent<>"",
indent<>indent<>"#Vector boson masses which require no diagonalization:",
Table[indent<>indent<>ToString[vMSq3DSinglesNames[[i]]]<>" = "<>ToString[vMSq3DSinglesElems[[i]],FortranForm],{i,nSingles}],
indent<>indent<>"",
indent<>indent<>"#Vector boson masses which require diagonalization:"}]];
For[i=1,i<=nBlocks,i++,
	AppendTo[vectMassSq3DUSLOExpr,indent<>indent<>vMSq3DBlocksMatrixNames[[i]]<>" = empty(_shape+("<>ToString[nBArr[[i]]]<>","<>ToString[nBArr[[i]]]<>"), _type)"];
	For[j=1,j<=nBArr[[i]],j++,
		For[k=1,k<=nBArr[[i]],k++,AppendTo[vectMassSq3DUSLOExpr,indent<>indent<>ToString[vMSq3DBlocksMatrixElems[[i]][[j]][[k]]]<>" = "<>vMSq3DBlocksElemsString[[i]][[j]][[k]]]]];	
	AppendTo[vectMassSq3DUSLOExpr,indent<>indent<>vMSq3DBlocksMatrixNames[[i]]<>"_eig = eigvalsh("<>vMSq3DBlocksMatrixNames[[i]]<>")"];
	AppendTo[vectMassSq3DUSLOExpr,indent<>indent<>StringReplace[ToString[vMSq3DBlocksNames[[i]]],Thread[{"{","}"}->""]]<>" = "<>StringReplace[ToString[vMSq3DBlocksMatrixNamesEig[[i]]],Thread[{"{","}"}->""]]];
	AppendTo[vectMassSq3DUSLOExpr,indent<>indent<>""]];
AppendTo[vectMassSq3DUSLOExpr,indent<>indent<>StringReplace[ToString[ToString[#]&/@vMSq3DNames],{"{"->"return array([","}"->"])"}]];
AppendTo[vectMassSq3DUSLOExpr,indent<>indent<>""];

Print["vectMassSq3DUSLO done!"];

(* Build the string for scalMassSq3DUSLO. *)
sMSq3D = tensorsVEVSaved[[1]]//Normal//Simplify;
sMSq3DIdx = ConnectedComponents@AdjacencyGraph[Unitize[sMSq3D] /. Unitize[x_] -> 1];
sMSq3DSinglesElems = {};
sMSq3DBlocksElems = {};
For[i=1,i<=Length[sMSq3DIdx],i++,If[Length[sMSq3DIdx[[i]]]==1 && ToString[sMSq3D[[sMSq3DIdx[[i]][[1]],sMSq3DIdx[[i]][[1]]]]]!="0",
	AppendTo[sMSq3DSinglesElems,sMSq3D[[sMSq3DIdx[[i]][[1]],sMSq3DIdx[[i]][[1]]]]],Null]];
For[i=1,i<=Length[sMSq3DIdx],i++,If[Length[sMSq3DIdx[[i]]]>1,AppendTo[sMSq3DBlocksElems,sMSq3D[[sMSq3DIdx[[i]],sMSq3DIdx[[i]]]]],Null]];
sMSq3DSinglesElems = sMSq3DSinglesElems/.gaugeSub/.params3DUSSub;
sMSq3DBlocksElems = sMSq3DBlocksElems/.gaugeSub/.params3DUSSub;
sMSq3DSinglesNames = Table[Symbol["mVSq"<>ToString[i]],{i,Length[sMSq3DSinglesElems]}];
nSingles = Length[sMSq3DSinglesElems];
nBlocks = Length[sMSq3DBlocksElems];
nBArr = Table[Length[sMSq3DBlocksElems[[i]]],{i,Length[sMSq3DBlocksElems]}];
nBArrAcc = Accumulate[nBArr]; 
sMSq3DBlocksElemsString = sMSq3DBlocksElems;
For[i=1,i<=nBlocks,i++,
	For[j=1,j<=nBArr[[i]],j++,
		For[k=1,k<=nBArr[[i]],k++,
			sMSq3DBlocksElemsString[[i]][[j]][[k]] = ToString[sMSq3DBlocksElems[[i]][[j]][[k]],FortranForm]]]];
sMSq3DBlocksNames = Table[Table[Symbol["mVSq"<>ToString[j]],{j,1+nSingles+nBArrAcc[[i]]-nBArr[[i]],nSingles+nBArrAcc[[i]]}],{i,nBlocks}];
sMSq3DBlocksMatrixNames = Table["A"<>ToString[i],{i,nBlocks}];
sMSq3DBlocksMatrixElems = Table[Table[Table["A"<>ToString[i]<>"[...,"<>ToString[j-1]<>","<>ToString[k-1]<>"]",{k,nBArr[[i]]}],{j,nBArr[[i]]}],{i,nBlocks}];
sMSq3DBlocksMatrixNamesEig = Table[Table["A"<>ToString[i]<>"_eig[...,"<>ToString[j-1]<>"]",{j,nBArr[[i]]}],{i,nBlocks}];

sMSq3DNames = Join[sMSq3DSinglesNames,Flatten[sMSq3DBlocksNames]];

scalMassSq3DUSLOExpr={
indent<>"def scalMassSq3DUSLO(self, X3D, params3DUS, *auxParams):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 3D field dependent scalar boson masses.",
indent<>indent<>"",
indent<>indent<>"This function is used to calculate the 3D field dependent scalar boson", 
indent<>indent<>"squared masses in the ultrasoft limit in terms of the vevs X3D and",
indent<>indent<>"the 3D parameters in the ultrasoft limit. The masses are calculated at",
indent<>indent<>"LO, i.e. from mass matrix derived from the LO potential VEff3DLO.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"X3D : array_like",
indent<>indent<>indent<>"The 3D vevs as either a single point or an array of points",
indent<>indent<>"params3DUS : array",
indent<>indent<>indent<>"Array of the 3D-parameters in the ultrasoft limit",
indent<>indent<>"auxParams : tuple",
indent<>indent<>indent<>"Tuple of auxiliary parameters",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The 3D field dependent scalar boson masses as an array",
indent<>indent<>"\"\"\"",
indent<>indent<>"X3D = asarray(X3D)"}//Flatten;
If[nVevs3D==1,scalMassSq3DUSLOExpr = Join[scalMassSq3DUSLOExpr,{indent<>indent<>"if X3D.shape != ():",indent<>indent<>indent<>ToString[vevs3D[[1]]]<>" = X3D[...,0]",
											                   indent<>indent<>"else:",indent<>indent<>indent<>ToString[vevs3D[[1]]]<>" = X3D"}],Null];
If[nVevs3D>1,
   AppendTo[scalMassSq3DUSLOExpr,
            indent<>indent<>StringReplace[ToString[vevs3D],Thread[{"{","}"}->""]]<>" = "<>StringReplace[ToString[Table["X3D[...,"<>ToString[i-1]<>"]",{i,nVevs3D}]],{"{","}"}->""]],Null];											                   
scalMassSq3DUSLOExpr = Join[scalMassSq3DUSLOExpr,Flatten[{											                   
indent<>indent<>"_shape = "<>ToString[vevs3D[[1]]]<>".shape",
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[params3DNamesUS],Thread[{"{","}"}->""]]<>" = params3DUS",
indent<>indent<>If[Length[auxParams]>0,If[Length[auxParams]>1,StringReplace[ToString[auxParams],Thread[{"{","}"}->""]]<>" = auxParams",ToString[auxParams[[1]]]<>", = auxParams"],""],
indent<>indent<>"_type = "<>ToString[params3DNamesUS[[1]]]<>".dtype",
indent<>indent<>"",
indent<>indent<>"#Scalar boson masses which require no diagonalization:",
Table[indent<>indent<>ToString[sMSq3DSinglesNames[[i]]]<>" = "<>ToString[sMSq3DSinglesElems[[i]],FortranForm],{i,nSingles}],
indent<>indent<>"",
indent<>indent<>"#Scalar boson masses which require diagonalization:"}]];
For[i=1,i<=nBlocks,i++,
	AppendTo[scalMassSq3DUSLOExpr,indent<>indent<>sMSq3DBlocksMatrixNames[[i]]<>" = empty(_shape+("<>ToString[nBArr[[i]]]<>","<>ToString[nBArr[[i]]]<>"), _type)"];
	For[j=1,j<=nBArr[[i]],j++,
		For[k=1,k<=nBArr[[i]],k++,AppendTo[scalMassSq3DUSLOExpr,indent<>indent<>ToString[sMSq3DBlocksMatrixElems[[i]][[j]][[k]]]<>" = "<>sMSq3DBlocksElemsString[[i]][[j]][[k]]]]];	
	AppendTo[scalMassSq3DUSLOExpr,indent<>indent<>sMSq3DBlocksMatrixNames[[i]]<>"_eig = eigvalsh("<>sMSq3DBlocksMatrixNames[[i]]<>")"];
	AppendTo[scalMassSq3DUSLOExpr,indent<>indent<>StringReplace[ToString[sMSq3DBlocksNames[[i]]],Thread[{"{","}"}->""]]<>" = "<>StringReplace[ToString[sMSq3DBlocksMatrixNamesEig[[i]]],Thread[{"{","}"}->""]]];
	AppendTo[scalMassSq3DUSLOExpr,indent<>indent<>""]];
AppendTo[scalMassSq3DUSLOExpr,indent<>indent<>StringReplace[ToString[ToString[#]&/@sMSq3DNames],{"{"->"return array([","}"->"])"}]];
AppendTo[scalMassSq3DUSLOExpr,indent<>indent<>""];

Print["scalMassSq3DUSLO done!"];

(* Build the string for pressure3DUS. *)
pressure3DUSLO = PrintPressureUS["LO"]/.couplingsSSub/.scalarMassesSSub/.temporalScalarCouplingsSub;
pressure3DUSNLOCombined = pressure3DUSLO + PrintPressureUS["NLO"]/.couplingsSSub/.scalarMassesSSub/.temporalScalarCouplingsSub;

pressure3DUSExpr={
indent<>"def pressure3DUS(self, T, mu4DH, mu3DS, params4D, order, *auxParams):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the pressure in the 3D effective theory in the ultrasoft limit.",
indent<>indent<>"",
indent<>indent<>"This function is used to calculate the pressure in the 3D effective", 
indent<>indent<>"theory in the ultrasoft limit, in terms of the temperature T, the hard",
indent<>indent<>"matching scale mu4DH (\[Mu]4DH), the hard-to-soft matching scale mu3DS",
indent<>indent<>"(\[Mu]3DS) and the values of the 4D-parameters at scale \[Mu]4DH.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"T : float",
indent<>indent<>indent<>"The temperature",
indent<>indent<>"mu4DH : float",
indent<>indent<>indent<>"The hard matching scale (i.e. \[Mu]4DH)",
indent<>indent<>"mu3DS : float",
indent<>indent<>indent<>"The hard-to-soft matching scale (i.e. \[Mu]3DS)",
indent<>indent<>"params4D : array",
indent<>indent<>indent<>"Array of the 4D-parameters at scale \[Mu]4DH",
indent<>indent<>"order : int",
indent<>indent<>indent<>"The order at which the dimensional reduction is performed (0 or 1)",
indent<>indent<>"auxParams : tuple",
indent<>indent<>indent<>"Tuple of auxiliary parameters",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The pressure in the 3D effective theory in the ultrasoft limit",
indent<>indent<>"\"\"\"",
indent<>indent<>"\[Mu] = mu4DH",
indent<>indent<>"\[Mu]3 = mu3DS",
indent<>indent<>"\[Mu]3US = mu3DS"<>" #Temporary fix due to error in DRalgo notation", (*Note! This can be removed after the DRalgo update. *)
indent<>indent<>"Lb = LbFunc(mu4DH,T)",
indent<>indent<>"Lf = LfFunc(mu4DH,T)",
indent<>indent<>"",
indent<>indent<>StringReplace[ToString[params4D],Thread[{"{","}"}->""]]<>" = params4D",
indent<>indent<>If[Length[auxParams]>0,If[Length[auxParams]>1,StringReplace[ToString[auxParams],Thread[{"{","}"}->""]]<>" = auxParams",ToString[auxParams[[1]]]<>", = auxParams"],""],
indent<>indent<>"",
indent<>indent<>"#The couplings in the soft limit:",
Table[indent<>indent<>ToString[couplingNamesS[[i]]]<>" = "<>ToString[couplingsS[[i]],FortranForm],{i,nCouplings}],
indent<>indent<>"",
indent<>indent<>"#The temporal scalar couplings:",
Table[indent<>indent<>ToString[temporalScalarCouplingNames[[i]]]<>" = "<>ToString[temporalScalarCouplings[[i]],FortranForm],{i,nTemporalScalarCouplings}],
indent<>indent<>"",
indent<>indent<>"#The Debye masses:",
indent<>indent<>"if order == 0:",
Table[indent<>indent<>indent<>ToString[DebyeMassNames[[i]]]<>" = "<>ToString[DebyeMassesLO[[i]],FortranForm],{i,nDebyeMasses}],
indent<>indent<>"elif order == 1:",
Table[indent<>indent<>indent<>ToString[DebyeMassNames[[i]]]<>" = "<>ToString[DebyeMassesNLOCombined[[i]],FortranForm],{i,nDebyeMasses}],
indent<>indent<>"",
indent<>indent<>"#The scalar masses in the soft limit:",
indent<>indent<>"if order == 0:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesS[[i]]]<>" = "<>ToString[scalarMassesSLO[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"elif order == 1:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesS[[i]]]<>" = "<>ToString[scalarMassesSNLOCombined[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"",
indent<>indent<>"#The couplings in the ultrasoft limit:",
Table[indent<>indent<>ToString[couplingNamesUS[[i]]]<>" = "<>ToString[couplingsUS[[i]],FortranForm],{i,nCouplings}],
indent<>indent<>"",
indent<>indent<>"#The scalar masses in the ultrasoft limit:",
indent<>indent<>"if order == 0:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesUS[[i]]]<>" = "<>ToString[scalarMassesUSLO[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"elif order == 1:",
Table[indent<>indent<>indent<>ToString[scalarMassNamesUS[[i]]]<>" = "<>ToString[scalarMassesUSNLOCombined[[i]],FortranForm],{i,nScalarMasses}],
indent<>indent<>"",
indent<>indent<>"if order == 0:",
indent<>indent<>indent<>"return "<>ToString[pressure3DUSLO,FortranForm],
indent<>indent<>"elif order == 1:",
indent<>indent<>indent<>"return "<>ToString[pressure3DUSNLOCombined,FortranForm],
indent<>indent<>""}//Flatten;

Print["pressure3DUS done!"];

(* Build the string for the template function for calculateParams4DRef*)
calculateParams4DRefExpr={
"",
"",
"def calculateParams4DRef(mu4DRef, *args, **kwargs):",
indent<>"\"\"\"",
indent<>"Returns the reference value params4DRef for the 4D MS-bar parameters.",
indent<>"",
indent<>"This is a template for a function that can be used to calculate the", 
indent<>"reference value params4DRef at the reference scale mu4DRef in terms",
indent<>"of suitable physical parameters. The main purpose of the template",
indent<>"function is to ensure that the output has the right format (in ",
indent<>"particular that the parameters appear in the right order.)",
indent<>"",
indent<>"Parameters",
indent<>"----------",
indent<>"mu4DRef : float",
indent<>indent<>"The reference value of the 4D RG scale parameter \[Mu]4D",
indent<>"*args",
indent<>indent<>"To be filled in",
indent<>"**kwargs",
indent<>indent<>"To be filled in",
indent<>"",
indent<>"Returns",
indent<>"-------",
indent<>"The reference value params4DRef as an array",
indent<>"\"\"\"",
indent<>"",
Table[indent<>"#"<>ToString[params4D[[i]]]<>" = ",{i,nParams4D}],
indent<>"",
indent<>StringReplace[ToString[ToString[#]&/@params4D],{"{"->"#return array([","}"->"])"}],
indent<>indent<>""}//Flatten;

Print["calculateParams4DRef done!"];

(* Build the string for params4DNames. *)
params4DNamesExpr={
indent<>"def paramsNames4D(self, indices):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 4D parameter names at given indices.",
indent<>indent<>"",
indent<>indent<>"This is an auxiliary function which simply returns the 4D parameter", 
indent<>indent<>"names for given parameter indices.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"indices : array_like",
indent<>indent<>indent<>"An array_like object with the indices of the 4D parameter names",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The 4D parameter names for the given indices as an array_like object",
indent<>indent<>"\"\"\"",
indent<>indent<>"names = np.array(["<>StringReplace[ToString[params4D],Thread[{"{","}"}->""]]<>"])",
indent<>indent<>"if isinstance(indices,int):",
indent<>indent<>indent<>"idx = indices",
indent<>indent<>"else:",
indent<>indent<>indent<>"idx = np.asarray(indices)",
indent<>indent<>"return names[idx]",
indent<>indent<>""}//Flatten;

Print["params4DNames done!"];

(* Build the string for params4DNames. *)
params4DIndicesExpr={
indent<>"def params4DIndices(self, names):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 4D parameter indices for given parameter names.",
indent<>indent<>"",
indent<>indent<>"This is an auxiliary function which simply returns the indices of the ", 
indent<>indent<>"4D parameters for given parameter names. ",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"names : array_like",
indent<>indent<>indent<>"The 4D parameter indices for the given names as an array_like object",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The 4D parameter names for the given indices as an array_like object",
indent<>indent<>"\"\"\"",
indent<>indent<>"storedNames = array(["<>StringReplace[ToString[params4D],Thread[{"{","}"}->""]]<>"])",
indent<>indent<>"if instance(names,str):",
indent<>indent<>indent<>"idx = np.where(storedNames == names)[0]",
indent<>indent<>indent<>"if idx.size == 0:",
indent<>indent<>indent<>indent<>"raise ValueError(\"Invalid parameter name.\")",
indent<>indent<>indent<>"else:",
indent<>indent<>indent<>indent<>"return idx[0]",
indent<>indent<>"soughtNames = np.asarray(names)",
indent<>indent<>"idx = np.empty(names.shape)",
indent<>indent<>"for i in range(0,names.size):",
indent<>indent<>indent<>"idxTemp = np.where(storedNames == soughtNames[i])[0]",
indent<>indent<>indent<>"if idxTemp.size == 0:",
indent<>indent<>indent<>indent<>"raise ValueError(\"Invalid parameter name.\")",
indent<>indent<>indent<>"else:",
indent<>indent<>indent<>indent<>"idx[i] = idxTemp[0]",
indent<>indent<>"return idx",
indent<>indent<>""}//Flatten;

Print["params4DIndices done!"];

(* Build the string for params4DIndices. *)
params3DNamesExpr={
indent<>"def paramsNames3D(self, indices):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 3D parameter names at given indices.",
indent<>indent<>"",
indent<>indent<>"This is an auxiliary function which simply returns the 3D parameter", 
indent<>indent<>"names for given parameter indices.",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"indices : array_like",
indent<>indent<>indent<>"An array_like object with the indices of the 3D parameter names",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The 3D parameter names for the given indices as an array_like object",
indent<>indent<>"\"\"\"",
indent<>indent<>"names = np.array(["<>StringReplace[ToString[params3D],Thread[{"{","}"}->""]]<>"])",
indent<>indent<>"if isinstance(indices,int):",
indent<>indent<>indent<>"idx = indices",
indent<>indent<>"else:",
indent<>indent<>indent<>"idx = np.asarray(indices)",
indent<>indent<>"return names[idx]",
indent<>indent<>""}//Flatten;

Print["params3DNames done!"];

(* Build the string for params3DIndices. *)
params3DIndicesExpr={
indent<>"def params3DIndices(self, names):",
indent<>indent<>"\"\"\"",
indent<>indent<>"Returns the 3D parameter indices for given parameter names.",
indent<>indent<>"",
indent<>indent<>"This is an auxiliary function which simply returns the indices of the ", 
indent<>indent<>"3D parameters for given parameter names. ",
indent<>indent<>"",
indent<>indent<>"Parameters",
indent<>indent<>"----------",
indent<>indent<>"names : array_like",
indent<>indent<>indent<>"The 3D parameter indices for the given names as an array_like object",
indent<>indent<>"",
indent<>indent<>"Returns",
indent<>indent<>"-------",
indent<>indent<>"The 3D parameter names for the given indices as an array_like object",
indent<>indent<>"\"\"\"",
indent<>indent<>"storedNames = np.array(["<>StringReplace[ToString[params3D],Thread[{"{","}"}->""]]<>"])",
indent<>indent<>"if instance(names,str):",
indent<>indent<>indent<>"idx = np.where(storedNames == names)[0]",
indent<>indent<>indent<>"if idx.size == 0:",
indent<>indent<>indent<>indent<>"raise ValueError(\"Invalid parameter name.\")",
indent<>indent<>indent<>"else:",
indent<>indent<>indent<>indent<>"return idx[0]",
indent<>indent<>"soughtNames = np.asarray(names)",
indent<>indent<>"idx = np.empty(names.shape)",
indent<>indent<>"for i in range(0,names.size):",
indent<>indent<>indent<>"idxTemp = np.where(storedNames == soughtNames[i])[0]",
indent<>indent<>indent<>"if idxTemp.size == 0:",
indent<>indent<>indent<>indent<>"raise ValueError(\"Invalid parameter name.\")",
indent<>indent<>indent<>"else:",
indent<>indent<>indent<>indent<>"idx[i] = idxTemp[0]",
indent<>indent<>"return idx",
indent<>indent<>""}//Flatten;

Print["params3DIndices done!"];


(* Print everything to file. *)
outExpr = Riffle[{fileHeaderExpr,classHeaderExpr,RGFuncs4DExpr,DebyeMassSqExpr,DRStepExpr,VEff3DLOExpr,vectMassSq3DUSLOExpr,scalMassSq3DUSLOExpr,pressure3DUSExpr,
				  params4DNamesExpr,params4DIndicesExpr,params3DNamesExpr,params3DIndicesExpr,If[printTemplate,calculateParams4DRefExpr,""]},""]//Flatten;
ostream = OpenWrite[fileName, PageWidth->\[Infinity]];
Export[fileName,StringReplace[outExpr,{"Sqrt"->"sqrt","Pi"->"pi","Log"->"log","Exp"->"exp","\[LetterSpace]"->"_"}],"List"];
Close[ostream];

Print["Export completed!"];

];




