(* ::Package:: *)

(*Quit[];*)


SetDirectory[NotebookDirectory[]];
$LoadGroupMath=True;
<<../DRalgo.m
<<../DRPythonExport.m


(* ::Chapter:: *)
(*2HDM-Two Higgs doublet model*)


(*See 1106.0034 [hep-ph] for a review*)


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2","U1"};
RepAdjoint={{1,1},{2},0};
HiggsDoublet1={{{0,0},{1},1/2},"C"};
HiggsDoublet2={{{0,0},{1},1/2},"C"};
RepScalar={HiggsDoublet1,HiggsDoublet2};
CouplingName={g3,g2,g1};


Rep1={{{1,0},{1},1/6},"L"};
Rep2={{{1,0},{0},2/3},"R"};
Rep3={{{1,0},{0},-1/3},"R"};
Rep4={{{0,0},{1},-1/2},"L"};
Rep5={{{0,0},{0},-1},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3,Rep4,Rep5};


RepFermion3Gen={RepFermion1Gen,RepFermion1Gen,RepFermion1Gen}//Flatten[#,1]&;


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepAdjoint,CouplingName,RepFermion3Gen,RepScalar];


(* ::Text:: *)
(*The first element is the vector self - interaction matrix :*)


InputInv={{1,1},{True,False}};
MassTerm1=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{2,2},{True,False}};
MassTerm2=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{1,2},{True,False}};
MassTerm3=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;
InputInv={{2,1},{True,False}};
MassTerm4=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;


VMass=(
	+M11*MassTerm1
	+M22*MassTerm2
	+M12(MassTerm3+MassTerm4)
	);


\[Mu]ij=GradMass[VMass[[1]]]//Simplify//SparseArray


QuarticTerm1=MassTerm1[[1]]^2;
QuarticTerm2=MassTerm2[[1]]^2;
QuarticTerm3=MassTerm1[[1]]*MassTerm2[[1]];
QuarticTerm4=MassTerm3[[1]]*MassTerm4[[1]];
QuarticTerm5=(MassTerm3[[1]]^2+MassTerm4[[1]]^2)//Simplify;

QuarticTerm6=MassTerm1[[1]]MassTerm3[[1]]+MassTerm1[[1]]MassTerm4[[1]];
QuarticTerm7=MassTerm2[[1]]MassTerm3[[1]]+MassTerm2[[1]]MassTerm4[[1]];


VQuartic=(
	+lam1*QuarticTerm1
	+lam2*QuarticTerm2
	+lam3*QuarticTerm3
	+lam4*QuarticTerm4
	+lam5*QuarticTerm5
	);


\[Lambda]4=GradQuartic[VQuartic];


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet1=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;


InputInv={{2,1,2},{False,False,True}}; 
YukawaDoublet2=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;


(*Ysff=-GradYukawa[yt1*YukawaDoublet1[[1]]+yt2*YukawaDoublet2[[1]]];*)
Ysff=-GradYukawa[yt*YukawaDoublet2[[1]]];


YsffC=SparseArray[Simplify[Conjugate[Ysff]//Normal,Assumptions->{yt1>0,yt2>0}]];


(* ::Section:: *)
(*Dimensional Reduction*)


ImportModelDRalgo[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];
PerformDRhard[]


BetaFunctions4D[]


PrintScalarMass["LO"]
PrintScalarMass["NLO"]


PrintDebyeMass["LO"]
PrintDebyeMass["NLO"]


PrintCouplings[]
PrintTemporalScalarCouplings[]


(* ::Text:: *)
(*Two active doublets:*)


PerformDRsoft[{}];


PrintCouplingsUS[]//Simplify


PrintScalarMassUS["LO"]
PrintScalarMassUS["NLO"]


BetaFunctions3DUS[]


(* ::Text:: *)
(*One active doublet :*)


PerformDRsoft[{5,6,7,8}];


PrintCouplingsUS[]


PrintScalarMassUS["LO"]
PrintScalarMassUS["NLO"]


BetaFunctions3DUS[]


PrintPressureUS["LO"]
PrintPressureUS["NLO"]


PosScalar = PrintScalarRepPositions[]


\[CurlyPhi]vev = Table[0,PosScalar[[-1,2]]];
\[CurlyPhi]vev[[PosScalar[[1,2]]]] = \[Phi]1;
\[CurlyPhi]vev[[PosScalar[[2,2]]]] = \[Phi]2;
\[CurlyPhi]vev


(* ::Section:: *)
(*Saving the model*)


(*result={};
AppendTo[result,Row[{
	TexFor["DRDRDRDRDRDRDRDRDRDRDRDRDRDR "],
	TexFor["DRalgo"],
	TexFor[" DRDRDRDRDRDRDRDRDRDRDRDRDRDRD"]}]];
	AppendTo[result,Row[{"Model: "//TexFor,"Two-Higgs doublet model. See hep-ph:1106.0034 for further details"//TexFor}]];
AppendTo[result,Row[{"Version: "//TexFor,"1.0 beta (16-05-2022)"//TexFor}]];
AppendTo[result,Row[{"Authors: "//TexFor,"Andreas Ekstedt, Philipp Schicho, Tuomas V.I. Tenkanen"//TexFor}]];
AppendTo[result,Row[{"Reference: "//TexFor,"2205.xxxxx [hep-ph]"//TexFor}]];
AppendTo[result,Row[{"Repository link: "//TexFor,
	Hyperlink[Mouseover[TexFor["github.com/DR-algo/DRalgo"],Style["github.com/DR-algo/DRalgo",Bold]],
	"https://github.com/DR-algo/DRalgo"]}]];
AppendTo[result,Style["DRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRDRD",{GrayLevel[0.3]}]];*)


(*ModelInfo=result;*)


(*SaveModelDRalgo[ModelInfo,"2hdm.txt"]*)


(* ::Section:: *)
(*Loading the model*)


(*{Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=LoadModelDRalgo["2hdm.txt"];*)


(*ImportModelDRalgo[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False];
PerformDRhard[]*)


(* ::Section:: *)
(*Exporting the model to DRPython*)


ImportModelDRalgo[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->True];

couplingNames = CouplingName;
auxParams = {};
\[Phi]VeV3D={0,0,0,\[Phi]1,0,0,0,\[Phi]2};  (* Note! No rescaling by Sqrt[T]. *)
className = "THDM"; (*Note! The name of a python class cannot start with a number, so 2hdm would not work.*)
indent = "    ";
printTemplate = True;
fileName = "./THDM_DRPython.py";

ExportModelToDRPython[couplingNames,auxParams,\[Phi]VeV3D,className,indent,printTemplate,fileName];



