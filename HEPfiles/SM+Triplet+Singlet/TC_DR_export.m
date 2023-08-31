(* ::Package:: *)

SetDirectory[NotebookDirectory[]];
$LoadGroupMath=True;
<</home/etlar/m22_ashar/.Mathematica/Applications/DRalgoUn/DRalgo/DRalgo.m
<</home/etlar/m22_ashar/.Mathematica/Applications/DRalgoUn/DRalgo/DRPythonExport.m


(* ::Chapter:: *)
(*Low-scale Technicolour*)


(* ::Section:: *)
(*Model*)


Group={"SU3","SU2","U1"};
Hdoublet={{{0,0},{1},1/2},"C"}; (* Why not all C or R? *)
aTriplet={{{0,0},{2},0},"R"};
fSinglet={{{0,0},{0},0},"R"};
RepScalar={Hdoublet,aTriplet,fSinglet};
CouplingName={gs,gw,gY};


Rep1={{{1,0},{1},1/6},"L"};
Rep2={{{1,0},{0},2/3},"R"};
Rep3={{{1,0},{0},-1/3},"R"};
Rep4={{{0,0},{1},-1/2},"L"};
Rep5={{{0,0},{0},-1},"R"};
RepFermion1Gen={Rep1,Rep2,Rep3,Rep4,Rep5};


RepFermion3Gen={RepFermion1Gen}//Flatten[#,1]&; (* Need to add two more families *)


RepVector={{1,1},{2},0}; (*For the vector bosons we have a colour octed {1,1}, a Weak Triplet {2}, and no charge under U1*)


(* ::Text:: *)
(*The input for the gauge interactions toDRalgo are then given by*)


{gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC}=AllocateTensors[Group,RepVector,CouplingName,RepFermion3Gen,RepScalar];


(* ::Chapter:: *)
(*Mass terms*)


InputInv={{1,1},{True,False}};  (*\[CapitalPhi]\[CapitalPhi]^+. False-> Hermitian conjugate*)
MassHH=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;

InputInv={{2,2},{True,True}};
MassAA=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;

InputInv={{3,3},{True,True}};
MassFF=CreateInvariant[Group,RepScalar,InputInv]//Simplify//FullSimplify;


VMass=(
	+m22*MassHH
	+mT2*MassAA
	+mS2*MassFF
	);


\[Mu]ij=GradMass[VMass[[1]]]//Simplify//SparseArray;


(* ::Chapter:: *)
(* scalar Cubic operators*)


InputInv={{3,3,3},{True,True,True}}; (*\[Lambda]1 f^3*)
CubicTerm1=CreateInvariant[Group,RepScalar,InputInv][[1]]//Simplify;

InputInv={{3,2,2},{True,True,True}}; (*\[Lambda]2 f Tr(aa)*)
CubicTerm2=CreateInvariant[Group,RepScalar,InputInv][[1]]//Simplify;

InputInv={{3,1,1},{True,False,True}}; (*\[Lambda]3 f H^dagger H*)
CubicTerm3=CreateInvariant[Group,RepScalar,InputInv][[1]]//Simplify;

CubicTerm4 = 1/3^(3/2) (2(
	DRalgo`Private`\[Phi]2[1,1]DRalgo`Private`\[Phi]1[1,1]DRalgo`Private`\[Phi]1[1,2] +
	DRalgo`Private`\[Phi]2[1,2]DRalgo`Private`\[Psi]1[1,1]DRalgo`Private`\[Phi]1[1,2] -
	DRalgo`Private`\[Phi]2[1,2]DRalgo`Private`\[Phi]1[1,1]DRalgo`Private`\[Psi]1[1,2] + 
	DRalgo`Private`\[Phi]2[1,1]DRalgo`Private`\[Psi]1[1,1]DRalgo`Private`\[Psi]1[1,2]) + 
	DRalgo`Private`\[Phi]2[1,3] (DRalgo`Private`\[Phi]1[1,1]^2+DRalgo`Private`\[Psi]1[1,1]^2-DRalgo`Private`\[Phi]1[1,2]^2-DRalgo`Private`\[Psi]1[1,2]^2)
	);


\[Lambda]3=GradCubic[\[Lambda]1H*CubicTerm1+\[Lambda]2H*CubicTerm2+\[Lambda]3H*CubicTerm3+\[Lambda]4H*CubicTerm4];


(* ::Chapter:: *)
(*Scalar Quartic operators*)


QuarticTerm1= MassFF[[1]]^2; (*\[Lambda]5 f^4*)
QuarticTerm2= MassFF[[1]]*MassAA[[1]]; (*\[Lambda]6 f^2 Tr(aa)*)
QuarticTerm3= MassFF[[1]]*MassHH[[1]]; (*\[Lambda]7 f^2 H^dagger H*)
QuarticTerm4= (1/2)*DRalgo`Private`\[Phi]3[1,1]*CubicTerm4; (*\[Lambda]8 f a H^dagger H*)
QuarticTerm5= MassHH[[1]]*MassHH[[1]];(*\[Lambda]9 H^4*)
QuarticTerm6= MassHH[[1]]*MassAA[[1]];(*\[Lambda]10 H^2 a^2*)
QuarticTerm7= (1/2)*MassAA[[1]]*MassAA[[1]];(*\[Lambda]11 a^4*)


VQuartic=(
	+\[Lambda]5H*QuarticTerm1
	+\[Lambda]6H*QuarticTerm2
	+\[Lambda]7H*QuarticTerm3
	+\[Lambda]8H*QuarticTerm4
	+\[Lambda]9H*QuarticTerm5
	+\[Lambda]10H*QuarticTerm6
	+\[Lambda]11H*QuarticTerm7
	);


\[Lambda]4=GradQuartic[VQuartic];


(* ::Chapter:: *)
(*Top Yukawa*)


InputInv={{1,1,2},{False,False,True}}; 
YukawaDoublet=CreateInvariantYukawa[Group,RepScalar,RepFermion3Gen,InputInv]//Simplify;


Ysff=-GradYukawa[yt*YukawaDoublet[[1]]];


YsffC=SparseArray[Simplify[Conjugate[Ysff]//Normal,Assumptions->{yt>0}]];


ImportModelDRalgo[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->False,Mode->2];
PerformDRhard[]//Timing


PConstants = PrintConstants[];


P\[Beta]F4d = BetaFunctions4D[];
Print["BetaFunctions4D"]
P\[Beta]F4d//TableForm;


PScalarMassLO = PrintScalarMass["LO"];
PScalarMassNLO = PrintScalarMass["NLO"];


PDebyeMassLO = PrintDebyeMass["LO"];
PDebyeMassNLO = PrintDebyeMass["NLO"];


PrintPressure["LO"];
PrintPressure["NLO"];


PTemporalScalarCouplings = PrintTemporalScalarCouplings[];


PerformDRsoft[{}];


PCouplingsUS = PrintCouplingsUS[]//Simplify;


PScalarMassUSLO = PrintScalarMassUS["LO"]
PScalarMassUSNLO = PrintScalarMassUS["NLO"]


PrintPressureUS["LO"]
PrintPressureUS["NLO"]


P\[Beta]F3DUS = BetaFunctions3DUS[]


PosScalar = PrintScalarRepPositions[]


\[CurlyPhi]vev = Table[0,PosScalar[[-1,2]]];
\[CurlyPhi]vev[[PosScalar[[1,2]]]] = \[Phi]1;
(*\[CurlyPhi]vev[[PosScalar[[2,2]]]] = \[Phi]2;
\[CurlyPhi]vev[[PosScalar[[3,1]]]] = \[Phi]3;*)
\[CurlyPhi]vev


(* ::Section:: *)
(*Exporting the model to DRPython*)


ImportModelDRalgo[Group,gvvv,gvff,gvss,\[Lambda]1,\[Lambda]3,\[Lambda]4,\[Mu]ij,\[Mu]IJ,\[Mu]IJC,Ysff,YsffC,Verbose->True];

couplingNames = CouplingName;
auxParams = {};
(*\[Phi]VeV3D={0,0,0,\[Phi]1,0,0,\[Phi]2,\[Phi]3};  (* Note! No rescaling by Sqrt[T]. *)*)
\[Phi]VeV3D={0,0,0,\[Phi]1,0,0,0,0};  (* Note! No rescaling by Sqrt[T]. *)
className = "LS_TColor"; (*Note! The name of a python class cannot start with a number, so 2hdm would not work.*)
indent = "    ";
printTemplate = True;
fileName = "./LS_TColor_DRPython.py";

ExportModelToDRPython[couplingNames,auxParams,\[Phi]VeV3D,className,indent,printTemplate,fileName];



