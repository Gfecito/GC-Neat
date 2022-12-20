#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Title::Initialization:: *)
(*Stereo Training Notebook*)


(* ::Input::Initialization:: *)
ec2 = ($Username=="gomezmas");
dataDir = "/Users/santi/Desktop/School/IFT6145/TP3_Stereo/SceneFlowMini"
If[ec2,(
dataDir = "/u/gomezmas/IFT6145/SceneFlowMini"
)]
Print[dataDir]


(* ::Section::Initialization:: *)
(*Functions*)


(* ::Subsection::Initialization:: *)
(*Dataset Functions*)


(* ::Input::Initialization:: *)
$HistoryLength=0;
exportMiniSceneFlow[dir_,i_,data_]:=Block[{idx,rsLeft,rsRight,rsDisp},(
idx=IntegerString[i,10,6];
rsLeft=ImageResize[data["iLeft"],Scaled[1/8]];
rsRight=ImageResize[data["iRight"],Scaled[1/8]];
rsDisp=ImageResize[data["dispLeft"],Scaled[1/8],Resampling->"Nearest"]/8.;
Export[dir<>"left/"<>idx<>".png",rsLeft];
Export[dir<>"right/"<>idx<>".png",rsRight];
Export[dir<>"disp/"<>idx<>".png",rsDisp/64.];
)]
fixDispPng[img_]:=Image[ImageData[img]*64]


(* ::Input::Initialization:: *)
getNamesSceneFlowMini[base_]:=Block[{type,sequence,leftNames,rightNames,dispNames},(
leftNames=FileNames[base<>"/left/*.png"];
rightNames=FileNames[base<>"/right/*.png"];
dispNames=FileNames[base<>"/disp/*.png"];
Transpose[{leftNames,rightNames,dispNames}]
)];


(* ::Input::Initialization:: *)
getDataSceneFlowMini[{leftName_,rightName_,dispName_}]:=Module[{iLeft,iRight,dispLeft},(
iLeft=ImagePartition[Import[leftName],{120,68}]//Flatten;
iRight=ImagePartition[Import[rightName],{120,68}]//Flatten;
dispLeft:=ImagePartition[fixDispPng[Import[dispName]],{120,68}]//Flatten;
MapThread[<|
"dims"->{120,68},
"leftName"->leftName,
"rightName"->rightName,
"dispName"->dispName,
"iLeft"->#1,
"iRight"->#2,
"dispLeft"->#3
|>&,{iLeft,iRight,dispLeft}]
)];


(* ::Subsection::Initialization:: *)
(*Graph Constructs*)


(* ::Input::Initialization:: *)
argmaxLayer[]:=NetGraph[FunctionLayer[Block[{sorted,position},(
sorted=Ordering[-#Input];
position = sorted[[1]];
<|"Output"->position|>
)]&]];
multiPortNest[net_,nb_,inports_,outports_]:=NetGraph[Table[net,{i,nb}],Flatten[MapThread[Table[NetPort[i,#2]->NetPort[i+1,#1],{i,nb-1}]&,{inports,outports}]]
]


(* ::Section::Initialization:: *)
(*Dataset*)


(* ::Input::Initialization:: *)
(*Filenames*)
fn=getNamesSceneFlowMini[dataDir];
fn//Dimensions
fn[[1]]


(* ::Input::Initialization:: *)
(*Import Dataset*)
data=Flatten[getDataSceneFlowMini/@fn];
data//Dimensions
data[[1]]


(* ::Input::Initialization:: *)
(*Split*)
SeedRandom[1];
dataAll=RandomSample[data];
dataTrain=dataAll[[1;;3600]];
dataValid=dataAll[[3601;;4000]];
dataTest=dataAll[[4001;;-1]];
dataTrain//Dimensions
dataValid//Dimensions
dataTest//Dimensions


(* ::Input::Initialization:: *)
If[$Username =="santi",(
dataTrain = dataTrain[[1;;10]];
dataValid = dataValid[[11;;20]];
dataTest = dataTest[[21;;30]];
)]


(* ::Section::Initialization:: *)
(*GC-Net*)


(* ::Input::Initialization:: *)
conv2d[f_,k_]:=Block[{p=(k-1)/2},NetChain[{PaddingLayer[{{0,0},{p,p},{p,p}},Padding->"Fixed"],ConvolutionLayer[f,{k,k}],Ramp}]];
conv2d[f_]:=NetChain[{PaddingLayer[{{0,0},{1,1},{1,1}},Padding->"Fixed"],ConvolutionLayer[f,{3,3}],Ramp}];
conv3d[f_,k_]:=Block[{p=(k-1)/2},NetChain[{PaddingLayer[{{0,0},{p,p},{p,p},{p,p}},Padding->"Fixed"],ConvolutionLayer[f,{k,k,k}],Ramp}]];
conv3d[f_]:=NetChain[{PaddingLayer[{{0,0},{1,1},{1,1},{1,1}},Padding->"Fixed"],ConvolutionLayer[f,{3,3,3}],Ramp}];
upconv3d[f_]:=NetChain[{ResizeLayer[{Scaled[2],Scaled[2],Scaled[2]}],PaddingLayer[{{0,0},{1,1},{1,1},{1,1}},Padding->"Fixed"],ConvolutionLayer[f,{3,3,3}]}]


(* ::Text::Initialization:: *)
(*End-to-End Learning of Geometry and Context for Deep Stereo Regression*)
(*https://arxiv.org/pdf/1703.04309.pdf*)


(* ::Text:: *)
(*Cost volume.*)


(* ::Input::Initialization:: *)
replicateLeft[maxDisp_]:=NetChain[{ReplicateLayer[maxDisp+1],FunctionLayer[Transpose[#,{4,1,2,3}]&]}]
rotateRight[{channels_,rows_,cols_,maxDisp_}]:=NetGraph[Join[
{PaddingLayer[{{0,0},{0,0},{maxDisp,0}},"Fixed"]},
Table[
FunctionLayer[{#[[All,All,1-d+maxDisp;;cols-d+maxDisp]]}&],
{d,0,maxDisp}],
{CatenateLayer[]},
{FunctionLayer[Transpose[#,{4,1,2,3}]&]}
],{
NetPort["Input"]->1->Table[d,{d,2,maxDisp+2}]->maxDisp+3->maxDisp+4->NetPort["Output"]
},"Input"->{channels,rows,cols}]
featuresToCostVolume[{features_,height_,width_},maxDisp_]:=NetGraph@FunctionLayer[Block[{leftVol,rightVol},(
leftVol=replicateLeft[maxDisp][#fLeft];
rightVol=rotateRight[{features,height,width,maxDisp}][#fRight];
<|"Output"->Join[leftVol,rightVol]|>
)]&, "fLeft"->{features,height,width},"fRight"->{features,height,width}];


(* ::Text:: *)
(*Unary features.*)


(* ::Input::Initialization:: *)
featureEncoder[features_]:=NetInsertSharedArrays[NetGraph@FunctionLayer[Block[{conv1,conv2,conv3,conv4,conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17,conv18, residual1, residual2, residual3, residual4,residual5,residual6,residual7,residual8},(
conv1=conv2d[features][#Input];
conv2=conv2d[features][conv1];
conv3=conv2d[features][conv2];
residual1=conv1+conv3;

conv4=conv2d[features][residual1];
conv5=conv2d[features][conv4];
residual2=residual1+conv5;

conv6=conv2d[features][residual2];
conv7=conv2d[features][conv6];
residual3=residual2+conv7;

conv8=conv2d[features][residual3];
conv9=conv2d[features][conv8];
residual4=residual3+conv9;

conv10=conv2d[features][residual4];
conv11=conv2d[features][conv10];
residual5=residual4+conv11;

conv12=conv2d[features][residual5];
conv13=conv2d[features][conv12];
residual6=residual5+conv13;

conv14=conv2d[features][residual6];
conv15=conv2d[features][conv14];
residual7=residual6+conv15;

conv16=conv2d[features][residual7];
conv17=conv2d[features][conv16];
residual8=residual7+conv17;

(* No relu *)
conv18 =ConvolutionLayer[features,{3,3},PaddingSize->1][residual8];

<|"Output"->conv18|>
)]&],"featureNet"]


(* ::Text:: *)
(*Learning regularization.*)


(* ::Input::Initialization:: *)
stereoMatcher[features_]:=NetGraph@FunctionLayer[Block[{conv19,conv20,conv21,conv22,conv23,conv24,conv25,conv26,conv27,conv28,conv29,conv30,conv31,conv32,conv33,conv34,conv35,conv36,conv37,residual1, residual2,residual3,residual4,residual5,residual6,residual7, out},(
conv19=conv3d[features][#Input];
conv20=conv3d[features][conv19];

conv21=conv3d[features*2][conv20];

conv22=conv3d[features*2][conv21];
conv23=conv3d[features*2][conv22];

residual1=conv21+conv23;
conv24=conv3d[features*2][residual1];

conv25=conv3d[features*2][conv24];
conv26=conv3d[features*2][conv25];

residual2=conv24+conv26;
conv27=conv3d[features*2][residual2];

conv28=conv3d[features*2][conv27];
conv29=conv3d[features*2][conv28];

residual3=conv27+conv29;
conv30=conv3d[features*4][residual3];

conv31=conv3d[features*4][conv30];
conv32=conv3d[features*4][conv31];

conv33=conv3d[features*2][conv32];
residual4=conv33+conv29;

conv34=conv3d[features*2][residual4];
residual5=conv34+conv26;

conv35=conv3d[features*2][residual5];
residual6=conv35+conv23;

conv36=conv3d[features][residual6];
residual7=conv36+conv20;

conv37=conv3d[features][residual7];

(* Transposed convolutions
conv33=upconv3d[features*2][conv20];

conv34=upconv3d[features*2][conv21];

conv35=upconv3d[features*2][conv19];

conv36=upconv3d[features][conv20];

conv37=conv3d[features][conv21];
*)

out=NetDelete[conv3d[1],"3"][conv37];
<|"Output"->out|>
)]&]


(* ::Input::Initialization:: *)
trainableGCNet[params_]:=Block[{featureEncoderLayer,features2CostLayer,stereoMatchLayer},(
(*---init---*)
featureEncoderLayer=featureEncoder[params["features"]];
features2CostLayer=featuresToCostVolume[{params["features"],params["inputHeight"],params["inputWidth"]},params["maxDisp"]];
stereoMatchLayer=stereoMatcher[params["features"]];
(*---forward---*)
NetGraph@FunctionLayer[Block[{fLeft,fRight,costVol,match,probabilities,range,prediction,loss},(
fLeft=featureEncoderLayer[#iLeft];
fRight=featureEncoderLayer[#iRight];
costVol=features2CostLayer[<|"fLeft"->fLeft,"fRight"->fRight|>];
match=stereoMatchLayer[costVol];
probabilities=SoftmaxLayer[-1][match];
range=NetArrayLayer["Array"->Range[0,params["maxDisp"]],LearningRateMultipliers->0][];
prediction=probabilities . range;
loss=MeanAbsoluteLossLayer[][<|"Input"->prediction,"Target"->#dispLeft|>];
<|"Output"->prediction,"Loss"->loss|>
)]&,
"iLeft"->NetEncoder[{"Image",{params["inputWidth"],params["inputHeight"]},"RGB"}],
"iRight"->NetEncoder[{"Image",{params["inputWidth"],params["inputHeight"]},"RGB"}],
"dispLeft"->NetEncoder[{"Image",{params["inputWidth"],params["inputHeight"]},"Grayscale"}]
])]


(* ::Input::Initialization:: *)
params=<|"inputHeight"->68,"inputWidth"->120,"features"->4,"maxDisp"->24|>;


(* ::Input::Initialization:: *)
trainableGCNet[params]


(* ::Input::Initialization:: *)
result=NetTrain[
trainableGCNet[params],
dataTrain,
All,
ValidationSet->dataValid,
BatchSize->If[ec2,16,1],
MaxTrainingRounds->If[ec2,128,1],
LossFunction->"Loss",
Method->{"ADAM",LearningRate->0.001},
TargetDevice->If[ec2,"GPU","CPU"],
WorkingPrecision->"Real32"
];


(* ::Input::Initialization:: *)
trainedNet=NetReplacePart[NetTake[result["TrainedNet"],"prediction"],"Output"->NetDecoder[{"Image","Grayscale"}]];
Export["GC-Net7.wlnet", trainedNet]


(* ::Input::Initialization:: *)
predictions=trainedNet[#,TargetDevice->"GPU"]&/@dataTest;
predictions//Dimensions
predictions[[1]]


(* ::Input::Initialization:: *)
i=4;
dataTest[[i,"dispLeft"]]/10
predictions[[i]]/10


(* ::Input::Initialization:: *)
EPE=MapThread[MeanAbsoluteLossLayer[][<|"Input"->ImageData[#1],"Target"->ImageData[#2["dispLeft"]]|>]&,{predictions,dataTest}];


(* ::Input::Initialization:: *)
Mean[EPE]
StandardDeviation[EPE]



