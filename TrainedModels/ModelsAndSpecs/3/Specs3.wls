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
(*DispNetSimple*)


(* ::Text::Initialization:: *)
(*A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation*)
(*https://arxiv.org/pdf/1512.02134.pdf*)


(* ::Input::Initialization:: *)
trainableDispNetSimple[params_,stereoNet_]:=Block[{},(
(*---init---*)

(*---forward---*)
NetGraph@FunctionLayer[Block[{prediction,loss},(
prediction=stereoNet[Join[#iLeft,#iRight]];
loss=MeanAbsoluteLossLayer[][<|"Input"->prediction,"Target"->#dispLeft|>];
<|"Output"->prediction,"Loss"->loss|>
)]&,
"iLeft"->NetEncoder[{"Image",{params["inputWidth"],params["inputHeight"]},"RGB"}],
"iRight"->NetEncoder[{"Image",{params["inputWidth"],params["inputHeight"]},"RGB"}],
"dispLeft"->NetEncoder[{"Image",{params["inputWidth"],params["inputHeight"]},"Grayscale"}]
])]


(* ::Input::Initialization:: *)
params=<|"inputHeight"->68,"inputWidth"->120|>;


(* ::Input::Initialization:: *)
upconv[f_]:=NetChain[{ResizeLayer[{Scaled[2],Scaled[2]}],PaddingLayer[{{0,0},{1,1},{1,1}},Padding->"Fixed"],ConvolutionLayer[f,{3,3}]}]
stereon=NetGraph@FunctionLayer[Block[{conv1,conv2,conv3,conv4,conv5,upconv4,upconv3,upconv2,upconv1},(
conv1=Ramp[ConvolutionLayer[8,{7,7},"Stride"->2,PaddingSize->{3,3}][#Input]];
conv2=Ramp[ConvolutionLayer[16,{5,5},"Stride"->2,PaddingSize->{2,2}][conv1]];
conv3=Ramp[ConvolutionLayer[32,{5,5},"Stride"->2,PaddingSize->{2,2}][conv2]];
conv4=Ramp[ConvolutionLayer[64,{3,3},PaddingSize->{1,1}][conv3]];
upconv3=Ramp[upconv[32][Join[conv3,conv4]]];
upconv2=Ramp[upconv[16][Join[conv2,upconv3[[All,1;;-2]]]]];
upconv1=upconv[1][Join[conv1,upconv2]];
<|"Output"->upconv1|>
)]&]



(* ::Input::Initialization:: *)
g=trainableDispNetSimple[params,stereon]


(* ::Input::Initialization:: *)
g=NetInitialize[g]


(* ::Input::Initialization:: *)
g[dataTrain[[1]]]


(* ::Input::Initialization:: *)
result=NetTrain[
trainableDispNetSimple[params,stereon],
dataTrain,
All,
ValidationSet->dataValid,
BatchSize->If[ec2,16,2],
MaxTrainingRounds->If[ec2,128,2],
LossFunction->"Loss",
Method->{"ADAM",LearningRate->0.001},
TargetDevice->If[ec2,"GPU","CPU"],
WorkingPrecision->"Real32"
];


(* ::Input::Initialization:: *)
trainedNet=NetReplacePart[NetTake[result["TrainedNet"],"prediction"],"Output"->NetDecoder[{"Image","Grayscale"}]]
Export["Simpleton-Net.wlnet", trainedNet]


(* ::Input::Initialization:: *)
predictions=trainedNet[#,TargetDevice->"CPU"]&/@dataTest;
predictions//Dimensions
predictions[[1]] // ImageAdjust


(* ::Input::Initialization:: *)
i=4;
dataTest[[i,"dispLeft"]]/10
predictions[[i]]/10


(* ::Input::Initialization:: *)
EPE=MapThread[MeanAbsoluteLossLayer[][<|"Input"->ImageData[#1],"Target"->ImageData[#2["dispLeft"]]|>]&,{predictions,dataTest}];


(* ::Input::Initialization:: *)
Mean[EPE]
StandardDeviation[EPE]


(* ::Section::Initialization:: *)
(*GC-Net*)


(* ::Input::Initialization:: *)
conv2d[f_,k_]:=Block[{p=(k-1)/2},NetChain[{PaddingLayer[{{0,0},{p,p},{p,p}},Padding->"Fixed"],ConvolutionLayer[f,{k,k}],Ramp}]];
conv2d[f_]:=NetChain[{PaddingLayer[{{0,0},{1,1},{1,1}},Padding->"Fixed"],ConvolutionLayer[f,{3,3}],Ramp}];
conv3d[f_,k_]:=Block[{p=(k-1)/2},NetChain[{PaddingLayer[{{0,0},{p,p},{p,p},{p,p}},Padding->"Fixed"],ConvolutionLayer[f,{k,k,k}],Ramp}]];
conv3d[f_]:=NetChain[{PaddingLayer[{{0,0},{1,1},{1,1},{1,1}},Padding->"Fixed"],ConvolutionLayer[f,{3,3,3}],Ramp}];


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
featureEncoder[features_]:=NetInsertSharedArrays[NetGraph@FunctionLayer[Block[{conv1,conv2,conv3,conv4,conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12, conv13, conv14, conv15, conv16, conv17},(
conv1=conv2d[features][#Input];
conv2=conv2d[features][conv1];
conv3=conv2d[features][conv2];

conv4=conv2d[features][conv3];
conv5=conv2d[features][conv4];
conv6=conv2d[features][conv5];
conv7=conv2d[features][conv6];
conv8=conv2d[features][conv7];
conv9=conv2d[features][conv8];
conv10=conv2d[features][conv9];
conv11=conv2d[features][conv10];
conv12=conv2d[features][conv11];
conv13=conv2d[features][conv12];
conv14=conv2d[features][conv13];
conv15=conv2d[features][conv14];
conv16=conv2d[features][conv15];
conv17=conv2d[features][conv16];

(* No relu *)
conv18 = NetChain[{PaddingLayer[{{0,0},{1,1},{1,1}},Padding->"Fixed"],ConvolutionLayer[32,{3,3}]}][conv17];

<|"Output"->conv17|>
)]&],"featureNet"]


(* ::Text:: *)
(*Learning regularization.*)


(* ::Input::Initialization:: *)
stereoMatcher[features_]:=NetGraph@FunctionLayer[Block[{conv1,conv2,conv3,conv4,out},(
conv1=conv3d[features][#Input];
conv2=conv3d[features][conv1];
conv3=conv3d[features][conv2];
conv4=conv3d[features][conv3];
out=NetDelete[conv3d[1],"3"][conv4];
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
BatchSize->If[ec2,16,2],
MaxTrainingRounds->If[ec2,128,2],
LossFunction->"Loss",
Method->{"ADAM",LearningRate->0.001},
TargetDevice->If[ec2,"GPU","CPU"],
WorkingPrecision->"Real32"
];


(* ::Input::Initialization:: *)
trainedNet=NetReplacePart[NetTake[result["TrainedNet"],"prediction"],"Output"->NetDecoder[{"Image","Grayscale"}]];
Export["GC-Net.wlnet", trainedNet]


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



