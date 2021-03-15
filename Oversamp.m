%The training and test datasets as well as training labels need to be imported into matlab first
%With variable names testdata and traindata, trainlabels
index1 = find(trainlabels{:,:} == 1);
index2 = find(trainlabels{:,:} == 2);
index3 = find(trainlabels{:,:} == 3);
index4 = find(trainlabels{:,:} == 4);
index5 = find(trainlabels{:,:} == 5);
index6 = find(trainlabels{:,:} == 6);
index7 = find(trainlabels{:,:} == 7);
index8 = find(trainlabels{:,:} == 8);
index9 = find(trainlabels{:,:} == 9);
index10 = find(trainlabels{:,:} == 10);

N_target = size(index1,1);
SData = Data;
SLabels = trainlabels;

N2 = size(index2,1);

for i = 1:(N2)
    j = randi(N2);
    SData = [SData; SData(index2(j),:)];
    SLabels = [SLabels; SLabels(index2(j),:)];
end

N3 = size(index3,1);

for i = 1:(N3)
    j = randi(N3);
    SData = [SData; SData(index3(j),:)];
    SLabels = [SLabels; SLabels(index3(j),:)];
end

N4 = size(index4,1);

for i = 1:(N4)
    j = randi(N4);
    SData = [SData; SData(index4(j),:)];
    SLabels = [SLabels; SLabels(index4(j),:)];
end

N5 = size(index5,1);

for i = 1:(N5)
    j = randi(N5);
    SData = [SData; SData(index5(j),:)];
    SLabels = [SLabels; SLabels(index5(j),:)];
end

N6 = size(index6,1);

for i = 1:(N6)
    j = randi(N6);
    SData = [SData; SData(index6(j),:)];
    SLabels = [SLabels; SLabels(index6(j),:)];
end

N7 = size(index7,1);

for i = 1:(N7)
    j = randi(N7);
    SData = [SData; SData(index7(j),:)];
    SLabels = [SLabels; SLabels(index7(j),:)];
end

N8 = size(index8,1);

for i = 1:(N8)
    j = randi(N8);
    SData = [SData; SData(index8(j),:)];
    SLabels = [SLabels; SLabels(index8(j),:)];
end
%% 

N9 = size(index9,1);

for i = 1:(N9)
    j = randi(N9);
    SData = [SData; SData(index9(j),:)];
    SLabels = [SLabels; SLabels(index9(j),:)];
end

N10 = size(index10,1);

for i = 1:(N10)
    j = randi(N10);
    SData = [SData; SData(index10(j),:)];
    SLabels = [SLabels; SLabels(index10(j),:)];
end

SLabels.Properties.VariableNames{1} = 'Labels';
SData_m = [SData SLabels];

writetable(SData, 'SData.csv')
writetable(SLabels, 'SLabels.csv')