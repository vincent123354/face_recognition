%% preprocess
mkdir('temp');
faceDetector = vision.CascadeObjectDetector();

% Get a list of all files and folders in this folder.
files = dir('member_face');

%temp_folders = zeros(size(files, 1)-2, 1)

% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);
% Print folder names to command window.
for k = 3 : length(subFolders)
    initial_folders{k-2} = fullfile(subFolders(k).folder, subFolders(k).name);
    temp_folders{k-2} = subFolders(k).name;
end

faceDetector = vision.CascadeObjectDetector();

for i=1:length(initial_folders)
    save_dir = fullfile('temp', temp_folders{i});
    mkdir(save_dir);
    
    current_dir = dir(initial_folders{i});
    for j=3:length(current_dir)
        image_name = current_dir(j).name;
        image_path = fullfile(initial_folders{i}, image_name);
        image = imread(image_path);
        bbox = step(faceDetector, image);
        if size(bbox, 1) == 1
            crop_img = imcrop(image, bbox);
            resize_img = imresize(crop_img, [256 256]);
            imwrite(resize_img, fullfile(save_dir, image_name));
        end
    end
end


%%  https://www.mathworks.com/help/matlab/ref/matlab.io.datastore.imagedatastore.html
imds = imageDatastore('temp', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

label_mapping = unique(imds.Labels);

% imds.ReadFcn = @(loc) imresize(imread(loc), [256,256]);

[imdsTrain,imdsValidation] = splitEachLabel(imds, 0.7, 'randomize');

%% https://www.mathworks.com/help/deeplearning/ref/imagedataaugmenter.html
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
input_size = [224 224 3];
augimdsTrain = augmentedImageDatastore(input_size,imdsTrain,'DataAugmentation',imageAugmenter);
augimdsVal = augmentedImageDatastore(input_size,imdsValidation);



%% vgg16
numClasses=5;
analyzeNetwork(vgg16);
net = vgg16;
lgraph = layerGraph(net.Layers);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers = freezeWeights(layers(1:37));
layers(38) = fullyConnectedLayer(numClasses, 'Name', 'custom_fc8');
layers(39) = softmaxLayer('Name', 'custom_softmax');
layers(40) = classificationLayer('Name', 'custom_classification_layer');
analyzeNetwork(layers);

%% https://www.mathworks.com/help/deeplearning/ug/train-deep-learning-network-to-classify-new-images.html
% https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html
% https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 3, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization', 0.01, ...
    'ValidationData',augimdsVal, ...
    'ValidationFrequency',1, ...
    'Verbose',1, ...
    'VerboseFrequency', 1, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,layers,options);
%%

save('model.mat', 'net')

%%
net = load(uigetfile(strcat(pwd, '\*.mat')));
net = net.net;

%%
[y_pred, y_prob] = classify(net, augimdsVal);
accuracy = mean(y_pred == imdsValidation.Labels);
disp(accuracy);
% https://www.mathworks.com/help/stats/confusionmat.html
cm = confusionchart(imdsValidation.Labels, y_pred);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';



%% https://www.mathworks.com/matlabcentral/answers/445244-googlenet-undefined-function-or-variable-freezeweights

function layers = freezeWeights(layers)

for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

end



