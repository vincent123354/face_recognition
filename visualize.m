%% preprocess
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



%% for display only
faceDetector = vision.CascadeObjectDetector();

img = imread(uigetfile(strcat(pwd, '\*.jpg')));
gray_img = rgb2gray(img);
bbox = step(faceDetector, gray_img);
crop_img = imcrop(img, bbox);

subplot(221);
imshow(img);
title('RGB Image');
subplot(222);
imshow(gray_img);
title('Grayscale Image')
subplot(223);
imshow(crop_img);
title('Cropped Image')
subplot(224);
imshow(insertObjectAnnotation(img, 'rectangle', bbox, 'test'));
title('Image with bounding box on face');

%% display faces
gallery = imageSet('member_face','recursive');
gallery_name = {gallery.Description};
displayFaceGallery(gallery, gallery_name);
disp(1);
disp(dataset(1).Description);


%% display cropped faces

gallery = imageSet('temp','recursive');
gallery_name = {gallery.Description};
displayFaceGallery(gallery, gallery_name);
disp(1);
disp(dataset(1).Description);

%% plot data distribution
imds = imageDatastore('member_face', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
label_mapping = {'Clement','KaChun','eunice','pyx','vincent'}
label_count = countcats(imds.Labels)
bar(categorical(label_mapping), label_count)

%% Display Face Gallery
%Copyright 2014-2015 The MathWorks, Inc. 
function displayFaceGallery(faceGallery,galleryNames)
% displaying all the gallery images
figure
for i = 1:length(faceGallery)
    I = cell(1, faceGallery(i).Count);
    % concatenate all the images of a person side-by-side
    for j = 1:faceGallery(i).Count
        image = read(faceGallery(i), j);        
        % scale images to have same height, maintaining the aspect ratio
        scaleFactor = 150/size(image, 1); 
        image = imresize(image, scaleFactor);        
        I{j} = image;
    end   
    subplot(length(faceGallery), 1, i);
    imshow(cell2mat(I));
    title(galleryNames{i}, 'FontWeight', 'normal');
end
annotation('textbox', [0 0.9 1 0.1], 'String', 'Face Gallery', ...
     'EdgeColor', 'none', 'FontWeight', 'bold', ...
     'FontSize', 12, 'HorizontalAlignment', 'center')
end
