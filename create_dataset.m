%%
name = "vincent";
folder_name = "webcam";
path = strcat(folder_name, '/',name);
mkdir(path)

%%
img_name = "img_";

face_detector = vision.CascadeObjectDetector();
avi_path = strcat(pwd, '\', folder_name,'\',name,'.avi');
avi = VideoWriter(convertStringsToChars(avi_path));
open(avi);

cam = webcam;
count = 0;

for i=1:50
    disp(i);
    img = snapshot(cam);
    bbox = step(face_detector, img);
    annot_img = insertObjectAnnotation(img, 'rectangle', bbox, 'face');
    if size(bbox, 1) > 0
        for j=1:size(bbox,1)
            crop_img = imcrop(img, bbox);
            imwrite(crop_img, convertStringsToChars(strcat(path, '\', img_name, int2str(count), '.jpg')));
            count = count + 1;
        end
    end
    writeVideo(avi, annot_img);
end
close(avi);
clear('cam');
disp(count);

