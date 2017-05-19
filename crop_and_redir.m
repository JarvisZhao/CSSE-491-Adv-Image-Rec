close all force
clear variables
directory = '/Users/jarvis/Desktop/CSSE491/';
load('cars_annos.mat');

for i = 1:size(annotations,2)
    
    names = getfield(annotations(i),'relative_im_path');
    class = getfield(annotations(i),'class');
    class_name = char(class_names(class));
    img = imread([directory '/' names]);
   
    info = imfinfo([directory '/' names]);
    x1 = getfield(annotations(i),'bbox_x1');
    y1 = getfield(annotations(i),'bbox_y1');
    x2 = getfield(annotations(i),'bbox_x2');
    y2 = getfield(annotations(i),'bbox_y2');
    if info.ColorType =='truecolor'
    i
    img = img(y1:y2,x1:x2,:);
    len = max(y2-y1+1,x2-x1+1);
    newImg = uint8(zeros(len,len,3));
    
    if y2-y1 ~= x2-x1
    if y2-y1 > x2-x1
        diff = (y2-y1) - (x2-x1);
        
        newImg(1:y2-y1+1, uint32(diff/2):uint32(diff/2)+x2-x1,1) = img(:,:,1);
        newImg(1:y2-y1+1, uint32(diff/2):uint32(diff/2)+x2-x1,2) = img(:,:,2);
        newImg(1:y2-y1+1, uint32(diff/2):uint32(diff/2)+x2-x1,3) = img(:,:,3);
        
        
    else  
        diff = (x2-x1) - (y2 - y1);
        newImg(uint32(diff/2):uint32(diff/2)+y2-y1, 1:x2-x1+1,1) = img(:,:,1);
        newImg(uint32(diff/2):uint32(diff/2)+y2-y1, 1:x2-x1+1,2) = img(:,:,2);
        newImg(uint32(diff/2):uint32(diff/2)+y2-y1, 1:x2-x1+1,3) = img(:,:,3);
   
    end
    end
    img = newImg;
    img = rgb2gray(img);
    if getfield(annotations(i),'test') == 1
        if size(strfind(class_name, 'SUV'),1) ~=0
            imwrite(img,fullfile('Images/Train/SUV','/', names(9:18)));
        elseif  size(strfind(class_name, 'Sedan'),1) ~=0
            imwrite(img,fullfile('Images/Train/Sedan','/', names(9:18)));
            
        elseif size(strfind(class_name, 'Hatchback'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Train/Hatchback','/', names(9:18)));
            
        elseif size(strfind(class_name, 'Coupe'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Train/Coupes','/', names(9:18)));
            
        elseif size(strfind(class_name, 'Convertible'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Train/Convertibles','/', names(9:18)));
            
        elseif size(strfind(class_name, 'Wagon'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Train/Station_Wagon','/', names(9:18)));
            
        elseif  size(strfind(class_name, 'Van'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Train/Van','/', names(9:18)));
            
        elseif  size(strfind(class_name, 'Minivan'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Train/Minivan','/', names(9:18)));
            
        elseif size(strfind(class_name, 'Cab'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Train/Truck','/', names(9:18)));
            
        end
    end
    
    if getfield(annotations(i),'test') == 0
        
        if size(strfind(class_name, 'SUV'),1) ~=0
            
            imwrite(img,fullfile('Images/Test/SUV','/', names(9:18)));
            
        elseif  size(strfind(class_name, 'Sedan'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Test/Sedan','/', names(9:18)));
            
        elseif  size(strfind(class_name, 'Hatchback'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Test/Hatchback','/', names(9:18)));
            
        elseif size(strfind(class_name, 'Convertible'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Test/Convertibles','/', names(9:18)));
            
        elseif size(strfind(class_name, 'Coupe'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Test/Coupes','/', names(9:18)));
            
        elseif  size(strfind(class_name, 'Wagon'),1) ~=0
            
            imwrite(img,fullfile('Images/Test/Station_Wagon','/', names(9:18)));
            
        elseif  size(strfind(class_name, 'Van'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Test/Van','/', names(9:18)));
            
            
        elseif  size(strfind(class_name, 'Minivan'),1) ~=0
            
            imwrite(img,fullfile('Images/Test/Minivan','/', names(9:18)));
            
        elseif  size(strfind(class_name, 'Cab'),1) ~=0
            
            
            imwrite(img,fullfile('Images/Test/Truck','/', names(9:18)));
            
        end
    end
    end
end