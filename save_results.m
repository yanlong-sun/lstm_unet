clc;
clear;

% masks' path
masks_path = '../Dataset/test_data/test_data_bmp/masks/'; % mask 
preds_path = '../predictions/';     % pred of the network
final_preds_path = '../png_pred_results_lstm/'; % classified by name
save_path = '../pred_compare/2p5d_lstm/';  % nii 

masks_folder=dir(masks_path);
masks_file= {masks_folder.name};
i = 0;  % index for the pred masks 
for num_masks = 3 : length(masks_file)
    
    mask_name = masks_file(num_masks);
   	mask = imread([masks_path char(mask_name)]);
    pred_path_name = [preds_path, num2str(i), '.png'];
    %%%%%%
    pred = im2gray(imread(pred_path_name));
    pred = imbinarize(pred); 
    %pred = bwareaopen(pred, 50);
    pred = imfill(pred, 'holes');
   
    case_name = char(mask_name);
    case_name = case_name(1:end-10);
    if exist([final_preds_path case_name, '/'],'dir')==0
        mkdir([final_preds_path case_name, '/']);
    end
    imwrite(pred, [final_preds_path case_name, '/', char(mask_name)]);
    disp([final_preds_path case_name, '/', char(mask_name)]);
    i = i+1;    
end


prediction_path = final_preds_path;
pred_folder= dir(prediction_path);
pred_file={pred_folder.name};

for num_pred= 3 : length(pred_file)
    case_name = pred_file(num_pred);
    case_name = char(case_name);
    finishing = [num2str(num_pred-3),'/',num2str(length(pred_file)-3)];
    disp(finishing)
    disp(case_name)
    
    v_orig = load_untouch_nii(['../Dataset/test_data/test_data_nii/slices/', case_name, '.nii.gz']);
    v2 = v_orig;
    v3 = v_orig;
    v4 = v_orig;
    [a1, a2, a3] = size(v_orig.img);
    
    %% Image Part
    first_slice_path = ['../Dataset/test_data/test_data_bmp/slices/', case_name,'_', num2str(10001), '.bmp' ];
    slices = im2gray(imread(first_slice_path));    
    slices = get_original_size(slices, a1, a2);
    
    for i = 2 : a3
        single_slice_path = ['../Dataset/test_data/test_data_bmp/slices/', case_name,'_', num2str(10000+i), '.bmp' ];
        single_slice = im2gray(imread(single_slice_path));
        single_slice = get_original_size(single_slice, a1, a2);
        figure(1)
        imshow(single_slice)
        slices = cat(3, slices, single_slice);
    end
    %% Pred Part
    for j = 1 : a3
        if j == 1
            preds = imread([prediction_path, case_name, '/', case_name,'_', num2str(10001), '.bmp' ]);
            %preds = im2gray(preds);
            masks = imread(['../Dataset/test_data/test_data_bmp/masks/', case_name,'_', num2str(10001), '.bmp' ]);
            
            preds = get_original_size(preds, a1, a2);
            masks = get_original_size(masks, a1, a2);
        else
            single_pred = imread([prediction_path case_name, '/', case_name,'_', num2str(10000+j), '.bmp' ]);
            single_mask = imread(['../Dataset/test_data/test_data_bmp/masks/', case_name,'_', num2str(10000+j), '.bmp' ]);
            
            single_pred = get_original_size(single_pred, a1, a2);
            single_mask = get_original_size(single_mask, a1, a2);
            
            preds = cat(3, preds, single_pred);
            masks = cat(3, masks, single_mask);
        end
    end


    %% SAVE part

    [x1, x2, x3] = size(slices);
    a = [x1, x2, x3];
    v2.hdr.dime.dim = [3, a1, a2, a3, 1, 1, 1, 1];
    v3.hdr.dime.dim = [3, a1, a2, a3, 1, 1, 1, 1];
    v4.hdr.dime.dim = [3, a1, a2, a3, 1, 1, 1, 1];
    v2.img = slices;
    v3.img = preds;
    v4.img = masks;
    if exist([save_path, case_name, '/'],'dir')==0
        mkdir([save_path, case_name, '/']);
    end
    save_untouch_nii(v2, [save_path, case_name, '/', case_name, '_image.nii']);
    save_untouch_nii(v3, [save_path, case_name, '/', case_name, '_pred.nii']);
    save_untouch_nii(v4, [save_path, case_name, '/', case_name, '_mask.nii']);

end

function [ image ] = get_original_size(image, n1, n2)
% n: original size

    

    
    if and(n1<256, n2<256) 
        num_pad_n1 = 256-n1;
        num_pad_n1_half = round(num_pad_n1/2);
        num_pad_n2 = 256-n2; 
        num_pad_n2_half = round(num_pad_n2/2);
        image = imcrop(image, [num_pad_n2_half, num_pad_n1_half, n2-1, n1-1]);
    end
    
    if and(n1<256, n2>=256)
        num_pad_n1 = 256-n1;
        num_pad_n1_half = round(num_pad_n1/2);
        image = imcrop(image, [1, num_pad_n1_half, n2 ,n1-1 ]);
    end
    
    if and(n1>=256, n2<256)
       num_pad_n2 = 256-n2; 
       num_pad_n2_half = round(num_pad_n2/2);       
       image = imcrop(image, [num_pad_n2_half, 1, n2-1, n1]);
    end
    
 
    if or(n1>256, n2>256)
        if n1>256
            num_pad_n1 = n1 - 256;
            num_pad_n1_half = round(num_pad_n1/2);
            image = padarray(image, [num_pad_n1_half, 0], 'pre');
            image = padarray(image, [num_pad_n1 - num_pad_n1_half, 0], 'post');
        end
 
        if n2>256
            num_pad_n2 = n2-256; 
            num_pad_n2_half = round(num_pad_n2/2);
            image = padarray(image, [0, num_pad_n2_half], 'pre');
            image = padarray(image, [0, num_pad_n2 - num_pad_n2_half ], 'post');          
        end   
    end
end