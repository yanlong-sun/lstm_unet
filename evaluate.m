clc;
clear;
%% Evaluate 2.5d dense unet model
prediction_path = '../pred_compare/2p5d_lstm/';
masks_path = '../Dataset/test_data/test_data_nii/masks/';
pred_folder= dir(prediction_path);
pred_file={pred_folder.name};
dice_coef_dl2p5d = zeros(1, length(pred_file)-3);
case_name_list = string(pred_file(3:length(pred_file)));
for num_pred= 3 : length(pred_file)
    case_name = pred_file(num_pred);
    case_name = char(case_name);   
    preds_nii = load_untouch_nii([prediction_path, case_name, '/',case_name, '_pred.nii']);  
    masks_nii = load_untouch_nii([masks_path, case_name, '.manual.mask.nii.gz']);
    pred = logical(preds_nii.img);
    mask = logical(masks_nii.img);  
    dice =  2*nnz(mask&pred)/(nnz(mask) + nnz(pred));
    dice_coef_dl2p5d(num_pred-2) = dice;

end
    dice_dl2p5d_avg = mean(dice_coef_dl2p5d)

    
%% Evaluate brainsuite
prediction_path = '../pred_compare/brainsuite/';
masks_path = '../Dataset/test_data/test_data_nii/masks/';
pred_folder= dir(prediction_path);
pred_file={pred_folder.name};
dice_coef_bse = zeros(1, length(pred_file)-3);
for num_pred= 3 : length(pred_file)
    case_name = pred_file(num_pred);
    case_name = char(case_name);   
    preds_nii = load_untouch_nii([prediction_path, case_name, '/',case_name, '.mask.nii.gz']);  
    masks_nii = load_untouch_nii([masks_path, case_name, '.manual.mask.nii.gz']);
    pred = logical(preds_nii.img);
    mask = logical(masks_nii.img);
    dice =  2*nnz(mask&pred)/(nnz(mask) + nnz(pred));
    dice_coef_bse(num_pred-2) = dice;
end
    dice_bse_avg = mean(dice_coef_bse)

%% Evaluate Denseunet
prediction_path = '../pred_compare/DACN/';
masks_path = '../Dataset/test_data/test_data_nii/masks/';
pred_folder= dir(prediction_path);
pred_file={pred_folder.name};
dice_coef_DACN = zeros(1, length(pred_file)-3);
for num_pred= 3 : length(pred_file)
    case_name = pred_file(num_pred);
    case_name = char(case_name);   
    preds_nii = load_untouch_nii([prediction_path, case_name, '/',case_name, '_pred.nii']); 
    masks_nii = load_untouch_nii([masks_path, case_name, '.manual.mask.nii.gz']);
    pred = logical(preds_nii.img);
    mask = logical(masks_nii.img);  
    dice =  2*nnz(mask&pred)/(nnz(mask) + nnz(pred));
    dice_coef_DACN(num_pred-2) = dice;
end
    dice_DACN_avg = mean(dice_coef_DACN)

%% plot
figure(1)
x = [[dice_coef_dl2p5d'], [dice_coef_bse'], [dice_coef_DACN']];
y = categorical(case_name_list);
barh(y, x)
set(gca,'FontSize',9);
xlabel('Dice Coefficient')
xlim([0.7, 1])
ylabel('Case Name')
xlabel('Dice Coefficient');
grid on;
ax = gca;
ax.LineWidth = 2;
ylim=get(gca,'Ylim');
line([dice_dl2p5d_avg, dice_dl2p5d_avg], ylim, 'Color','blue','LineStyle','--', 'LineWidth',2 );
line([dice_bse_avg, dice_bse_avg], ylim, 'Color','red','LineStyle','--', 'LineWidth',2 );
line([dice_DACN_avg, dice_DACN_avg], ylim, 'Color','#EDB120','LineStyle','--', 'LineWidth',2 );

legend({['2p5D LSTM: ', num2str(dice_dl2p5d_avg)], ['Brainsuite: ', num2str(dice_bse_avg)], ['DACN: ', num2str(dice_DACN_avg)]}, 'Location','southwest');
saveas(gcf,'result_final.png')