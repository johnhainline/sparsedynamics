relpath = 'Pre-Clinical_Data_Samples/Seimens Inveon Scanner/4 Bed/Dynamic/PET/mpet3721a_em1_v1.pet';
filepath = fullfile(pwd,relpath);
[img_data,frame_duration] = getimg(relpath);

disp(size(img_data));