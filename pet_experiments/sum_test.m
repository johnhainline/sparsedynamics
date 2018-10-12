files = dir('single_mouse_pet');

allSums = [];
coordinate_avgs = [];

nfiles = 5;


for k=1:length(files)
    fname = files(k).name;
    
    if ~endsWith(fname,'img')
        continue
    end
    
    strPieces = strsplit(fname,'.');
    imgName = string(strPieces(1));
    
    
    fpath = strcat('single_mouse_pet/',imgName,'.pet');
    [img_data,frame_duration] = getimg(fpath);
    
    % translate data to min at 0
    img_data = img_data + abs(min(img_data(:)));
    
    % sum each frame and divide by max
    sums = sum(reshape(img_data,128*128*159,40),1);
    sums = sums/max(sums);
    
    allSums = [allSums; sums];
    
    frame_avgs = [];
    
    for f =1:40
        avg = [0 0 0];
        xavg=0;yavg=0;zavg=0;
        for x=1:128
            for y=1:128
                for z=1:159
                    
                    avg = avg + [x y z]*img_data(x,y,z,f);
                    
                end
            end
        end
        frame_data = img_data(:,:,:,f);
        avg = avg/sum(frame_data(:));
        frame_avgs = [frame_avgs;avg];
    end
    
    % coordinate averages for all frames
    coordinate_avgs = cat(3,coordinate_avgs,frame_avgs);
    
    
    
end

% plot total frame intensity
figure;
for k=1:nfiles
    plot(1:40,allSums(k,:)); hold on;
end

% plot x,y,z average coords
figure;
for k=1:nfiles
    plot(coordinate_avgs(:,1,k), coordinate_avgs(:,3,k),'.'); hold on;
end