function [imagemat1,frame_duration] = getimg(file, pl, fr)
% Input:
% File name without suffix (.img, .hdr);
% Plane range in form [pl1 pl2]
% Frame range in form [fr1 fr2]
% Output:
% Image matrix
% frame duration

ext1='.img.hdr';		% Header file
ext2='.img';
headfile=strcat(file,ext1);
imagefile=strcat(file,ext2);
pfile=file;

% *****************************************
% Reading the Header file of microPET
% *****************************************

fidh=fopen(headfile,'r');
nframe=1;
nnframe=1;

if fidh == -1
 error('Cannot open file.')
end

for i = 1:20000
   line = fgetl(fidh);
   if line(1) ~= '#'
      if line == -1 
         break;
      end
      
      blk=findstr(line,' ');
            
      switch line(1:blk)
      case 'axial_blocks '
            axial_blocks = str2num(line(blk+1:length(line)));
      case 'axial_crystals_per_block '
            axial_crystals_per_block = str2num(line(blk+1:length(line)));
      case 'axial_crystal_pitch '
            axial_crystal_pitch = str2num(line(blk+1:length(line)));
      case 'data_type '
            data_type = str2num(line(blk+1:length(line)));  
      case 'z_dimension '   
            z_dimension = str2num(line(blk+1:length(line)));
      case 'x_dimension '
            x_dimension = str2num(line(blk+1:length(line)));
      case 'y_dimension '
            y_dimension = str2num(line(blk+1:length(line)));
      case 'pixel_size '
            pixel_size = str2num(line(blk+1:length(line)));
      case 'total_frames '
            total_frames = str2num(line(blk+1:length(line)));
      case 'calibration_factor '
            calibration_factor = str2num(line(blk+1:length(line)));
      case 'scale_factor '
         	scale_factor_from_header(nframe,1) = str2num(line(blk+1:length(line)));
         	nframe=nframe+1;
      case 'isotope_branching_fraction '
         	isotope_branching_fraction = str2num(line(blk+1:length(line)));
      case 'frame_duration '
            frame_duration(nnframe,1) = str2num(line(blk+1:length(line)));
            nnframe=nnframe+1;
      end
   end
end

fclose(fidh);


axial_fov=axial_blocks*axial_crystals_per_block*axial_crystal_pitch+axial_crystal_pitch;
Iz_size=z_dimension;
Iz_pixel=axial_fov/z_dimension;
matsize=x_dimension*y_dimension*z_dimension;
aspect=Iz_pixel/pixel_size;

scale_factor=scale_factor_from_header(1:total_frames,1).*calibration_factor/isotope_branching_fraction;

if nargin==1
    if z_dimension>1 pl=[1 z_dimension]; else pl=1; end
    if total_frames>1 fr=[1 total_frames]; else fr=1; end
elseif nargin==2;
    if total_frames>1 fr=[1 total_frames]; else fr=1; end
end
  
% *****************************************
% Reading the Static or Dynamic Images file of microPET
% *****************************************

npl=length(pl); 
nfr=length(fr); 

if npl>2 disp('Wrong plane input.');
else
    if npl==1
        pl1=pl; pl2=pl;
        PL=1; nPL=1;
    else
        pl1=pl(1); pl2=pl(2);
        PL=[pl1:pl2]; nPL=length(PL);
    end
end

if nfr>2 disp('Wrong frame input.');
else
    if nfr==1
        fr1=fr; fr2=fr;
        FR=1; nFR=1;
    else
        fr1=fr(1); fr2=fr(2);
        FR=[fr1:fr2]; nFR=length(FR);
    end
end

fiddy=fopen(imagefile,'rb');
matsize=x_dimension * y_dimension * nPL;
imgmat=[];
pl_offset=(pl1-1)*(x_dimension*y_dimension);
for ifr=fr1:fr2
    fr_offset=(ifr-1)*(x_dimension*y_dimension*z_dimension);
    switch data_type
    case 1
        fseek(fiddy, (fr_offset+pl_offset), -1);  % 1 bytes integer Intel
        [mat, count]=fread(fiddy,matsize,'int8');
    case 2
        fseek(fiddy, 2*(fr_offset+pl_offset), -1);  % 2 bytes integer Intel
        [mat, count]=fread(fiddy,matsize,'int16');
    case 3
        fseek(fiddy, 4*(fr_offset+pl_offset), -1);  % 4 bytes integer Intel
        [mat, count]=fread(fiddy,matsize,'int32');
    case 4
        fseek(fiddy, 4*(fr_offset+pl_offset), -1);  % 4 bytes float Intel
        [mat, count]=fread(fiddy,matsize,'float32');
    otherwise
        error('Unsupported Datatype\n');
    end
    imgmat=[imgmat; mat];
end
fclose(fiddy);

if (npl==2) & (nfr==1)
    imagemat=reshape(imgmat,x_dimension,y_dimension, nPL);
    for j=1:nPL
        imagemat1(:,:,j)=imagemat(:,:,j)'*scale_factor(fr);
    end
elseif (npl==1) & (nfr==2)
    imagemat=reshape(imgmat,x_dimension,y_dimension, nFR);
    for j=1:nFR
        imagemat1(:,:,j)=imagemat(:,:,j)'*scale_factor(FR(j));
    end
elseif (npl==1) & (nfr==1)
    imagemat=reshape(imgmat,x_dimension,y_dimension);
    imagemat1(:,:)=imagemat(:,:)'*scale_factor(fr);
elseif (npl==2) & (nfr==2)
    imagemat=reshape(imgmat,x_dimension,y_dimension, nPL, nFR);
    for i=1:nFR
        for j=1:nPL
            imagemat1(:,:,j,i)=imagemat(:,:,j,i)'*scale_factor(FR(i));
        end
    end
end


return