
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from baseimage import PETImage, CTImage, normalize



# put filepath to image to open
filepath = os.path.join("Pre-Clinical_Data_Samples",
						"Seimens Inveon Scanner",
						"1 Bed",
						"Dynamic",
						"PET",
						"mpet3715b_em1_v1.pet.img")

# load image into memory
my_img = PETImage(filepath=filepath)
my_img.load_image()



# get_axis is just a map from x,y,z,t to the 0,1,2,3 which numpy uses
view_axis = my_img.get_axis("z")
time_axis = my_img.get_axis("t")

# we can view the 3D image in time from the x,y, or z axes
my_img.img_data = normalize(my_img.img_data)*10 # multiply by 10, makes video easier to see
frame_blocks = my_img.split_on_axis(time_axis)	# get 3D frames
frames = [fb.max(axis=view_axis) for fb in frame_blocks]	# get 2D frames; can also use .sum(axis=view_axis), but i think max looks better


# view as video
fig = plt.figure()
ims = [[plt.imshow(frame, cmap="gray",clim=(0,1), animated=True)] for frame in frames]
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
plt.show()


# the PETImage object uses np.memmap to hold data in "memory"
# this just cleans up tempfiles used in np.memmap
# good to do this before exiting program
my_img.unload_image()