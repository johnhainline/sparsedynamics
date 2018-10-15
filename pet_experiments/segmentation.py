import os
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from baseimage import PETImage, CTImage, normalize

# should probably implement as method for PETImage ultimately
def pca_kmeans_segmentation(pet_img, **kwargs):  # nclusters=10, nfeatures=None, options=None):

	default_options = {
		'nclusters' : 10,
		'n_init' : 20,
		'plot' : True
	}

	# load and unload image outside of this routine
	pet_img.check_data()

	# initialize ROI as whole matrix
	roi = pet_img.img_data
	zd,yd,xd,td = roi.shape

	# get kwargs
	keys = kwargs.keys()

	# specify limits of region to segment
	if 'roi_lims' in keys:
		(zmin,zmax),(ymin,ymax),(xmin,xmax) = kwargs['roi_lims']
		roi = roi[zmin:zmax,ymin:ymax,xmin:xmax,:]
		zd,yd,xd,td = roi.shape

	# other kwargs		
	nclusters = kwargs['nclusters'] if 'nclusters' in keys else default_options['nclusters']
	n_init = kwargs['n_init'] if 'n_init' in keys else default_options['n_init']
	plot = kwargs['plot'] if 'plot' in keys else default_options['plot']
	nfeatures = kwargs['nfeatures'] if 'nfeatures' in keys else td

	# default to use all principal components if not specified
	nfeatures = td if nfeatures is None else nfeatures

	# number of voxels to analyze
	N = zd*yd*xd

	# raw data matrix
	X = roi.reshape(N,td)

	# normalize each voxel over time
	X = np.apply_along_axis(lambda x: x/np.linalg.norm(x,ord=2), 0, X)

	# SVD of data matrix
	U, S, Vh = np.linalg.svd(X.T.dot(X))

	# transform project each voxel's time series onto eigenvectors
	# U^T: X -> Xh
	X_h = U[:,:nfeatures].T.dot(X.T)

	# use kmeans clustering on transformed data
	kmeans = KMeans(init='k-means++', n_clusters=nclusters, n_init=n_init)
	kmeans.fit(X_h.T)
	Z = kmeans.predict(X_h.T)

	# create masks for clustered data
	root_mask = np.linspace(0,N-1,N)	# a skeleton for actual masks
	masks = []
	for clust in set(Z):
		mask_func = np.vectorize(lambda x:int(Z[int(x)]==clust))
		masks.append(mask_func(root_mask).reshape(zd,yd,xd))

	print('Created {} masks of image'.format(len(masks)))

	if plot:
		plot_masks(roi,masks,**kwargs)

	return masks, roi




def plot_masks(roi, masks, **kwargs):

	ax_map = {'z':0,'y':1,'x':2,'t':3}

	default_options = {
		'view_ax' : 'y',
		'mask_collapse' : np.sum,
		'time_collapse' : np.max,
		'spatial_collapse' : np.sum,
		'fig_size' : (40,50)
	}

	keys = kwargs.keys()
	view_ax = kwargs['view_ax'] if 'view_ax' in keys else default_options['view_ax']
	mask_collapse = kwargs['mask_collapse'] if 'mask_collapse' in keys else default_options['mask_collapse']
	time_collapse = kwargs['time_collapse'] if 'time_collapse' in keys else default_options['time_collapse']
	spatial_collapse = kwargs['spatial_collapse'] if 'spatial_collapse' in keys else default_options['spatial_collapse']
	fig_size = kwargs['fig_size'] if 'fig_size' in keys else default_options['fig_size']

	# check and set np.ndarray axes
	view_ax = ax_map[view_ax] if view_ax in ax_map.keys() else view_ax
	if view_ax not in list(ax_map.keys()) + list(ax_map.values()):
		raise ValueError('Bad view_ax: {}'.format(view_ax))
	time_ax = 3

	# setup axes for plots
	fig = plt.figure()
	axes = []
	nplots = len(masks)+1
	for k in range(nplots):
	    ax = fig.add_subplot(1+nplots//3,3,k+1)
	    axes.append(ax)

	# figures size
	plt.rcParams["figure.figsize"] = fig_size
	
	# plot original image
	roi_image = normalize(spatial_collapse(time_collapse(roi,axis=time_ax),axis=view_ax))
	axes[0].imshow(roi_image, cmap="gray",clim=(0,1))

	# plot each mask
	for ax,premask in zip(axes[1:],masks):	
	    fmask = np.apply_along_axis(mask_collapse,view_ax,premask)
	    ax.imshow(fmask)
	plt.tight_layout()
	plt.show()





if __name__ == '__main__':
	filepath = os.path.join("single_mouse_pet",
							"mpet3715b_em1_v1.pet.img")
							# "mpet3721a_em1_v1_s4.pet.img")

	img = PETImage(filepath=filepath)
	img.load_image()
	Ns = img.img_data.shape
	Ws = (60,40,40)
	roi_lims = [(int((N-W)/2),int((N+W)/2)) for N,W in zip(Ns,Ws)]

	pca_kmeans_segmentation(img,roi_lims=roi_lims,nfeatures=20,nclusters=10)

	img.unload_image()