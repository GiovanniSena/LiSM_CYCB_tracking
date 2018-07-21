from skimage.restoration import denoise_wavelet
from skimage.measure import regionprops,compare_psnr,shannon_entropy,simple_metrics
from scipy.ndimage import label,gaussian_filter,maximum_filter,gaussian_gradient_magnitude
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
import numpy as np
import pandas as pd
from . import pipe_object, analysis


class preprocessing(object):  
    @staticmethod
    @pipe_object
    def denoise(im, ctx, props={}):
        trange = ctx["noise_trange"]
        perc = ctx["last_frame_stats"]["percentiles_509599"][-1]
        pseudo_ent = ctx["last_frame_stats"]["pseudo_ent"]   
        noise = ctx["last_frame_stats"]["noise"]     
        if ctx.is_frame_degenerate:return im
            
        if trange != None:
            if noise > 0.085:
                ctx.log("estimated noise standard deviation of {:.4f} exceeds 0.085. Thresholding aggressively and skipping denoise (EXPERIMENTAL)".format(noise, trange[1],perc))
                im/=im.max()
                im[im<0.7] = 0
                im/=im.max()
                return im
            elif noise > trange[1]:#excessive
                ctx.log("estimated noise standard deviation of {:.4f} exceeds [*,{}]. Thresholding @ 99 percentile {} before denoise...".format(noise, trange[1],perc))
                im[im<perc] = 0
                im/=im.max()
                return denoise_wavelet(im, multichannel=False,  mode='soft')
            elif pseudo_ent < 0.1:#low entropy - i tried this on .2 but .1 seems more sensible looking at plot
                ctx.log("estimated pseudo entropy {:.4f} less than 0.2. Thresholding @ 99 percentile {} before denoise...".format(pseudo_ent, perc))
                im[im<perc] = 0
                im/=im.max()
                return denoise_wavelet(im, multichannel=False,  mode='soft')
            elif noise> trange[0] and noise < trange[1]: #no point denoising if its too noisy perhaps??
                ctx.log("estimated noise standard deviation of {:.4f} is in range of [{},{}]. Applying denoise...".format(noise, *trange))
                return denoise_wavelet(im, multichannel=False,  mode='soft')
            return im
        return denoise_wavelet(im, multichannel=False,  mode='soft')
    
    @staticmethod
    @pipe_object
    def smoothing(im,ctx,props={}): return gaussian_filter(im,sigma=props["sigma"])
    
    @staticmethod
    @pipe_object
    def as_dst_transform(im,ctx,props={}): return distance_transform_edt(im)
    
    @staticmethod
    @pipe_object
    def dog(j,ctx,props={}): return preprocessing.__dog__(j,ctx,props)
    
    def __dog__(j,ctx=None, props = {}):
        """diff of gaussian - this is an internal method wrapped by the pipe object"""
        sigma_range = [5,8]
        sigma_smoothing=8
        threshold=0.2 if "threshold" not in props else props["threshold"]
        g2 = gaussian_filter(j,sigma=sigma_range[0]) - gaussian_filter(j,sigma=sigma_range[1])
        g2=gaussian_filter(g2,sigma=8)
        g2 = g2/g2.max() 
        g2[g2<threshold] = 0
        return g2

    @staticmethod
    def lowband_histogram_thresholding(im,ctx,props={}):
        pre_threshold = analysis.percentiles(im)
        im[im>pre_threshold[-1]]=pre_threshold[-1]#kll above the 99th
        im[im<pre_threshold[-2]] = 0 #kill below the 95th
        im /= im.max()
        return im
        
    @staticmethod
    def annealing_thresholding(im,ctx,props={}):
        def label_distribution(img):
            l = label(img.sum(0))[0]
            for r in regionprops(l):
                if r.perimeter != 0:yield r.perimeter
           
        mask_threshold, tmax, waiting =0.1, 0.3, False
        perim_range=[1000,2000]
        
        try:
            for t in np.arange(0.01,tmax, 0.01):
                g2 = im.copy()
                g2 /= g2.max()
                g2[g2<t]=0
                props = np.array(list(label_distribution(g2)))
                if len(props)==0: 
                    ctx.log("Image appears to contain little or no data - breaking out of annealing threshold", mtype="WARN") 
                    return im
                m = props.max()
                if m > perim_range[1] and not waiting: waiting=True
                if waiting and m < perim_range[0]: 
                    ctx.log_stats({"anealThresh":t})
                    ctx.log("setting the clipped region adaptive threshold to {0:.2f} based on maximum label perimeters".format(t))
                    g2 /=g2.max()
                    mask = g2.copy()
                    mask[mask<mask_threshold] = 0
                    mask[mask>0] = 1
                    return g2 
        except: pass # next in workflow is fine
        perc99 = ctx["last_frame_stats"]["percentiles_509599"][-1]
        ctx.log("Unable to find adaptive threshold for frame - aggressively removing 99th percentile", mtype="WARN") 
        im[im<perc99] = 0
        return im

    @staticmethod
    @pipe_object
    def gradient_filter(im,ctx,props={}):
        im = gaussian_gradient_magnitude(im, sigma=1)
        im /= im.max()
        return im
    
    @staticmethod
    @pipe_object
    def pinpoint(im,ctx,props={},size=15):
        """
        label the local peaks in the image - this is a presegmentation
        """        
        im = im.astype(float)
        im/=im.max()
        im = preprocessing.__dog__(im)#call internal one
        mim = maximum_filter(im, size=size, mode='constant')
        local_maxi =peak_local_max(mim, indices=False, min_distance=5)
        return label(local_maxi)[0]
        
    @staticmethod
    @pipe_object
    def select_filtered_by_2d_lowband_largest_component(im,ctx,props={}):
        #histogram thresholding to select a low band slice -sum to take 2d projection
        im_ = preprocessing.lowband_histogram_thresholding(im.sum(0),ctx) 
        #aggresive smoothing to find rough background
        im_ = preprocessing.smoothing(im_,ctx,props={"sigma":8})
        #find the largest connected component using low band threshold
        region,mask = preprocessing.largest_region(im_>0.01,ctx)
        #select the original 3d using the mask
        return preprocessing.select(im, region,mask)
    
    @staticmethod
    def largest_region(im,ctx,props={}):
        """
        Returns the largest region and a mask on the original data
        """
        largest,lab,PP,R=0,0,None,None
        #mark regions
        sub_markers = label(im)[0]
        #find the largest region
        for c, p in enumerate(regionprops(sub_markers)):
            if p.area > largest:largest,R = p.area,p   
        #record usefull frame statistics
        if R is None: return None,im
        ctx["last_frame_stats"]["bbox_area"], ctx.bbox = largest,R.bbox
        ctx.log_stats({"bbox_area": largest})
        ctx.log_stats({"bbox": R.bbox})
        
        return R, sub_markers == R.label
        
    @staticmethod
    def select(data, region, mask):#this pattern is annoying. we should only need one param
        if region is None: return data
        #copy tabula rasa
        output = np.zeros_like(data)
        #3d projection
        if len(mask.shape) == 2 and len(output.shape)==3: mask = np.tile(mask,(output.shape[0],1,1))
        #copy only what we want
        output[mask] = data[mask]    
        if len(region.bbox) == 6:
            z1,y1,x1, z2,y2,x2 = region.bbox[0],region.bbox[1],region.bbox[2],region.bbox[3],region.bbox[4],region.bbox[5]
            return output[z1:z2, y1:y2,x1:x2]
        else:# im assuming data is always 3d but i could put in more cases
            y1,x1, y2,x2 = region.bbox[0],region.bbox[1],region.bbox[2],region.bbox[3]
            return output[:, y1:y2,x1:x2]
    
    @staticmethod
    def remove_small_objects(im,ctx,props={}):
        """TODO"""
        max_label = 0
        l = label(im.sum(0))[0]
        for r in regionprops(l):
            if r.perimeter > max_label:max_label = r.perimeter
        if max_label < 100:
            ctx.log("detected bad image - marking degenerate",mtype="WARN")
            ctx["last_frame_stats"]["is_degenerate"] = True
            im[im>0] = 0
        return im

    @staticmethod
    def point_cloud_emphasis(im,ctx, props={}):
         #threshold 
        im = preprocessing.annealing_thresholding(im,ctx)
        #small objects cause lots of problems downstream but only occur in degenerate frames based on upstream
        im = preprocessing.remove_small_objects(im,ctx)
        #return distance threshold of 
        to_dst = False if "to_dst" not in props else props["to_dst"]
        return preprocessing.as_dst_transform(im,ctx) if to_dst else im
        
#pipeline = [
#    select_filtered_by_2d_lowband_largest_component,
#    point_cloud_emphasis
#    #get the x region and update the context
#    #note we will be plotting on the original 
#]

#decorator_general: normlalize, log before and after

class xlabel(object):
    """
    A single object which maky contain multiple sub objects
    a "pre-segmentation" strategy is used to find the keypoints or blob centroids in the image
    """
    def __init__(self,region,ctx):
        self.region = region
        self.region2d = region if len(region.image.shape)==2 else xlabel._label_2d_(region.image.sum(0))
        self._num_keypoints = 1 #default
        self._kp = self.__build_key_points__(ctx)
            
    @property
    def radius(self): return xlabel.__radius_from_bbox__(self.region.bbox)
    
    @property
    def coords(self): return xlabel.__coords_from_bbox__(self.region.bbox)
    
    def __radius_from_bbox__(bb):
        return np.sqrt( (bb[3]-bb[0])**2+(bb[4]-bb[1])**2+(bb[5]-bb[2])**2)
    
    def __coords_from_bbox__(bb,offset=None):
        r = xlabel.__radius_from_bbox__(bb)
        #sometimes we want to offset into a sub region
        aoffset = 0 if offset is None else offset[0]
        boffset = 0 if offset is None else offset[1]
        coffset = 0 if offset is None else offset[2]       
        return [aoffset+(bb[3]+bb[0])/2,boffset+(bb[4]+bb[1])/2,coffset+(bb[5]+bb[2])/2, r]
    
    @property
    def props(self):
        r = self.region2d
        if r == None:return None
        return {
            "key_points": self._num_keypoints,
            "area" : r.area,
            "eccentricity" : r.eccentricity,
            "bbox" : self.region.bbox,
            "2dbbox" : r.bbox,
            "coords" : self.coords,
            "perimeter" : r.perimeter,
            "minior_axis_length" : r.minor_axis_length,
            "major_axis_length" : r.major_axis_length
        }
    
    def prop_is_above(self, propname, value):
        if self.props == None:return None
        p = self.props
        if propname in p: return p[propname] > value  
        return False
    
    def prop_is_below(self, propname, value):
        p = self.props
        if "propname" in p: return p[propname] < value  
        return False
    
    def _label_2d_(im):
        try:
            return regionprops(label(im)[0])[0]
        except:
            return None
                
    def __build_key_points__(self,ctx):
        """Using a max filter or otherwise"""   
        kps = preprocessing.pinpoint(self.region.image,ctx=ctx)#empty context passed
        kps = regionprops(kps)
        if len(kps)>0:
            self._num_keypoints = len(kps)
            for item in kps:  yield xlabel.__coords_from_bbox__(item.bbox,self.region.bbox)   
        #if we can find sub keypoints, return those, oterwise return self centroid
        else: yield self.coords
    
    def __iter__(self):
        """return keypoints in the label based on intensity"""
        for k in self._kp:  yield k
    
class xregion(object):
    """
    For 3d objects, we also need some useful 2D properties
    This class makes it easy to compile all these details supporting blob detection strategies
    The XLabels are returned but these can contain one or more blob-like objects
    We can then decide how to subdiv these
    """
    def __init__(self,im,ctx,of_keypoints=True):           
        #we can either globally pinpoint or recursively pinpoint
        self._ctx = ctx
        im_ = im if of_keypoints == False else preprocessing.pinpoint(im,ctx)
        self._labels = self.__build_xlabels__(im_) 
        
        
    def update_context(self):
        #lst = [p.coords for p in  self.xlabels]
        self._ctx.add_blobs(pd.DataFrame(list(self),columns=["z","y", "x","r"]))
        
    def __build_xlabels__(self,im): 
        l = [xlabel(r,self._ctx) for r in regionprops(label(im)[0])]
        l = [_l for _l in l if _l.prop_is_above("area", 100)]
        return l
    
    @property
    def region_attributes(self):
        return pd.DataFrame([d.props for d in self.xlabels if d.region2d != None]) 

    @property
    def xlabels(self): return self._labels
    
    def __iter__(self):
        for l in self._labels:
            for keypoint in l:
                yield keypoint

    def __repr__(self):
        return str(len(self._labels))+" regions"

###OTHER FUNCTIONS############
def blob_centroids_from_labels(im,ctx,props={}):
    kps = regionprops(im)
    centroids = [xlabel.__coords_from_bbox__(r.bbox,None)  for r in kps]
    
    return pd.DataFrame(centroids,columns=["z","y", "x","r"])
    
#