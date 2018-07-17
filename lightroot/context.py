from scipy.ndimage import gaussian_filter
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.measure import compare_psnr
import numpy as np
import os
from .io import io_manager
from .tracking import tpctree
from . import analysis
from .detection import *
from tqdm import tqdm

from matplotlib import pyplot as plt

class context(dict):
    def __init__(self,*args, **kwargs):
        self["lightroot_folder"] = "./"
        self["logging"] = True
        self.__dict_refresh__(*args, **kwargs)
        self._index = -1
        self._stats = {}#key on index, merge dicts
        
        self._iom = io_manager(self)
        self._tree = None
        self._frame_stats = []
        self._blobs = None
        self.bbox = None
        
    def add_blobs(self,df):
        """
        Add the bounding box offset to the incoming blobs - this needs some more thought
        """
        df["x"] = df["x"] + self.bbox[1]
        df["y"] = df["y"] + self.bbox[0]
        self._blobs = df
    
    @property
    def show_progress(self):   return False if "show_progress" not in self else self["show_progress"]
        
    @property
    def stats(self):return pd.DataFrame([d for d in self._stats.values()]).set_index("index").sort_index() #may have to create a master dict depending on how dataframe works
    
    @property
    def blobs(self):  return self._blobs
    
    @property
    def index(self): return self._index
        
    def folder_context(proc_dir):
        return context({"proc_dir":proc_dir})
    
    def __dict_refresh__(self,*args, **kwargs):
        for k, v in dict(*args, **kwargs).items():  self[k] = v
        
    def setup(self):
        self.log("***********BEGIN PROCESSING LOOP***********")
        self.__load_settings__()
        self.tree=None
        
    def log(self,m,mtype="INFO"):self._iom.log_message(m,mtype)  
        
    def log_stats(self,sdict):
        sdict["index"] = self.index
        if self.index not in self._stats:self._stats[self.index] = sdict
        else: self._stats[self.index].update(sdict)
        
    def __load_settings__(self):
        import json
        fname = os.path.join(self["lightroot_folder"], "settings.json")
        with open(fname) as _f:
            try: 
                self.log("loading settings file "+fname)
                self.update( json.load(_f) )
            except Exception as ex:  
                self.log("unable to parse the ./settings.json file:"+repr(ex), "ERROR")
            
    def load_frame(self, i):
        self.setup()
        f =  self._iom._get_stack_(i)
        analysis.set_context_frame_statistics(f,self)
        return f
    
    def plot(self,f,props={},callback=None,):  return self._iom.plot(f,callback=callback,props=props)
    
    @property
    def is_frame_degenerate(self):
        if "is_degenerate" in self["last_frame_stats"]:return self["last_frame_stats"]["is_degenerate"]
        return False
    
    @property
    def frame_warning(self):
        if self._index == self._last_good_frame:return None
        return "Raw Frame Missing Here! Using last good one: {}".format(self._last_good_frame)
    
    @property
    def wrapped_iterator(self):
        self.setup()
        iomanager = tqdm(list(self._iom)) if self.show_progress else self._iom
        for f in iomanager:
            analysis.set_context_frame_statistics(f,self)
            #log them in a list but this is less important then the _stat dict merge
            self._frame_stats.append(self["last_frame_stats"])
            yield f
    
    def detect(f,c,show=False):
        f = preprocessing.denoise(f,c)
        if c.is_frame_degenerate:return f#breaking condition
        #the bounding box is recorded on the context for latter offset
        g = preprocessing.select_filtered_by_2d_lowband_largest_component(f,c)
        h = preprocessing.point_cloud_emphasis(g,c,props={"to_dst":False})
        h = preprocessing.dog(h,c,props={"threshold":0.2})
        #find and update the centroids in the context -they will be offset in the context
        #in some cases we just take key points if we do not trust dogs. Param=False
        xregion(h,False).update_context(c)
        if show: c._iom.plot(f,c.blobs, bbox=c.bbox) #show the results 
        return f#we jump back to original frame context here
    
    def make_pipeline(pipes=None,capture_stats_callback=None):
        if pipes == None: pipes = [context.detect]#default pipeline
        if not isinstance(pipes,list):pipes = [pipes]
        
        def pipeline(im,ctx):
            for p in pipes: 
                im = p(im,ctx)
                #breaking condition
                #if ctx.is_frame_degenerate:return im
            if capture_stats_callback != None: capture_stats_callback(ctx)
            return im
            
        return pipeline
            
    def _update_tree_(self):
        if self._tree ==None: self._tree = tpctree(self._blobs,self)
        else: self._tree.update(self._blobs)
        #self.update_stats(self._tree.stats) # merge statistics from the tree
        return self._tree.blobs
    
    def run(self,pipeline=None,capture_stats_callback=None):  
        """
        main entry point - see inline comments
        Either runs the detault pipeline or a pipeline passed in by the user
        Uses internal objects such as file manager to iterate files
        A number of objects are saved in the cached data folder
        """
        pipeline = context.make_pipeline(pipeline,capture_stats_callback)

        for stack in self.wrapped_iterator:
            #process stack with top level pipeline
            stack = pipeline(stack,self)
            #compute pairing using the tracker - the tracker knows the context blobs so no arg
            blob_annotations = self._update_tree_()
            #plot previous blobs when they exist using a red blobs
            #>if the stack was not loaded properly, an older frame is used and we add a warning on the frame
            ax = self._iom.plot(stack,self._tree.prev_blobs,props={"c":"r", "frame_warning":self.frame_warning}, bbox=self.bbox)
            #overlay the new blobs with annotations
            ax = self._iom.plot(stack,self._tree.blobs,blob_annotations,self.bbox,ax=ax)
            #save the image
            self._iom.save(ax)
            #save the tracks to check point
            self._iom.save_file(self._tree.data,"data.csv", as_check_point=True)
            plt.close()
            
        #save the blob data actual
        self._iom.save_file(self._tree.data, "data.csv")
        #write extra files
        self._iom.save_file(self._tree.life_matrix, "life_matrix.csv")
        self._iom.save_file(tpctree.make_life_matrix(self._tree.data, restricted=True), "life_matrix_restricted.csv")
        
        self._iom.save_file(pd.DataFrame([f for f in self._frame_stats]), "frame_statistics.csv")
        #cleanup
        self._iom.remove_check_points()
        
        print("Done!")