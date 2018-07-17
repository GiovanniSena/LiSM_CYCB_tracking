import os
from glob import glob
from skimage import io,draw
from os.path import basename
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import close
import matplotlib.patches as patches
import time

# io - build in the missing file handler logics and the format determination
class io_manager(object):
    """
    Handles all file and image IO. This includes logging, folder logic, loading settings or defaults, ...
    """
    def __init__(self, options):
        #self._root_ = "./test/"
        #pass in the context with settings and runtime context
        self._ctx = options
        self._log_to_file=False if "logging" not in options else options["logging"] == "file"
        #we should always have a folder to process but default to root
        self._data_dir = self._root if not "proc_dir" in options else options["proc_dir"]
        
        #set the output directory
        out_dir_name = "cached_datax" if "out_dir_name" not in options else options["out_dir_name"]
        self._out_dir = os.path.join(self._ctx["lightroot_folder"], out_dir_name)  
        if not os.path.exists(self._out_dir): os.makedirs(self._out_dir)
        self.ensure_empty_out_dir()
         
        #treat the folder entered by user so it is behaving properly
        self._data_dir = self._data_dir.replace("\\", "/").replace("\"", "")
        #some default options which can be overriden by settings file or runtime context
        self.__set_defaults__()
        #report some details about the folder
        self.log_message("Data dir is set to {}".format(self._data_dir))
        self.log_message("Output dir is set to {}".format(self._out_dir))
        #some feedback on what we see in the data folder
        self.describe_data_dir()
        
    def ensure_empty_out_dir(self):
        path = self._out_dir 
        if os.listdir(path) != []: 
            if  input("The directory "+path+" should be empty. Do you want to clear it? (y/n)").lower() == "y":
                for f in glob(path+"/*.*"): os.remove(f)
    
    def __infer_file_format__(self,token="_tp"): #token is not a general idea - this is sort of like PREFIX_PADDEDINT_SUFFIX
        """
        The file name is inferred based on TIFF files that exist in the data folder with a consistent padded index.
        For example files could be called Frame0001.tif, Frame0002.tif etc. Here the padding is length [4] and the prefix is [Frame]
        The prefix and the padding should be the same for all TIFF files. In this example, the 101st frame must be called Frame0101.tif
        If a frame formatting for the stacks does not match this but instead another consistent rule exists, the settings.json file should reflect this
        """
        search = self._data_dir+"/*.tif"
        fsample = glob(search)[-1] 
        f = basename(fsample).split(".")[0]
        prefix = f[:f.index(token)+len(token)]
        return os.path.join(self._data_dir, prefix+ "{:0>3}.tif")
           
    def __set_defaults__(self):
        """
        These are lightroot defaults that can be overridden in the settings.json file or passed in on the context
        """
        defaults = {}
        defaults["noise_trange"] = [0.01,0.045] 
        defaults["max_files"] = 10000
        defaults["max_file_gap"] = 100 #if this is set to 0 then we do not allow a gap at all
        defaults["stack_file_format"] = self.__infer_file_format__()
        
        found_good = -1
        for i in range(defaults["max_file_gap"]+1):
            if os.path.isfile(defaults["stack_file_format"].format(i)):
                found_good = i
                break
        
        if found_good == -1 and "stack_file_format" not in self.ctx:
            raise Exception("Fatal problem: The stack file format could not be inferred and no file format was set in the settings file.")
    
        for k,v in defaults.items():
            if k not in self._ctx: 
                self.log_message("Setting a default value {}:{}".format(k,v))
                self._ctx[k] = v             
        
    def describe_data_dir(self):
        d = {}               
        max_good = -1
        first_good = -1
        gaps = {}
        missing = 0
        for i in range(self._ctx["max_files"]):
            if not os.path.isfile(self._ctx["stack_file_format"].format(i)):
                #link the missing one to the last good one
                gaps[i] = max_good
                missing+=1
            else:
                #track the first and last good frames witnessed
                max_good = i
                if first_good == -1:first_good = i
            
            #depending on allowed gap, terminate and purge the gap dictionary
            #here we are trying to find the last tiff based on the format so we allow a possible gap or give up
            if missing == self._ctx["max_file_gap"]:
                for k in list(gaps.keys()):
                    if k >= max_good: gaps.pop(k)
                break
        
        d["Total_files"] = self.count_files(self._data_dir)
        d["Total_tiff_files"] = self.count_files(self._data_dir, "tif")
        d["Max_good_index"] = max_good
        d["First_good_index"] = first_good
        d["Frame_gaps"] = gaps
        
        for k,v in d.items():  self.log_message("{}:{}".format(k,v))
        
        self._ctx.update(d)
                
    def __list_files__(self,folder,ftype="*"):
        search = "{}/*.{}".format(folder,ftype)
        return list(glob(search))
    
    def count_files(self, folder,ftype="*"):
        return len(self.__list_files__(folder,ftype="*"))

    def _get_stack_(self,i):
        """Using the file format and the index passed in, load the frame"""
        fname = self._ctx["stack_file_format"].format(i)
        return self._get_stack_by_name_(fname)
    
    def _get_stack_by_name_(self,fname,norm=True):
        f = io.imread(fname)  
        if norm: f = f / f.max()
        return f
    
    def __file_indices_with_filled_gaps(self):
        """
        This is used by the class iterator which loads frames
        This helper method simply finds frame indices that are valid
        It also uses gap information to allow the context to replace frame j with some previous frame if frame j is missing
        This is to allow for smooth analysis due to corruputed frames which need to be removed.
        This behaviour can be turned off by setting the alowed gap property "max_file_gap" to 0
        """
        #iterate the (open) range
        for i in range(self._ctx["First_good_index"], self._ctx["Max_good_index"]+1):
            #update the context frame state (although we should minimize used of this if we can)
            self._ctx._index = i
            if i in self._ctx["Frame_gaps"]:  
                self.log_message("Filling a frame gap using last frame that we loaded properly")
                self.log_stat({"frame_missing":1})
                yield self._ctx["Frame_gaps"][i]
            else: 
                self._ctx._last_good_frame = i
                yield i
                
    def __iter__(self):
        for index in self.__file_indices_with_filled_gaps(): yield self._get_stack_(index)
        
    def draw_bounding_box(self, ax, rect):
        inlay = 10
        ax.add_patch( patches.Rectangle(  list(reversed(rect[0:2])),   rect[3]-rect[1]-inlay,   rect[2]-rect[0]-inlay,  fill=False, edgecolor='white' ))
    
    def plot(self, im, blob_overlay=None, annotations=None, bbox=None, ax=None, callback=None, props={},):
        """
        plotting with overlays - some examples follow assuming df has columns "x", "y", "z" and possible "key" for annotatations
        ax=c._iom.plot(frm[0], blob_overlay=df)
        ax=c._iom.plot(frm[0], blob_overlay=df2, props={"c":"r"},ax=ax)
        ax=c._iom.plot(frm[0], blob_overlay=df2, annotations=df2, props={"c":"r", "frame_warning": "this is a warning"},ax=ax)
        """
        #plot a 2D image or 3D
        #create the plot and add the image only if now ax is passed in
        if ax is None: 
            fig,ax = plt.subplots(1,figsize=(20,10))
            if im is not None: 
                plottable = im.sum(0) if len(im.shape) == 3 else im
                if bbox is not None:self.draw_bounding_box(ax, bbox)
                if "palette" in props:ax.imshow(plottable,props["palette"])
                else: ax.imshow(plottable)
            else: #this is just a helper canvas
                _x = [0,0,1000,1000] if "canvasx" not in props else props["canvasx"]
                _y = [0,1200,1200,0] if "canvasy" not in props else props["canvasy"]
                ax.scatter(x=_x,y=_y,c='w')
                ax.minorticks_on()
                ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
                ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

        #plot the blob overlay
        if blob_overlay is not None:
            c = "w" if "c" not in props else props["c"]
            s = 30 if "s" not in props else props["s"]
            ax.scatter(x=blob_overlay.x, y=blob_overlay.y, c=c, s=s,label="objects at time t")
            
        #add any annotations
        if annotations is not None:
            for k,r in annotations.iterrows(): 
                key = int(r["key"]) if "key" in annotations.columns else -1
                ax.annotate(str(key), (r["x"],r["y"]+5),  ha='center', va='top', size=14)

        #add frame warning if its in the options
        if "frame_warning" in props and props["frame_warning"] is not None:
            ax.text(30, 30, props["frame_warning"], color='red', fontsize=15)
                
        if callback != None: callback(ax)
    
        return ax
    
    def plot_3d(self,im):
        pass
    
    def log_message(self, m, mtype="INFO"):
        index = "({})".format(self._ctx.index) if self._ctx.index >= 0 else ""
        record = "{} {}{}:{}".format(time.strftime('%d/%m/%Y %H:%M:%S'), mtype, index, m)
        if not self._ctx.show_progress: print(record)
        if self._log_to_file:
            with open(os.path.join(self._out_dir, "log.txt"), "a") as f:
                f.write(record+"\n");
 
    def ensure_dir(self,d): 
        if not os.path.exists(d): os.makedirs(d)
    
    def _localise_file_(self,file):  os.path.join(self._root_, file)
        
    def save(self, data, dispose=False):
        prefix = "Frame"
        file_template = os.path.join(self._out_dir, prefix+ "{:0>4}.png")
        fig = data.figure
        fig.savefig(file_template.format(self._ctx.index))
        close(fig)
    
    def remove_check_points(self):
        file_template = os.path.join(self._out_dir, "*.cpt")
        for f in glob(file_template):
            self.log_message("removing check point file "+f)
            os.remove(f)
            
    
    #currently assumed to be a dataframe
    def save_file(self,obj,name,as_check_point=False):
        if as_check_point: name = name + ".cpt"
        file_template = os.path.join(self._out_dir, name)
        self.log_message("saving file "+file_template)
        obj.to_csv(file_template)
        
    
    