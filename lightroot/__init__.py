from functools import wraps
from scipy.ndimage import gaussian_filter
from skimage.restoration import  estimate_sigma
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class views(object):
    def __init__(self, df, options={}):
        self._df = df.copy()
        #apply scalings, renamings, 
        #axis limits,
        
    @property
    def view(self):
        return self._df #modified
    
    @property
    def name(self):
        pass
    
def plot_views(title, views):
    #add all the views to a grid - there may be 1 or more
    #set all the styles that we want
    #combine pandas plotting and styles with grid
    pass

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.patches as patches

class plots(object):


    def __init__(self,stats):
        self._stats = stats
       
    def plot_quad(ims,pals=["gist_earth", "gist_earth", "gist_earth", "gist_earth"],callbacks={}):
        im = np.arange(100)
        im.shape = 10, 10
        fig = plt.figure(1, (10., 8.))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                         axes_pad=0.1,  # pad between axes in inch.
                         )
        labels=["A", "B", "C", "D"]
        for i in range(4): 
            I = ims[i] if len(ims[i].shape)==2 else ims[i].sum(0)
            c = "white" if pals[i] != "Blues" else "k"

            if i in callbacks:
                callbacks[i](I,grid[i])
            else:
                grid[i].imshow(  I, pals[i],)  # The AxesGrid object work as a list of axes.

            grid[i].annotate(  s = labels[i],  xy=(5, 5),  xytext=(0, 0), va='top', ha='left', color=c,  fontsize = 30)

            grid[i].set_xticklabels ([])
            grid[i].set_yticklabels ([])

        return fig

    def frame_stats(file="./frame_stats.pdf"):
        """
        This will be refactored into something more general - here we hard code just to keep track of the charts
        """
        
        temp = self._stats
        
        matplotlib.rcParams.update({'font.size': 12})
        fig, (ax1, ax2,ax3) = plt.subplots(3,  figsize=(10,8), sharex=True)

        ax1.plot(temp.index, temp.noise, ':')
        ax1.plot(temp.index, [.1 for i in temp.index], '--',color='k')
        ax1.fill_between(temp.index, 0.01,0.045, facecolor='blue', alpha=0.1)
        ax1.set( ylabel='Noise')
        ax1.yaxis.label.set_size(16)

        ax2.plot(temp.index, temp.pseudo_ent/10, ':')
        ax2.plot(temp.index, [0.01 for i in temp.index], '--',color='k')
        ax2.set(ylabel='Signal',)
        ax2.yaxis.label.set_size(16)
        ax2.xaxis.label.set_size(16)

        ax3.plot(temp.index, temp["2dhist95pre"]/10, ':')
        ax3.plot(temp.index, temp["2dhist99pre"]/10, ':')
        ax3.fill_between(temp.index,temp["2dhist95pre"]/10,temp["2dhist99pre"]/10, facecolor='blue', alpha=0.1)
        ax3.set( ylabel='Narrow-band')#xlabel='Frame index',
        ax3.yaxis.label.set_size(16)
        ax3.set_ylim(0,1)

        plt.xlim(0,100)
        plt.subplots_adjust(hspace=0.1)
        fig.savefig(file,bbox_inches='tight')
        
    def example1():
        test_data = pd.read_csv("./test_points.csv").drop("Unnamed: 0",1)
        d1 = test_data[test_data.t==9].reset_index().drop("index",1)
        d2 = test_data[test_data.t==10].reset_index().drop("index",1)
        t = tpctree(d1)
        t.update(test_data[test_data.t==10])
        t.data.head()
        p = {"c":"b", "canvasx": [400,800,800,400],"canvasy": [700,700,350,350] }
        cn = list(tpctree.constellation_sampler(d1,N=6))
        _call=utils.render_constellation_callback(cn)
        c._iom.plot(None, blob_overlay=d1, props=p,callback=_call)
        #
        cn = list(tpctree.constellation_sampler(d2,N=50))
        _call=utils.render_constellation_callback(cn)
        c._iom.plot(None, blob_overlay=d2, props=p,callback=_call)
        #
        c._iom.plot(None, blob_overlay=d1,  props=p,callback=add_second(d2, pj=projected))
    
    #temp = df
    #temp["noise"] = temp["noise"] * 10
    #temp["top percentile"] = temp["perc"]
    #temp["percentile difference"] = temp["pseudo_ent"] * 5 #scale arbitrarily just to plot things - the abslute values are meaningless
    #temp["noise max threshold"] = 0.95
    #ax = temp[['noise', 'top percentile', 'percentile difference', 'noise max threshold']].plot( title="Normalized frame statistics by frame index", figsize=(30,10),style=['--', '--', '--', ':'],ylim=(0,2))
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.title.set_size(14)

class analysis(object):

    @staticmethod
    def set_context_frame_statistics(im,ctx):
        ctx["last_frame_stats"] = {}
        #this is the upper bound on acceptable noise levels
        excessive_noise_th = 0.1 if "excessive_noise_th" not in ctx else ctx["excessive_noise_th"]
        noise = estimate_sigma(im, multichannel=False, average_sigmas=True)
        ctx["last_frame_stats"]["noise"] = noise
        ctx["last_frame_stats"]["percentiles_509599"] =  [round(p,3) for p in np.percentile(im, [50,95, 99])]
        perc = ctx["last_frame_stats"]["percentiles_509599"][-1]
        #the pseudo entropy is a measure based on summed 2d projection
        analysis.log_pseudo_entropy(im,ctx)
        #degeneracy checks
        if perc == 1.:
            ctx.log("Too much saturation, 99th percentile is 1.0 - marking frame as degenerate!", mtype="WARN")
            ctx["last_frame_stats"]["is_degenerate"] = True
        if noise > excessive_noise_th: 
            ctx.log("noise level of {:.4f} is excessive - marking frame as degenerate!".format(noise), mtype="WARN")
            ctx["last_frame_stats"]["is_degenerate"] = True
                
        ctx.log_stats({"noise":ctx["last_frame_stats"]["noise"],
                       "degenerate":ctx.is_frame_degenerate,
                       "perc": perc, 
                       "pseudo_ent": ctx["last_frame_stats"]["pseudo_ent"], 
                       "2dhist99post": ctx["last_frame_stats"]["2dHisto_postnorm"][-1],
                       "2dhist95post": ctx["last_frame_stats"]["2dHisto_postnorm"][-2],
                       "2dhist99pre": ctx["last_frame_stats"]["2dHisto_prenorm"][-1],
                       "2dhist95pre": ctx["last_frame_stats"]["2dHisto_prenorm"][-2]})  
    
    @staticmethod
    def percentiles(im): return [round(p,3) for p in np.percentile(im, [50, 95, 99])]
    
    def log_pseudo_entropy(stack, ctx,props={}):
        #using the 2d projection, we can look at histograms
        noise = ctx["last_frame_stats"]["noise"] if "last_frame_stats" in ctx else 0           
 
        im = stack.sum(0)
        #before norm, absolute values are useful - following is the same as histogram thresholding function - could refactor but consider logging
        pre_threshold = analysis.percentiles(im)
        ctx["last_frame_stats"]["2dHisto_prenorm"] =  pre_threshold
        im[im>pre_threshold[-1]]=pre_threshold[-1]#fill above the 99th   
        #im[im<pre_threshold[-2]]=0 - check sensitivitiy
        #we threshold and norm before taking new readings
        im /= im.max()
        post_ent = analysis.percentiles(im)
        ctx["last_frame_stats"]["2dHisto_postnorm"] =  post_ent
        #this is a type of entropy - 99th percentile -95 percentile. if all the information is in the 99th, the entropy is low
        ctx["last_frame_stats"]["pseudo_ent"] = post_ent[-1] - post_ent[-2]
        return stack
    
def pipe_object(original_function=None,
                post_normalize=True,
                copy_image = False,
                proj_2d=False,
                pre_hist_threshold=False,
                pre_smoothing=-1,
                logging=True,
                trap_exceptions=False  ):
    
    def _decorate_single(function):         
        @wraps(function)
        def single_wrapped_function(_gen, ctx, props={}):   
            gen = _gen if copy_image == False else _gen.copy()
            if proj_2d: gen = gen.sum(0)
            try:  
                #if there is a valid threshold value we will use it otherwise turn off tthresholding
                if "last_frame_stats" in ctx: pre_threshold = ctx["last_frame_stats"]["percentiles_509599"] 
                else: pre_hist_threshold = False
                
                #smoothing option 
                if pre_smoothing > 0: gen=gaussian_filter(gen,sigma=pre_smoothing)
                
                #gen[gen>pre_threshold[1]]=pre_threshold[1]
                #gen[gen<pre_threshold[0]]=0
                
                if logging: ctx.log("running stage {}".format(function.__name__))
                im = function(gen,ctx,props) 
                if post_normalize and im.dtype == np.float: im/= im.max()
                return im
            except Exception as e: 
                ctx.log(repr(e),mtype="ERROR")
                raise e              
        return single_wrapped_function
        
    if original_function == None: return _decorate_single
    return _decorate_single(original_function)

from itertools import combinations
from scipy.spatial.distance import euclidean
from numpy import (array, dot, arccos, clip, degrees)
from numpy.linalg import norm

class utils(object):
    """
    Given sets of points which correspond to constellations, finds vectors lengths and angles
    A = np.array([0,2,0])
    B = np.array([2,0,0])
    C = np.array([0,0,0])
    All functions that accept 'c' param accept a list of k points and will construct vector relationships
    """
    @staticmethod
    def pttv(tups): 
        """point tuples to vector e.g. ([x1,x2,x3], [y1,y2,y3]) -> Y-X
           we take two tuples because we want to make sure the vectors originate from a common point
           If they dont we reverse the tuple representation of the vector
        """
        a,b = tups[0],tups[1]
        if not np.array_equal(a[0],b[0]): a = (a[1], a[0])  
        return [np.array(a[1]) - np.array(a[0]),
                np.array(b[1]) - np.array(b[0])  ]

    @staticmethod
    def angle(v,u): return degrees(arccos(clip(dot(u,v)/norm(u)/norm(v), -1, 1)))
    @staticmethod
    def vector_lengths(c): return [ euclidean(p[0], p[1]) for p in list(combinations(c,2))]   
    @staticmethod
    def vector_inter_angles(c): 
        vectors = list(combinations(c,2))
        return [ utils.angle(*utils.pttv(v)) for v in list(combinations(vectors,2))]
    @staticmethod
    def sort_constellation(c):
        a = np.stack(c)
        return a[np.lexsort((a[:,0],a[:,1]))]
    @staticmethod
    def render_constellation_callback(cset):
        def callback(ax):
            for c_ in cset:
                for v in combinations(c_,2):  ax.plot([v[0][0], v[1][0]], [v[0][1], v[1][1]], 'k:',lw=0.8)
        return callback
    
    