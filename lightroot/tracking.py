# the tree code - focus on this now
#also more image statistics and different strategies
from scipy.spatial import KDTree as scKDTree
import pandas as pd 
import numpy as np
from numba import jit
from itertools import combinations
from . import utils
#
class tpctree(object):
    """
    In online mode, pass blocks of coordinates for each time point and update the tracker
    
    #SAMPLE TREE USAGE
    import pandas as pd
    data = pd.read_csv("./test_tree_blobs.csv")[["t","x","y","z"]]
    tree=None
    for g,d in data.groupby("t"):
        print(".", end="")
        if g %10 ==0:print()
        if tree == None: 
            tree = tpctree(d,options={"epsilon":25, "transforms":True})
        else: tree.update(d) 
        if g == 100:break
    """
    def __init__(self, data,options={}):
        ##init
        self._t = 0 
        self.data = None
        if data is not None:
            data.index.name = "key"
            self.data = data.reset_index()
            self.data["t"] = self._t#we do not trust any t passed in but we should not really overwrite it - think about this
        self._last_transform = None
        self._detect_ratio = []
        #could take any of these from options
        #10 seems to small and 20 could be too big - double counting and also an opposite direction in subtle movements on boundary to admit debris
        #need to do some fnal checks over extended frames to be sure
        self._epsilon = 15 if "epsilon" not in options else options["epsilon"]
        self._alt_lags_enabled = False if "lags" not in options else options["lags"]
        self._transforms_enabled = False if "transforms" not in options else options["transforms"]
        
        self._columns = ["x","y","z"]
        #add a small translation to the priors - this is effectively a void transform
        self._transform_priors = [tpctree.translation_transform_for(0.01*np.ones((1,len(self._columns))))]
        self._transformer = tpctree.__default_transformer__
        #self._sampler = tpctree.constellation_sampler#simple_sampler
        self._sampler = tpctree.simple_sampler
        self._stats = {}
      
    def merge_statistic(self, k,v):
        d = {k:v}
        if self._t not in self._stats: self._stats[self._t] = d
        else:self._stats[self._t].update(d)
        
    def __getitem__(self, k): 
        if self.data is None: return None
        return self.data[self.data["t"]==k]
    
    @property
    def life_matrix(self):return tpctree.make_life_matrix(self.data)
      
    def get_restricted(lm):
            restricted = lm.sum(axis=1).reset_index()#.set_index("index")
            restricted = lm[restricted[0]>1]
            restricted.astype(int).head()
            restricted.index = restricted.index.astype(int)
            return restricted
        
    def make_life_matrix(df,restricted=False):
        
        #bandd = lm.diff(1,axis=1,).fillna(0) #* lifetime_matrix
        #births = bandd[bandd==1].fillna(0)
        #deaths = np.abs(bandd[bandd==-1]).fillna(0)
        #ageMat = lm.cumsum(axis=1) * lm

        def matrix_life_mask(times,max_time):
            ar = np.zeros(max_time)
            ar[times] = 1
            return ar
        
        maxt = df.t.dropna().max()+1
        index = []
        mat = []
        for k,g in df.groupby("key"):
            l = list(g.t.dropna().values)
            vals = matrix_life_mask(l, maxt)
            index.append(k)
            mat.append(vals)
        lm =  pd.DataFrame(mat, index=index)
        return lm if not restricted else tpctree.get_restricted(lm)
        
    def _make_fluc_mat_(df):
        return df[["epsilon", "key", "t"]].groupby(["t", "key"])[["epsilon"]].\
        mean().reset_index().pivot("key", "t", "epsilon").fillna(0)
    
    @property
    def stats(self): return self._stats
    
    @property
    def blobs(self):
        """Last known t, return slice"""
        return self[self._t]
    
    @property
    def prev_blobs(self):
        """Last known t, return slice for last t"""
        if self._t ==0:return None
        return self[self._t-1]
    
    #####TRANSFORMATIONS#######    
    #@jit
    def _get_transforms_(self,t1,t2):
        #for v in self._sampler(t1[self._columns]):
        #    for w in self._sampler(t2[self._columns]):
        #        yield self._transformer(w,v)
        
        #translations
        for v in tpctree.simple_sampler(t1[self._columns]):
            for w in tpctree.simple_sampler(t2[self._columns]):
                yield tpctree.find_translation(w,v)
                
        if self._transforms_enabled:
            try:
                #generalised transforms
                for v in tpctree.constellation_sampler(t1[self._columns]):
                    for w in tpctree.constellation_sampler(t2[self._columns]):
                        yield tpctree.find_transform(w,v)
            except:
                pass
                #self.merge_statistic("failed_sample_transform", True)
    
    def find_transform(X,Y): 
        """
        Given two matrices, try to compute the transformation
        """
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:,:-1]
        A, res, rank, s = np.linalg.lstsq(pad(X), pad(Y),rcond=None)
        transform = lambda x: unpad(np.dot(pad(x), A))
        transform.__ttype__ = "transform"
        return transform
    
    def find_translation(X,Y): 
        """Given two matrices, compute the simple translation
           it as assumed the point subsets represent likely congruences and the mean translation vector us used
        """
        mean_vec = np.array(Y)-np.array(X)
        if len(mean_vec.shape) == 1:  mean_vec = np.stack([mean_vec])
        mean_vec = np.stack([mean_vec.mean(axis=0)])
        tr= tpctree.translation_transform_for(mean_vec)
        tr.__ttype__ = "translation"
        return tr
    
    def translation_transform_for(v):
        """
        Generate a simple translation transform for a vector that represents an average displacement
        This is a fall-back for the more generalized transform
        """
        def translation_matrix():
            Q = np.eye(v.shape[1]+1)
            Q[:v.shape[1],-1] = np.array(v)
            return np.matrix(Q)
        
        def translation_transform(m):
            m = np.matrix(np.hstack((m,np.ones((len(m),1)))))
            A = translation_matrix()
            m = A * m.T
            return m.T[:,:-1]
        
        #return the transform functon
        return translation_transform
    
    def __default_transformer__(X,Y):
        try:   
            #print("returning reg transform")
            return tpctree.find_transform(Y,X)
        except Exception as ex:
            print(repr(ex))
            return tpctree.find_translation(Y,X)
    
    def best_transform(self, collection,lag=1):
        """
        Simply return the transform that minimises the objective with respect to the data
        The data will be the point cloud at time [t] and [t-1]
        """
        best_index, best_tr, best_score = None, None, self._epsilon+1
        #print(len(collection),"transforms")
        for counter,tr in enumerate(collection):
            if tr is None: continue
            score = self.apply_transform(tr,False,lag=lag)
            if score <= best_score: 
                best_index, best_tr , best_score= counter,tr,score
        
        #print("best is",best_index,str(best_tr))  - need to find some way to show that we used a certain transform e.g. affine or void
        
        self.merge_statistic("best_tr_index", best_index)
        self.merge_statistic("best_tr_type", tr.__ttype__)
        
        return best_tr
    #####END TRANSFORMATIONS#######
        
    #def __eps_from_min_intertarget_distance__(self):
    #    #for each point in target, find the smallest distance between any two points
    #    #half this vaue and take the min of the max_epsilon (bio-feasible) and this value
    #    #rationale: adapt to the data up to a point. if we have some points very close together then we got the bio wrong or there is noise
        
    def update(self, data,t=None): #unless t is given, auto-inc
        """
        Update the tracker with new points
        """
        if data is None: return None
        #increment the flock
        self._t += 1
        data.loc[:,'t']  = self._t
        self.data = pd.concat([self.data, data])
        #some local vars
        t,new_transforms = self._t, []
        
        #add to the list, the list of priors and include the last chosen (best) transform
        for tr in [self._last_transform]+  self._transform_priors : 
            new_transforms.append(tr)   
            
        #get all transforms generated from the data
        for tr in self._get_transforms_(self[t], self[t-1]):
            new_transforms.append(tr)
        
        #select the best of the transforms by evalaution against the data and save the best
        self._last_transform = self.best_transform(new_transforms)
        #apply the best transform and assign ids
        self.apply_transform(self._last_transform)

        if self._alt_lags_enabled:#analytics
            #if checking lags enabled
            lag2,lag3 = [],[]
            if t>1:
                for tr in self._get_transforms_(self[t], self[t-2]): lag2.append(tr)
                self.apply_transform(self.best_transform(lag2,lag=2),lag=2)
            if t>2:
                for tr in self._get_transforms_(self[t], self[t-3]): lag3.append(tr)
                self.apply_transform(self.best_transform(lag3,lag=3),lag=3)
            
        #return the state
        return self.blobs #here we only return the latest info
    
    def __normed_distance__(self,D):
        """
        simple helper methods to replace infinity values by the model worst-score
        """
        D[(np.isinf(D)) ] = self._epsilon+1
        return D
    
    def __validate_ids__(self,res,lag=1):
        """
        If the result is infinite, there is no neighbour within the ball of radius epsilon
        In such cases we should set the id to newly generated ids TODO
        """
        next_key = int(self.data.key.max() + 1)
        ids = res[1]
        invalid_mask = np.where(res[0]>=self._epsilon+1)[0]
        valid_mask = np.where(res[0]<self._epsilon+1)[0]
        
        self._detect_ratio = [len(valid_mask),len(invalid_mask)]
        self.merge_statistic("detections_ratio", self._detect_ratio)
                
        num_orphans = len(invalid_mask) 
        #for the valid ids, take the key for them (off the prev)
        ids[valid_mask] = self[self._t-lag].iloc[ids[valid_mask]].key
        #for the none-valid ids, generate new keys
        newids = list(range(next_key, next_key+num_orphans))
        ids[invalid_mask ] = newids
        return np.array(ids,np.int)
    
    def apply_transform(self, tr, accept_pair=True,lag=1):
        #get the actual frame-frame data points
        X,Y = self[self._t-lag],self[self._t]
        #compute the transfomration from Y-> Back to X (Ys are the ones that need id pairings)
        #make sure that we compute the transform that way too
        TR = tr(Y[self._columns].as_matrix())
        #index Y, i.e. the target vector - a suitable result from this is the pairing
        #we do not deal with marriage problem and simply assign closest - could improve this but fine based on our data
        target = scKDTree(X[self._columns].as_matrix())
        #get the first nearest neighbour within threshold
        res=target.query(TR, k=1,distance_upper_bound=self._epsilon)
        #the score is the bounded distance - see called function - kdtree returns vector [distance, ids]
        score = self.__normed_distance__(res[0]).mean().round(4)
        #assign ids
        key_name = "key" if lag == 1 else "key"+str(lag)
        #if key_name not in self.data.columns:self.data[key_name] = None #do i need this
            
        if accept_pair:
            #print("Accepting pair for lag", lag)
            self._temp_res = np.array(res)
            #get the keys corresponding to the integer index returned from the kdtree
            self.data.loc[self.data["t"] == self._t, key_name] = self.__validate_ids__(res,lag=lag)
            #assign epsilons
            self.data.loc[self.data["t"] == self._t, "epsilon"+str(lag)] = self.__normed_distance__(res[0])
            #should always be possible to either assign or generate - null key not acceptable
            if lag == 1: self.data.key = self.data.key.astype(int)
            
        #return score of this transform
        return score
    
    ###Samplers###
    def simple_sampler(df,columns=["x", "y", "z"]):  
        """simply sample all the points"""
        return df[columns].as_matrix()

    
    def constellation_sampler(df, k=3, N=50,min_angle=20,min_length=20):
        """
        Find constellations - parameters are
        k: number of points in the constellation
        N:Number of random constellations to sample
        min_angle: the min allowed angle between two vectors
        min_length: the min alowed vector length
        """
        ps = [v for v in df.as_matrix()]

        def _validate_(ps):
            for a in utils.vector_lengths(ps):
                if a < min_length:return False
            for a in utils.vector_inter_angles(ps):
                if a < min_angle:return False        
            return True

        perms = list(combinations(ps,k))
        #perms = np.random.choice(perms, len(perms))

        counter = 0
        for i in np.random.choice(range(len(perms)),len(perms)):
            item = np.stack(list(perms[i]))
            #print(item)
            if _validate_(item): 
                counter += 1
                yield utils.sort_constellation(item)
                if counter == N:return
                
    def show_projection_callback(df,tr,r=30):
        from matplotlib import pyplot as plt
        import math

        projected = tr(df[["x", "y", "z"]])
        projected = pd.DataFrame(projected, columns=["x","y","z"])
        def _call(ax): 
            area = 2*math.pi*r
            ax.scatter(x=df.x, y=df.y, c='r', s=30,marker='x', label="objects at time t+1")
            ax.scatter(x=projected.x, y=projected.y, facecolors='none', edgecolors='b', s=area, label='projected')      
            plt.legend(loc=4, fontsize=20)
        return _call
