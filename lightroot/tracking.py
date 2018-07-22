# the tree code - focus on this now
#also more image statistics and different strategies
from scipy.spatial import KDTree as scKDTree
import pandas as pd 
import numpy as np
from numba import jit
from itertools import combinations,product
from . import utils
from tqdm import tqdm
#
DEFAULT_EPSILON = 18
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
        self._t_offset= 0
        self.merge_data(data)
       
        self._last_transform = None
        self._detect_ratio = []
        #could take any of these from options
        #10 seems to small and 20 could be too big - double counting and also an opposite direction in subtle movements on boundary to admit debris
        #need to do some fnal checks over extended frames to be sure
        self._epsilon = DEFAULT_EPSILON if "epsilon" not in options else options["epsilon"]
        self._alt_lags_enabled = False if "lags" not in options else options["lags"]
        self._transforms_enabled = False if "transforms" not in options else options["transforms"]
        self._use_tr_concensus = True if "use_tr_concensus" not in options else options["use_tr_concensus"]
        self._angle_tresh = 90 if "angle_tresh" not in options else options["angle_tresh"]
        
        self._columns = ["x","y","z"]
        #add a small translation to the priors - this is effectively a void transform
        self._transform_priors = [tpctree.translation_transform_for(0.01*np.ones((1,len(self._columns))))]
        self._transformer = tpctree.__default_transformer__
        #self._sampler = tpctree.constellation_sampler#simple_sampler
        self._sampler = tpctree.simple_sampler
        self._stats = {}
        self._transform_collection = []
      
    def process_file(file_name, start=0,end=-1, options={"epsilon":15, "transforms":True}):
        data = pd.read_csv(file_name).reset_index()[["t","x","y","z"]]
        tree=None
        maxt = data.t.max()+1 if end==-1 else end
        for g,d in tqdm(data[(data.t>=start)&(data.t<=end)].groupby("t")):
            if tree == None: tree = tpctree(d[["x", "y", "z"]],options)
            else: tree.update(d[["x", "y", "z"]]) 
            tree.data.to_csv(file_name+".updated")
        return tree
        
    def merge_data(self, data):
        if data is not None:
            if self.data is None or len(self.data)==0:#first time case
                data.index.name = "key"
                self.data = data.reset_index()
                self.data["t"] = self._t#we do not trust any t passed in but we should not really overwrite it - think about this
                self.data["angles"] = None
            else:
                data.loc[:,'t']  = self._t
                self.data = pd.concat([self.data, data])
        #set the ordinal offset for this page of data at time t        
        temp = self.data.reset_index()
        self._t_offset = temp[temp.t==self._t].index[0]
        
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
    def stats(self): return pd.DataFrame([self._stats[k] for k in self._stats.keys()])
    
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
        #translations
        for v in tpctree.simple_sampler(t1[self._columns]):
            for w in tpctree.simple_sampler(t2[self._columns]):
                yield tpctree.find_translation(w,v)
              
        #TODO - a smart constellation pair sampler where we first evaluate translations to create points that exist in both frames and then select 
        #the same id permutation in each one possibly with some randomness??
        #can the general affine transform improve the match??
        
        #general
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
        transform.ref_vector = tpctree._tr_ref_displacement_(transform)
        transform.ref_angle = tpctree._tr_ref_angle_(transform)
        
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
        tr.ref_vector = tpctree._tr_ref_displacement_(tr)
        tr.ref_angle = tpctree._tr_ref_angle_(tr)
        
        return tr
    
    def tr_type(tr):
        try:  return tr.__type__
        except:  return "general"
    
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
        
        tr =  translation_transform
        tr.ref_vector = tpctree._tr_ref_displacement_(tr)
        tr.ref_angle = tpctree._tr_ref_angle_(tr)
        return tr
    
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
        self._transform_collection = []
        for counter,tr in enumerate(collection):
            if tr is None: continue
            score = self.apply_transform(tr,False,lag=lag)
            if score <= best_score: 
                best_index, best_tr , best_score= counter,tr,score
            #if collecting
            self._transform_collection.append({"score": score, "tr": tr, "type":  tpctree.tr_type(tr)})
            
        self.merge_statistic("best_tr_index", best_index)
        self.merge_statistic("best_tr_type", tpctree.tr_type(tr))
        
        return best_tr
    #####END TRANSFORMATIONS#######
        
    #def __eps_from_min_intertarget_distance__(self):
    #    #for each point in target, find the smallest distance between any two points
    #    #half this value and take the min of the max_epsilon (bio-feasible) and this value
    #    #rationale: adapt to the data up to a point. if we have some points very close together then we got the bio wrong or there is noise
        
    def update(self, data,t=None): #unless t is given, auto-inc
        """
        Update the tracker with new points
        """
        #increment the clock
        self._t += 1
        if data is None or len(data)==0: return None
        self.merge_data(data)
        #safety i.e. even though the clock has been ticking, we have only just found blobs, so nothing to transform
        if len(self.prev_blobs) == 0:
            #we know they are all new because there is nothing in the last frame to match
            self.auto_fill_ids()
            return self.blobs
        
        ##REFACTOR best transform here for any LAG -> apply_best_Transform
        t = self._t
        scores = tpctree.translations_from_marriages(self[t-1],self[t] ,name="random_tr",eps = self._epsilon, add_concensus_best=self._use_tr_concensus)
        #apply the best transform and assign ids
        self.apply_transform(scores.iloc[0]["tr"])
        
        #consider check the last transform and new global affines here and re-apply
        if self._transforms_enabled:
            aftr_scores = tpctree.transformations_from_stored_marriages(self[t-1],self[t], name="random_affine", eps=self._epsilon)
            if aftr_scores is not None and aftr_scores.iloc[0]["score"] < scores.iloc[0]["score"]:
                scores = pd.concat([scores,aftr_scores]).sort_values("score")
                self.apply_transform(scores.iloc[0]["tr"])
          
        #from the sorted list take the best
        self._last_transform = scores.iloc[0]["tr"]
        self.merge_statistic("best_tr", scores.iloc[0]["score"])
        self.merge_statistic("best_tr_type", scores.iloc[0]["type"])
        self._transform_collection = scores
        ## END BEST REFACTOR

        #if self._alt_lags_enabled:#analytics
        #    for _lag in [2,3]:  self.apply_best_transform(lag2,lag=_lag)
            
        return self.blobs
    
    def __normed_distance__(D,epsilon):
        """
        simple helper methods to replace infinity values by the model worst-score
        """
        D[(np.isinf(D)) ] = epsilon+1
        return D
    
    def auto_fill_ids(self):
        next_key = int(self.data.key.max() + 1)
        mask = self.blobs["key"].isnull()
        num_orphans = len(mask)
        newids = list(range(next_key, next_key+num_orphans))
        self.data.loc[self.data["t"] == self._t, "key"] = newids
    
    def __validate_ids__(self,res,lag=1, angles=None):
        """
        If the result is infinite, there is no neighbour within the ball of radius epsilon
        In such cases we should set the id to newly generated ids TODO
        """
        #this is fixed param for reducing tolerance for anistopric catchment areas
        beta = 0.2
        next_key = int(self.data.key.max() + 1)
        ids = res[1]
        #handle anisop - if either the score fails or the angle is excessive based on settings
        __epsilon__ = np.array([self._epsilon for i in res[0]])
        #we scale down the tolerance by a factor beta wherever the angle is too large
        #__epsilon__[np.where(angles>self._angle_tresh)[0]] = self._epsilon*beta#should shrink only matches
        #gen mask
        invalid_mask = res[0]>=(__epsilon__+1)
        #set the valid/invalid "masks" in ordinal space  
        valid_mask = np.where(~invalid_mask)[0]
        invalid_mask = np.where(invalid_mask)[0]
        
        self._detect_ratio = [len(valid_mask),len(ids)]
        self.merge_statistic("detections_ratio", self._detect_ratio)
                
        num_orphans = len(invalid_mask) 
        #for the valid ids, take the key for them (off the prev)
        ids[valid_mask] = self[self._t-lag].iloc[ids[valid_mask]].key
        #for the none-valid ids, generate new keys
        newids = list(range(next_key, next_key+num_orphans))
        ids[invalid_mask ] = newids
        return np.array(ids,np.int)
        
    def _tr_ref_angle_(tr):
        ref_vector = np.array([[2,2,2]])
        u = (np.array([[4,2,2]]) - ref_vector)[0]
        v =  tr(ref_vector)-ref_vector#returns a matrix
        v = np.array(v)[0]
        return utils.angle(u,v)
    
    def _tr_ref_displacement_(tr):
        ref_vector = np.array([[2,2,2]])
        u =  tr(ref_vector)-ref_vector#returns a matrix
        return np.array(u)[0]
    
    def determine_angles(self,res,tr,lag=1):
        """
        For anisotropic catchment areas we need an angle between the reference transform angle and the actual married objects
        If the angle is larger than a configured allowed the valid_ids will skip the marriage even if the objects are less than epsilon apart
        """
        def compute_angle(row):
            first_ = np.array([row["x"], row["y"], row["z"]])
            next_= np.array([row["xnext"], row["ynext"], row["znext"]])
            v = next_-first_
            #we are getting angle between this aribtrary reference vector along the supplied transform "tr" and the point displacement
            u = tpctree._tr_ref_displacement_(tr)
            self.merge_statistic("tr_ref_vector", tr.ref_vector)
            self.merge_statistic("tr_ref_angle", tr.ref_angle)
            
            return utils.angle(u,v)

        #self.merge_statistic("best_ref_vec", tr(np.array([[0,0,0]])))
        
        ids = res[1] #pointer to index on prev in ordinal not df.index space
        valid_mask = np.where(res[0]<self._epsilon+1)[0]
        #reset_index so we can join the same-size datasets in ordinal space
        prev=self[self._t-lag].iloc[ids[valid_mask]].reset_index()
        cur = self[self._t].iloc[valid_mask].reset_index()
        cur = cur.join(prev, rsuffix="next")
        #set the angles where valid in ordinal space using the t_offset for current data page
        ids = np.array(valid_mask) + self._t_offset
        angle_col_ordinal = list(self.data.columns).index("angles")
        angles = list(cur.apply(compute_angle,axis=1))
        try: self.data.iloc[ids, [angle_col_ordinal]] = angles
        except: pass #not sure about this yet
 
        return self[self._t]["angles"]#return here because it gives index properly - some nan
        
    def __apply__(tr, X,Y,epsilon,columns=["x","y","z"]):
        """
        supply a dataframe with labelled columns for data - from Y->X
        do not currently support passing just numpy arrays even though it is a simple switch
        #returns the tree scores, the score overall and the projected 
        """
        #compute the transfomration from Y-> Back to X (Ys are the ones that need id pairings)
        #make sure that we compute the transform that way too
        TR = tr(Y[columns].as_matrix())
        #index Y, i.e. the target vector - a suitable result from this is the pairing
        #we do not deal with marriage problem and simply assign closest - could improve this but fine based on our data
        target = scKDTree(X[columns].as_matrix())
        #get the first nearest neighbour within threshold
        res=target.query(TR, k=1,distance_upper_bound=epsilon) # TODO better abstraction might add normed scored to res[2] so it is contained here
        #the score is the bounded distance - see called function - kdtree returns vector [distance, ids]
        score = tpctree.__normed_distance__(res[0],epsilon).mean().round(4)  
        #assign ids
        return res, score, TR
        
    def apply_transform(self, tr, accept_pair=True,lag=1):
        #get the actual frame-frame data points
        X,Y = self[self._t-lag],self[self._t]
        #defensive 
        if len(X) == 0 or len(Y) == 0:return -1
   
        res, score,TR = tpctree.__apply__(tr, X,Y,self._epsilon, self._columns)
        
        key_name = "key" if lag == 1 else "key"+str(lag)
        #if key_name not in self.data.columns:self.data[key_name] = None #do i need this
        #minimise the number of things outside the base below as called much more often!
        if accept_pair:
            #for anisop calc angles - haven chosen best independant of angles
            angles = self.determine_angles(res,tr,lag)
            #print("Accepting pair for lag", lag)
            self._temp_res = np.array(res)
            #get the keys corresponding to the integer index returned from the kdtree
            self.data.loc[self.data["t"] == self._t, key_name] = self.__validate_ids__(res,lag=lag,angles=angles)
            #assign epsilons
            self.data.loc[self.data["t"] == self._t, "epsilon"+str(lag)] = tpctree.__normed_distance__(res[0],self._epsilon)
            #should always be possible to either assign or generate - null key not acceptable
            if lag == 1: self.data.key = self.data.key.astype(int)
            
        #return score of this transform
        return score
    
    def concensus_best_from_scores(t1, t2, scores, eps=DEFAULT_EPSILON, head_size=5):
        average_disp = np.stack(scores[0:head_size]["trref"].values).T.mean(axis=1)
        crowd_tr = tpctree.find_translation(np.array([[0,0,0]]),average_disp)
        res, score,TR = tpctree.__apply__(crowd_tr, t1,t2,eps) 
        
        return crowd_tr, score

    def find_transform_from_dataframe_rows(df1,df2,point_ordinal_tuple):
        #print(point_ordinal_tuple, len(df1), len(df2))
        X= df1.iloc[point_ordinal_tuple[1]][["x","y","z"]].as_matrix()
        Y= df2.iloc[point_ordinal_tuple[0]][["x","y","z"]].as_matrix()    
        return tpctree.find_translation(Y,X)

    def transformations_from_stored_marriages(t1,t2, mar=None, name="handpicked",eps=DEFAULT_EPSILON):
        #this is done because we compare based on ordinals and we must ignore any other index
        t1 = t1.reset_index()
        t2 = t2.reset_index()
        rws = []
        for x,y in tpctree.marriage_constellation_sampler(t1,t2):
            try:
                tr = tpctree.find_transform(y,x)
                res, score,TR = tpctree.__apply__(tr, t1,t2, epsilon=eps)
                d = {"score":score, "trang":tr.ref_angle, "trref": tr.ref_vector, "marriage":np.stack([x,y]), "tr":tr}
                rws.append(d)
            except:#not always possible for LSq to converge
                pass
        if len(rws)==0: return None
        scores = pd.DataFrame(rws).sort_values("score").reset_index()
        scores["type"] = name
        scores = scores.join(pd.DataFrame(np.stack(scores["trref"].values),columns=["x","y","z"]))
        return scores
    
    def translations_from_marriages(t1,t2, mar=None, name="handpicked",eps=DEFAULT_EPSILON,add_concensus_best=True,include_identity=True):
        """
        if mar is None, computes cartesian
        t1 and t2 should be euclidean point matrices
        """
        #this is done because we compare based on ordinals and we must ignore any other index
        t1 = t1.reset_index()
        t2 = t2.reset_index()
        rws = []
        
        #emphhasise that the transform is from the 2nd dataframe to the first
        if mar is None: mar =[list(element) for element in product(*[t2.index.values,t1.index.values])]
        
        for m in mar:
            tr = tpctree.find_transform_from_dataframe_rows(t1,t2,m)
            res, score,TR = tpctree.__apply__(tr, t1,t2, epsilon=eps)
            d = {"score":score, "trang":tr.ref_angle, "trref": tr.ref_vector, "marriage":m, "tr":tr, "type": name}
            rws.append(d)
            
        if include_identity:
            #columns are assumed to be length 3 - need to update these static methods with a way to pass col assumptions
            #here i am using a small translation instead of idenity
            tr = tpctree.translation_transform_for(0.01*np.ones((1,3)))
            res, score,TR = tpctree.__apply__(tr, t1,t2, epsilon=eps)
            d = {"score":score, "trang":tr.ref_angle, "trref": tr.ref_vector, "marriage":m, "tr":tr, "type": "identity"}
            rws.append(d)
            
        scores = pd.DataFrame(rws).sort_values("score").reset_index()
        scores = scores.join(pd.DataFrame(np.stack(scores["trref"].values),columns=["x","y","z"]))

        if add_concensus_best:
            tr, score = tpctree.concensus_best_from_scores(t1,t2, scores, eps=eps)
            cb = pd.DataFrame([{"score":score, "trang":tr.ref_angle, "trref": tr.ref_vector, "marriage":None, "tr":tr}])
            cb["type"] = "tr_conc_best"
            scores = pd.concat([scores,cb]).sort_values("score")
        
        return scores
    
    ###Samplers###
    def simple_sampler(df,columns=["x", "y", "z"]):  
        """simply sample all the points"""
        return df[columns].as_matrix()

    def marriage_constellation_sampler(t1,t2, N=50, k=3):
        #pair on marriages to find coordinates along a common spine
        pairing = t1.join(t2,on='key',rsuffix='next')
        if(len(pairing)<k):return #nothing to do
        #find permutations of these 
        perms = np.array(list(combinations(pairing.index.values,k)))
        #randomize
        perms = perms[np.random.choice(range(len(perms)),len(perms))]
        #select the max that we want
        N = -1 if N > len(perms) else N
        for p in perms[:N]:
            u = pairing.iloc[p][["x","y","z"]].as_matrix()
            v = pairing.iloc[p][["xnext","ynext","znext"]].as_matrix()
            yield u,v

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
