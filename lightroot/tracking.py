# the tree code - this can be refactored - many static transformation  bits could go in their own class/file
# some plotting things could also be moved out into a tree plotting helper
from scipy.spatial import KDTree as scKDTree
import pandas as pd 
import numpy as np
from numba import jit
from itertools import combinations,product
from . import utils,plots
from tqdm import tqdm
#
DEFAULT_EPSILON = 18
class tpctree(object):
    """
    In online mode, pass blocks of coordinates for each time point and update the tracker
    See the process_file loop
    """
    def __init__(self, data,options={}):
        ##init
        self._t = 1 #start at one - presume something to track from here backwards 
        self.data = None
        self._t_offset= 0
        self._stats = {}
        self._transform_collection = []
        self._detect_ratio = []
        self._test_scores = []
        
        self._epsilon = DEFAULT_EPSILON if "epsilon" not in options else options["epsilon"]
        self._lags = [1] if "lags" not in options else options["lags"]
        self._transforms_enabled = False if "transforms" not in options else options["transforms"]
        self._use_tr_concensus = True if "use_tr_concensus" not in options else options["use_tr_concensus"]
        self._angle_tresh = 90 if "angle_tresh" not in options else options["angle_tresh"]
        self._use_data_t = True if "use_data_t" not in options else options["use_data_t"]
        self._sample_size =  50 if "sample_size" not in options else options["sample_size"]
        #temp - need an aspects for this
        self._debug = False if "debug" not in options else options["debug"]
        
        self.merge_data(data)
       
        self._last_transform = None        
        self._columns = ["x","y","z"]
        #add a small translation to the priors - this is effectively a void transform
        self._transform_priors = []
        self._transformer = tpctree.__default_transformer__
        #self._sampler = tpctree.constellation_sampler#simple_sampler
        self._sampler = tpctree.simple_sampler
        
    def process_file(file_name, start=0,end=-1, options={"epsilon":15, "transforms":True}):
        data = pd.read_csv(file_name).reset_index()[["t","x","y","z"]]
        tree=None
        maxt = data.t.max()+1 if end==-1 else end
        for g,d in tqdm(data[(data.t>=start)&(data.t<=end)].groupby("t")):
            if tree == None: tree = tpctree(d[["x", "y", "z", "t"]],options)
            else: tree.update(d[["x", "y", "z", "t"]]) 
            tree.data.to_csv(file_name+".updated")
        return tree
        
    def __handle_data_tvals__(self, data):
        if self._use_data_t:
            if "t" in data:
                t_in = data.iloc[0]["t"]
                self.merge_statistic("data_t_index", t_in)
                
    def merge_data(self, data):
        if data is not None:
            self.__handle_data_tvals__(data)
            if self.data is None or len(self.data)==0:#first time case
                data.index.name = "key"
                self.data = data.reset_index()
                self.data["t"] = self._t
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
    
    def _add_object_ages_in_place_(data):
        """
        given a set of data points with t and key values, use the full life matrix to add the age details onto the dataframe
        """
        lm = tpctree.make_life_matrix(data,restricted=False).astype(int)
        ageMat = lm.cumsum(axis=1) * lm
        def age_at(t,i):return ageMat[t][i]
        def age_at_px(row): return age_at(row["t"],row["key"])
        def max_age_at(t,i):return ageMat.iloc[i].max()
        def max_age_at_px(row): return max_age_at(row["t"],row["key"]) 
        data["age"]=data.reset_index().apply(age_at_px, axis=1)
        data["max_age"]=data.reset_index().apply(max_age_at_px, axis=1)
        return data

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
        print("getting transforms")
        #general
        if self._transforms_enabled:
            try:
                print("applying transforms", self._sample_size)
                #generalised transforms
                for v in tpctree.constellation_sampler(t1[self._columns],N=self._sample_size):
                    for w in tpctree.constellation_sampler(t2[self._columns], N=self._sample_size):
                        yield tpctree.find_transform(w,v)
                print("done applying transforms", self._sample_size)
            except Exception as ex:
                print("failed sampling", repr(ex))
                pass
                #self.merge_statistic("failed_sample_transform", True)
    
    def find_transform(X,Y): 
        """
        Given two matrices, try to compute the transformation
        """
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:,:-1]
        A, res, rank, s = np.linalg.lstsq(pad(X), pad(Y),rcond=None)
        tr = lambda x: unpad(np.dot(pad(x), A))
        tr.__ttype__ = "transform"
        tr.ref_vector,tr.ref_angle = tpctree._tr_refs_(tr)

        return tr
    
    def find_translation(X,Y): 
        """Given two matrices, compute the simple translation
           it as assumed the point subsets represent likely congruences and the mean translation vector is used
        """
        mean_vec = np.array(Y)-np.array(X)
        if len(mean_vec.shape) == 1:  mean_vec = np.stack([mean_vec])
        mean_vec = np.stack([mean_vec.mean(axis=0)])
        tr= tpctree.translation_transform_for(mean_vec)
        tr.__ttype__ = "translation"
        tr.ref_vector,tr.ref_angle = tpctree._tr_refs_(tr)
        
        return tr
    
    def tr_type(tr):
        try:  return tr.__type__
        except:  return "general"
    
    def identity_transform(s=3):
        def _identity_(m):
            m = np.matrix(np.hstack((m,np.ones((len(m),1)))))
            A = np.matrix(np.eye(s+1))
            m = A * m.T
            return m.T[:,:-1]
        tr =  _identity_
        tr.ref_vector = np.array([0.,0.,0.])
        tr.ref_angle = None
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
        
        tr =  translation_transform
        tr.ref_vector,tr.ref_angle = tpctree._tr_refs_(tr)
        
        return tr
    
    def __default_transformer__(X,Y):
        try:   
            #print("returning reg transform")
            return tpctree.find_transform(Y,X)
        except Exception as ex:
            print(repr(ex))
            return tpctree.find_translation(Y,X)
    
    #def best_transform(self, collection,lag=1):
    #    """
    #    Simply return the transform that minimises the objective with respect to the data
    #    The data will be the point cloud at time [t] and [t-1]
    #    """
    #    best_index, best_tr, best_score = None, None, self._epsilon+1
    #    self._transform_collection = []
    #    for counter,tr in enumerate(collection):
    #        if tr is None: continue
    #        score = self.apply_transform(tr,False,lag=lag)
    #        if score <= best_score: 
    #            best_index, best_tr , best_score= counter,tr,score
    #        #if collecting
    #        self._transform_collection.append({"score": score, "tr": tr, "type":  tpctree.tr_type(tr)})
    #        
    #    self.merge_statistic("best_tr_index", best_index)
    #    self.merge_statistic("best_tr_type", tpctree.tr_type(tr))
    #    
    #    return best_tr
    #####END TRANSFORMATIONS#######
        
    #def __eps_from_min_intertarget_distance__(self):
    #    #for each point in target, find the smallest distance between any two points
    #    #half this value and take the min of the max_epsilon (bio-feasible) and this value
    #    #rationale: adapt to the data up to a point. if we have some points very close together then we got the bio wrong or there is noise
      
    #sort in a consisten way
    def __score_sort__(scores): 
        #alot comes down to this - prefer a vector way to review this too - otherwise nightmare to analyse and debug 
        #if i scale the score by a distance kernel i would have to choose it but that would allow sort on only score
        #as presume the matches cannot be radically different in the space between
        #eturn scores.round(3).sort_values(["matches", "tr_norm", "score"],    ascending=[False,True,True])
        
        #integer norming is a laz tolerance which should always have matches sorted - but lets see
        
        scores["score_int"] = scores["score"].astype(int)
        return scores.sort_values(["score_int", "tr_norm", "matches"],  ascending=[True,True,False])
      
    def __update__(self, lag=1):
        t = self._t
        if t-lag < 0:return
        
        scores = tpctree.translations_from_marriages(self[t-1],self[t] ,name="random_tr",eps = self._epsilon, add_concensus_best=self._use_tr_concensus)
        #apply the best transform and assign ids
        self.apply_transform(scores.iloc[0]["tr"],lag=lag)
        
        #consider check the last transform and new global affines here and re-apply
        if self._transforms_enabled:
            
            aftr_scores = tpctree.transformations_from_stored_marriages(self[t-lag],self[t], name="random_affine", eps=self._epsilon, sample_size=self._sample_size)
            if aftr_scores is not None and aftr_scores.iloc[0]["score"] < scores.iloc[0]["score"]:
                scores = tpctree.__score_sort__(pd.concat([scores,aftr_scores]))
                self.apply_transform(scores.iloc[0]["tr"],lag=lag)
          
        if lag == 1:
            self._last_transform = scores.iloc[0]["tr"]
            self.merge_statistic("best_tr_score", scores.iloc[0]["score"])
            self.merge_statistic("best_tr_type", scores.iloc[0]["type"])
            self.merge_statistic("best_tr_angle", scores.iloc[0]["trang"])
            self.merge_statistic("best_tr_disp", scores.iloc[0]["trref"])
            self._transform_collection = scores
            self._rec_ref_angles_from_marriages_()
            
        if self._debug: 
            scores["t_index"] = t-lag
            self._test_scores.append(scores)
        
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
        
        for l in reversed(sorted(self._lags)):  self.__update__(l)
            
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
    
    def __validate_ids__(self,res,lag=1):
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
        
    def _tr_refs_(tr):
        """
        Get the displacement ref vector and the angle taking the displacement into account
        it is either clockwise or anti-clocwise from the x-axis depending on the sign of the disp vector
        """
        d = tpctree._tr_ref_displacement_(tr)
        sign = 1 if d[1] < 0 else -1
        return d,tpctree._tr_ref_angle_(tr,sign)
    
    def _tr_ref_angle_(tr,sign=1):
        ref_vector = np.array([[2,2,2]])
        u = (np.array([[4,2,2]]) - ref_vector)[0]
        v =  tr(ref_vector)-ref_vector#returns a matrix
        v = np.array(v)[0]
        return utils.angle(u,v)*sign
    
    def _tr_ref_displacement_(tr):
        ref_vector = np.array([[2,2,2]])
        u =  tr(ref_vector)-ref_vector#returns a matrix
        return np.array(u)[0]
    
    def _rec_ref_angles_from_marriages_(self,lag=1):
        key_name = "key" if lag == 1 else "key"+str(lag)
        pairing = self[self._t].set_index(key_name).join(self[self._t-lag].set_index(key_name),rsuffix='next')
        pairing_angles = []
        for k,row in pairing.iterrows():
            u = row[["x","y","z"]].as_matrix()
            v = row[["xnext","ynext","znext"]].as_matrix()
            disp = v-u
            sign = 1 if disp[1] < 0 else -1
            xaxis = np.array([1,0,0])
            angle = utils.angle(xaxis,disp)*sign
            pairing_angles.append({"key": k, "angle": angle })
        self.merge_statistic("marriage_angles", pairing_angles)

    #def determine_angles(self,res,tr,lag=1):
    #    """
    #    For anisotropic catchment areas we need an angle between the reference transform angle and the actual married objects
    #    If the angle is larger than a configured allowed the valid_ids will skip the marriage even if the objects are less than epsilon apart
    #    """
    #    def compute_angle(row):
    #        first_ = np.array([row["x"], row["y"], row["z"]])
    #        next_= np.array([row["xnext"], row["ynext"], row["znext"]])
    ##        v = next_-first_
    #       #we are getting angle between this aribtrary reference vector along the supplied transform "tr" and the point displacement
    #       u = tpctree._tr_ref_displacement_(tr)
    #       self.merge_statistic("tr_ref_vector", tr.ref_vector)
    #        self.merge_statistic("tr_ref_angle", tr.ref_angle)
    #        
    #        return utils.angle(u,v)#
    #
    #    #self.merge_statistic("best_ref_vec", tr(np.array([[0,0,0]])))
    #    
    #    ids = res[1] #pointer to index on prev in ordinal not df.index space
    #    valid_mask = np.where(res[0]<self._epsilon+1)[0]
    #    #reset_index so we can join the same-size datasets in ordinal space
    #    prev=self[self._t-lag].iloc[ids[valid_mask]].reset_index()
    #    cur = self[self._t].iloc[valid_mask].reset_index()
    #    cur = cur.join(prev, rsuffix="next")
    #    #set the angles where valid in ordinal space using the t_offset for current data page
    #    ids = np.array(valid_mask) + self._t_offset
    #    
    #    angle_col_ordinal = list(self.data.columns).index("angles")
    #    angles = list(cur.apply(compute_angle,axis=1))
    #    try: self.data.iloc[ids, [angle_col_ordinal]] = angles
    #    except: pass #not sure about this yet
    # 
    #    return self[self._t]["angles"]#return here because it gives index properly - some nan
        
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
            #get the keys corresponding to the integer index returned from the kdtree
            self.data.loc[self.data["t"] == self._t, key_name] = self.__validate_ids__(res,lag=lag)
            #assign epsilons
            self.data.loc[self.data["t"] == self._t, "epsilon"+str(lag)] = tpctree.__normed_distance__(res[0],self._epsilon)
            #should always be possible to either assign or generate - null key not acceptable
            if lag == 1: self.data.key = self.data.key.astype(int)
            
        #return score of this transform
        return score
    
    #a more useful and general way to do this would be to cluster the top h by displacement vector
    #then we could have them compete in clusters and we could take the best
    #this avoids taking a silly displacement mean and also smoothes out scores   
    def concensus_best_from_scores(t1, t2, scores, eps=DEFAULT_EPSILON, head_size=5):
        average_disp = np.stack(scores[0:head_size]["trref"].values).T.mean(axis=1)
        crowd_tr = tpctree.find_translation(np.array([[0,0,0]]),average_disp)
        res, score,TR = tpctree.__apply__(crowd_tr, t1,t2,eps)       
        return crowd_tr, score, tpctree._matches_(res,eps)

    def find_transform_from_dataframe_rows(df1,df2,point_ordinal_tuple):
        #print(point_ordinal_tuple, len(df1), len(df2))
        X= df1.iloc[point_ordinal_tuple[1]][["x","y","z"]].as_matrix()
        Y= df2.iloc[point_ordinal_tuple[0]][["x","y","z"]].as_matrix()    
        return tpctree.find_translation(Y,X)

    #wrapper function just to count how many matches in the res result[[],[]] are less than epsilon +1 the capped score 
    def _matches_(res, eps): return len(np.where(res[0]<eps+1)[0])
    
    def __apply_get_dict__(tr, t1,t2,epsilon,trtype,seed_marriages=None):
        res, score,TR = tpctree.__apply__(tr, t1,t2, epsilon=epsilon)
        return {"score":score,  
                "matches": tpctree._matches_(res,epsilon), 
                "trang":tr.ref_angle, 
                "trref": tr.ref_vector, 
                "tr_norm":np.linalg.norm(tr.ref_vector,ord=2), 
                "marriage":seed_marriages, 
                "tr":tr, 
                "type": trtype}
    
    def transformations_from_stored_marriages(t1,t2, lag=1, mar=None, name="handpicked",eps=DEFAULT_EPSILON,sample_size=None):
        #this is done because we compare based on ordinals and we must ignore any other index
        t1 = t1.reset_index()
        t2 = t2.reset_index()
        rws = []
        #print("appling transforms with samples", sample_size)
        for x,y in tpctree.marriage_constellation_sampler(t1,t2,lag=lag,N=sample_size):
            try:
                tr = tpctree.find_transform(y,x)
                rws.append(tpctree.__apply_get_dict__(tr, t1,t2, epsilon=eps,trtype="affine_general", seed_marriages=np.stack([x,y])))
                #print("added gen affine")
            except Exception as ex:#not always possible for LSq to converge but need to be carefull to re-check this after refactor
                #print(repr(ex))
                pass
        
        if len(rws)==0: return None
        
        scores = pd.DataFrame(rws).sort_values("score").reset_index()
        scores["type"] = name
        scores = scores.join(pd.DataFrame(np.stack(scores["trref"].values),columns=["x","y","z"]))
        
        return tpctree.__score_sort__(scores)
            
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
            rws.append(tpctree.__apply_get_dict__(tr, t1,t2, epsilon=eps,trtype="tr",seed_marriages=m))
            
        if include_identity:
            tr = tpctree.identity_transform(3)
            rws.append(tpctree.__apply_get_dict__(tr, t1,t2, epsilon=eps, trtype="identity"))
            
        #TODO sort by score and then displacement so we dinstinguis small disp - also update identity to actual identity    
        scores = pd.DataFrame(rws).sort_values("score").reset_index()
        scores = scores.join(pd.DataFrame(np.stack(scores["trref"].values),columns=["x","y","z"]))

        if add_concensus_best:
            tr, score, matches = tpctree.concensus_best_from_scores(t1,t2, scores, eps=eps)
            cb = pd.DataFrame([{"score":score, "tr_norm":np.linalg.norm(tr.ref_vector,ord=2), 
                                "matches": matches, "trang":tr.ref_angle, "trref": tr.ref_vector, "marriage":None, "tr":tr}])
            cb["type"] = "tr_conc_best"
            scores = pd.concat([scores,cb]).sort_values("score")
        
        return tpctree.__score_sort__(scores)
    
    ###Samplers###
    def simple_sampler(df,columns=["x", "y", "z"]):  
        """simply sample all the points"""
        return df[columns].as_matrix()

    def marriage_constellation_sampler(t1,t2,lag=1, N=50, k=3):
        #pair on marriages to find coordinates along a common spine
        #can deal with marriages made at each lag
        key_name = "key" if lag == 1 else "key"+str(lag)
        pairing = t1.join(t2,on=key_name,rsuffix='next')
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

    def constellation_sampler(df, k=3, N=100,min_angle=20,min_length=20):
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
    
    #refactor this out somewhere 
    def plot_transition(self, t, transform_seed=None,epsilon=15, use_keys=False, splot_tuple=None):
        from matplotlib import pyplot as plt
        fig, (ax1, ax2,ax3)  = None, (None,None,None)
        if splot_tuple != None:   fig, (ax1, ax2,ax3) = splot_tuple[0], splot_tuple[1] 
        else:plt.subplots(1, 3, figsize=(20, 8), tight_layout=True)
            
        t1 = self[t].reset_index()
        t2 = self[t+1].reset_index()
        
        #tpc_tree.list_married_betwee(t1,t2) -> returns key and t1_ordinal and t2_ordinal
        if transform_seed is None:
            #wip: if nothing suggested, find one that works and use it
            t1.index.name="aind"
            t1_=t1.reset_index().set_index("key")
            t2.index.name="bind"
            t2_=t2.reset_index().set_index("key")
            binding = t1_[["aind"]].join(t2_["bind"]).dropna().astype(int)
            if len(binding) > 0:#sample the first one
                transform_seed = np.array([[binding.iloc[0][1]],[binding.iloc[0][0]]])

        if transform_seed is not None and ax3 is not None:
            tr = tpctree.find_transform_from_dataframe_rows(t1,t2,transform_seed)
            plots.add_projection_from_points(tr, t2, epsilon=epsilon, ax=ax3)
            res, score,TR = tpctree.__apply__(tr, t1,t2,epsilon)
            ax3.scatter(x=t2.x, y=t2.y, s=30, c='b')
            ax3.scatter(x=t1.x, y=t1.y, s=20, c='g')
            if self._debug:
                title = "proposed by taking {}->{} scoring {} (avg. dist.)".format(transform_seed[0],transform_seed[1],score)
                #print(title)
                ax3.set_title(title)
            for k,r in t1.iterrows(): 
                _k = k if not use_keys else r["key"]
                ax3.annotate(str(_k), (r["x"],r["y"]+8),  ha='center', va='top', color='g', size=14)
            for k,r in t2.iterrows(): 
                _k = k if not use_keys else r["key"]
                ax3.annotate(str(_k), (r["x"],r["y"]+4),  ha='center', va='top', color='b', size=14)
        
        if ax1 is not None:
            ax1.scatter(x=t1.x, y=t1.y, s=30, c='g')
            for k,r in t1.iterrows(): 
                _k = k if not use_keys else r["key"]
                ax1.annotate(str(_k), (r["x"],r["y"]+8),  ha='center', va='top', color='g', size=14)
          
        if ax2 is not None:
            ax2.scatter(x=t2.x, y=t2.y, s=30, c='b')
            ax2.scatter(x=t1.x, y=t1.y, s=20, c='g')
            for k,r in t2.iterrows(): 
                _k = k if not use_keys else r["key"]
                ax2.annotate(str(_k), (r["x"],r["y"]+4),  ha='center', va='top', color='b', size=14)

        for ax in [ax1,ax2,ax3]:
            if ax is not None:
                ax.minorticks_on()
                ax.grid(which='major', linestyle=':', linewidth='0.5', color='grey')
                ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
                ax.set_ylim(ax.get_ylim()[::-1]) #flip
