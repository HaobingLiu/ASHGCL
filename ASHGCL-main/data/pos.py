
from collections import Counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import cosine_similarity
####################################################
# This tool is to generate positive set with a thre-shold "pos_num".
# dataset  pos_num   similarity_threshold
# acm      5                  0.3
# dblp     700                0.08
# aminer   17                 0
#
#########################Topologically positive sample###############################
pos_num = 5
p = 4019
pap = sp.load_npz("./acm/pap.npz")
pap = pap / pap.sum(axis=-1).reshape(-1,1)
psp = sp.load_npz("./acm/psp.npz")
psp = psp / psp.sum(axis=-1).reshape(-1,1)
all = (pap + psp).A.astype("float32")
all_ = (all>0).sum(-1)

pos = np.zeros((p,p))
k=0#

for i in range(len(all)):
    one = all[i].nonzero()[0]
    if len(one) > pos_num:
        oo = np.argsort(-all[i, one])
        sele = one[oo[:pos_num]]
        pos[i, sele] = 1
        k += 1
    else:
        pos[i, one] = 1

pos = sp.coo_matrix(pos)


#############################Attribute positive sample#############################
feat_p = sp.load_npz("./acm/p_feat.npz").tocsr()
similarity_matrix = cosine_similarity(feat_p)
similarity_threshold = 0.3

pos1 = np.zeros((p, p))
for i in range(p):

    similarities = similarity_matrix[i]
    similar_nodes = np.where(similarities > similarity_threshold)[0]
    if len(similar_nodes) > pos_num:
        top_similar = similar_nodes[np.argsort(-similarities[similar_nodes])[:pos_num]]
    else:
        top_similar = similar_nodes
    pos1[i, top_similar] = 1
pos1 = sp.coo_matrix(pos1)

#combine the two samples
combined_pos = sp.coo_matrix((pos + pos1) > 0)
sp.save_npz("./acm/combined_pos.npz", combined_pos)

