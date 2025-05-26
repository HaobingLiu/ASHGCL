# cluster.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def cluster(embeds, label, nb_classes, dataset):
    nmi_list = []
    ari_list = []
    runs = 20


    all_embeds = embeds.cpu().numpy()
    all_labels = label.argmax(dim=-1).cpu().numpy()

    for _ in range(runs):

        kmeans = KMeans(n_clusters=nb_classes, n_init=10)

        y_kmeans = kmeans.fit_predict(all_embeds)


        nmi = normalized_mutual_info_score(all_labels, y_kmeans)
        ari = adjusted_rand_score(all_labels, y_kmeans)
        nmi_list.append(nmi)
        ari_list.append(ari)


    nmi_mean = np.mean(nmi_list)
    ari_mean = np.mean(ari_list)
    nmi_std = np.std(nmi_list)
    ari_std = np.std(ari_list)

    print("\t[Clustering] NMI_mean: {:.4f} var: {:.4f}  ARI_mean: {:.4f} var: {:.4f}"
          .format(nmi_mean, nmi_std, ari_mean, ari_std))


    with open(f"cluster_{dataset}.txt", "a") as f:
        f.write(f"{nmi_mean}\t{ari_mean}\n")
