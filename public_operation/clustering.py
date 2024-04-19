import numpy as np

import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# import ElbowVisualizer
# from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score

import scipy.cluster.hierarchy as shc
from matplotlib import pyplot
import pandas as pd


# def output_hidden_states(model):
#

class Clustering:

    def __init__(self, do_layer=0, do_neuron=0):
        self.do_layer = do_layer
        self.do_neuron = do_neuron

    def clustering(self, inputs):
        inputs = np.array(inputs)
        km = KMeans(n_clusters=7, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(inputs)
        labels = km.labels_
        print("-->labels:", km.labels_)
        centers = km.cluster_centers_
        print("-->center:", centers, type(centers))
        print(centers.shape)
        inertia = km.inertia_
        print("-->inertia_", km.inertia_)

        return labels, centers.tolist(), inertia

        # distortions = []
        # for i in range(2, 40):
        #     km = KMeans(n_clusters=i,
        #                 init='k-means++',
        #                 n_init=10,
        #                 max_iter=300,
        #                 random_state=0)
        #     km.fit(inputs)
        #     # distortions.append(km.inertia_)
        #     # distortions.append(metrics.silhouette_score(inputs, km.labels_, metric='euclidean'))
        #     distortions.append(metrics.calinski_harabasz_score(inputs, km.labels_))
        #     # score = metrics.calinski_harabasz_score(X, y_pre)
        #
        # plt.plot(range(2, 40), distortions, marker='o')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('Distortion')
        # plt.savefig("clustering_result ch_index.png")
        # plt.close('all')
        # # plt.show()

    def clustering_gap_statistic(self, inputs):
        print("-->clustering_gap_statistic")
        inputs = np.array(inputs)
        # Gap Statistic for K means
        def optimalK(data, nrefs=2, maxClusters=40):
            """
            Calculates KMeans optimal K using Gap Statistic
            Params:
                data: ndarry of shape (n_samples, n_features)
                nrefs: number of sample reference datasets to create
                maxClusters: Maximum number of clusters to test for
            Returns: (gaps, optimalK)
            """
            gaps = np.zeros((len(range(1, maxClusters)),))
            resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
            for gap_index, k in enumerate(range(2, maxClusters)):
                print("-->k", k)
                # Holder for reference dispersion results
                refDisps = np.zeros(nrefs)
                # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
                for i in range(nrefs):
                    # Create new random reference set
                    randomReference = np.random.random_sample(size=data.shape)

                    # Fit to it
                    km = KMeans(k)
                    km.fit(randomReference)

                    refDisp = km.inertia_
                    refDisps[i] = refDisp
                # Fit cluster to original data and create dispersion
                km = KMeans(k)
                km.fit(data)

                origDisp = km.inertia_
                # Calculate gap statistic
                gap = np.log(np.mean(refDisps)) - np.log(origDisp)
                # Assign this loop's gap statistic to gaps
                gaps[gap_index] = gap

                resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

            return (gaps.argmax() + 1, resultsdf)

        score_g, df = optimalK(inputs, nrefs=29, maxClusters=50)
        plt.figure()
        plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b')
        plt.xlabel('K')
        plt.ylabel('Gap Statistic')
        plt.title('Gap Statistic vs. K')
        # plt.savefig("hidden_states/clustering_result gap statistic.png")
        plt.savefig("hidden_states/clustering_result gap statistic(pooler_output)" + str(self.do_layer) + "_" + str(self.do_neuron) + ".png")
        plt.close('all')

    def clustering_elbow(self, inputs):
        print("-->clustering_elbow")
        inputs = np.array(inputs)
        # Elbow Method for K means
        model = KMeans()
        # k is range of number of clusters.
        from yellowbrick.cluster import KElbowVisualizer
        plt.figure()
        visualizer = KElbowVisualizer(model, k=(2, 40), timings=True)
        visualizer.fit(inputs)  # Fit data to visualizer
        path = "hidden_states/clustering_result elbow" + str(self.do_layer) + "_" + str(self.do_neuron) + ".png"
        # path = "hidden_states/clustering_result elbow(pooler_output).png"
        visualizer.show(outpath=path)  # Finalize and render figure

    def clustering_Silhouette(self, inputs):
        print("-->clustering_Silhouette")
        inputs = np.array(inputs)
        model = KMeans()
        # k is range of number of clusters.
        from yellowbrick.cluster import KElbowVisualizer
        plt.figure()
        visualizer = KElbowVisualizer(model, k=(2, 30), metric='silhouette', timings=True)
        visualizer.fit(inputs)  # Fit the data to the visualizer
        visualizer.show(outpath="clustering_result silhouetts1.png")  # Finalize and render the figure

    def clustering_ch(self, inputs):
        print("-->clustering_ch")
        inputs = np.array(inputs)
        model = KMeans()
        # k is range of number of clusters.
        from yellowbrick.cluster import KElbowVisualizer
        plt.figure()
        visualizer = KElbowVisualizer(model, k=(2, 30), metric='calinski_harabasz', timings=True)
        visualizer.fit(inputs)  # Fit the data to the visualizer
        visualizer.show(outpath="clustering_result ch1.png")  # Finalize and render the figure

    def clustering_db(self, inputs):
        print("-->clustering_db")
        inputs = np.array(inputs)
        def get_kmeans_score(data, center):
            '''
            returns the kmeans score regarding Davies Bouldin for points to centers
            INPUT:
                data - the dataset you want to fit kmeans to
                center - the number of centers you want (the k value)
            OUTPUT:
                score - the Davies Bouldin score for the kmeans model fit to the data
            '''
            # instantiate kmeans
            kmeans = KMeans(n_clusters=center)
            # Then fit the model to your data using the fit method
            model = kmeans.fit_predict(data)

            # Calculate Davies Bouldin score

            score = davies_bouldin_score(data, model)
            return score

        scores = []
        centers = list(range(2, 30))
        for center in centers:
            print("-->center", center)
            scores.append(get_kmeans_score(inputs, center))

        plt.figure()
        plt.plot(centers, scores, linestyle='--', marker='o', color='b');
        plt.xlabel('K')
        plt.ylabel('Davies Bouldin score')
        plt.title('Davies Bouldin score vs. K')
        plt.savefig("clustering_result db1.png")


    import numpy as np

    def clustering_gap_statistic1(self, inputs):
        print("-->clustering_gap_statistic1")
        inputs = np.array(inputs)
        def calculate_Wk(data, centroids, cluster):
            K = centroids.shape[0]
            wk = 0.0
            for k in range(K):
                data_in_cluster = data[cluster == k, :]
                center = centroids[k, :]
                num_points = data_in_cluster.shape[0]
                for i in range(num_points):
                    wk = wk + np.linalg.norm(data_in_cluster[i, :] - center, ord=2) ** 2

            return wk

        def bounding_box(data):
            dim = data.shape[1]
            boxes = []
            for i in range(dim):
                data_min = np.amin(data[:, i])
                data_max = np.amax(data[:, i])
                boxes.append((data_min, data_max))

            return boxes

        def gap_statistic(data, max_K, B):
            num_points, dim = data.shape
            K_range = np.arange(1, max_K, dtype=int)
            num_K = len(K_range)
            boxes = bounding_box(data)
            data_generate = np.zeros((num_points, dim))

            ''' 写法1
            log_Wks = np.zeros(num_K)
            gaps = np.zeros(num_K)
            sks = np.zeros(num_K)
            for ind_K, K in enumerate(K_range):
                cluster_centers, labels, _ = cluster_algorithm(data, K)
                log_Wks[ind_K] = np.log(calculate_Wk(data, cluster_centers, labels))
                # generate B reference data sets
                log_Wkbs = np.zeros(B)
                for b in range(B):
                    for i in range(num_points):
                        for j in range(dim):
                            data_generate[i][j] = \
                                np.random.uniform(boxes[j][0], boxes[j][1])
                    cluster_centers, labels, _ = cluster_algorithm(data_generate, K)
                    log_Wkbs[b] = \
                        np.log(calculate_Wk(data_generate, cluster_centers, labels))
                gaps[ind_K] = np.mean(log_Wkbs) - log_Wks[ind_K]
                sks[ind_K] = np.std(log_Wkbs) * np.sqrt(1 + 1.0 / B)
            '''

            ''' 写法2
            '''
            log_Wks = np.zeros(num_K)
            for indK, K in enumerate(K_range):
                print("-->K", K)
                km = KMeans(K)
                km.fit(data)
                cluster_centers = km.cluster_centers_
                labels = km.labels_
                # cluster_centers, labels, _ = cluster_algorithm(data, K)
                log_Wks[indK] = np.log(calculate_Wk(data, cluster_centers, labels))

            gaps = np.zeros(num_K)
            sks = np.zeros(num_K)
            log_Wkbs = np.zeros((B, num_K))

            # generate B reference data sets
            for b in range(B):
                print("-->b", B)
                for i in range(num_points):
                    for j in range(dim):
                        data_generate[i, j] = \
                            np.random.uniform(boxes[j][0], boxes[j][1])
                for indK, K in enumerate(K_range):
                    print("-->K", K)
                    km = KMeans(K)
                    km.fit(data_generate)
                    cluster_centers = km.cluster_centers_
                    labels = km.labels_
                    # cluster_centers, labels, _ = cluster_algorithm(data_generate, K)
                    log_Wkbs[b, indK] = \
                        np.log(calculate_Wk(data_generate, cluster_centers, labels))

            for k in range(num_K):
                gaps[k] = np.mean(log_Wkbs[:, k]) - log_Wks[k]
                sks[k] = np.std(log_Wkbs[:, k]) * np.sqrt(1 + 1.0 / B)

            return gaps, sks, log_Wks

        max_K = 11
        B = 1
        K_range = np.arange(1, max_K, dtype=int)
        gaps, stds, log_wks = gap_statistic(inputs, max_K, B)

        num_gaps = len(gaps) - 1
        gaps_diff = np.zeros(num_gaps)
        for i in range(num_gaps):
            gaps_diff[i] = gaps[i] - (gaps[i + 1] - stds[i + 1])

        print("-->gaps", gaps)
        print("-->K_range", K_range)
        print("-->log_wks", log_wks)

        select_K = K_range[np.argmax(gaps)]
        print('Select K: {}'.format(select_K))
        print('Gaps Diff: {}'.format(gaps_diff))

        plt.plot(K_range, gaps, 'o-')
        plt.xlabel('K')
        plt.ylabel('Gap')
        plt.savefig("gap.png")
        print("saved gap.png")

        plt.figure()
        plt.plot(K_range, log_wks, 'o-')
        plt.xlabel('K')
        plt.ylabel('Log(Wk)')
        plt.savefig("log(Wk).png")
        print("saved log(Wk).png")

        plt.figure()
        plt.xlabel('K')
        plt.ylabel('Gap(k) - (Gap(k+1) - s(k+1))')
        plt.bar(K_range[:num_gaps], gaps_diff)
        plt.savefig("gap_bar.png")
        print("saved gap_bar.png")
