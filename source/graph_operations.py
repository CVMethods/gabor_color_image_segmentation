from source.computation_support import *


def update_edges_weight(regions, rag, gabor_energies, ground_dist, method):
    """
    Obtain 3D color histogram of each superpixel region, then it computes the color distance between neighbor regions.
    :param img: Input image in RGB
    :param regions: label of each region
    :param rag: Region adjacency graph of the image
    :param convert2lab: Boolean parameter to indicate if work on the LAB color space. If false it work on RGB color space
    :param n_bins: Number of bins per color axe to quantize the color image
    :return: Regions adjacency graph with the edges weights updated
    """

    i_superpixel = np.unique(regions)
    superpixel_signatures = []
    for ii in i_superpixel:
        superpixel_signatures.append(gabor_energies[regions == ii].sum(axis=0)/gabor_energies[regions == ii].shape[0])  # Normalized w/suppix size
        # superpixel_signatures.append(gabor_energies[regions == ii].sum(axis=0))  # Not Normalized

    num_cores = multiprocessing.cpu_count()
    rag_weighted = rag.copy()
    if method == 'OT':
        # texture_dist = np.array(Parallel(n_jobs=num_cores)(
        #     delayed(em_dist_Rubner)(np.float64((superpixel_signatures[e[0]], superpixel_signatures[e[1]])),
        #                           ground_dist) for e in list(rag.edges)))
        # texture_dist = np.array(Parallel(n_jobs=num_cores)(
        #     delayed(em_dist_Pele)(np.float64((superpixel_signatures[e[0]], superpixel_signatures[e[1]])),
        #                           ground_dist) for e in list(rag.edges)))
        texture_dist = np.array(Parallel(n_jobs=num_cores)(
            delayed(em_dist_mine)(np.float64((superpixel_signatures[e[0]], superpixel_signatures[e[1]])),
                                  ground_dist) for e in list(rag.edges)))

        for ii, e in enumerate(list(rag.edges)):
            rag_weighted[e[0]][e[1]]['weight'] = texture_dist[ii]
        # # Get 3d color histograms
        # hist = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(color_3dhistogram)(img[regions == i], n_bins) for i in regions_ids))

        # # Compute the Optimal Transport (EMD) between neighbor regions
        # for e in list(rag.edges()):
        #     cost_matrix = ot.dist(np.array(hist[e[0]][1], dtype='int').T, np.array(hist[e[1]][1], dtype='int').T, 'sqeuclidean')
        #     dist = ot.emd2(hist[e[0]][0], hist[e[1]][0], cost_matrix, processes=num_cores)
        #     # dist += np.abs(np.mean(text_model[np.where(regions == e[0])]) - np.mean(text_model[np.where(regions == e[1])])) * 100
        #     # pdb.set_trace()
        #
        #     rag_weighted[e[0]][e[1]]['weight'] = dist

    # if method == 'KL':
    #     # Get 3d color histograms
    #     hist = np.array(Parallel(n_jobs=num_cores, require='sharedmem')(delayed(color_histogram)(img[regions == i], n_bins) for i in regions_ids), dtype=np.float32)
    #
    #     # # Compute the Optimal Transport (EMD) between neighbor regions
    #     # for e in list(rag.edges()):
    #     #     divergence = cv2.compareHist(hist[e[0]], hist[e[1]], 5)
    #     #     rag_weighted[e[0]][e[1]]['weight'] = divergence
    #
    #     # Compute the Optimal Transport (EMD) between neighbor regions
    #     for e in list(rag.edges()):
    #         divergence = cv2.compareHist(hist[e[0]], hist[e[1]], 5)
    #         rag_weighted[e[0]][e[1]]['weight'] = divergence

    return rag_weighted


def update_groundtruth_edges_weight(regions, rag, segments):
    num_cores = multiprocessing.cpu_count()

    i_superpixel = np.unique(regions)
    superpixel_labels = []
    for ii in i_superpixel:
        values, counts = np.unique(segments[regions == ii], return_counts=True)
        ind = np.argmax(counts)
        superpixel_labels.append(values[ind])

    groundtruth_dist = np.array(Parallel(n_jobs=num_cores)(
            delayed(dist_label)((superpixel_labels[e[0]], superpixel_labels[e[1]])) for e in list(rag.edges)))

    rag_weighted = rag.copy()
    for ii, e in enumerate(list(rag.edges)):
        rag_weighted[e[0]][e[1]]['weight'] += groundtruth_dist[ii]

    return rag_weighted


def get_pixel_graph(neighbors, img_shape):
    rows, cols, channels = img_shape
    pixels = np.arange(rows * cols)
    nodes = pixels.reshape((rows, cols))
    yy, xx = np.where(nodes >= 0)
    centroids = np.column_stack((yy, xx))
    knn_mat = kneighbors_graph(centroids, neighbors, mode='connectivity', include_self=False)
    _, indices, indptr = knn_mat.data, knn_mat.indices, knn_mat.indptr

    edges = [(yy[indices[j]], xx[indices[j]], yy[i], xx[i]) for i in pixels for j in range(indptr[i], indptr[i + 1]) if
             indices[j] <= i]
    edges_index = np.array([(nodes[e[0], e[1]], nodes[e[2], e[3]]) for e in edges])
    neighbor_edges = [[] for i in pixels]

    for ii, e in enumerate(edges_index):
        neighbor_edges[e[0]].append(ii)
        neighbor_edges[e[1]].append(ii)

    return edges_index, np.array(neighbor_edges)


def get_graph(img, regions, graph_type, neighbors, radius):

    if graph_type == 'rag':
        graph = rag_networkx(regions)
        graph = set_node_edge_attr(graph, img, regions)

    if graph_type == 'complete':
        graph = graph_complete_networkx(regions)
        graph = set_node_edge_attr(graph, img, regions)

    if graph_type == 'knn':
        graph = graph_knn(regions, neighbors)
        graph = set_node_edge_attr(graph, img, regions)

    if graph_type == 'eps':
       graph = graph_epsilon(regions, radius)
       graph = set_node_edge_attr(graph, img, regions)

    return graph


def graph_epsilon(regions, radius):
    region_props = regionprops(regions + 1)
    centroid = np.array([region_props[ii].centroid for ii in range(len(region_props))])
    knn_mat = radius_neighbors_graph(centroid, radius, mode='connectivity', include_self=False)
    epsilon_graph = nx.from_scipy_sparse_matrix(knn_mat, 'weight')

    return epsilon_graph


def graph_knn(regions, neighbors):
    region_props = regionprops(regions + 1)
    centroid = np.array([region_props[ii].centroid for ii in range(len(region_props))])
    knn_mat = kneighbors_graph(centroid, neighbors, mode='connectivity', include_self=False)
    knn_graph = nx.from_scipy_sparse_matrix(knn_mat, 'weight')

    return knn_graph


def set_node_edge_attr(graph, img, regions):
    region_props = regionprops(regions + 1)

    for n in graph:

        graph.nodes[n].update({'labels': [n],
                            'pixel count': 0,
                            'total color': np.array([0, 0, 0], dtype=np.double),
                            'centroid': np.array([0, 0], dtype=np.double)})

    for index in np.ndindex(regions.shape):
        current = regions[index]
        graph.nodes[current]['pixel count'] += 1
        graph.nodes[current]['total color'] += img[index]

    for n in graph:
        graph.nodes[n]['mean color'] = (graph.nodes[n]['total color'] / graph.nodes[n]['pixel count'])
        graph.nodes[n]['centroid'] = (region_props[n].centroid)

    nx.set_edge_attributes(graph, 0, 'weight')
    # nx.set_edge_attributes(rag, 1, 'similarity')

    return graph


def rag_networkx(regions):
    rag = grph.RAG(regions, connectivity=2)
    print('Number of edges:', rag.number_of_edges())

    return rag


def graph_complete_networkx(regions):
    complete_graph = nx.complete_graph(len(np.unique(regions)))
    print('Number of edges:', complete_graph.number_of_edges())

    return complete_graph


def graph2gradient(img, rag, weights, regions):
    img_grad = np.zeros((img.shape[0], img.shape[1]))
    for i_edge, e in enumerate(list(rag.edges)):
        rag[e[0]][e[1]]['weight'] = weights[i_edge]
        cnt = np.logical_and(dilation(regions == e[0], disk(1)), dilation(regions == e[1], disk(1)))
        img_grad[cnt] = np.maximum(img_grad[cnt], rag[e[0]][e[1]]['weight'])

    return img_grad