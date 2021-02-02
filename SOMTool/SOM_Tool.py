import numpy as np

import math
from itertools import product
from collections import defaultdict


import geom_plot


def minmax_normalize(datum, data):
    """
    Minmax normalizes a datum based on a transposed 2d numpy dataset.
    This works for a single datum, as well as multiple datums where each variable
    contains a vector of values. If datum and data are the same, this returns the normalized dataset.
    """
    normed_datum = []
    for i in range(datum.shape[0]):
        normed_datum.append((datum[i]-np.min(data[i]))/(np.max(data[i])-np.min(data[i])))
    return np.array(normed_datum)


def minmax_unnormalize(datum, data):
    """
    Minmax unnormalizes a datum based on a transposed 2d numpy dataset.
    """
    unnormed_datum = []
    for i in range(datum.shape[0]):
        unnormed_datum.append(datum[i]*(np.max(data[i])-np.min(data[i]))+np.min(data[i]))
    return np.array(unnormed_datum)


def center_on_mean(datum, data):
    """
    Centers at the mean a datum based on a transposed 2d numpy dataset.
    This works for a single datum, as well as multiple datums where each variable
    contains a vector of values. If datum and data are the same, this returns the normalized dataset.
    """
    normed_datum = []
    for i in range(datum.shape[0]):
        normed_datum.append(datum[i]-np.mean(data[i]))
    return np.array(normed_datum)


def uncenter_on_mean(datum, data):
    """
    Uncenters at the mean a datum based on a transposed 2d numpy dataset.
    """
    unnormed_datum = []
    for i in range(datum.shape[0]):
        unnormed_datum.append(datum[i]+np.mean(data[i]))
    return np.array(unnormed_datum)


def standardize(datum, data):
    """
    Z-Score normalizes a datum based on a transposed 2d numpy dataset.
    This works for a single datum, as well as multiple datums where each variable
    contains a vector of values. If datum and data are the same, this returns the normalized dataset.
    """
    normed_datum = []
    for i in range(datum.shape[0]):
        normed_datum.append((datum[i]-np.mean(data[i]))/np.std(data[i]))
    return np.array(normed_datum)


def unstandardize(datum, data):
    """
    Z-score unnormalizes a datum based on a transposed 2d numpy dataset.
    """
    unnormed_datum = []
    for i in range(datum.shape[0]):
        unnormed_datum.append(datum[i]*np.std(data[i])+np.mean(data[i]))
    return np.array(unnormed_datum)


def outlier_standardize(datum, data, s=2):
    """
    Scale to [0,1] so that the upper value is some upper limit or s standard deviations from the mean 
    (whichever is lower) and the lower value is some lower limit or s standard deviations from the mean 
    (whichever is higher).
    """
    normed_datum = []
    for i in range(datum.shape[0]):
        upper_val = min((np.mean(data[i])+s*np.std(data[i])), np.max(data[i]))
        lower_val = max((np.mean(data[i])-s*np.std(data[i])), np.min(data[i]))
        normed_datum.append((datum[i]-lower_val)/(upper_val-lower_val))
    return np.array(normed_datum)


def outlier_unstandardize(datum, data, s=2):
    """
    Reverse the outlier standardize function.
    """
    normed_datum = []
    for i in range(datum.shape[0]):
        upper_val = min((np.mean(data[i])+s*np.std(data[i])),np.max(data[i]))
        lower_val = max((np.mean(data[i])-s*np.std(data[i])),np.min(data[i]))
        normed_datum.append(datum[i]*(upper_val-lower_val)+lower_val)
    return np.array(normed_datum)


def PCA_init(data, m, n, s=1):
    """
    Return an m by n grid along the first two principal components of the data with a width of s standard deviations
    """
    cov = np.zeros((data.shape[0],data.shape[0]))
    for i,j in product(range(data.shape[0]),range(data.shape[0])):
        cov[i,j] = np.dot(data[i,:]-np.mean(data[i,:]),data[j,:]-np.mean(data[j,:]))/data.shape[1]
    eigenValues, eigenVectors = np.linalg.eig(cov)
    idx = np.flip(eigenValues.argsort())
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    grid = []
    
    for i in range(m):
        if m>1:
            m_val = (2*i/(m-1)-1)*s*np.sqrt(eigenValues[int(not m>n)])*eigenVectors[:,int(not m>n)]
        else:
            m_val = np.zeros(data.shape[0])
        grid_row = []
        for j in range(n):
            if n>1:
                n_val = (2*j/(n-1)-1)*s*np.sqrt(eigenValues[int(m>n)])*eigenVectors[:,int(m>n)]
            else:
                n_val = np.zeros(data.shape[0])
            grid_row.append(m_val+n_val)
        grid.append(grid_row)
    return np.real(np.array(grid))


def find_bmu(t, net):
    """
        Find the best matching unit for a given vector, t, in the SOM
        Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
                 and bmu_idx is the index of this vector in the SOM
    """
    t = t.reshape(net.shape[2], 1)
    bmu_idx = np.array([0, 0])
    # set the initial minimum distance to a huge number (the hugest possible int)
    min_dist = np.iinfo(np.int).max
    # calculate the high-dimensional distance between each neuron and the input
    for row in range(net.shape[0]):
        for col in range(net.shape[1]):
            w = net[row, col, :].reshape(net.shape[2], 1)
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sum((w - t) ** 2)
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([row, col])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(net.shape[2], 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)


def decay_radius(initial_radius, i, time_constant):
    """
    Decay the neighborhood radius over time. 
    """
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    """
    Decay the learning rate over time so that the algorithm takes progressively smaller steps.
    """
    return initial_learning_rate * np.exp(-i / n_iterations)


def calculate_influence(distance, radius):
    """
    Calculates the influence of a neighboring cell update at a given distance given the current radius.
    """
    return np.exp(-distance / (2*(radius**2)))


def hex_idx_to_pos(row, col):
    """
    Source: https://www.redblobgames.com/grids/hexagons/
    Because our hexagons are built from the top left corner,
    we need to do some math to get the center of the hexagon. 
    Here index (idx) refers to (row, col) and position (pos)
    refers to the center of the hexagon.
    """
    x = math.sqrt(3)+2*math.sqrt(3)*col+(row&1)*math.sqrt(3)
    y = -3*row-1
    return x, y

def idx_to_pos(row, col):
    """
    Because our squares are built from the top left corner,
    we need to do some math to get the center of the square. 
    Here index (idx) refers to (row, col) and position (pos)
    refers to the center of the square.
    """
    x = col+0.5
    y = -row-0.5
    return x, y


def get_neighbors(row, col, net, hex_grid=False):
    """
    Get the set of vectors stored in the neighboring cells to the cell at (row, col).
    """
    neighbors = set()
    
    if hex_grid == False:
        ncoords = [(row-1,col-1), (row-1, col), (row-1, col+1), (row, col-1), (row, col+1), (row+1,col-1), (row+1, col), (row+1, col+1)]
    else:
        if row%2==0:
            ncoords = [(row-1,col-1), (row-1, col), (row, col-1), (row, col+1), (row+1,col-1), (row+1, col)]
        else:
            ncoords = [(row-1,col), (row-1, col+1), (row, col-1), (row, col+1), (row+1,col), (row+1, col+1)]
        
    for nrow, ncol in ncoords:
        if nrow in range(net.shape[0]) and ncol in range(net.shape[1]):
            neighbors.add(tuple(net[nrow][ncol]))

    return neighbors


def get_unified_distance(row, col, net, hex_grid=False):
    """
    Get the average distance from the cell at (row, col) to its neighbors.
    """
    neighbors = get_neighbors(row, col, net, hex_grid=hex_grid)
    dist = 0
    w = net[row, col, :].reshape(net.shape[2], 1)
    for neigh in neighbors:
        t = np.array(neigh).reshape(net.shape[2], 1)
        dist+=math.sqrt(np.sum((w - t) ** 2))
    return dist / len(neighbors)


def normed_u_dist(net, hex_grid=False):
    """
    Normalize the unified (aka averaged) distances of each cell to a number in [0,1].
    """
    normed_u_dist = []
    for row in range(net.shape[0]):
        u_row = []
        for col in range(net.shape[1]):
            u_row.append(get_unified_distance(row, col, net, hex_grid=hex_grid))
        normed_u_dist.append(u_row)
    normed_u_dist = np.array(normed_u_dist)
    max_dist = max(normed_u_dist.reshape(normed_u_dist.shape[0]*normed_u_dist.shape[1]))
    min_dist = min(normed_u_dist.reshape(normed_u_dist.shape[0]*normed_u_dist.shape[1]))
    normed_u_dist = (normed_u_dist-min_dist) / (max_dist-min_dist)
    return normed_u_dist


def sum_of_squares_distance(data, net):
    m = data.shape[0]
    n = data.shape[1]
    sum_of_squares_dist = 0
    for i in range(n):
        t = data[:, i].reshape(np.array([m, 1]))
        
        # find its Best Matching Unit
        bmu, bmu_idx = find_bmu(t, net)
        
        sum_of_squares_dist += np.sum(t - bmu) ** 2
    return sum_of_squares_dist


# http://blog.yhat.com/posts/self-organizing-maps-2.html
def run_SOM(data, net, iter_ratio=10, init_learning_rate = 0.01, hex_grid=False, sum_of_squares=False, sos_measurements=10):
    """
    Run the SOM algorithm based on the given data and an initial network grid and other parameters.
    """
    m = data.shape[0]
    n = data.shape[1]
    sum_of_squares_idx = []
    sum_of_squares_vals = []
    n_iterations = iter_ratio*n
    sos_spacing = math.floor(n_iterations/sos_measurements)
    # initial neighbourhood radius, half of the wider dimension of the grid (meaning that the initial neighborhood size is on the order of the entire grid)
    if hex_grid == False:
        init_radius = max(np.abs(idx_to_pos(net.shape[0], abs(net.shape[1])))) / 2
    else:
        init_radius = max(np.abs(hex_idx_to_pos(net.shape[0], abs(net.shape[1])))) / 2
    # radius decay parameter
    time_constant = n_iterations / np.log(init_radius)

    for i in range(n_iterations):
        if sum_of_squares==True:
            if i%sos_spacing==0:
                sum_of_squares_idx.append(i)
                sum_of_squares_vals.append(sum_of_squares_distance(data, net))
        
        # select a training example at random
        t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))

        # find its Best Matching Unit
        bmu, bmu_idx = find_bmu(t, net)

        # decay the SOM parameters
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, n_iterations)

        # now we know the BMU, update its weight vector to move closer to input
        # and move its neighbours in 2-D space closer
        # by a factor proportional to their 2-D distance from the BMU
        for x in range(net.shape[0]):
            for y in range(net.shape[1]):
                w = net[x, y, :].reshape(m, 1)
                # get the 2-D distance (again, not the actual Euclidean distance)
                if hex_grid == False:
                    w_dist = np.sum((np.array(idx_to_pos(x, y)) - np.array(idx_to_pos(bmu_idx[0], bmu_idx[1]))) ** 2)
                else:
                    w_dist = np.sum((np.array(hex_idx_to_pos(x, y)) - np.array(hex_idx_to_pos(bmu_idx[0], bmu_idx[1]))) ** 2)
                # if the distance is within the current neighbourhood radius
                if w_dist <= r**2:
                    # calculate the degree of influence (based on the 2-D distance)
                    influence = calculate_influence(w_dist, r)
                    # now update the neuron's weight using the formula:
                    # new w = old w + (learning rate * influence * delta)
                    # where delta = input vector (t) - old w
                    new_w = w + (l * influence * (t - w))
                    # commit the new weight
                    net[x, y, :] = new_w.reshape(1, m)
    if sum_of_squares==False:
        return net
    else:
        return net, sum_of_squares_idx, sum_of_squares_vals


default_colormap = [geom_plot.triple_normed_to_hex(i/256, i/256, i/256) for i in range(256)]

def get_cell_shapes(net, data, udist=True, dim_index=-1, colormap=default_colormap, out_stand=False, hex_grid=False, as_circles=False, s=2):
    """
    Get the cell shapes from a grid network.
    """
    
    rows = net.shape[0]
    cols = net.shape[1]
    
    if udist==True:
        u_dist = normed_u_dist(net, hex_grid=hex_grid)

    cell_shapes = []
    for row in range(rows):
        for col in range(cols):
            if hex_grid == False:
                if as_circles == False:
                    x,y = geom_plot.get_square_cell(row, col)
                else:
                    ox,oy = idx_to_pos(row, col)
                    x,y = geom_plot.get_circle(ox, oy, r=0.5, div=12)
            else:
                if as_circles == False:
                    x,y = geom_plot.get_hexagonal_cell(row, col)
                else:
                    ox,oy = hex_idx_to_pos(row, col)
                    x,y = geom_plot.get_circle(ox, oy, r=math.sqrt(3), div=12)
            if udist==True:
                color = get_colormap_color(1-u_dist[row][col], colormap)
                name = '('+str(row)+','+str(col)+')'
            else:
                if out_stand == True:
                    color = get_colormap_color(outlier_standardize(net[row][col], data, s=s)[dim_index], colormap)
                else:
                    color = get_colormap_color(net[row][col][dim_index], colormap)
                name = '('+str(row)+','+str(col)+')'+': '+str(round(net[row][col][dim_index],2))
            if as_circles == False:
                cell_shapes.append({'x':x, 'y':y, 'fillcolor':color, 'name':name})
            else:
                cell_shapes.append({'x':x, 'y':y, 'fillcolor':color, 'name':name, 'line_shape':'spline'})
    return cell_shapes


def get_node_shapes(net, data, node_data, radius_scale=0.1, node_names=[], node_colors=[], color_by_value=False, dim_index=-1, colormap=default_colormap, out_stand=False, hex_grid=False, s=2):
    """
    From a list of datapoints in the high-dimensional space, plot them as circles
    on their closest corresponding grid cell.
    """
        
    # Transpose the data back to the more conventional orientation with rows representing data points
    node_data = node_data.transpose()
    if len(node_names)!=node_data.shape[0]:
        node_names = list(range(node_data.shape[0]))
    
    rows = net.shape[0]
    cols = net.shape[1]
    
    node_shapes = []
    for i in range(len(node_data)):
        bmu, bmu_idx = find_bmu(node_data[i], net)
        if hex_grid == False:
            tile_rad = 0.5
            radius = radius_scale*tile_rad
            ox,oy = idx_to_pos(bmu_idx[0], bmu_idx[1])
        else:
            tile_rad = math.sqrt(3)
            radius = radius_scale*tile_rad
            ox,oy = hex_idx_to_pos(bmu_idx[0], bmu_idx[1])
        ox,oy = geom_plot.jitter(ox,oy,.5*tile_rad,seed=rows*cols*i+bmu_idx[0]*rows+bmu_idx[1])
        x,y = geom_plot.get_circle(ox, oy, radius)
        if dim_index == -1:
            name = node_names[i]
        else:
            name = node_names[i]+': '+str(node_data[i][dim_index])
        if color_by_value==False:
            if len(node_colors)!=node_data.shape[0]:
                color = 'red'
            else:
                color = node_colors[i]
        else:
            if out_stand == True:
                color = get_colormap_color(outlier_standardize(node_data[i], data, s=2)[dim_index], colormap)
            else:
                color = get_colormap_color(node_data[i][dim_index], colormap)
        node_shapes.append({'x':x, 'y':y, 'fillcolor':color, 'line_shape':'spline',
                                        'line_width':0.5, 'line_color':'white', 'name':name})
    return node_shapes


def get_node_cells(net, data):
    """
    From a list of datapoints in the high-dimensional space,
    find their closest corresponding grid cell.
    """
    
    # Transpose the data back to the more conventional orientation with rows representing data points
    data = data.transpose()
    
    rows = net.shape[0]
    cols = net.shape[1]
    
    node_cells = defaultdict(list)
    for i in range(len(data)):
        bmu, bmu_idx = find_bmu(data[i], net)
        node_cells[tuple(bmu_idx)].append(data[i])
    
    return node_cells


def get_colormap_color(fraction, colormap):
    """
    Convert from a point on the interval [0,1] to a color from a colormap.
    """
    fraction = min(max(fraction, 0), (len(colormap)-1)/len(colormap))
    return colormap[math.floor(fraction*len(colormap))]




