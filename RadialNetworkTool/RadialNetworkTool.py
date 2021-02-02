import math
from math import nan
import random
from collections import defaultdict


import geom_plot


# Colors to use
COLORS = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']
COLORS_LTD = ['blue', 'purple', 'red', 'green', 'orange']

def default_vertex():
    return {'x':None, 'y':None, 'radius':None, 'color':None, 'level':None}

def default_arc():
    return {'weight':None, 'color':None}

class Network:
    def __init__(self, vertices=defaultdict(default_vertex), arcs=defaultdict(lambda: defaultdict(default_arc)),
                 random=False, num_roots=12, numv=261, line_weight=0.1, minr=.5, maxr=.5, levels=False):
        
        # Import vertices and arcs into the correct format
        self.vertices = defaultdict(default_vertex)
        for vertex in vertices.keys():
            for key in vertices[vertex].keys():
                if key in default_vertex().keys():
                    self.vertices[vertex][key] = vertices[vertex][key]
        
        self.arcs = defaultdict(lambda: defaultdict(default_arc))
        for v1 in arcs.keys():
            if v1 in self.vertices.keys():
                for v2 in arcs[v1].keys():
                    if v2 in self.vertices.keys():
                        for key in arcs[v1][v2].keys():
                            if key in default_arc().keys():
                                self.arcs[v1][v2][key] = arcs[v1][v2][key]
        
        if random==True:
            self.generate_random_network(num_roots, numv, line_weight, minr, maxr)
                    
        # Store inverse arcs for convenience
        self.inverse_arcs = defaultdict(lambda: defaultdict(default_arc))
        for v1 in self.arcs.keys():
            for v2 in self.arcs[v1].keys():
                self.inverse_arcs[v2][v1] = self.arcs[v1][v2]
                    
        # Defines roots as vertices with no incoming arcs
        self.roots = self.get_roots()
        # Defines leaves as vertices with no outgoing arcs
        self.leaves = self.get_leaves()
        # Checks if network is forest
        self.is_forest = self.is_forest()
        # Get leaf counts (only relevant if network is a forest)
        self.leaf_counts = self.get_leaf_counts()
        # Vertices to generation. Root is 1, leaf is tree height.
        self.v2g = self.get_generations()              
        # Update level according to generation if no levels are given in the input
        if levels==False:
            # v2g will only contain None values if the network is not a tree
            if None not in set(self.v2g.values()):
                for vertex in self.vertices.keys():
                    self.vertices[vertex]['level'] = self.v2g[vertex]
        # Generation to vertices.
        self.g2v = defaultdict(set)
        for vertex in self.v2g.keys():
            self.g2v[self.v2g[vertex]].add(vertex)
        
        if levels==True:
            if not self.is_forest:
                print("ERROR: Cannot work with levels if network is not a forest.")
            else:
                for v1 in self.arcs.keys():
                    for v2 in self.arcs[v1].keys():
                        if self.vertices[v1]['level'] >= self.vertices[v2]['level']:
                            print("ERROR: All nodes must be at a higher level than their parents.")
                            print("Parent vertex '"+str(v1)+" has level "+str(self.vertices[v1]['level'])+
                                  " and child vertex '"+str(v2)+" has level "+str(self.vertices[v1]['level'])+".")

    # Here we generate random tree networks for testing purposes
    # Outputs vertices, arcs, and roots data structures
    # num_roots: the number of roots in our network
    # numv: the total number of vertices in our network
    # line_weight: a constant line_weight shared by all our edges
    # minr: the minimum radius for a vertex
    # maxr: the maximum radius for a vertex
    def generate_random_network(self, num_roots, numv, line_weight, minr, maxr):
        # Clear existing
        self.vertices = defaultdict(default_vertex)
        self.arcs = defaultdict(lambda: defaultdict(default_arc))
        
        # Add num_roots root vertices
        for i in range(num_roots):
            self.vertices[i]['radius'] = random.uniform(minr, maxr)
            self.vertices[i]['color'] = COLORS_LTD[math.floor(len(COLORS_LTD)*random.random())]
        
        # While there are less vertices than we want
        while len(self.vertices.keys()) < numv:
            # Add a new vertex as a leaf somewhere
            self.arcs[math.floor(len(self.vertices.keys())*random.random())][len(self.vertices.keys())]['weight'] = line_weight
            # Add the new vertex to the vertices
            self.vertices[len(self.vertices.keys())]['radius'] = random.uniform(minr, maxr)
        return
    
    # Compute the radius of the largest vertex which is actually present in the arcs data structure
    def get_max_radius(self):
        rmax = 0
        for vertex in self.vertices.keys():
            if self.vertices[vertex]['radius'] > rmax:
                rmax = self.vertices[vertex]['radius']
        return rmax

    # Compute the maximum arc weight
    def get_max_weight(self):
        wmax = 0
        for v1 in self.arcs.keys():
            for v2 in self.arcs[v1].keys():
                if self.arcs[v1][v2]['weight'] > wmax:
                    wmax = self.arcs[v1][v2]['weight']
        return wmax
    
    # Defines roots as vertices with no incoming arcs.
    def get_roots(self):
        roots = list(self.vertices.keys())
        for v1 in self.arcs.keys():
            for v2 in self.arcs[v1].keys():
                if v2 in roots:
                    roots.remove(v2)
        return set(roots)
    
    # Defines leaves as vertices with no outgoing arcs.
    # A root is a leaf if it is unconnected to any other node.
    def get_leaves(self):
        leaves = list(self.vertices.keys())
        for vertex in self.arcs.keys():
            if vertex in leaves:
                leaves.remove(vertex)
        return set(leaves)
    
    # Check if this network is a forest.
    def is_forest(self):
        # If there are no roots, it is not a forest.
        if self.roots == []:
            return False
        
        # If any vertex has two or more parents, it is not a forest.
        for vertex in self.inverse_arcs.keys():
            if len(list(self.inverse_arcs[vertex].keys())) > 1:
                return False
        
        # If any vertex has itself as a descendent, it is not a forest.
        visited = set()
        visiting = [root for root in self.roots]
        num_vertices = len(list(self.vertices.keys()))
        while len(visited) < num_vertices and visiting != []:
            vertex = visiting.pop(0)
            if vertex in visited:
                return False
            visited.add(vertex)
            if vertex in self.arcs.keys():
                children = list(self.arcs[vertex].keys())
                visiting.extend(children)
        
        # Otherwise, it's a forest.
        return True

    # Leaf counts will be used to evenly space radial nodes.
    def get_leaf_counts(self):
        if not self.is_forest:
            return {vertex:None for vertex in self.vertices.keys()}
        else:
            counts = {}
            for vertex in self.roots:
                self.count_leaves(vertex, counts)
            return counts

    # Recursive helper method
    def count_leaves(self, vertex, counts):
        # If the vertex has no outgoing arcs, it is a leaf and has a leaf count of 1.
        if vertex not in self.arcs.keys():
            counts[vertex] = 1
            return 1
        # Otherwise, it is not a leaf and a has a leaf count which is the sum of the leaf counts of its children.
        else:
            leaves = 0
            for child in self.arcs[vertex].keys():
                if child in counts.keys():
                    leaves += counts[child]
                else:
                    num_children = len(list(self.arcs[vertex].keys()))
                    leaves += self.count_leaves(child, counts)
            counts[vertex] = leaves
            return leaves
    
    # Get generations for each node
    def get_generations(self):
        if not self.is_forest:
            return {vertex:None for vertex in self.vertices.keys()}
        else:
            generations = {vertex:1 for vertex in self.roots}
            for root in self.roots:
                self.get_generation(root, generations)
                
            return generations
    
    # Helper method for get_generations()
    def get_generation(self, vertex, generations):
        if vertex in self.leaves:
            return
        else:
            for child in self.arcs[vertex].keys():
                generations[child] = generations[vertex]+1
                self.get_generation(child, generations)
            return
    
    # Assumes a tree structure of the graph
    # Updates colors so that trees have same color as their roots
    def color_components(self):
        if not self.is_forest:
            print('ERROR: Network not a forest.')
            return
        else:
            for root in self.roots:
                self.color_descendents(root)
            for v1 in self.arcs.keys():
                for v2 in self.arcs[v1].keys():
                    self.arcs[v1][v2]['color'] = self.vertices[v1]['color']
            return
    
    # Helper method for color_components()
    def color_descendents(self, vertex):
        for child in self.arcs[vertex].keys():
            if 'color' in self.vertices[vertex].keys() and self.vertices[vertex]['color']!=None:
                self.vertices[child]['color'] = self.vertices[vertex]['color']
            else:
                # Give a default color
                self.vertices[vertex]['color'] = 'black'
                self.vertices[child]['color'] = 'black' 
            self.color_descendents(child)
        return
    
    # ring_width: the ratio of the ring width to the maximum vertex radius
    # spread: the overall angle of the visualization (add this to the start_angle counterclockwise to get the end angle)
    # start_angle: 0 is at 3 oclock, PI/2 (90 degrees) is at 12 oclock, PI (180 degrees) is at 9 oclock, etc.
    # vlog: if vlog is True we plot the vertex area logarithmically
    # vlogval: the base of the v logarithm
    def radialize(self, ring_width=20, spread=math.pi, start_angle=0, vlog=False, vlogval=math.e, max_radius=nan):
        if not self.is_forest:
            print('ERROR: Network not a forest.')
            return
        else:
            self.color_components()
            
            if math.isnan(max_radius):
                max_radius = self.get_max_radius()

            if vlog:
                # Reconstitute the value from the radius. Remember, the value is stored in the area.
                rec_val = math.pi*(max_radius**2)
                # The new value will be the logarithm of the reconstituted value.
                new_val = math.log(rec_val+1, vlogval)
                # Get the radius from the area.
                max_radius = math.sqrt(new_val/math.pi)

            # Construct a sorted list of the levels
            level_list = sorted(list(set(filter(None, [self.vertices[vertex]['level'] for vertex in self.vertices.keys()]))))
            
            # Initialize vertex_range
            vertex_range = {}
            total_leaf_count = sum([self.leaf_counts[vertex] for vertex in self.roots])
            edge = start_angle
            increment = spread/total_leaf_count
            # Sort the initial roots by color, just for fun
            for vcolor, vertex in sorted([(self.vertices[vertex]['color'], vertex) for vertex in self.roots]):
                vertex_range[vertex] = (edge, edge+self.leaf_counts[vertex]*increment)
                edge = vertex_range[vertex][1]

            # Do while loop for each level
            index=0
            level_num = level_list[index]
            while True:
                # Calculate the x and y for the current layer
                curr_level = [vertex for vertex in self.vertices.keys() if self.vertices[vertex]['level'] == level_list[index]]
                for vertex in curr_level:
                    # Place it in the middle of its range
                    angle = (vertex_range[vertex][0]+vertex_range[vertex][1])/2
                    # Compute distance from center
                    level_radius = level_num*(ring_width*max_radius)
                    # Convert this to x and y coordinates
                    x = level_radius*math.cos(angle)
                    y = level_radius*math.sin(angle)
                    self.vertices[vertex]['x'] = x
                    self.vertices[vertex]['y'] = y

                # Go to the next level and check the condition
                index = index+1
                
                # If the index is out of range, we have gone through all the levels and thus all the nodes
                if index not in range(len(level_list)):
                    break
                
                # Update level_num
                level_num = level_list[index]
                
                # Update vertex_range for children
                for parent in curr_level:
                    if parent in self.arcs.keys():
                        edge = vertex_range[parent][0]
                        increment = (vertex_range[parent][1]-vertex_range[parent][0])/self.leaf_counts[parent]
                        for child in self.arcs[parent].keys():
                            vertex_range[child] = (edge, edge+self.leaf_counts[child]*increment)
                            edge = vertex_range[child][1]
                        
        return
    
    # divs is the divisions in each plotted circle
    # offset is the offset of arc lines from vertex circles as a fraction of the maximum vertex radius
    # va_ratio is the the ratio of vertex radius to arc thickness
    # vlog: if vlog is True we plot the vertex area logarithmically
    # vlogval: the base of the v logarithm
    # alog: if alog is True we plot the arcs thicknesses logarithmically
    # alogval: the base of the a logarithm
    # opacity: the opacity of the graph

    def plot(self, divs=12, offset=0.5, va_ratio=5, vlog=False, vlogval=math.e, alog=False, alogval=math.e, opacity=1, max_radius=nan, max_weight=nan, plot_height=750, plot_width=1250):
        # Shapes to plot
        shape_list = []
        
        # Used to compute relative sizes of offsets and line thickness
        if math.isnan(max_radius):
            max_radius = self.get_max_radius()
        if math.isnan(max_weight):
            max_weight = self.get_max_weight()
        
        if vlog:
            # Recovered value. Remember, the value is stored in the area, not the radius.
            rec_val = math.pi*(max_radius**2)
            new_val = math.log(rec_val+1, vlogval)
            max_radius = math.sqrt(new_val/math.pi)
        if alog:
            max_weight = math.log(max_weight+1, alogval)
        
        # Get node circles
        for vertex_id in self.vertices.keys():
            ox = self.vertices[vertex_id]['x']
            oy = self.vertices[vertex_id]['y']
            r = self.vertices[vertex_id]['radius']
            if vlog:
                # Reconstitute the original value. We use the area to store the value
                # since areas give a better sense of size than radii.
                rec_val = math.pi*(r**2)
                # Get the new value as the logarithm of the reconstituted value.
                # Add 1 to avoid taking the log of zero.
                new_val = math.log(rec_val+1, vlogval)
                # Get the radius from the new value which is the new area.
                r = math.sqrt(new_val/math.pi)
            x, y = geom_plot.get_circle(ox, oy, r, divs)
            
            shape_list.append({'x':x, 'y':y, 'fillcolor':self.vertices[vertex_id]['color'], 'line_shape':'spline',
                                    'line_width':0, 'opacity':opacity, 'name':vertex_id})
            
        # Get arc lines
        for o_id in self.arcs.keys():
            for d_id in self.arcs[o_id].keys():
                ox = self.vertices[o_id]['x']
                oy = self.vertices[o_id]['y']
                dx = self.vertices[d_id]['x']
                dy = self.vertices[d_id]['y']
                if not alog:
                    weight = self.arcs[o_id][d_id]['weight']
                else:
                    weight = math.log(self.arcs[o_id][d_id]['weight'], alogval)
                th = (max_radius/va_ratio)*(weight/max_weight)
                orad = self.vertices[o_id]['radius'] 
                if vlog:
                    rec_val = math.pi*(orad**2)
                    new_val = math.log(rec_val+1, vlogval)
                    orad = math.sqrt(new_val/math.pi)
                drad = self.vertices[d_id]['radius'] 
                if vlog:
                    rec_val = math.pi*(drad**2)
                    new_val = math.log(rec_val+1, vlogval)
                    drad = math.sqrt(new_val/math.pi)
                o_off = offset*max_radius
                d_off = offset*max_radius

                x, y = geom_plot.get_line(ox, oy, dx, dy, th, orad, drad, o_off, d_off)
                shape_list.append({'x':x, 'y':y,'fillcolor':self.arcs[o_id][d_id]['color'], 'line_width':0,
                                   'opacity':opacity, 'name':str(o_id)+'__'+str(d_id)})

        # Display the plot        
        geom_plot.plot_shapes(shape_list, height=plot_height, width=plot_width)
