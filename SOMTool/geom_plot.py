import math
import random

import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo

pyo.init_notebook_mode(connected=True)


def svg_to_xy(base_x, base_y, svg_string):
    """
    Takes in an x coordinate and a y coordinate as well as an svg path.
    The svg path is drawn with respect to origin (0,0) and then shifted to new origin (base_x,base_y).
    Returns an x array and a y array of equal length containing x and y coordinates respectively.
    This format is for use in the plotly Scatter function.
    """
    svg = svg_string.split(' ')
    x = []
    y = []
    for i in range(len(svg)):
        if svg[i] == 'M' or svg[i] == 'm':
            x.append(None)
            y.append(None)
            x.append(float(svg[i+1]))
            y.append(float(svg[i+2]))
        if svg[i] == 'L':
            x.append(float(svg[i+1]))
            y.append(float(svg[i+2]))
        if svg[i] == 'V':
            x.append(float(x[-1]))
            y.append(float(svg[i+1]))
        if svg[i] == 'H':
            x.append(float(svg[i+1]))
            y.append(float(y[-1]))
        if svg[i] == 'l':
            x.append(x[-1]+float(svg[i+1]))
            y.append(y[-1]+float(svg[i+2]))
        if svg[i] == 'v':
            x.append(x[-1])
            y.append(y[-1]+float(svg[i+1]))
        if svg[i] == 'h':
            x.append(x[-1]+float(svg[i+1]))
            y.append(y[-1])
        if svg[i] == 'Z' or svg[i] == 'z':
            x.append(None)
            y.append(None)

    for i in range(len(x)):
        if type(x[i]) == float:
            x[i] = float(base_x) + x[i]
        if type(y[i]) == float:
            y[i] = -float(base_y) - y[i]

    return x,y


def remove_nones(arr):
    """
    This is a helper method for xy_to_svg. Because the plotly format uses 'None's where svg uses 'M' for move,
    we have to deal with the 'None's specifically.
    """
    res = []
    temp = []
    for val in arr:
        if val!=None:
            temp.append(val)
        else:
            if temp!=[]:
                res.append(temp)
                temp = []
    if temp!=[]:
        res.append(temp)
    return res


def xy_to_svg(x, y):
    """
    Converts from the plotly x coordinate list and y coordinate list to an svg path at origin (0, 0).
    """
    if not len(x)==len(y):
        print(False)
    x = remove_nones(x)
    y = remove_nones(y)
    svg = ''
    for i in range(len(x)):
        svg = svg + 'M'
        svg = svg +str(x[i][0])+' '+str(-y[i][0])
        for j in range(1,len(x[i])):
            svg = svg+' L'+str(x[i][j])+' '+str(-y[i][j])
        svg = svg+' Z'
    return svg


def get_circle(ox, oy, r, div=12):
    """
    Gets the x and y coordinates of a circle centered at point (ox, oy) with radius r.
    The div parameter represents the number of divisions in the circle. 
    We are approximating the circle with a regular polygon with number of sides equal to div.
    """
    angle = 2.0*math.pi/div
    x = []
    y = []
    for i in range(div):
        x.append(ox+r*math.cos(i*angle))
        y.append(oy+r*math.sin(i*angle))
    x.append(x[0])
    y.append(y[0])
    return x, y


def get_line(ox, oy, dx, dy, th, orad, drad, o_off, d_off):
    """    
    Gets the x and y coordinates of a line between origin point (ox, oy) and destination point (dx, dy) with thickness th.
    We offset the line from the origin point by a radius orad and an offset o_off.
    We offset the line from the destination point by a radius drad and an offset d_off.
    """    
    # The line between the origin (ox, oy) and the destination (dx, dy)
    dir = (dx-ox, dy-oy)
    # Normalize the direction
    dir = (dir[0]/math.sqrt(dir[0]**2+dir[1]**2), dir[1]/math.sqrt(dir[0]**2+dir[1]**2))
    # This direction points perpendicular to the line between the origin (ox, oy) and the destination (dx, dy)
    perp = (-dir[1], dir[0])

    # Offset the end points
    o_x = ox+(orad+o_off)*dir[0]
    o_y = oy+(orad+o_off)*dir[1]
    d_x = dx-(drad+d_off)*dir[0]
    d_y = dy-(drad+d_off)*dir[1]

    # Make sure we do not allow offsets to eat into each other
    new_dir = (d_x-o_x, d_y-o_y)
    if new_dir[0]*dir[0]+new_dir[1]*dir[1] < 0:
        return [], []

    # Store points of the rectangle along the line from origin (ox, oy) to dest (dx, dy) with thickness 'th'
    x = [o_x+th*perp[0], d_x+th*perp[0], d_x-th*perp[0], o_x-th*perp[0]]
    y = [o_y+th*perp[1], d_y+th*perp[1], d_y-th*perp[1], o_y-th*perp[1]]

    return x, y


def get_bordered_rect(ox, oy, w, h, th):
    """
    Gets the x and y coordinates of a bordered rectangle whose innermost top left corner is located at (ox, oy)
    with inner width w, inner height h, and border thickness th.
    """
    x = [ox, ox+w, ox+w, ox, ox, ox-th, ox-th, ox+w+th, ox+w+th, ox-th, ox-th, ox]
    y = [oy, oy, oy-h, oy-h, oy, oy, oy-h-th, oy-h-th, oy+th, oy+th, oy, oy]
    return x, y


def get_filled_rect(ox, oy, w, h):
    """
    Gets a filled rectangle with top left corner located at (ox, oy), with width w, and height h.
    """
    x = [ox, ox+w, ox+w, ox, ox]
    y = [oy, oy, oy-h, oy-h, oy]
    return x, y


def get_filled_hexagon(ox, oy, s):
    """
    Gets a filled "points-up" hexagon with top left corner located at (ox, oy), with side length of s.
    """
    x = [ox,ox+s*math.sqrt(3),ox+2*s*math.sqrt(3),ox+2*s*math.sqrt(3),ox+s*math.sqrt(3),ox]
    y = [oy,oy+s,oy,oy-2*s,oy-3*s,oy-2*s,oy]
    return x, y


def get_square_cell(row, col):
    """
    Gets a filled square based on its row, col position in a grid.
    """
    return [col+1,col+1,col,col], [-row-1,-row,-row,-row-1]


def get_hexagonal_cell(row, col):
    """
    Gets a filled "points-up" hexagon based on its row, col position in a grid.
    """
    return list(np.asarray([0,math.sqrt(3),2*math.sqrt(3),2*math.sqrt(3),math.sqrt(3),0])+2*math.sqrt(3)*col+row%2*math.sqrt(3)), list(np.asarray([0,1,0,-2,-3,-2,0])-3*row)


def jitter(x, y, jitter_radius, seed=None):
    """
    Jitter around a point based on a given range. Set the same seed to repeatedly get the same random number.
    """
    random.seed(seed)
    return x+random.uniform(-jitter_radius, jitter_radius), y+random.uniform(-jitter_radius, jitter_radius)


def normed_to_hex(val):
    """
    Convert from a number in [0,1] to a hexadecimal representation of its RGB value (when scaled by 255).
    """
    return '%02x' % int(255*(min(max(0,val),1)))


def triple_normed_to_hex(val1, val2, val3):
    """
    Convert from three numbers in [0,1] to a hexadecimal representation of its RGB values (when scaled by 255).
    """
    return '#'+normed_to_hex(val1)+normed_to_hex(val2)+normed_to_hex(val3)


def plot_fig(fig, showticklabels=False, showgrid=False, width=1250, height=750):
    """
    This is a helper method to plot a given figure.
    """
    # Display fig       
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(
            l=10,
            r=10,
            b=100,
            t=100,
            pad=0
        ),
        paper_bgcolor="LightSteelBlue",
        yaxis = dict(scaleanchor = "x",
                    scaleratio = 1)
    )

    # Update axes properties
    fig.update_xaxes(
        showticklabels=showticklabels,
        showgrid=showgrid,
        zeroline=False,
    )

    fig.update_yaxes(
        showticklabels=showticklabels,
        showgrid=showgrid,
        zeroline=False,
    )

    fig.show()

    return


default_shape = {'x':[], 'y':[],
                 'fill':'toself', 'fillcolor':'blue',
                 'line_shape':'linear', 'line_width':1, 'line_color':'gray',
                 'mode':'lines', 'name':'',
                 'opacity':1}

def plot_shapes(shape_list, default_shape=default_shape, showticklabels=False, showgrid=False, offline=False, width=1250, height=750):
    """
    Plot multiple shapes with assorted properties.
    """
    # Initialize figure
    fig = go.Figure()
    fig.update_layout(showlegend=False)

    # Make sure shapes have all necessary entries
    for shape in shape_list:
        for key in default_shape.keys():
            if key not in shape.keys():
                shape[key] = default_shape[key]

    num_shapes = len(shape_list)
    for i in range(num_shapes):
        fig.add_trace(go.Scatter(x=shape_list[i]['x'], y=shape_list[i]['y'],
                                 fill=shape_list[i]['fill'], fillcolor=shape_list[i]['fillcolor'],
                                 line_shape=shape_list[i]['line_shape'], line_width=shape_list[i]['line_width'], line_color=shape_list[i]['line_color'],
                                 mode=shape_list[i]['mode'], name=shape_list[i]['name'],
                                 opacity=shape_list[i]['opacity']))
    # Plot the figure
    plot_fig(fig, showticklabels, showgrid, width, height)

    return fig


def hermite_interpolate(y0, y1, y2, y3, mu, tension, bias):
    """
    Source: http://paulbourke.net/miscellaneous/interpolation/#:~:text=Linear%20interpolation%20is%20the%20simplest,in%20between%20the%20data%20points.&text=Often%20a%20smoother%20interpolating%20function,smooth%20transition%20between%20adjacent%20segments.
    Hermite Interpolation: Takes four points y0, y1, y2, and y3. We can imagine that these are the vertical 
    values of four points whose x values are 0, 1, 2, and 3 respectively. Imagine a segment of our curve 
    between y1 and y2. mu is a number between 0 and 1 which gives us how far along this curve the point we are 
    computing is. tension gives us the amount of slack in the curve and bias gives us whether the curve bends 
    more towards segment [y0,y1] or segment [y2,y3].
    """
    m0  = (y1-y0)*(1+bias)*(1-tension)/2
    m0 += (y2-y1)*(1-bias)*(1-tension)/2
    m1  = (y2-y1)*(1+bias)*(1-tension)/2
    m1 += (y3-y2)*(1-bias)*(1-tension)/2
    a0 = 2*mu**3 - 3*mu**2 + 1
    a1 = mu**3 - 2*mu**2 + mu
    a2 = mu**3 -   mu**2
    a3 = -2*mu**3 + 3*mu**2
    
    return a0*y1+a1*m0+a2*m1+a3*y2


def interpolate_multiple(points, divs):
    """
    This interpolates between a list of several multi-dimensional points
    according to a certain pattern of divisions.
    If points contains 4 points in a 3 dimensional space, points is a list of lists [x, y, z]
    where x = [x1, x2, x3, x4], y = [y1, y2, y3, y4], and z = [z1, z2, z3, z4].
    divs, in this case, will contain 5 numbers. If points contains n points,
    divs will contain n+1 divisions. This includes the n-1 divisions between the n points
    and two divisions on either end. If we want no interpolation in a segment, we put -1.
    For simplicity we leave the tension and bias of the Hermite interpolation at 0.
    """
    # Make a copy of the points list which will be locally modified
    points = points.copy()
    # For each dimension, add two synthetic points to the beginning and the end of the list.
    # These will simply continue the line of the first and last segments.
    for i in range(len(points)):
        points[i].insert(0,2*points[i][0]-points[i][1])
        points[i].insert(0,2*points[i][0]-points[i][1])
        points[i].append(2*points[i][-1]-points[i][-2])
        points[i].append(2*points[i][-1]-points[i][-2])
    
    
    newpoints = []
    # For each dimension
    for i in range(len(points)):
        newpoint_i = []
        # For each segment
        for j in range(len(divs)):
            y0 = points[i][j]
            y1 = points[i][j+1]
            y2 = points[i][j+2]
            y3 = points[i][j+3]
            # Add an interpolated point as specified
            for k in range(divs[j]+1):
                mu = k/divs[j]
                newpoint_i.append(hermite_interpolate(y0, y1, y2, y3, mu, 0, 0))
        newpoints.append(newpoint_i)
    
    # newpoints is of the same form as the points input 
    # but containing all the interpolated points
    return newpoints


def rgb_to_hex(colors_rgb):
    return ['#'+'0'*(2-len(hex(min(255,max(0,int(colors_rgb[0][i]))))[2:]))+hex(min(255,max(0,int(colors_rgb[0][i]))))[2:]+'0'*(2-len(hex(min(255,max(0,int(colors_rgb[1][i]))))[2:]))+hex(min(255,max(0,int(colors_rgb[1][i]))))[2:]+'0'*(2-len(hex(min(255,max(0,int(colors_rgb[2][i]))))[2:]))+hex(min(255,max(0,int(colors_rgb[2][i]))))[2:] for i in range(len(colors_rgb[0]))]


def plot_colormap(colormap):
    """
    Allows you to visualize a colormap.
    """
    shape_list = []
    for i in range(len(colormap)):
        x, y = get_filled_rect(0, i, len(colormap), 1)
        shape_list.append({'x':x, 'y':y, 'fillcolor':colormap[i], 'line_width':0})

    return plot_shapes(shape_list, width=950, height=650)
    


