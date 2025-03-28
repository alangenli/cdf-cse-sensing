"""
@author: alan-gen.li
"""

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update({'font.family':'Franklin Gothic Medium'})

plt.close('all')
##############################################################################
"""
FUNCTIONS
"""
##############################################################################
def set_xy_lims(x_lims=0, y_lims=0, file_name=0):
    ax = plt.gca()
    if x_lims!=0:
        ax.set_xlim(x_lims)
    if y_lims!=0:
        ax.set_ylim(y_lims)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')  

def add_right_yaxis(x, y, ylabel, colour='r', file_name=0):
    """
    FUNCTION
    
    to add a secondary right-side yaxis to a SINGLE plot
    """
    ax1 = plt.gca()
    ax1colour = ax1.get_lines()[-1].get_c()
    ax1.tick_params(axis='y',  labelcolor=ax1colour)
    ax1.spines['left'].set(color=ax1colour)
    
    ax2 = ax1.twinx()
    ax2.plot(x, y, alpha=.75, color=colour)
    ax2.set_ylabel(ylabel)
    ax2.tick_params(axis='y', labelcolor=colour)
    ax2.spines['right'].set(color=colour)
    ax2.spines[['left','top']].set_visible(False)
    plt.tight_layout()
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')    

def add_scatter(x, y, colour='r', file_name=0):
    ax = plt.gca()
    ax.plot(x, y, color=colour, linestyle='', marker='o', markersize=5, fillstyle='none')
    
    
def invert_xaxis(sel_ax=0, file_name=0):
    """
    FUNCTION
    
    to invert x axis
    
    inputs
    -------
    sel_ax, index to select axes to invert
        DEAFULT 0
        = 'all' to invert all axes
    """
    if sel_ax=='all':
        for i in range(len(plt.gcf().get_axes())):
            plt.gcf().get_axes()[i].xaxis.set_inverted(True)
    else:
        plt.gcf().get_axes()[sel_ax].xaxis.set_inverted(True)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')    


def set_xy_labels_N(N, fig, axs, x_lab, y_lab, sel_ax_x_label='sup', sel_ax_y_label=0):
    """
    FUNCTION
    
    to set x and y labels in multi-plot
    varying or shared x or y labels
    
    inputs
    -------
    N, number of plots
    fig, axs, the figure and axes objects
    x_lab, y_lab, the x and y labels (list or string)
    sel_ax_y_label=0
        axes to set shared y label
    """
    #X LABEL
    if len(x_lab)==N:
        #VARIED X LABEL
        for i in range(N): 
            axs[i].set_xlabel(x_lab[i])
    else:
        if sel_ax_x_label=='sup':
            #SHARED X LABEL
            fig.supxlabel(x_lab)
        else:
            #SHARED Y LABEL
            axs[sel_ax_x_label].set_xlabel(x_lab)
    #Y LABEL
    if len(y_lab)==N:
        #VARIED Y LABEL
        for i in range(N): 
            axs[i].set_ylabel(y_lab[i])
    else:
        if sel_ax_y_label=='sup':
            fig.supylabel(y_lab)
        else:
            #SHARED Y LABEL
            axs[sel_ax_y_label].set_ylabel(y_lab)




def set_titles_N(N, fig, axs, titles):
    """
    FUNCTION
    
    to set titles in multi-plot
    
    unique title per plot
    overall figure title
    
    """
    if len(titles)>=N:
        for i in range(N):
            axs[i].set_title(titles[i])
    if len(titles) % N==1:
        fig.suptitle(titles[-1])
        


def ex_labels(lab_codes, titles, key):
    """
    FUNCTION
    to find string(s) in list containing the key
    """
    lab = [i for i in lab_codes if key in i]
    titles = [titles[lab_codes.index(i)] for i in lab]
    num = len(lab)
    return lab, titles, num

def calc_arrow_coord(x, y, direction):
    """
    FUNCTION
    to calculate arrow coordinates on a line
    arrow is placed offset from the middle
    #0 = right
    #1 = left
    """
    ind_increment = round(.05*len(x))
    if direction==1:
        ind_increment *= -1
    ind_mid = round(len(x)/2)

    startx = x[ind_mid]
    starty =  y[ind_mid]
    endx = x[ind_mid + 2*round(ind_increment)]
    endy = y[ind_mid + 2*round(ind_increment)]

    return startx, starty, endx, endy       


def calc_colour_bar(z, colour_map='gist_rainbow', reverse=False):
    """
    FUNCTION
    to calculate colors from data and the colorbar object
    """
    if reverse:
        colour_map +='_r'
    #calculate colours
    if type(z)==list:
        max_colour = max([max(i) for i in z])
        min_colour = min([min(i) for i in z])
        #calculate colour vector
        colours = [getattr(mpl.cm, colour_map)((i-min_colour)/(max_colour-min_colour)) for i in z]
        
    else:
        max_colour = max(z)
        min_colour = min(z)
        #calculate colour vector
        colours = getattr(mpl.cm, colour_map)((z-min_colour)/(max_colour-min_colour))
        
    #map colours to bar
    sm = plt.cm.ScalarMappable(cmap=colour_map, norm=mpl.colors.Normalize(vmin=min_colour, vmax=max_colour))
    
    return colours, sm



from matplotlib.collections import LineCollection

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    FUNCTION
    
    from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)

    return ax.add_collection(lc)


##############################################################################
"""
GRAPHS
"""
##############################################################################

def plot_y(y, x = [], xy_lab=[], title=[], colour='b', linewidth=1, marker_style='', scale=[.7*6, .7*4.8]):
    """
    FUNCTION
    
    for producing SINGLE plot of y data 
    
    Input:
        y
            if array, then each COLUMN is plotted as separate line
    """
    #plot 
    fig, axs = plt.subplots(figsize=scale)
    axs.spines[['right', 'top']].set_visible(False)
    if len(x)!=0:
        axs.plot(x, y, color=colour, linewidth=linewidth, marker=marker_style)
    else:
        axs.plot(y, color=colour, linewidth=linewidth, marker=marker_style)
        axs.set_xlabel('index')
    #LABELS
    if len(xy_lab)!=0:
        axs.set_xlabel(xy_lab[0])
        axs.set_ylabel(xy_lab[1])
    else:
        axs.set_xlabel('x')
        axs.set_ylabel('y')
    if len(title)!=0:
        axs.set_title(title)
    plt.tight_layout()
    
    
def scatter_xy_trend(x, y, xyt_label, alt_colour=0, scale=[.7*6, .7*4.8], file_name=0):
    """
    FUNCTION
    
    for plotting SINGLE SCATTER PLOT of y data against a common x axis 
    WITH TREND LINE
    
    Input:
        x, vector of x data
        y, vector of y data
        xyt_label, list of 3 elements. 
            0 is string to label x data
            1 is string to label y data
            2 is legend label for trendline
        alt_colour, to use jet instead of gist_rainbow
        file_name, list of strings for saving the plot
        scale, scale of the plots
        save, boolean to save
    """
    colours = ['b', 'red', 'green', 'violet', 'cyan', 'orange', 'indigo']
    nC = len(y)
    #define t distribution
    t_dist = lambda p, df : abs(stats.t.ppf(p/2, df))
    #t-statistic
    ts = t_dist(.1, nC-2)
    fig, axs = plt.subplots(figsize=(scale[0], scale[1]))
    axs.scatter(x, y,  facecolors='none', edgecolors=colours[alt_colour], label=xyt_label[2])
    res = stats.linregress(x, y)
    y_trend = x*res.slope + res.intercept
    y_UB = x*(res.slope + res.stderr*ts) + res.intercept + res.intercept_stderr*ts
    y_LB = x*(res.slope - res.stderr*ts) + res.intercept - res.intercept_stderr*ts
    axs.plot(x, y_trend, linewidth=2, color='k', label='trendline')
    ind_sort = np.argsort(x)
    axs.fill_between(x[ind_sort], y_LB[ind_sort], y_UB[ind_sort], color='grey', alpha=.25)
    axs.set_xlabel(xyt_label[0])
    axs.set_ylabel(xyt_label[1])
    axs.spines[['top', 'right']].set_visible(False)
    axs.ticklabel_format(style='plain', useOffset=False, axis='y')
    #axs.spines['right'].set_visible(False)
    #axs.set_ylim(0, .11)
    plt.legend(frameon=False) 
    if len(xyt_label)!=3:
        axs.set_title(xyt_label[-1])
    plt.tight_layout()
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')    



def plot_xy_N(x, y, xy_lab, altstyle=0, scale=[.7*6, .7*4.8], file_name=0):
    """
    FUNCTION
    
    Plot MULTIPLE x y plots
    
    Input:
        x, list of x vectors
        y, (list) of 2-row matrices
            row 0 is for the line
            row 1 is for the scatter
        xyz_label
            0 is x label
            1 is ylabel
            2 is title (optional)
        leglabels
    """
    if altstyle % 5==0:
        colours = 4*['crimson',  'orangered', 'gold', 'limegreen', 'aqua', 'cornflowerblue', 'blueviolet', 'violet', 'indigo']
    elif altstyle % 5==1:
        colours = ['indigo', 'tab:orange', 'green', 'violet', 'cyan', 'orange', 'indigo', 'pink']
    elif altstyle % 5==2:
        colours = 4*['blue', 'violet', 'red', 'green']
    elif altstyle % 5==3:
        colours = ['limegreen', 'deepskyblue', 'violet', 'crimson']
    elif altstyle % 5==4:
        colours = ['green', 'cornflowerblue', 'limegreen', 'red', 'orangered']
    
    if altstyle%10 < 5:
        lstyle = '-'
        lwidth = 2
        marker = None
        markersize = 0
    elif altstyle%10>=5:
        lstyle = '-'
        lwidth = 1
        marker = '.'
        markersize = 5
    N = len(y)
    #SINGLE/ROW PLOTS
    if N==1:
        #SINGLE PLOT
        fig, axs = plt.subplots(sharex=True, figsize=(scale[0], scale[1]))
        axs.plot(x[0], y[0], linestyle=lstyle, linewidth=lwidth, marker = marker, markersize=markersize, color=colours[altstyle])
        axs.set_ylabel(xy_lab[1])
        axs.spines[['top', 'right']].set_visible(False)
        axs.ticklabel_format(style='plain', useOffset=False, axis='y')
        axs.set_xlabel(xy_lab[0])
        #titles
        if len(xy_lab)>=3:
            axs.set_title(xy_lab[-1][0])
    else:
        if altstyle<10:
            #COLUMN PLOT
            fig, axs = plt.subplots(N, 1, sharex=True, figsize=(scale[0], scale[1]))
            for i, ydata in enumerate(y):
                axs[i].plot(x[i], ydata, linestyle=lstyle, linewidth=lwidth, marker = marker, markersize=markersize, color=colours[i])
                axs[i].spines[['top', 'right']].set_visible(False)
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xy_lab[0], xy_lab[1], sel_ax_x_label=-1, sel_ax_y_label='sup')
        else:
            #ROW PLOTS
            #SHARED y axis
            if altstyle>=10 and altstyle<20:
                fig, axs = plt.subplots(1, N, sharex=True, sharey=True, figsize=(scale[0], scale[1]))
            #VARIED y axis
            elif altstyle>=20 and altstyle<30:
                fig, axs = plt.subplots(1, N, sharex=True, figsize=(scale[0], scale[1]))
            #VARIED x and y axis
            elif altstyle>=30:
                fig, axs = plt.subplots(1, N, figsize=(scale[0], scale[1]))
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xy_lab[0], xy_lab[1])
        
        #PLOT LINES
        for i, ydata in enumerate(y):
            axs[i].plot(x[i], ydata, linestyle=lstyle, linewidth=lwidth, marker = marker, markersize=markersize, color=colours[i])
            axs[i].spines[['top', 'right']].set_visible(False)
            axs[i].ticklabel_format(style='plain', useOffset=False, axis='y')
            
        '''
        #PLOT DATA MARKERS
        else:
            for i, ydata in umerate(y):
                axs[i].plot(x[i], ydata, alpha=0.8, marker='.',  markersize=10, color=colours[i])
                axs[i].spines[['top', 'right']].set_visible(False)
                axs[i].ticklabel_format(style='plain', useOffset=False, axis='y')
        '''
        #titles
        if len(xy_lab)>=3:
            set_titles_N(N, fig, axs, xy_lab[2])

    plt.tight_layout(w_pad=1)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')






def plot_xy_N_leg(x, y, xy_lab, leglabels, altstyle=0, scale=[.7*6, .7*4.8], file_name=0, ax_legend = 0):
    """
    FUNCTION
    
    for N scatter plots
    with legend
    
    Input:
        x, x vector
        y, array of y data matching x
        xy_lab, list of 2 elements
            0 is string of x label
            1 is string for y data
            2 (optional) is plot title
        leglabels, list of strings of labels
        altstyle
            0-19, SINGLE plot
                0-9, shared x axis
                10-19, variable x axis
            >=20, HORIZONTALLY STACKED plots, variable x axis
        file_name, string for saving the plot
        scale, scale of the plots
        save, boolean to save
                    
    """
    if altstyle % 5==0:
        colours = ['crimson',  'orange', 'darkgoldenrod', 'limegreen', 'darkgreen', 'aqua', 'cornflowerblue', 'blueviolet', 'violet', 'indigo']
    elif altstyle % 5==1:
        colours = ['indigo', 'tab:orange', 'green', 'violet', 'cyan']
    elif altstyle % 5==2:
        colours = ['green', 'r', 'deepskyblue', 'indigo','crimson', 'orangered', 'goldenrod']
    elif altstyle % 5==3:
        colours = ['green', 'r', 'deepskyblue', 'indigo','crimson', 'orangered', 'goldenrod']
    elif altstyle % 5==4:
        colours = ['lightskyblue', 'b', 'k']
    if altstyle>=5 and altstyle<10:
        lstyles = 10*[None]
        lwidths = 10*[0]
        markers = 10*['.']
        markersizes = 10*[5]
    elif altstyle>=10 and altstyle<15:
        lstyles = ['-', '', '-']
        lwidths = 10*[1]
        markers = ['', '.', '']
        markersizes = [0, 4, 0]
    else:
        lstyles = 10*['-']
        lwidths = 10*['1']
        markers = 10*['']
        markersizes = 10*[0]
    if altstyle<20:
        #SINGLE PLOT
        fig, axs = plt.subplots(figsize=(scale[0], scale[1]))
        axs.spines[['top', 'right']].set_visible(False)
        #SHARED X VECTOR
        if len(x)!=len(y):
            for j, ydata in enumerate(y):
                axs.plot(x, ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j], alpha=.75)
        #VARYING X VECTOR
        else:
            for j, ydata in enumerate(y):
                axs.plot(x[j], ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j], alpha=.75)
    
        axs.set_xlabel(xy_lab[0])
        axs.set_ylabel(xy_lab[1])
        #LEGEND
        if len(leglabels)>len(y):
            axs.legend(frameon=False, title=leglabels[-1])
        else:
            axs.legend(frameon=False)
        #TITLE
        if len(xy_lab)==3:
            axs.set_title(xy_lab[2])
    elif altstyle>=20:
        #HORIZONTAL STACK, SINGLE LEGEND
        N = len(y)
        fig, axs = plt.subplots(1, N, figsize=(scale[0], scale[1]))
        for i in range(N):
            #VARIED X VECTOR
            if len(x[i])==len(y[i]):
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i][j], ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j])
                    axs[i].spines[['top', 'right']].set_visible(False)
            #SHARED x vector
            else:
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i], ydata, label=leglabels[j], linestyle=lstyles[j], linewidth=lwidths[j], marker = markers[j], markersize=markersizes[j], color=colours[j])
                    axs[i].spines[['top', 'right']].set_visible(False)
        
        #SET X Y LABELS
        set_xy_labels_N(N, fig, axs, xy_lab[0], xy_lab[1])

        #LEGEND
        if len(leglabels)>len(y[0]):
            #LEGEND WITH TITLE
            axs[ax_legend].legend(frameon=False, title=leglabels[-1])
        else:
            #LEGEND NO TITLE
            axs[ax_legend].legend(frameon=False)
        #TITLES
        if len(xy_lab)>=3:
            set_titles_N(N, fig, axs, xy_lab[2])

    plt.tight_layout(w_pad=1)
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')  

        

def plot_xy_N_colour(x, y, z, xyz_label, altstyle=0, scale=[.7*6, .7*4.8], file_name=0, reverse_colour=False):
    """
    FUNCTION
    
    for N horizontally or vertically stacked plots
    SHARED COLOUR BAR
    
    Input:
        x, x vector
        y, N element list or matrix containing ROW vectors of y data
        z, single vector or LIST of vector(s) by which to colour the data
        xyz_label
            0 is the xlabels for all plots
            1 is the ylabel for all plots
            2 is colourbar label
            3 is list of titles
        altstyle
            if <10 then single plot
            mod 5==0 then rainbow
            mod 5==1 then jet
        scale, scale of the plots
        file_name, string for saving the plot
    """
    N = len(y)

    #define colour vector
    if altstyle % 5==0:
        colours, sm = calc_colour_bar(z, 'gist_rainbow', reverse_colour)
    elif altstyle % 5==1:
        colours, sm = calc_colour_bar(z, 'jet', reverse_colour)
    elif altstyle % 5==2:
        colours, sm = calc_colour_bar(z, 'cool', reverse_colour)
    elif altstyle % 5==3:
        colours, sm = calc_colour_bar(z, 'viridis', reverse_colour)
    elif altstyle % 5==4:
        colours, sm = calc_colour_bar(z, 'YlOrRd', reverse_colour)
    if altstyle % 10 < 5:
        l_style='-'
        m_style = ''
        m_size = 0
    elif altstyle % 10 >= 5:
        l_style='-'
        m_style = 'o'
        m_size = 5
    #SINGLE PLOT
    if altstyle<10:
        fig, axs = plt.subplots(figsize = scale)
        axs.spines[['top', 'right']].set_visible(False)
        #SHARED X VECTOR
        if len(x)!=len(y):
            for j, ydata in enumerate(y):
                axs.plot(x, ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[j])
        #VARYING X VECTOR
        else:
            for j, ydata in enumerate(y):
                axs.plot(x[j], ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[j])
        #LABELS
        axs.set_xlabel(xyz_label[0])
        axs.set_ylabel(xyz_label[1])
        #TITLE
        if len(xyz_label)>=4:
            axs.set_title(xyz_label[3])
    #SINGLE ROW OR COLUMN
    elif altstyle>=10 and altstyle<50:
        #HORIZONTAL STACK
        if altstyle>=10 and altstyle<40:
            #SHARED X AND Y TICKS, LABELS
            if altstyle<20:
                fig, axs = plt.subplots(1, N, sharex=True, sharey=True, figsize = scale)
            #VARIED Y AXIS LABEL, SHARED X LABEL
            elif altstyle>=20:
                fig, axs = plt.subplots(1, N, figsize = scale)
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xyz_label[0], xyz_label[1])
        #VERTICAL STACK
        elif altstyle>=40:
            fig, axs = plt.subplots(N, 1, sharex=True, figsize = scale)
            axs[-1].set_xlabel(xyz_label[0])
            for i in range(N):
                axs[i].set_ylabel(xyz_label[1][i])
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xyz_label[0], xyz_label[1], sel_ax_x_label=-1, sel_ax_y_label='sup')
            
        #PLOT
        for i in range(N):
            axs[i].spines[['top', 'right']].set_visible(False)
            #VARYING X VECTOR
            if len(x[i])==len(y[i]) and type(x[i][0]) is not float:
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i][j], ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[i][j])
            #SHARED X VECTOR
            else:
                for j, ydata in enumerate(y[i]):
                    axs[i].plot(x[i], ydata, alpha=0.75, linestyle=l_style, marker = m_style, markersize=m_size, color=colours[i][j])
        
        #TITLE
        if len(xyz_label)>=4:
            set_titles_N(N, fig, axs, xyz_label[3])

    plt.tight_layout(w_pad=2)
    #COLOUR BAR
    cbar = plt.colorbar(sm, ax=axs, pad = 0.02)
    #cbar = plt.colorbar(sm, ax=axs, aspect=25, pad=0.2)
    cbar.set_label(xyz_label[2])
    #save figure
    if file_name!=0:
        fig.savefig(file_name+'.png', dpi=300, bbox_inches='tight')
        
        
        
        
def scatter_xy_colour_N(x, y, z, xyz_label, altstyle=0, scale=[.7*6, .7*4.8], file_name=0, reverse_colour=False):
    #define colour vector
    if altstyle % 5==0:
        cmap = 'gist_rainbow'
    elif altstyle % 5==1:
        cmap = 'jet'
    elif altstyle % 5==2:
        cmap = 'cool'
    elif altstyle % 5==3:
        cmap = 'viridis'
    elif altstyle % 5==4:
        cmap = 'YlOrRd'
    #set colour vector and bar object
    colours, sm = calc_colour_bar(z, cmap, reverse_colour)
    if altstyle % 10 < 5:
        m_style = 'o'
        m_size = 5
    #SINGLE PLOT
    if altstyle<10:
        fig, axs = plt.subplots(figsize = scale)
        axs.spines[['top', 'right']].set_visible(False)
        #SHARED X VECTOR
        axs.scatter(x, y, alpha=0.75, marker = m_style, s=m_size, c=colours)
        colored_line(x, y, z, axs, linewidth=2, cmap=cmap)
        #LABELS
        axs.set_xlabel(xyz_label[0])
        axs.set_ylabel(xyz_label[1])
        #TITLE
        if len(xyz_label)>=4:
            axs.set_title(xyz_label[3])
    #SINGLE ROW OR COLUMN
    elif altstyle>=10 and altstyle<50:
        N = len(y)
        #HORIZONTAL STACK
        if altstyle>=10 and altstyle<40:
            #SHARED X AND Y TICKS, LABELS
            if altstyle<20:
                fig, axs = plt.subplots(1, N, sharex=True, sharey=True, figsize = scale)
            #VARIED Y AXIS LABEL, SHARED X LABEL
            elif altstyle>=20:
                fig, axs = plt.subplots(1, N, figsize = scale)
            #SET X Y LABELS
            set_xy_labels_N(N, fig, axs, xyz_label[0], xyz_label[1])
        
        #PLOT
        for i in range(N):
            axs[i].spines[['top', 'right']].set_visible(False)                    
            #SHARED X VECTOR
            axs[i].scatter(x[i], y[i], alpha=0.75, marker = m_style, s=m_size, c=colours[i])
            colored_line(x[i], y[i], z[i], axs[i], linewidth=2, cmap=cmap)

        #TITLE
        if len(xyz_label)>=4:
            set_titles_N(N, fig, axs, xyz_label[3])
    plt.tight_layout(w_pad=2)
    #COLOUR BAR
    cbar = plt.colorbar(sm, ax=axs, pad = 0.02)
    cbar.set_label(xyz_label[2])
    
    #save figure
    if file_name!=0:
        fig.savefig(file_name+'.png', dpi=300, bbox_inches='tight')
        
        
        
        
        
def plot_2arrow_1line(x, y, xy_label, leglabels, altstyle=0, scale=[.7*6.4, .7*4.8], file_name=0):
    colours = ['green', 'r', 'b']
    fig, axs = plt.subplots(figsize = scale)
    axs.spines[['top', 'right']].set_visible(False)
    #SHARED X VECTOR
    a_styles = 10*["-|>"]
    leg_styles = [">", "<"]
    #SHARED X VECTOR
    #MUST BE ASCENDING
    for i, ydata in enumerate(y):
        if i<2:
            ind_sort = np.argsort(x[i])
            axs.plot(x[i][ind_sort], ydata[ind_sort], color=colours[i])
            startx, starty, endx, endy = calc_arrow_coord(x[i][ind_sort], ydata[ind_sort], i)
            axs.annotate('', xytext=(startx, starty), xy=(endx, endy),
                         arrowprops=dict(arrowstyle=a_styles[i], color=colours[i]),size=15)
            axs.plot([], label=leglabels[i], marker=leg_styles[i], color=colours[i])
        elif i==2:
            axs.plot(x[i], ydata, color=colours[i], linestyle=':')
            axs.plot([], label=leglabels[i], linestyle=':', color=colours[i])
    #legend
    if len(leglabels)>len(y):
        axs.legend(frameon=False, title=leglabels[-1])
    else:
        axs.legend(frameon=False)
    #LABELS
    axs.set_xlabel(xy_label[0])
    axs.set_ylabel(xy_label[1])
    #TITLE
    if len(xy_label)>=3:
        axs.set_title(xy_label[2])
    plt.tight_layout()
    #save figure
    if file_name!=0:
        fig.savefig(file_name+'.png', dpi=300, bbox_inches='tight')

        
        

def plot_3D_lines(x, y, z, xyz_label=['x', 'y', 'z'], lab_pad=2, x_lim=0, y_lim=0, colour_map='viridis', scale=[6, 4.8], file_name=0):
    """
    FUNCTION
    
    to plot lines in 3D

    Parameters
    ----------
    x : x axis data (plotted HORIZONTALLY)
    y : y axis data (plotted VERTICALLY)
    z : z axis data (plotted INTO THE SCREEN)
    xyz_label : labels for x, y, z
    lab_pad : optional, to pad distance from axis to labels
    x_lim : optional, list of 2 values to limit x data
    y_lim : optional, list of 2 values to limit y data
    colour_map : 
        eg, 'jet', 'cool', 'viridis'
    scale : optional, list of 2 values to change figure size

    """
    #3D PLOT
    colours, sm = calc_colour_bar(z, colour_map)
    #INITIALISE
    axs = plt.figure(figsize=scale).add_subplot(projection='3d')
    
    #SET X LIMITS
    if x_lim!=0:
        if len(x)!=len(y):
            x_bool = (x >= min(x_lim)) & (x <= max(x_lim))
            x[np.invert(x_bool)] = None
        else:
            for j, _ in enumerate(y):
                x[j][np.invert(x_bool)] = None
    #SET Y LIMITS
    if y_lim!=0:
        for j, ydata in enumerate(y):
            y_bool = (ydata >= min(y_lim)) & (ydata <= max(y_lim))
            y[j][np.invert(y_bool)] = None
    
    #SHARED X VECTOR
    if len(x)!=len(y):
        #PLOT
        for j, ydata in enumerate(y):
            axs.plot(x, ydata, zs=z[j], zdir='y', alpha=0.75, color=colours[j])
    #VARYING X VECTOR
    else:
        for j, ydata in enumerate(y):
            axs.plot(x[j], ydata, zs=z[j], zdir='y',alpha=0.75, color=colours[j])
    #LABELS
    axs.set_xlabel(xyz_label[0], labelpad=lab_pad)
    axs.set_ylabel(xyz_label[2], labelpad=lab_pad)
    axs.set_zlabel(xyz_label[1], labelpad=lab_pad)
    #title
    if len(xyz_label)==4:
        axs.set_title(xyz_label[3])
    plt.tight_layout()
    #save figure
    if file_name!=0:
        plt.savefig(file_name+'.png', dpi=300, bbox_inches='tight')




def plot_heatmap(x, y, z, xyz_label=['x', 'y', 'z'], colour_map = 'viridis', scale=[6, 4.8], file_name=0):
    """
    FUNCTION
    
    to plot heatmap on x-y grid
    
    z is intensity

    """
    fig, axs = plt.subplots(figsize = scale)
    im = axs.pcolormesh(x, y, z, cmap=colour_map)
    cbar = fig.colorbar(im, ax=axs)
    cbar.set_label(xyz_label[2])
    #axis labels
    axs.set_xlabel(xyz_label[0])
    axs.set_ylabel(xyz_label[1])
    #title
    if len(xyz_label)==4:
        axs.set_title(xyz_label[3])
    plt.tight_layout()
    #save figure
    if file_name!=0:
        fig.savefig(file_name+'.png', dpi=300, bbox_inches='tight')
    
    '''
    im = axs.imshow(z, extent=[x[0], x[-1], y[0], y[-1]], interpolation='none', origin='lower', cmap=colour_map, aspect='auto')
    #axis labels
    axs.set_xlabel(xyz_label[0])
    axs.set_ylabel(xyz_label[1])
    #color bar, adjust location/size to match image axis
    #bounds = [x0, y0, width, height]. with default transform=axs.transAxes
    cbar = fig.colorbar(im, cax=axs.inset_axes(bounds = [1.02, 0, 0.02, 1.]))
    '''
    
    