import matplotlib.pyplot as plt
import matplotlib
import tempfile
import shutil
import os

###################################################################################################
# Adjust Spines (Dickinson style, thanks to Andrew Straw)
###################################################################################################

# NOTE: smart_bounds is disabled (commented out) in this function. It only works in matplotlib v >1.
# to fix this issue, try manually setting your tick marks (see example below) 
def adjust_spines(ax,spines, spine_locations={}, smart_bounds=True, xticks=None, yticks=None, linewidth=1, spineColor='black'): # ivo: spineColor
    if type(spines) is not list:
        spines = [spines]
        
    # get ticks
    if xticks is None:
        xticks = ax.get_xticks()
    if yticks is None:
        yticks = ax.get_yticks()
        
#    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    spine_locations_dict = {'top': 6, 'right': 6, 'left': 6, 'bottom': 6}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]
        
    if 'none' in spines:
        for loc, spine in ax.spines.items():
        #for loc, spine in ax.spines.iteritems(): #ivo:  was used in python 2.7
            spine.set_color('none') # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return
    
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',spine_locations_dict[loc])) # outward by x points
            spine.set_linewidth(linewidth)
            spine.set_color(spineColor) #ivo
            ax.tick_params(colors=spineColor) #ivo
            ax.tick_params(length=linewidth*4) #ivo
            ax.tick_params(pad=linewidth*4) #ivo
            ax.tick_params(direction="in") #ivo
        else:
            spine.set_color('none') # don't draw spine
            
    # smart bounds, if possible
    if int(matplotlib.__version__[0]) > 0 and smart_bounds: 
        for loc, spine in ax.spines.items():
            if loc in ['left', 'right']:
                ticks = yticks
            if loc in ['top', 'bottom']:
                ticks = xticks
            spine.set_bounds(ticks[0], ticks[-1])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    if 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])    
    
    if 'left' in spines or 'right' in spines:
        ax.set_yticks(yticks)
    if 'top' in spines or 'bottom' in spines:
        ax.set_xticks(xticks)
    
    for line in ax.get_xticklines() + ax.get_yticklines():
        #line.set_markersize(6)
        line.set_markeredgewidth(linewidth)

        
def kill_spines(ax):
    return adjust_spines(ax,'none', 
                  spine_locations={}, 
                  smart_bounds=True, 
                  xticks=None, 
                  yticks=None, 
                  linewidth=1)

def kill_labels(ax):
    #ax = ax['axis']
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

def kill_all_spines(layout):
    [kill_spines(ax) for ax in layout.axes.values()]

def kill_all_labels(layout):
    [kill_labels(ax) for ax in layout.axes.values()]
    
def set_fontsize(fig,fontsize):
    """
    For each text object of a figure fig, set the font size to fontsize
    """
    def match(artist):
        return artist.__module__ == "matplotlib.text"

    for textobj in fig.findobj(match=match):
        textobj.set_fontsize(fontsize)
        
def set_fontfamily(fig,fontfamily): #ivo
    """
    For each text object of a figure fig, set the font size to fontsize
    """
    def match(artist):
        return artist.__module__ == "matplotlib.text"

    for textobj in fig.findobj(match=match):
        textobj.set_family(fontfamily)

def fix_mpl_svg(file_path, pattern='miterlimit:100000;', subst='miterlimit:1;'):
    """used to fix problematic outputs from the matplotlib svg
    generator, for example matplotlib creates exceptionaly large meterlimis"""
    fh, abs_path = tempfile.mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    os.close(fh)

    os.remove(file_path)

    shutil.move(abs_path, file_path)
    return