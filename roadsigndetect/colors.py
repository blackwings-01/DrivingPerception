
import six
import numpy as np
import matplotlib

colormap = {} # rgb value from 0 to 1
for name, hex_ in six.iteritems(matplotlib.colors.cnames):
    colormap[name] = matplotlib.colors.hex2color(hex_)

# Add the single letter colors.
for name, rgb in six.iteritems(matplotlib.colors.ColorConverter.colors):
    colormap[name] = rgb

def rgb(name):
    return [min(c,255) for c in np.floor(np.asarray(colormap[name]) * 256).astype(int)]

def bgr(name):
    [r,g,b] = rgb(name)
    return [b,g,r]
