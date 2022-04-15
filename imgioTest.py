import imageio
imageio.plugins.freeimage.download()
# x = imageio.imread(r"/Users/kepeihou/Objectron/rendertest2/test2.png")
x = imageio.imread(r'/Users/kepeihou/Objectron/rendertest2/test2_zdepth_1.exr', 'exr')

print(x)