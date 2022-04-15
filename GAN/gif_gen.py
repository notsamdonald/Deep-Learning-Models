import imageio

from os import walk

gan_dir = "GAN_outputs"

filenames = next(walk(gan_dir), (None, None, []))[2]  # [] if no file

import imageio
images = []

for filename in filenames:
    images.append(imageio.imread("{}/{}".format(gan_dir,filename)))
imageio.mimsave('GAN_outputs/gan_output.gif', images)
