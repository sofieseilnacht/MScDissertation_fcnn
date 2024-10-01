import os
import numpy as np
import configparser as ConfigParser

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import colors

import galsim
from rwl_tools import load_catalogue
from astropy.io import fits

# Set matplotlib parameters
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all')  # tidy up any unshown plots

# Read the existing configuration file
config = ConfigParser.ConfigParser(inline_comment_prefixes=";")
config.read('./inis/test.ini')

# Get paths and other parameters from the configuration file
output_path = config.get("pipeline", "output_path")
figure_path = config.get("pipeline", "figure_path")
pixel_scale = config.getfloat("skymodel", "pixel_scale") * galsim.arcsec
fov = config.getfloat("skymodel", "field_of_view") * galsim.arcmin
image_size = int((fov / galsim.arcmin) / (pixel_scale / galsim.arcmin))

psf_image = galsim.fits.read(config.get("skymodel", "psf_filepath"))
psf = galsim.InterpolatedImage(psf_image, flux=1, scale=pixel_scale / galsim.arcsec)

# Load the catalogue
cat = load_catalogue(config)

# Cut the catalogue to the number of sources we want
nobj = len(cat)
if config.getint("skymodel", "ngals") > -1:
    nobj = config.getint("skymodel", "ngals")
    cat = cat[:nobj]

# Function to update the configuration file
def update_config_ellipticity(config, value):
    config.set("skymodel", "constant_mod_e", str(value))
    new_output_suffix = f"test_mod_e{value:.3f}"
    config.set("pipeline", "output_suffix", new_output_suffix)

# Loop through the desired range of ellipticity values
for ellipticity_value in np.arange(0.0, 0.405, 0.005):
    update_config_ellipticity(config, ellipticity_value)
    
    constant_mod_e = config.get("skymodel", "constant_mod_e")
    
    # Write out catalogue with updated ellipticity
    output_filename = (
        f"truthcat_{constant_mod_e}ellipticity.fits"
    )
    output_filepath = os.path.join(output_path, output_filename)

    if not os.path.exists(output_filepath):
        print(f"Writing truthcat to {output_filepath}")
        cat.write(output_filepath, format="fits", overwrite=True)
    else:
        print(f"Appending truthcat data to {output_filepath}")
        with fits.open(output_filepath, mode='update') as existing_cat:
            existing_data = existing_cat[1].data
            new_data = cat.as_array()
            new_data_standardized = np.array(new_data, dtype=existing_data.dtype)
            combined_data = np.concatenate((existing_data, new_data_standardized))
            existing_cat[1].data = combined_data
            existing_cat.flush()

    # Loop through each galaxy in the catalogue and generate images
    for i, cat_gal in enumerate(cat):
        print(f"Drawing galaxy {i}...")

        full_image = galsim.ImageF(image_size, image_size,
                                   scale=pixel_scale/galsim.arcsec)
        im_center = full_image.bounds.true_center

        if config.get("skymodel", "galaxy_profile") == "exponential":
            gal = galsim.Exponential(
                scale_radius=cat_gal["Maj"] / 2.0,
                flux=cat_gal["Total_flux"],
            )
        elif config.get("skymodel", "galaxy_profile") == "gaussian":
            gal = galsim.Gaussian(
                fwhm=cat_gal["Maj"],
                flux=cat_gal["Total_flux"],
            )

        # Calculate the total ellipticity
        ellipticity = galsim.Shear(e1=cat_gal["e1"], e2=cat_gal["e2"])

        if config.getboolean("skymodel", "doshear"):
            if config.get("skymodel", "shear_type") == 'trecs':
                shear = galsim.Shear(g1=cat_gal["g1_shear"], g2=cat_gal["g2_shear"])
                if i == 0:
                    print('Applying shear read from trecs catalogue...')
            elif config.get("skymodel", "shear_type") == 'constant':
                g1 = config.getfloat("skymodel", "constant_shear_g1")
                g2 = config.getfloat("skymodel", "constant_shear_g2")
                shear = galsim.Shear(g1=g1, g2=g2)
                if i == 0:
                    print(f'Applying constant shear g1 = {g1}, g2 = {g2}')

            total_shear = ellipticity + shear
        else:
            total_shear = ellipticity

        # Get the galaxy size and add its ellipticity
        maj_gal = cat_gal["Maj"]
        q_gal = cat_gal["Min"] / cat_gal["Maj"]
        A_gal = np.pi * maj_gal ** 2.0
        maj_corr_gal = np.sqrt(A_gal / (np.pi * q_gal))

        gal = gal.shear(total_shear)
        gal = gal.dilate(maj_gal / maj_corr_gal)

        # Convolve gal with psf
        gal = galsim.Convolve([gal, psf])

        cn = galsim.CorrelatedNoise(psf_image*1.e-8)

        stamp = gal.drawImage(scale=pixel_scale / galsim.arcsec)

        if i == 0:
            print("Drawing PSF...")
            plt.title('PSF')
            psf_stamp = psf.drawImage(nx=512, ny=512, scale=pixel_scale / galsim.arcsec)
            psf_stamp.setCenter((image_size / 2, image_size / 2))
            plt.figure()
            plt.imshow(psf_stamp.array, origin="lower")
            plt.savefig(os.path.join(figure_path, "psf.png"), bbox_inches="tight", dpi=300)

        stamp.setCenter(image_size / 2, image_size / 2)
        bounds = stamp.bounds & full_image.bounds
        full_image[bounds] += stamp[bounds]

        full_image.addNoise(cn)

        # Write out this image
        if config.get("pipeline", "output_type") == 'txt':
            while True:
                filename = f"{constant_mod_e}ellipticity_{i}.txt"
                filepath = os.path.join(output_path, filename)
                if not os.path.exists(filepath):
                    np.savetxt(filepath, full_image.array)
                    break
                else:
                    i += 1

        if config.getboolean("pipeline", "do_thumbnails"):
            while True:
                filename = f"{constant_mod_e}ellipticity_{i}.png"
                filepath = os.path.join(figure_path, filename)
                if not os.path.exists(filepath):
                    plt.figure()
                    plt.title(f'Source {i}')
                    plt.imshow(full_image.array, origin="lower")
                    plt.savefig(filepath, bbox_inches="tight", dpi=300)
                    plt.close()
                    break
                else:
                    i += 1
