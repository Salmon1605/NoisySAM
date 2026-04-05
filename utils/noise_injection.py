import os 
import sys 
import glob 
import re
import cv2  

import torch
import PIL 
import numpy as np 
import matplotlib.pyplot as plt 
import albumentations as A 
import torchvision.datasets as tds
import torchvision.transforms as transforms 
import skimage as sk 

from torch.utils.data import Dataset, DataLoader 
from utils.dataLoader import COCOLoader 
from PIL import Image
from scipy.ndimage import zoom as scizoom 
from skimage.filters import gaussian

from wand.image import Image as WandImage
from wand.api import library as wandlibrary
from io import BytesIO


class NoiseHelpers: 
    def __init__(self):
        pass

    ### Helpers 
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    def clipped_zoom(img, zoom_factor):
        h, w = img.shape[:2]

        ch = int(np.ceil(h / zoom_factor))
        cw = int(np.ceil(w / zoom_factor))

        top = (h - ch) // 2
        left = (w - cw) // 2

        img = scizoom(
            img[top:top + ch, left:left + cw],
            (zoom_factor, zoom_factor, 1),
            order=1
        )

        trim_top = (img.shape[0] - h) // 2
        trim_left = (img.shape[1] - w) // 2

        return img[
            trim_top:trim_top + h,
            trim_left:trim_left + w
        ]

    def plasma_fractal(mapsize=256, wibbledecay=3):
        """
        Generate a heightmap using diamond-square algorithm.
        Return square 2d array, side length 'mapsize', of floats in range 0-255.
        'mapsize' must be a power of two.
        """
        assert (mapsize & (mapsize - 1) == 0)
        maparray = np.empty((mapsize, mapsize), dtype=np.float32)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

        def fillsquares():
            """For each square of points stepsize apart,
            calculate middle value as mean of points + wibble"""
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize] 
            squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize // 2:mapsize:stepsize,
            stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            """For each diamond of points stepsize apart,
            calculate middle value as mean of points + wibble"""
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum + tulsum
            maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        return maparray / maparray.max()

    def motion_blur_kernel(size, angle):
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[size // 2, :] = 1
        kernel /= size

        rot_mat = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, rot_mat, (size, size))

        return kernel
    


noise_helper = NoiseHelpers()
class NoiseInjection: 
    def __init__(self):
        pass 

    def _inject_gaussian_noise(self, image, severity:int=1):
        """
        Inject Gaussian Noise into image with certain severity severity.

        Args: 
            image (np.array): Input image that need to be noise injected.
            severity (int): Severity severity (1 - 3).

        Returns: 
            np.array: Gaussian Noise injected Image.
        """
        temp_image = image

        if severity == 1:
            std_range = (0.00, 0.02)
        elif severity == 2:
            std_range = (0.02, 0.05)
        elif severity == 3:
            std_range = (0.05, 0.10)
        elif severity == 4:
            std_range = (0.10, 0.18)
        elif severity == 5:
            std_range = (0.18, 0.30)
        else:
            raise ValueError("severity must be in [1, 5]")

        transform = A.GaussNoise(std_range=std_range, p=1)
        transformed_image = transform(image=temp_image)['image']
        return transformed_image
        
    
    def _inject_poisson_noise(self, image, severity:int=1): 
        """
        Inject Poisson Noise into image with certain severity severity.

        Args: 
            image (np.array): Input image that need to be noise injected.
            severity (int): Severity severity (1 - 3).

        Returns: 
            np.array: Poisson Noise injected Image.
        """
        temp_image = image
        if severity == 1:
            scale_range = (0.005, 0.02)
        elif severity == 2:
            scale_range = (0.02, 0.05)
        elif severity == 3:
            scale_range = (0.05, 0.10)
        elif severity == 4:
            scale_range = (0.10, 0.18)
        elif severity == 5:
            scale_range = (0.18, 0.30)
        else:
            raise ValueError("severity must be in [1, 5]")

        transform = A.ShotNoise(scale_range=scale_range, p=1)
        return transform(image=temp_image)['image']
     
    def _inject_salt_and_pepper_noise(self, image, severity:int=1): 
        """
        Inject Salt-and-Pepper Noise into image with certain severity severity.

        Args: 
            image (np.array): Input image that need to be noise injected.
            severity (int): Severity severity (1 - 3).

        Returns: 
            np.array: Salt-and-Pepper Noise injected Image.
        """
        # Implementation for Salt-and-Pepper noise to be added
        temp_image = image 
        salt_and_pepper_transform = None 
        if severity == 1: 
            salt_and_pepper_transform = A.SaltAndPepper(amount=(0.01, 0.05), p=1)
        elif severity == 2: 
            salt_and_pepper_transform = A.SaltAndPepper(amount=(0.06, 0.10), p=1)
        elif severity == 3: 
            salt_and_pepper_transform = A.SaltAndPepper(amount=(0.11, 0.15), p=1)
        elif severity == 4: 
            salt_and_pepper_transform = A.SaltAndPepper(amount=(0.15, 0.25), p=1)
        elif severity == 5: 
            salt_and_pepper_transform = A.SaltAndPepper(amount=(0.26, 0.35), p=1)
        
        transformed_image = salt_and_pepper_transform(image=temp_image)['image']
        return transformed_image
    
    def _inject_speckle_noise(self, image, severity:int=1):
        transformed_image = image 
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
        # print(f'severity severity: {c}')
        transformed_image = np.array(transformed_image) / 255 
        transformed_image = transformed_image + (transformed_image * np.random.normal(size=transformed_image.shape, scale=c))
        transformed_image = np.clip(transformed_image, 0, 1) 
        transformed_image = transformed_image * 255 
        return transformed_image.astype(np.uint8)

    def gaussian_blur(self, image, severity:int=1):
        transformed_image = image 
        c = [1, 2, 3, 4, 6][severity - 1]
        transformed_image = np.array(transformed_image) / 255.
        transformed_image = gaussian(transformed_image, sigma=c, channel_axis=-1) 
        transformed_image = np.clip(transformed_image, 0, 1) 
        transformed_image = transformed_image * 255
        return transformed_image.astype(np.uint8) 

    def _inject_defocus_noise(self, image, severity:int=1): 
        transformed_image = image 
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        transformed_image = np.array(transformed_image) / 255 
        kernel = noise_helper.disk(radius=c[0], alias_blur=c[1])
        channels = [] 

        for d in range(3): 
            channels.append(cv2.filter2D(transformed_image[:,:,d], -1, kernel=kernel))
        channels = np.array(channels).transpose((1, 2, 0)) 

        channels = np.clip(channels, 0, 1) 
        channels = channels * 255
        return channels.astype(np.uint8) 

    def _motion_blur(self, image, severity: int = 1):
        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
        radius, sigma = c
        angle = np.random.uniform(-45, 45)

        if not isinstance(image, np.ndarray):
            transformed_image = np.array(image)
        else:
            transformed_image = image.copy()

        k = radius
        center = k // 2

        kernel = np.zeros((k, k), dtype=np.float32)

        xs = np.arange(k) - center
        gauss = np.exp(-(xs**2) / (2 * sigma**2))
        gauss /= gauss.sum()

        kernel[center, :] = gauss

        rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (k, k))

        transformed_image = cv2.filter2D(transformed_image, -1, kernel)

        if transformed_image.ndim == 2:  # grayscale
            transformed_image = np.stack([transformed_image]*3, axis=-1)

        transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)

        return transformed_image

    def _inject_frosted_glass_blur(self, image, severity:int=1): 
        transformed_image = image 
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        transformed_image = np.array(transformed_image) / 255. 
        transformed_image = gaussian(transformed_image, c[0], channel_axis=-1) 
        transformed_image = transformed_image * 255. 
        transformed_image = transformed_image.astype(np.uint8) 
        img_height = image.shape[0] 
        img_width = image.shape[1]

        for i in range(c[2]): 
            for h in range(img_height - c[1], c[1], -1): 
                for w in range(img_width - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    transformed_image[h, w], transformed_image[h_prime, w_prime] = transformed_image[h_prime, w_prime], transformed_image[h, w] 

        transformed_image = np.clip(gaussian(transformed_image/255., sigma=c[0], channel_axis=-1), 0, 1) * 255
        return transformed_image.astype(np.uint8)
            
    def _inject_zoom_blur(self, image, severity: int = 1):
        c = [
                np.arange(1, 1.06, 0.01), 
                np.arange(1, 1.10, 0.01),
                np.arange(1, 1.15, 0.02),
                np.arange(1, 1.20, 0.02),
                np.arange(1, 1.25, 0.03)
                ][severity - 1]

        transformed_image = np.array(image).astype(np.float32) / 255.
        out = np.zeros_like(transformed_image, dtype=np.float32)

        for zoom_factor in c:
            out += noise_helper.clipped_zoom(transformed_image, zoom_factor)

        transformed_image = (transformed_image + out) / (len(c) + 1)
        transformed_image = np.clip(transformed_image, 0, 1) * 255

        return transformed_image.astype(np.uint8)

    def _inject_snow(self, image, severity:int=1): 
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
            (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
            (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
            (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
            (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]
        
        transformed_image = np.array(image, dtype=np.float32) / 255 
        h, w = transformed_image.shape[:2]

        snow_layer = np.random.normal(loc=c[0], scale=c[1], size=(h, w))
        snow_layer = noise_helper.clipped_zoom(snow_layer[..., np.newaxis], c[2]).squeeze()
        snow_layer[snow_layer < c[3]] = 0
        kernel = noise_helper.motion_blur_kernel(c[4], angle=np.random.uniform(-135, -45))
        snow_layer = cv2.filter2D(snow_layer, -1, kernel)
        snow_layer = np.clip(snow_layer, 0, 1)
        snow_layer = snow_layer[..., np.newaxis]
        gray = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

        transformed_image = c[6] * transformed_image + (1 - c[6]) * np.maximum(transformed_image, gray * 1.5 + 0.5) 
        transformed_image = transformed_image + snow_layer + np.rot90(snow_layer, k=2) 
        transformed_image = np.clip(transformed_image, 0, 1) * 255 
        return transformed_image.astype(np.uint8)

    def fog(self, image, severity:int=1):
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
        transformed_image = image
        transformed_image = np.array(transformed_image) / 255.
        max_val = transformed_image.max()
        fractal = noise_helper.plasma_fractal(mapsize=256)
        fractal = cv2.resize(fractal, (transformed_image.shape[1], transformed_image.shape[0]))
        fractal = fractal[..., np.newaxis]  # (H, W, 1)
        transformed_image += c[0] * fractal
        transformed_image = np.clip(transformed_image * max_val / (max_val + c[0]), 0, 1) * 255 
        return transformed_image.astype(np.uint8) 

    def _inject_brightness(self, image, severity:int=1): 
        c = [.1, .2, .3, .4, .5][severity - 1]
        transformed_image = image 
        transformed_image = np.array(transformed_image) / 255 
        transformed_image = sk.color.rgb2hsv(transformed_image) 
        transformed_image[:, :, 2] = np.clip(transformed_image[:, :, 2] + c, 0, 1)
        transformed_image = sk.color.hsv2rgb(transformed_image)
        transformed_image = np.clip(transformed_image, 0, 1) * 255
        return transformed_image.astype(np.uint8)

    def _inject_contrast(self, image, severity: int = 1):
        c = [0.75, 0.6, 0.5, 0.4, 0.3][severity - 1]  

        transformed_image = np.array(image).astype(np.float32) / 255.
        means = np.mean(transformed_image, axis=(0, 1), keepdims=True)

        transformed_image = (transformed_image - means) * c + means
        transformed_image = np.clip(transformed_image, 0, 1) * 255

        return transformed_image.astype(np.uint8)

    def _inject_pixelate(self, image, severity: int = 1):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

        # đảm bảo là PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))

        w, h = image.size  # (width, height)

        new_w = max(1, int(w * c))
        new_h = max(1, int(h * c))

        # downscale
        image_small = image.resize((new_w, new_h), Image.BOX)

        # upscale
        image_pixelated = image_small.resize((w, h), Image.BOX)

        return np.array(image_pixelated, dtype=np.uint8)   

    def _inject_JPEG(self, image, severity:int=1): 
        c = [40, 30, 25, 18, 12][severity - 1]

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        image = image.astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format='JPEG', quality=c)
        transformed_image = np.array(Image.open(buffer))
        return transformed_image  

 