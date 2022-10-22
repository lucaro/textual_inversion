import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

imagenet_templates_small = [

    'a picture in the style of {}',
    'a photo in the style of {}',
    'a photograph in the style of {}',
    'a photographic picture in the style of {}',
    'an image in the style of {}',
    'a portrait in the style of {}',
    'a portrait picture in the style of {}',

    'a picture by {}',
    'a photo by {}',
    'a photograph by {}',
    'a photographic picture by {}',
    'an image by {}',
    'a portrait by {}',
    'a portrait picture by {}',
    
    'a black and white picture by {}',
    'a black and white photo by {}',
    'a black and white photograph by {}',
    'a black and white photographic picture by {}',
    'a black and white image by {}',
    'a black and white portrait by {}',
    'a black and white portrait picture by {}',
    
    'an analogue picture in the style of {}',
    'an analogue photo in the style of {}',
    'an analogue photograph in the style of {}',
    'an analogue photographic picture in the style of {}',
    'an analogue image in the style of {}',
    'an analogue portrait in the style of {}',
    'an analogue portrait picture in the style of {}',
    
    'a picture of a person by {}',
    'a photo of a person by {}',
    'a photograph of a person by {}',
    'a photographic picture of a person by {}',
    'an image of a person by {}',
    'a portrait of a person by {}',
    'a portrait picture of a person by {}',
    
    'a cropped picture in the style of {}',
    'a cropped photo in the style of {}',
    'a cropped photograph in the style of {}',
    'a cropped photographic picture in the style of {}',
    'a cropped image in the style of {}',
    'a cropped portrait in the style of {}',
    'a cropped portrait picture in the style of {}',
    
    'a dark picture by {}',
    'a dark photo by {}',
    'a dark photograph by {}',
    'a dark photographic picture by {}',
    'a dark image by {}',
    'a dark portrait by {}',
    'a dark portrait picture by {}',
    
    'a grainy picture in the style of {}',
    'a grainy photo in the style of {}',
    'a grainy photograph in the style of {}',
    'a grainy photographic picture in the style of {}',
    'a grainy image in the style of {}',
    'a grainy portrait in the style of {}',
    'a grainy portrait picture in the style of {}',
    
    'a close-up picture by {}',
    'a close-up photo by {}',
    'a close-up photograph by {}',
    'a close-up photographic picture by {}',
    'a close-up image by {}',
    'a close-up portrait by {}',
    'a close-up portrait picture by {}',
    
    'a rendition in the style of {}',
    'a piece of art in the style of {}',
    'a piece in the style of {}',
    'a portrait print in the style of {}',
    'a rendition by {}',
    'a piece of art by {}',
    'a piece by {}',
    'a portrait print by {}',

]

imagenet_dual_templates_small = [
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBaseArtist(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.per_image_tokens and np.random.uniform() < 0.25:
            text = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(self.placeholder_token)
            
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example