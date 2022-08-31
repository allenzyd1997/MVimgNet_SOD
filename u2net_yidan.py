"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib
from typing import List, Union
import PIL.Image
import numpy as np
import torch
from PIL import Image

from carvekit.ml.arch.u2net.u2net import U2NETArchitecture, _upsample_like, _size_map
from carvekit.ml.files.models_loc import u2net_full_pretrained
from carvekit.utils.image_utils import load_image, convert_image
from carvekit.utils.pool_utils import thread_pool_processing, batch_generator

__all__ = ["U2NET"]


class U2NET(U2NETArchitecture):
    """U^2-Net model interface"""

    def __init__(self,
                 layers_cfg="full",
                 device='cpu',
                 input_image_size: Union[List[int], int] = 320,
                 batch_size: int = 30,
                 load_pretrained: bool = True):
        """
            Initialize the U2NET model

            Args:
                layers_cfg: neural network layers configuration
                device: processing device
                input_image_size: input image size
                batch_size: the number of images that the neural network processes in one run
                load_pretrained: loading pretrained model

        """
        super(U2NET, self).__init__(cfg_type=layers_cfg, out_ch=1)
        self.device = device
        self.batch_size = batch_size
        if isinstance(input_image_size, list):
            self.input_image_size = input_image_size[:2]
        else:
            self.input_image_size = (input_image_size, input_image_size)
        self.to(device)
        if load_pretrained:
            self.load_state_dict(torch.load(u2net_full_pretrained(), map_location=self.device))
        self.train()

    def data_preprocessing(self, data: PIL.Image.Image) -> torch.FloatTensor:
        """
            Transform input image to suitable data format for neural network

            Args:
                data: input image

            Returns:
                input for neural network

        """
        resized = data.resize(self.input_image_size, resample=3)
        # noinspection PyTypeChecker
        resized_arr = np.array(resized, dtype=float)
        temp_image = np.zeros((resized_arr.shape[0], resized_arr.shape[1], 3))
        if np.max(resized_arr) != 0:
            resized_arr /= np.max(resized_arr)
        temp_image[:, :, 0] = (resized_arr[:, :, 0] - 0.485) / 0.229
        temp_image[:, :, 1] = (resized_arr[:, :, 1] - 0.456) / 0.224
        temp_image[:, :, 2] = (resized_arr[:, :, 2] - 0.406) / 0.225
        temp_image = temp_image.transpose((2, 0, 1))
        temp_image = np.expand_dims(temp_image, 0)
        return torch.from_numpy(temp_image).type(torch.FloatTensor)

    @staticmethod
    def data_postprocessing(data: torch.tensor,
                            original_image: PIL.Image.Image) -> PIL.Image.Image:
        """
            Transforms output data from neural network to suitable data
            format for using with other components of this framework.

            Args:
                data: output data from neural network
                original_image: input image which was used for predicted data

            Returns:
                Segmentation mask as PIL Image instance

        """
        data = data.unsqueeze(0)
        mask = data[:, 0, :, :]
        ma = torch.max(mask)  # Normalizes prediction
        mi = torch.min(mask)
        predict = ((mask - mi) / (ma - mi)).squeeze()
        predict_np = predict.cpu().data.numpy() * 255
        mask = Image.fromarray(predict_np).convert("L")
        mask = mask.resize(original_image.size, resample=3)
        return mask

    def __call__(self, images: List[Union[str, pathlib.Path, PIL.Image.Image]]) -> List[PIL.Image.Image]:
        """
            Passes input images though neural network and returns segmentation masks as PIL.Image.Image instances

            Args:
                images: input images

            Returns:
                segmentation masks as for input images, as PIL.Image.Image instances

        """
        collect_masks = []
        image_batch = images
        images = thread_pool_processing(lambda x: convert_image(load_image(x)), image_batch)
        batches = torch.vstack(thread_pool_processing(self.data_preprocessing, images))
        # with torch.no_grad():
        batches = batches.to(self.device)
        # masks, d2, d3, d4, d5, d6, d7 = super(U2NET, self).__call__(batches)
        # masks size (batchsize, 1,320,320)
        # masks_cpu = masks.cpu()
        # del d2, d3, d4, d5, d6, d7, batches, masks
        # masks = thread_pool_processing(lambda x: self.data_postprocessing(masks_cpu[x], images[x]), range(len(images)))
        # collect_masks += masks
        # return collect_masks
        
        return  super(U2NET, self).__call__(batches)


    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f'stage{height}')(x)
                x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                x = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
                side(x, height)
                return _upsample_like(x, sizes[height - 1]) if height > 1 else x
            else:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps


if __name__ == "__main__":
    import os 
    import torch 
    import numpy as np 

    from pathlib import Path 
    from PIL import Image

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u2 = U2NET(device=device,batch_size=1)

    input_pth = '/data/zhangyidan/mask_mvImgNet/imgs/images0/001.jpg'
    output_pth = '/data/zhangyidan/mask_mvImgNet/imgs/test_out/'
    output_pth = os.path.join(output_pth, "a.png")
    # input_pth = Path(input_pth)
    img = Image.open(input_pth)
    out_img = u2([img])
    print('end')


    
