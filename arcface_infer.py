import math
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms as trans

from .model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from pathlib import Path

THIS_DIR=Path(__file__).parent.absolute()

class Arcface(object):
    #TODO: build arcface for batch processing
    def __init__(self, model_path = None, use_gpu=True, tta=True):
        '''
        model_path : path to model
        use_gpu : use gpu or cpu
        tta : test time augmentation (hfilp, that's all)
        '''
        self.tta = tta
        if use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        net_depth = 50
        drop_ratio = 0.6
        net_mode = 'ir_se' # or 'ir'
        if model_path is None:
            model_path = THIS_DIR / 'model_ir_se50.pth'
        assert model_path.is_file(),'Model not found at {}'.format(model_path)
        
        self.model = Backbone(net_depth, drop_ratio, net_mode).to(self.device)
        self.model.load_state_dict(torch.load(str(model_path)))
        self.model.eval()
        print('{}_{} FR model (arcface) generated'.format(net_mode, net_depth))

        input_size = [112,112]
        self.preproc_transform = trans.Compose([
                        trans.Resize(input_size),
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                                ])
        print('warming up..')
        import time
        tic = time.time()
        # zeros = np.zeros((20,20,3), dtype='uint8')
        warmup_image = self.preproc_transform(Image.fromarray(np.zeros((20,20,3), dtype='uint8'))).to(self.device).unsqueeze(0)
        self.model.forward(warmup_image)
        toc = time.time()
        print('warmed up! {:0.3f}s'.format(toc - tic))

    def get_embeds(self, faces):
        '''
        passthrough function to fit legacy api, TODO implement a proper batch inference 
        faces : list of ndarray (BGR channels)
        '''
        return self.embed(faces)

    def embed(self, faces):
        '''
        faces : list of ndarray (BGR channels)
        '''
        embs = np.zeros((len(faces), 512))
        for i, img in enumerate(faces):
            img = Image.fromarray(img)
            if self.tta:
                mirror = trans.functional.hflip(img)
                input_ = self.preproc_transform(img).to(self.device).unsqueeze(0)
                input_mirror = self.preproc_transform(mirror).to(self.device).unsqueeze(0)
                emb = self.model(input_)
                emb_mirror = self.model(input_mirror)
                emb = l2_norm(emb + emb_mirror).cpu().data.numpy()
            else:              
                input_ = self.preproc_transform(img).to(self.device).unsqueeze(0)
                emb = self.model(input_).cpu().data.numpy()
            embs[i, ] = emb
        return embs

    # def infer(self, conf, faces, target_embs, tta=True):
    #     '''
    #     faces : list of PIL Image
    #     target_embs : [n, 512] computed embeddings of faces in facebank
    #     names : recorded names of faces in facebank
    #     tta : test time augmentation (hfilp, that's all)
    #     '''
    #     embs = []
    #     for img in faces:
    #         if tta:
    #             mirror = trans.functional.hflip(img)
    #             emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
    #             emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
    #             embs.append(l2_norm(emb + emb_mirror))
    #         else:                        
    #             embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
    #     source_embs = torch.cat(embs)
        
    #     diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
    #     dist = torch.sum(torch.pow(diff, 2), dim=1)
    #     minimum, min_idx = torch.min(dist, dim=1)
    #     min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
    #     return min_idx, minimum               