import math
import torch
from pathlib import Path
from torchvision import transforms as trans

from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm

class Arcface(object):
    def __init__(self, conf):
        print(conf)
        # if conf.use_mobilfacenet:
        #     self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
        #     print('MobileFaceNet model generated')
        # else:
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        self.threshold = conf.threshold
        model_path = conf.model_path
        assert Path(model_path).is_file()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def embed(self, conf, faces, tta=True):
        '''
        faces : list of PIL Image
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
                # embs.append(self.model(conf.test_transform(img).to(conf.device)))
        source_embs = torch.cat(embs)
        return source_embs.cpu().data.numpy()
        
    def infer(self, conf, faces, target_embs, tta=True):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               