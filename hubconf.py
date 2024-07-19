dependencies = ['torch', 'torchvision']

from perspective2d import PerspectiveFields


version = 'Paramnet-360Cities-edina-centered'
# version = 'Paramnet-360Cities-edina-uncentered'
# version = 'PersNet_Paramnet-GSV-centered'
# version = 'PersNet_Paramnet-GSV-uncentered'
# version = 'PersNet-360Cities'

def perspective_fields(pretrain=True, **kwargs):
    '''
    Return a PerspectiveFields model.
    For usage examples, refer to:
    '''
    version = kwargs.get('version', 'Paramnet-360Cities-edina-centered')
    pf_model = PerspectiveFields(version).eval().cuda()
    return pf_model


'''Usage:
import cv2
import torch
model = torch.hub.load('mkocabas/PerspectiveFields', "perspective_fields", pretrained=True)

bgr = cv2.imread(img_path)
pred = model.inference(bgr)
print(f"roll: {pred['pred_roll'].item()}\npitch: {pred['pred_pitch'].item()}\nvfov: {pred['pred_vfov'].item()}\ncx: {pred['pred_rel_cx'].item()}\ncy: {pred['pred_rel_cy'].item()}")
'''