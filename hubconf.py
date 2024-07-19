dependencies = ['torch', 'torchvision', 'pyequilib==0.3.0']

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