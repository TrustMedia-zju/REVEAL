

_BASE = '/mnt/shenzhen2cephfs/mm-base-vision/sanmucao/dataset/Chameleon'
_SECTIONS = (
    'dalle', 'glide_50_27', 
    'glide_100_10','glide_100_27',
     'guided', 'ldm_100',
    'test',
)

IMAGE_FOLDERS = []
for sec in _SECTIONS:
    IMAGE_FOLDERS.append(f'{_BASE}/{sec}/1_fake')
    IMAGE_FOLDERS.append(f'{_BASE}/{sec}/0_real')