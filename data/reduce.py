import glob
import numpy as np
import imageio
import matplotlib.pyplot as plt
import PIL.Image

cap = 'GE-15_full/labels/cap'
cg = 'GE-15_full/labels/cg'
prostate = 'GE-15_full/labels/prostate'
pz = 'GE-15_full/labels/pz'
sample = 'GE-15_full/samples'

cap_d = 'GE-15_reduced/labels/cap'
cg_d = 'GE-15_reduced/labels/cg'
prostate_d = 'GE-15_reduced/labels/prostate'
pz_d = 'GE-15_reduced/labels/pz'
sample_d = 'GE-15_reduced/samples'

sorted_patients = sorted(glob.glob('GE-15_full/samples/*.png'), key=lambda x: list(int(i) for i in x.split('GE-15_full/samples\\')[1].split('.png')[0].split('_')))
names = [i.split('GE-15_full/samples\\')[1] for i in sorted_patients]

i=0
for name in names:
    print(name)
    cap_img = np.array(imageio.imread(f'{cap}/{name}'))
    cg_img = np.array(imageio.imread(f'{cg}/{name}'))
    prostate_img = np.array(imageio.imread(f'{prostate}/{name}'))
    pz_img = np.array(imageio.imread(f'{pz}/{name}'))
    combined_labels = np.concatenate((cap_img.reshape(cap_img.shape[0], cap_img.shape[1], 1),
                                      cg_img.reshape(cg_img.shape[0], cg_img.shape[1], 1),
                                      prostate_img.reshape(prostate_img.shape[0], prostate_img.shape[1], 1),
                                      pz_img.reshape(pz_img.shape[0], pz_img.shape[1], 1)),
                                     axis=2)
    sample_img = np.array(imageio.imread(f'{sample}/{name}'))
    if np.max(combined_labels) > 0:
        PIL.Image.fromarray(cap_img).save(f'{cap_d}/{name}')
        PIL.Image.fromarray(cg_img).save(f'{cg_d}/{name}')
        PIL.Image.fromarray(prostate_img).save(f'{prostate_d}/{name}')
        PIL.Image.fromarray(pz_img).save(f'{pz_d}/{name}')
        PIL.Image.fromarray(sample_img).save(f'{sample_d}/{name}')
