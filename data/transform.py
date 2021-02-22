# Naming Schema: NN_AA, where NN Patient number and AA sample number
# Every Patient has 64 samples

import pydicom
import numpy as np
import png
from glob import glob


def convert_dicom_to_png(source, target):
    ds = pydicom.dcmread(source)
    # if 'WindowWidth' in ds:
    #    print('Dataset has windowing')
    #    windowed = apply_voi_lut(ds.pixel_array, ds)

    shape = ds.pixel_array.shape
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    # Write the PNG file
    with open(target, 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)


# Samples
patient_number = 1
patients = sorted(glob('siemens_source/Patient*'), key=lambda x: int(x.split('siemens_source\\Patient ')[1]))
for patient in patients:
    sample_number = 1
    images = sorted(glob(f'{patient}/T2W/*.dcm'), key=lambda x: int(x.split(f'{patient}/T2W\\image')[1].split('.dcm')[0]))
    for image in images:
        convert_dicom_to_png(image, f'siemens_full/samples/{patient_number}_{sample_number}.png')
        sample_number += 1
    patient_number += 1

# Labels
patient_number = 1
patients = sorted(glob('siemens_source/Patient*'), key=lambda x: int(x.split('siemens_source\\Patient ')[1]))
for patient in patients:
    sample_number = 1
    images = sorted(glob(f'{patient}/T2W/*.dcm'), key=lambda x: int(x.split(f'{patient}/T2W\\image')[1].split('.dcm')[0]))
    indices = list(map(lambda x: int(x.split(f'{patient}')[1].split('/T2W\\image')[1].split('.dcm')[0]), images))
    for index in indices:
        for clazz in ['cap', 'cg', 'prostate', 'pz']:
            cap_s = f'{patient}/GT/{clazz}/image{index}.dcm'
            cap_d = f'siemens_full/labels/{clazz}/{patient_number}_{sample_number}.png'
            convert_dicom_to_png(cap_s, cap_d)
        sample_number += 1
    patient_number += 1
