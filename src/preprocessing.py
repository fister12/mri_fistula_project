import SimpleITK as sitk
import numpy as np
from pathlib import Path

def load_all_series_sitk(dicom_dir):

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    print(f"Found {len(series_ids)} series")

    images = []

    for i, sid in enumerate(series_ids):
        print(f"\nLoading series {i}")
        file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)
        reader.SetFileNames(file_names)
        img = reader.Execute()

        print("  Size:", img.GetSize())
        print("  Spacing:", img.GetSpacing())

        images.append((i, img))

    return images

def resample_image(image,
                   new_spacing=(1.0, 1.0, 1.0),
                   interpolator=sitk.sitkBSpline,
                   default_value=0):

    image = sitk.Cast(image, sitk.sitkFloat32)

    original_spacing = np.array(image.GetSpacing())
    new_spacing = np.array(new_spacing)

    shrink_factor = new_spacing / original_spacing  #check for downsampling(gaussian lpf)

    if np.any(shrink_factor > 1):
        sigma = [
            max((new_spacing[i] - original_spacing[i]) / 2.0, 1e-6)
            if shrink_factor[i] > 1 else 1e-6
            for i in range(3)
        ] #gaussian sigma for standard deviation 
        image = sitk.SmoothingRecursiveGaussian(image, sigma)

    original_size = np.array(image.GetSize(), dtype=np.int32)

    new_size = [
        int(round((original_size[i] - 1) * original_spacing[i] / new_spacing[i])) + 1
        for i in range(3)
    ]

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetOutputPixelType(image.GetPixelID())

    return resampler.Execute(image)



#Biascorrection
def bias_correct(image, shrink_factor=4):

    print(f"Bias correction (shrink factor = {shrink_factor})")

    image = sitk.Cast(image, sitk.sitkFloat32)

    mask = sitk.OtsuThreshold(image, 0, 1, 200) #seperate foreground and background
    mask = sitk.BinaryFillhole(mask) #solid and continous mask

    image_small = sitk.Shrink(image, [shrink_factor]*3) #downsampling

    mask_small = sitk.Resample(
        mask,
        image_small,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID()
    ) #matching same grid as shrunk image

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50,50,50,50]) #no of iteration at each resolution level

    print("Estimate bias field")
    corrector.Execute(image_small, mask_small)

    log_bias = corrector.GetLogBiasFieldAsImage(image)

    corrected = sitk.Divide(image, sitk.Exp(log_bias)) #converting log bias to normal bias

    print("Bias correction done")
    return corrected



#denoise 
def denoise(image, method="nlm", shrink_factor=4): #nlm model takes time , use curvature for faster result

    image = sitk.Cast(image, sitk.sitkFloat32)

    if method == "nlm":

        print(f"NLM denoising, shrink factor = {shrink_factor}")

        #body mask
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        mask = sitk.BinaryFillhole(mask)

        #downsampling
        image_small = sitk.Shrink(image, [shrink_factor]*3)

        #resampling to smaller image grid
        mask_small = sitk.Resample(
            mask,
            image_small,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )
        image_small_masked = sitk.Mask(image_small, mask_small)
        f = sitk.PatchBasedDenoisingImageFilter()
        f.SetKernelBandwidthSigma(1.0)
        f.SetPatchRadius(2)
        f.SetNumberOfIterations(3)

        denoised_small = f.Execute(image_small_masked)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform())

        denoised_full = resampler.Execute(denoised_small)

        result = sitk.Mask(denoised_full, mask) + sitk.Mask(image, sitk.Not(mask))

        return result
    
    if method == "curvature":
        f = sitk.CurvatureFlowImageFilter()
        f.SetNumberOfIterations(5)
        f.SetTimeStep(0.125)
        return f.Execute(image)

    return image


# Z SCORE NORMALIZATION
def zscore(image):

    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)

    mean = stats.GetMean()
    std = stats.GetSigma() + 1e-6

    arr = sitk.GetArrayFromImage(image)
    arr = (arr - mean) / std  

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out

# PERCENTILE CLIP
def percentile_clip(image, low=1, high=99):

    arr = sitk.GetArrayFromImage(image)
    l, h = np.percentile(arr, [low, high])
    arr = np.clip(arr, l, h)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out

#Preprocessing Pipeline
def preprocess(image):

    print("Reorient to RAI")
    image = sitk.DICOMOrient(image, "RAI")

    print("Resampling")
    image = resample_image(image, (1,1,1))

    print("Bias correction")
    image = bias_correct(image)

    print("Denoising")
    image = denoise(image)

    print("Percentile clipping")
    image = percentile_clip(image)

    print("Z-score")
    image = zscore(image)

    return image



def save_volume(image, out_path):
    sitk.WriteImage(image, str(out_path))
    print("Saved:", out_path)

def preprocess_patient(dicom_folder, output_root):

    dicom_folder = Path(dicom_folder)
    patient_name = dicom_folder.parent.name

    series_list = load_all_series_sitk(dicom_folder)

    for idx, img in series_list:

        print("\nProcessing series", idx)
        processed = preprocess(img)

        out_dir = Path(output_root) / patient_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"series_{idx}_preprocessed.nii.gz"
        save_volume(processed, out_file)













