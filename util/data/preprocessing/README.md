# Data preparation
To extract images data and meta information in ProTECT/UMICH dataset.            
CT scans stored in dicom format and annotations by clinicians will be extracted and a matlab structure array will be used to store annotated patient's head CT. Fields in the structure array includes: 'brains' (brain images after standard contrast adjustment);  'annots': manual hematoma annotations from clinicians; 'dicomImgs': images in CT value; 'meta': meta data such as pixel spacing and slice thickness. 'Pid', patient id, 'Datatype': the dataset name.     


## How to use?
1. Generate struct arrays for annotated cases   
run preprocess_annotated_images.mlx      
2. Generate struct arrays for unannotated cases   
run runPreprocess.m      

## Preprocessing Steps (`preprocess_annotated_images.mlx`)
1. (Before running script) organize annotated data so that each patient has it's own folder with 
the patient id as the folder name. Within each patient's folder should be 
a subfolder named `DICOM` which contains the individual slices in `DICOM` 
format. The annotated images are not in a subfolder. Currently, only 
annotations in `.tiff` or `.tif` format are allowed
2. Get a list of all dicom file names
3. Get dicom metadata from first dicom file
4. Run `read_dicoms()` (we care mostly about `dicomImgs` and `sliceThickness` outputs).
   1. Get/compute image location/orientation information
   2. Crop image to 512x512
   3. Convert pixel values to Hounsefield Units and perform contrast adjustment (0,140)
   4. Compute slice thickness using info from (1). Not sure why when it is in dicom metadata?
5. Extract annotation mask
6. Rotate images and masks based on `regionprops3()` volume, orientation, and cetroid

## Notes
- Original data directory structure. The original unannotated data for these scripts were in `/nfs/turbo/med-kayvan-lab/Datasets/Polytrauma/mTBI/ProTECT III PUD/Images/` and the seem to be organized by patientID/someID-CT/Series-ID/dicomFileName
- `runReadWriteProTECT.m` uses parallel processesing and must run on Great Lakes. 
- I need to make a job script. looks like in the past they used PBS protocol batch system?
- the parameters of `readWriteProTECT.m` seem to be
  - `startI` - patient ID to start at. 1, 101, 201, 301,... 701 (they did batches of 100)
  - `endI` - patient ID to end at. 100, 200, 300,... 800
  - `fileSize` - it's used at the very end of the script to find indices (all in between `startI` and `endI`) that are mulitples of it (using `mod()`). So this must save the patients into manageable sized file batches?
  - `fold` - this also seems to be used for some sort of batch saving?
  - `pernum` - also used for saving batches.