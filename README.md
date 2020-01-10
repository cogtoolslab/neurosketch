# Relating Visual Production and Recognition of Objects in Human Visual Cortex

This repo contains the code used to produce the results in [Fan et al., Journal of Neuroscience (2019)](https://www.jneurosci.org/content/early/2019/12/23/JNEUROSCI.1843-19.2019). We also provide links to download preprocessed fMRI data used in the main analysis notebooks. 

1. [Abstract](#abstract)
2. [Experimental procedure](#experimental-procedure)
3. [Scanning parameters](#scanning-parameters)
4. [fMRI data preprocessing](#fmri-data-preprocessing)
5. [Input to main analyses](#input-to-main-analyses)
6. [List of main analysis notebooks](#list-of-main-analysis-notebooks)

-----
# Abstract
Drawing is a powerful tool that can be used to convey rich perceptual information about objects in the world. 
What are the neural mechanisms that enable us to produce a recognizable drawing of an object, and how does this visual production experience influence how this object is represented in the brain? 
Here we evaluate the hypothesis that producing and recognizing an object recruit a shared neural representation, such that repeatedly drawing the object can enhance its perceptual discriminability in the brain.
We scanned human participants (N=31; 11 male) using fMRI across three phases of a training study: during training, participants repeatedly drew two objects in an alternating sequence on an MR-compatible tablet; 
before and after training, they viewed these and two other control objects, allowing us to measure the neural representation of each object in visual cortex.
We found that: (1) stimulus-evoked representations of objects in visual cortex are recruited during visually cued production of drawings of these objects, even throughout the period when the object cue is no longer present; (2) the object currently being drawn is prioritized in visual cortex during drawing production, while other repeatedly drawn objects are suppressed; and (3) patterns of connectivity between regions in occipital and parietal cortex supported enhanced decoding of the currently drawn object across the training phase, suggesting a potential neural substrate for learning how to transform perceptual representations into representational actions. 
Taken together, our study provides novel insight into the functional relationship between visual production and recognition in the brain.

# Experimental procedure

- Each participant completed 10 runs in total: 2 (localizer recognition runs) + 2 (pre recognition runs) + 4 (drawing runs) + 2 (post recognition runs)
- There were 4 3D objects used in this study: bed, bench chair, table.
- During recognition runs, participants viewed each object 20 times per run (stimulus duration = 1000ms). 
- During production runs, participants drew two objects in alternating sequence. Each drawing trial lasted 23 TR (each TR = 1.5s).

# Scanning parameters

For full information about scanning parameters, see: `/metadata/fmri_sequence_info`

# fMRI data preprocessing
    Note: ALL OF THE OUTPUTS FROM THIS STAGE HAVE BEEN PROVIDED AND ARE 
    READILY AVAILABLE, SO THESE STEPS ARE NOT NECESSARY

### INPUT:
* DICOM archives for each subject: `data/raw/$SUBJ/raw.tar.gz`
* Regressor files for each object for each recognition run: `data/regressor/$SUBJ/$RUN/$object.txt`
* Regressor files for each object for the drawing runs (same across runs): `data/regressor/$SUBJ/production/$object.txt`  
.nii.gz, .bxh files, and trial metadata stored on mongo (currently hosted on kanefsky)
* Run order files for each subject: `data/run-orders/$SUBJ/run-order.txt`

### RUNNING:
*Note: Follows generally from [neuropipe framework](https://github.com/ntblab/neuropipe)*

* This analysis assumes you have conda, FSL 6.0.1, BXH, Matlab and Freesurfer installed

* To start, create a conda environment using the provided file:
	
	$ conda env create -f environment.yml

* There will be cluster-specific changes required. Specifically, if you open `preprocessing/prototype/link/globals.sh`, 
you'll find the following paths that will need to be updated to reflect the appropriate locations for BXH/XCEDE Tools, ImageMagick, 
BIAC Matlab Tools respectively.

        # add necessary directories to the system path
        export BXH_DIR=/jukebox/ntb/packages/bxh_xcede_tools/bin
        export MAGICK_HOME=/jukebox/ntb/packages/ImageMagick-6.5.9-9
        export BIAC_HOME=/jukebox/ntb/packages/BIAC_matlab/mr

* In `analysis/preprocessing`, there is a script called `scaffold`, which generates a new directory 
in `analysis/preprocessing/subjects/` when run, pulling raw data, run-order files and regressor 
files into their appropraite locations. This script duplicates a folder structure, and copies or links 
a number of analysis scripts contained in the `analysis/preprocessing/prototype` folder.
* The main handler script to run an entire subjects' analysis is `drawing.sh`, which is copied into the
newly created `analysis/preprocessing/subjects/$SUBJ` directory. This script will:
    1. Complete some basic preparation (`prep.sh`), including:
        1. convert data to nifti (`scripts/convert-and-wrap-raw-data.sh`)
        2. run quality assurance (`scripts/qa-wrapped-data.sh`)
        3. reorient data to radiological convention (`scripts/reorient-to-las.sh`)
    2. Create `.fsf` files to feed in to FSL and run low-level feat analyses:
        1. For recognition runs (`scripts/render_fsf_templates_glm4.sh`) 
        2. and production runs (`scripts/render_fsf_templates_draw.sh`)
    3. Skull strip the subjects' T1 scans (`brain_extract_first.sh`)
    4. Start freesurfer's `recon-all` to later develop anatomical ROIs
    5. Run FSL's `feat` for every recognition and production run. These are stored in the directory 
    `analysis/firstlevel/${RUN}.feat`
    6. Run a script that awaits the string "Finished at" in the `report.html` file generated by `feat` 
    for each run (`scripts/wait-for-feat.sh`) before advancing
    7. Renders a template for the second level analysis, which combines across all of the production runs,
    then runs the second level `feat`, which will be stored in the directory `analysis/secondlevel/draw.gfeat`
    8. Converts recognition cope maps (`scripts/convert_copes.sh`) and filtered functional data (`scripts/flirt_filt.sh`)
    9. Wait for confirmation that freesurfer has finished (`scripts/wait-for-surf.sh`)
    10. Creates ROI masks based on freesurfer results (`scripts/surfROI.sh`)
    11. Creates feature matrices and metadata for recognition (`scripts.recog_features.py`) and production 
    (`scripts/draw_features.py`) runs
* Within `analysis/preprocessing`, one can batch `scaffold` and subsequently batch run `drawing.sh`
in all subjects, using the following steps:

        $ cd analysis/preprocessing
        $ sbatch scripts/scaffold_all.sh
        
        # wait until finished
        
        $ sbatch scripts/analyze_all.sh
        
* To check in on which analysis stage was most recently completed for each of the subjects' analyses, 
one can run:

        $ bash scripts/checkin.sh
        
* Once all subjects are completed, the next step is to run group-level production run analyses. This 
next script will create a production run univariate task mask for each held out subject, derived from all
of the remaining subjects' lower level analyses. With these individualized task masks, the intersect with
subjects' anatomical ROIs are computed and connectivity features generated. To accomplish this, one can run:

        $  sbatch scripts/lv1out.sh
        
* If all has run successfully, all of the following output should be present.

<!---
Motion correction
Projection of filtered_func into anatomical space to yield 4D timeseries
ROIs:
Freesurfer used to derive ROIs from each participants’ T1
Univariate comparison (drawing vs. not drawing) used to make task mask
ROIs with ‘draw’ suffix are intersects of ROI masks and draw task mask
All in anatomical space
-->

### OUTPUT: 
	For each subject and phase (e.g. 12, 34, or 56):
* Voxel matrices (m timepoints x n voxels) saved as .npy for each region of interest (ROI).
    * e.g. `data/features/recognition/$SUBJ_$ROI_$PHASE_featurematrix.npy`
    * or `data/features/production/$SUBJ_$ROI_$PHASE_featurematrix.npy`
* Connectivity matrices (m trials x n correlations, where n is the product of # voxels in ROI1 and # voxels in ROI2), saved as .npy
    * e.g. `data/features/connectivity/$SUBJ_$ROI1_$ROI2_$PHASE_featurematrix.npy`
* Metadata (m timepoints x 4 columns `[subj,label,run_num,TR_num]`) saved as .csv
    * e.g. `data/features/recognition/metadata_$SUBJ_$ROI_$PHASE.csv`
    * or `data/features/production/metadata_$SUBJ_$ROI_$PHASE.csv`
    * or `data/features/connectivity/metadata_$SUBJ_$PHASE.csv`
* ROI Masks for each ROI saved as niftis
    * For those derived from freesurfer: `rois/freesurfer/$SUBJ_ROI.nii.gz`
    * For those derived from production univariate task masks `rois/production/$SUBJ_draw_task_mask.nii.gz`
    * For those derived from intersect between freesurfer and univariate masks: `rois/intersect/$SUBJ_${ROI}Draw.nii.gz`

<!---Note: before 4/23/18, canonical voxel matrices + metadata were in `neurosketch/data/neurosketch_voxelmat3mm_freesurfer_drawing`, renamed to `neurosketch/data/features/drawing`. 

    Note: path on jukebox (accessible via Spock) is: `/jukebox/ntb/projects/sketchloop02/data`
-->

# Input to main analyses

There is a master DATA dir containing all processed fMRI data in standard format:
* `DATA/features` contains all feature matrices and metadata from all phases, ROIs, and subjects.
	* `DATA/features/drawing` includes drawing filtered func data (represented as .npy feature matrices) + corresponding metadata from full freesurfer anatomical ROIs.
	* Also includes drawing filtered func data + metadata from freesurfer ROIs intersected with drawing taskGLM group-level cope map
	* `DATA/features/recog` contains recognition filtered func data (represented as .npy feature matrices) + corresponding metadata
* `DATA/copes` contains all cope NIFTI files from GLM fit to recognition and drawing runs for all subjects.
	* `DATA/copes/recog/objectGLM` recognition cope maps from object GLM
	* `DATA/copes/draw/taskGLM` drawing cope maps from drawing_task GLM

**Link to download data will be provided here soon.**

# List of main analysis notebooks

Link to [analysis notebooks](https://github.com/cogtoolslab/neurosketch/tree/master/analysis/notebooks).

1. [Measure Object Evidence on Recognition Run Data in Visual Cortex](https://github.com/cogtoolslab/neurosketch/blob/master/analysis/notebooks/1_object_evidence_during_recognition.ipynb)

2. [Measure Object Evidence on Drawing Run Data in Visual Cortex](https://github.com/cogtoolslab/neurosketch/blob/master/analysis/notebooks/2_object_evidence_during_production.ipynb)

3. [Measure Patterns of Connectivity Between Early Visual and Parietal Regions](https://github.com/cogtoolslab/neurosketch/blob/master/analysis/notebooks/3_connectivity_pattern_during_drawing.ipynb)

