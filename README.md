# Short description

The goal of this project is to cluster letter images provided via the Moodle platform.

The notebook magdalena_czapiewska_zal1.ipynb contains the solution to the first assignment (clustering letters).
It is divided into 6 sections:

1. Creating execution environment
2. Custom options adjustment
3. Data preparation
4. Features extraction
5. Clustering algorithm
6. Evaluation (this section does not execute, it just presents the experiments I have made to choose the best custom options)

The pipeline begins with resizing images so each of them is of the same size. Then features are extracted using DAISY method (Histogram of oriented gradients method was also considered, but not chosen). Feature vectors are clustered using Agglomerative clustering with linkage method `ward` and `euclidean` distance metric.

In each section I describe in detail the algorithms used. Custom options can be changed in the `Custom options adjustment` section (paths to input and output files can be changed here if they differ from the assignment description).

I provide the files ground_truth_clusters.txt and clean_clusters.html with clusters that were constructed by manually changing results of my imperfect clustering for the data in the training_samples folder which was provided on Moodle. I used this ground truth clustering to evaluate my solution and choose hyperparameters.

# How to run

#### Step 1. Environment setup
I have run the notebook magdalena_czapiewska_zal1.ipynb using jupyter-notebook. To successfully run the notebook, please run the code in the `Creating execution environment` section. Then a new kernel called `magdalena_czapiewska_sus_1_env` will be added to jupyter-notebook.

#### Step 2. Switching the kernel
Please select from the menu `Kernel -> Change Kernel -> magdalena_czapiewska_sus_1_env`.

#### Step 3. Execution
Please run all cells starting from the first cell in the `Custom options adjustment` section (`Run -> Run Selected Cell and All Below`). If it does not work, please execute each cell independently (time of execution is less than 5 minutes for the whole notebook after kernel choice).

#### Step 4. In case of failure of kernel registration / activation
If you encounter any problems with activating a kernel, please use an alternative option. I provided a python script magdalena_czapiewska_zal1.py that has the same functionality as this notebook (custom options are constants at the beginning of the script, so paths to input and output files can be changed there if needed). To execute it, please create conda virtual environment:

conda create -y -n magdalena_czapiewska_sus_1_conda -c conda-forge python=3.10 "numpy<2" "opencv<4.10" matplotlib scikit-learn scikit-image

conda activate magdalena_czapiewska_sus_1_conda

python magdalena_czapiewska_zal1.py

#### Time of execution for 7620 images

For python script in conda environment:

real	1m17,799s
user	1m13,336s
sys	0m1,381s

For ipynb file:

Imports:
CPU times: user 1.41 s, sys: 203 ms, total: 1.61 s
Wall time: 1.16 s

Loading images:
CPU times: user 580 ms, sys: 186 ms, total: 766 ms
Wall time: 774 ms

DAISY feature extraction:
CPU times: user 1min 11s, sys: 11.2 ms, total: 1min 11s
Wall time: 1min 11s

Clustering:
CPU times: user 19.3 s, sys: 158 ms, total: 19.5 s
Wall time: 19.5 s
