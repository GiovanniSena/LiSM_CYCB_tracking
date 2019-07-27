# Welcome to lightroot

## What is lightroot and how do I used it?

`lightroot` is a project for tracking transient objects in video frames. To use it, clone the project locally. Assuming python and pip are installed, run the following commands to install the dependencies and run the code to process a given frame data directory. From the directory `LSFM_CYCB_analysis_v2`,
~~~~
pip install -r requirements.txt
python run.py [DATA_DIRECTORY]
~~~~

## From where can I get sample data?
The data directory is a folder containing TIFF files. Sample data can be downloaded from [here](https://imperialcollegelondon.box.com/v/Amarteifio2019). Download the data to [DATA_DIRECTORY]. For example, within the project directory `LSFM_CYCB_analysis_v2`, create a folder called `samples` and download and extract the data to 'samples/Cut_Root_192. _This assumes there is a folder called `Cut_Root_192` containing TIFF files at the top level_. Then run
~~~~
python run.py ./samples/Cut_Root_192
~~~~
_Processing 3D images can be slow and you might want to try out the end-end process on a folder containing just a few frames!_

If you are downloading individual frames from the sample data repository, be sure to check the file names for contiguous files for sensible results!
## How do I interpret the output?
Data are dumped to the folder `./cached_datax`
- `data.csv`: contains the locations of tracked objects with a `key` column giving the object identifier
- `life_matrix.csv`: transforms the data into time on the x-axis and the object id on the y-axis and shows if an object was observed at a moment in time
- `life_matrix_restricted.csv`: as above but restricted to objects that were observed for more than one frame only

There are various other logs and details output. The ones listed are the important ones.