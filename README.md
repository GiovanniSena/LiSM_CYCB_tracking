# Welcome to lightroot

## What is lightroot?

`lightroot` is a project for tracking transient objects in video frames. To use it locally from this project, clone the project and assuming python and pip are installed run the following commands to install the dependencies and run the code to process a frame data directory
~~~~
pip install -r requirements.txt
~~~~
And the from the project base directory
~~~~
python run.py [DATA_DIRECTORY]
~~~~
The data directory is a folder containing TIFF files. Sample data can be downloaded [here](https://imperialcollegelondon.box.com/v/Amarteifio2019). Donwload the data to [DATA_DIRECTORY]. For example, within the project directory, create a folder called 'samples' and download and exctract the data to 'samples/Cut_Root_192. Then run
~~~~
python run.py ./samples/Cut_Root_192
~~~~
Processing 3D images can be slow and you might want to try out the end-end process on just a few frames. For example to test against the first 10 frames run
~~~~
python run.py ./samples/Cut_Root_192 10
~~~~
