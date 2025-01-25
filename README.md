For my challenge, I used two main approaches. 

First, I did a non ML method using OpenCV and hough circle detention. This was motivated by the small amount of available training data. After finishing this, I wanted to try working with more advanced AI techniques, so I wrote a solution using a Mask R-CNN model.

Hough Circle Detection:
Location: Jupyter Notebook file sean_data_challenge.ipynb. 
To run this, install the dependencies listed in the environment file dali.yml. This environment can be recreated using conda. Install conda, download the dali.yml file to downloads folder, then in terminal use the following commands. (NOTE: These instructions are for MacOS. May be different on Windows. Ensure Terminal has Full Disk Access -> System Preference->Privacy and Security->Full Desk Access)

conda env create --name data_env -f Downloads/dali.yml

conda activate data_env

jupyter notebook

This will allow you to navigate to the sean_data_challenge.ipynb file and run it.

This approach is an end-to-end process that processes images, identifies barnacles, allows for editing barnacle images, and uses the output to do some analysis. I factored most classes and functions out into barnacle.py. Other folders are used to save images and outputs.

Mask R-CNN:
Location: Google Colab file barnacle_detection.ipynb
To run this, use Colab’s T4 GPU. The code also requires the user to upload “barnacle_resized.zip,” which is saved in the demo folder. All dependencies will be installed through the notebook.

This approach uses Detectron2, a Facebook library based on Mask R-CNN. It uses a dataset augmented with Roboflow and includes hyperparameter tuning with Optuna.

Further thoughts and explanations are within these two notebooks!
