reproducingcodes
================

Project Repo for Reproducing Codes Class 
Joe Ellis PhD Candidate
Columbia Univeristy - DVMM Lab

Paper
===========
D. Parikh and K. Grauman,
Relative Attributes,
International Conference on Computer Vision (ICCV), 2011.

CODE
====

################Running the Code############
The two files that are used to reproduce the results of this work are the matlab scripts 
titled osr_script and pubfig_script.  These files are simply run in the matlab command line 
by typing osr_script or pubfig_script.  To run these files please move into either the osr or pubfig directory
in tha matlab command line then type the commands provided.
When the code is run we will randomly choose matches between the images and then 10 iterations will run, and the accuracy will be found then averaged over 
the 10 iterations and output to the screen.  Other output to the screen for debug purposes can be seen.

*Some people have asked for the ability to automatically create each figure from the paper, but I do not want to put this functionality into this version of the code.
Every time either  osr_script or pubfig_script is run corresponds to one point within one of the plots presented in the paper, by manually creating each data point the 
person using this code is able to get a feel for how each parameter changes the run time, and what is happening in the code.  This is imporant to me, and therefore this
implementation contains only the ability to recreate a single data point from one of the available plots.

######Changing the parameters##############
To recreate the graphs that are present within the above paper you will need to change the parameters that are 
given within the beginning of each script.  The parameters that need to be changed to create the graphs are pasted below.
Each attribute can be changed to the specifications stated within the paper to create each data point. 

Note: The variable held_out_attributes corresponds to the # of attributes used plots that are presented in "Relative Attributes".
Therefore if the number of used attributes desired is 5, and the total number of attributes for a task is 6, then held_out_attributes should be 1 (6 - 1 = 5). 
	
% These are the num of unseen classes and training images per class
num_unseen = 2;
trainpics = 30;
num_iter = 10;
held_out_attributes = 0;
labeled_pairs = 4;
looseness_constraint = 1;


DATA
=====
This folder contains the folders that have the data that is to be used.
The images used can be seen in pubfig and osr.
The data from this folder has been preprocessed for your convenience and is loaded 
by running osr_script or pubfig_script.


License
======
The MIT License (MIT)

Copyright (c) 2012, Joseph G. Ellis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

* The original license applied to the OSR and PubFig Datasets still applies

Citation
========
If you use this code in any research please cite
*Ellis, Joseph G. "Implementation of Relative Attributes", Web Resource, https://github.com/jellis505, 2013*
