This repository contains the code for my third year project on the feasibility of training machine learning models on a BBC micro:bit.  
It contains the code for 4 different programs: a gesture recognition program using a kNN classifier; a voice activity detection program using a logistic regression model; an environment recognition model using a naive Bayes classifier; and a speaker recognition program using a decision tree.  

# Cloning this repository
In order to clone this repository run the command below in a terminal.  
```
git clone --recurse-submodules https://github.com/siclayton/typ.git
```  

# Building 
- Clone this repository
- Select and build one of the four programs
    - Copy the code for one of the programs into the source folder
        - In the root of this repository type `cp models/<program_name>/* source/`
        - Replace <program_name> with the name of the program you wish to build
    - In the root of this repository type `python build.py`
- The hex file will be built `MICROBIT.hex` and placed in the root folder.
