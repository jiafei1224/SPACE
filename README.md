# SPACE: A Simulator for Physical Interactions and Causal Learning in 3D Environments

![newsetup (1)](https://user-images.githubusercontent.com/51585075/126741271-45b0f2df-03ef-49c1-aab9-ad6bc505a1e7.jpg)

## SPACE Simulator 
Run the Blender Code for each of the physical interactions tasks from the BlenderGeneration file using the Blender Python API. Please ensure that the object files is in the same file directory as the `.blend` file.

## SPACE Dataset
Download the respective dataset along with the annotation from here: 

[[Contact]](https://drive.google.com/drive/folders/1nb8e63H78-FjF_ErxrtWfa0fCrvlGjPP?usp=sharing) 

[[Containment]](https://drive.google.com/drive/folders/1-wOgkW69odhein5RSQd1ObI9emULoHG8?usp=sharing)

[[Stability]](https://drive.google.com/drive/folders/1TrbHI0hV8tyLSfppJkQJrIV1zvzisgJl?usp=sharing)

You may download down and put it into your own drive,then you can `gdown <Drive Url to upload to server>` 

## Dataset Statistics
Dataset statistics are obtained from running `python ./util/get_stats.py`.

## System Requirements
- Blender 2.83 with eevee engine
- **Python 3.6**
- External Python libraries in `PhyDNet_Add_On/requirement.txt`

## Creating Images from Data
Run `./util/replay_and_save_frames.py` file using frames from `./dataset/contact/video` and save it to `./dataset/contactframes`.

## For evaluation results via PhyDNet, please view the PhyDNet_Add_On for instruction



