# Prompt Log

1. Prompt 1: I would like to build this well designed publishable notebook showing the tests comparing out the box pretrained deeplabcut for our workflow of (X-Ray Of Moving Morphology). You will help me write this file with proper documentsation following FAIR Guidelines. I have layed out the introguidelines, with most of the other ipynb notebooks holding the logic I want implemented into this study.

I want to first split all the data in C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCutExperiments\Data into training and testing. which will require each {Bird}_{Trial} Data folder to contain inside a test and train folder that chooses 300 random frames for testing.

to do this you will want to select 300 frames from the available photos using random seed 42 and pull each image at the 300 indicies as well as each corresponding row of the labeledBodyPartsCoordinates.csv. these will be saved in the same folder structure as the CompleteSet but under training data folder. Keep a list of those indicies in a metadata file in the same folder so we can leverage this information later and compare across trials.

I Will be adding new data folders as they come in so keep that in mind while you are designing this experimental workflow.

2. Prompt 2: Also Document all my prompts in a readme file word for word with your only addition being the number of prompt. Save that in DeepLabCutExperiments folder.

3. Prompt 3: The test split looks good. make sure you number each prompt i make. Now we will generate 3 training sets(to input to the model) for each bird_trial from the now constructed training data set. therefore we will have three folders named random_train, displacement_train, dino_train.

The first will come from the Deeplabcut generate training dataset. This requires direct input into the #sym:train_update_model (which you should create a new version of this code in a new python folder in this notebook)  I have created. follow the logic of train_update_model until the dlcs.create_and_train. I want to seperate dataset creation and training. 
So we should input bird name and trial number found in the parent to each data folder {bird}_T{trialnum} ( DB = DavidBowie, Tulio = Tulio) epochs will be fixed at 125(used for all training sets). we should take two number of frames: 100 and 50 (these values will be used for all training data sets)
updateset = Random

For the second, we should focus on point distance selection method I developed here along with its helper functions from postanlaysis_review.py #sym:_build_displacement_selection . the zone size used for point distance calculation and frame selection  should be 150 rather than 400.

third is the dino method which follows this logic and associated:#sym:_compute_dino_embeddings  for maximization of image embedding differences.

for both displacement and dino, then run it through the same beginning of train

add this to readme, maybe that requires a rule so you dont forget

4. yes, since you didnt install deeplab cut or torch/torchvision we could not generate dino frames of model infrastructure. Do so now, and remember to apply bird config bodypart adjustments as I have done before.  


5. Prompt 4: ok now the data has been prepared. I want to do a comparison of the data that has been selected. Using the logic and visualization of cell 67 from  test_updated_experiments(FrameSelectionForUpdate) i want to visualize the embeddings of all data images in a dark color with the frames selected by each training module in unique bright colors along with the testing data (that way we can visualize the splits of data being utilized in different components of the modeling process.)

6. Prompt 5: add a cell that makes this run across all trials and plots that data. The dataset_build_metadata for dino models did not update so it reads that there was an error. Make sure these are updated.

7. Prompt 6: I will work on the visualization myself as time goes, but now lets focus on running the training of all these models. Run it through a loop leveraging dlcs.create_and_train. Please inform me if there is a way to parallelize/optimize this training process. Currently it takes a significant amount of time.

