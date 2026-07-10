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

8. Prompt 7: There was a running error on all DB_T16. Lets solve that and train the models.


9. Prompt 8: let us test the model performance now. We always want to test it on the same bird, and we want to test it within trial and across trial. this means we should use #sym:predict_trial_from_jpg_stacks  where trial_dir is the Test dir for each file and using the config path of each trained model (slection_of x Number_of frames). This saves them all as csvs but perhaps we should not do this and rather just immediately compute the metrics for scoring. Since they are already returned in XMA lab format, we can compare the true Test data vs the predictions simply. We should store one score results df per trial encompassing both within and cross trial predictions. Then I want to save a entire score results.

Score metrics should be drawn from test_updated_experiments.ipynb cell 21 where we check RMSE, percent points within 5 pixels, and percent of frames with all points within 5 pixels, and average number +std of predictions across all frames that are within 5 pixels

10. Prompt 9: make this a cell that I run in my notebook.

11. Prompt 10: there is an error in the #sym:predict_trial_from_jpg_stacks  logic currently as it is pulling more frames than we have. It would be more wise to create a Predictions.py module that reformulates this and any dependent functions to fit it for our current experimentation. Then we can easily run our program for all birds in the notebook without this error.

12. Prompt 11: there is an issue with the predictions not allowing more than once the video to be analyzed

we are getting this error:Analyzing videos with [C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCutExperiments\Data\DB_T17\dino_train\nframes_100\ModelsToTune\Canari-FineTuner-2026-07-09\dlc-models-pytorch\iteration-0\CanariJul9-trainset95shuffle1\train\snapshot-125.pt]
Using scorer: DLC_Resnet50_CanariJul9shuffle1_snapshot_125
Video [C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCutExperiments\Data\DB_T15\test\Cam1.avi] already analyzed at [C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCutExperiments\Data\DB_T15\test\Cam1DLC_Resnet50_CanariJul9shuffle1_snapshot_125_full.pickle]!
Video [C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCutExperiments\Data\DB_T15\test\Cam2.avi] already analyzed at [C:\Users\Salle-Cineradio\Documents\MachineLearning\BirdSongs-MNHN\Testing\DeepLabCutExperiments\Data\DB_T15\test\Cam2DLC_Resnet50_CanariJul9shuffle1_snapshot_125_full.pickle]!
No .h5 files were created during video analysis. Please check your code and ensure that the video inference and output generation are correct.

