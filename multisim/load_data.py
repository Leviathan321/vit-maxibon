import os
from multisim.global_log import GlobalLog
from multisim.dataset_utils import load_all_into_dataset
import numpy as np

# load images from matteos dataset
logg = GlobalLog("train_model")

# archive_path = r"C:\\Users\\sorokin\\Downloads\\training_datasets\\"
#archive_path = r"/home/lev/Downloads/training_datasets/"
archive_path = fr"C:\Users\levia\Downloads\training_datasets\\"

seed = 1
test_split = 0
predict_throttle = False

env_names = [
             #"beamng", 
             #"udacity", 
             "donkey"
             ]

for env_name in env_names:
    ######################

    if env_name == "udacity":
        archive_names =  [r"udacity-2022_05_31_12_17_56-archive-agent-autopilot-seed-0-episodes-50.npz"]
    elif env_name == "donkey":
        archive_names = [r"donkey-2022_05_31_12_45_57-archive-agent-autopilot-seed-0-episodes-50.npz"]
    elif env_name == "beamng":
        archive_names = [r"beamng-2022_05_31_14_34_55-archive-agent-autopilot-seed-0-episodes-50.npz"]

    ##################
    if seed == -1:
        seed = np.random.randint(2**30 - 1)

    logg.info("Random seed: {}".format(seed))

    dataset, labels = load_all_into_dataset(
        archive_path=archive_path,
        archive_names=archive_names,
        predict_throttle=predict_throttle,
        env_name=None if env_name != "mixed" else "mixed")
    
    logg.info("shape dataset:{}".format(dataset.shape))
    logg.info("shape labels: {}".format(labels.shape))

    # # num = 10000
    # num = None

    # ############## write steering labels
    
    # labels_folder = os.getcwd() + os.sep + "/labels_matteo/" + env_name + os.sep
    # Path(labels_folder).mkdir(parents = True, exist_ok = True)
    
    # # Write array to a JSON file
    # with open(labels_folder + 'labels.json', 'w') as json_file:
    #     json.dump(labels.tolist(), json_file)

    # ############### perform segmentation

    # images_folder = os.getcwd() + os.sep + "/images_matteo/" + env_name + os.sep
    # Path(images_folder).mkdir(parents = True, exist_ok = True)

    # # write images down
    # for i, image in enumerate(dataset[:num]):
    #     img = Image.fromarray(image)
    #     img.save(images_folder + os.sep + f"image_{i}.jpg")

    # output_folder = os.getcwd() + os.sep + "/output_matteo/" + os.sep

    # # run colour masks based segmentation scipt
    # segmented = segment_from_arrays(image_arrays = dataset[:num],
    #                     env_name = env_name, 
    #                     output_folder = output_folder,
    #                     image_names = None,
    #                     do_write = True)