from multisim.plot import plot_steering_distribution


archive_path = "/home/lev/Downloads/training_datasets/raw/"

additional_data_paths = [
    # maxibon - seed2000 - 25
    # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/udacity_2025-07-30_18-13-59",
    # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/donkey_2025-07-30_14-04-44",
    # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/beamng_2025-07-30_14-17-01",

    # maxibon - seed3000 - 25
    # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/beamng_2025-07-31_22-59-29/",
    # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/donkey_2025-07-31_22-47-17/",
    # "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/udacity_2025-08-02_01-55-41"

    #"/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/20-07-2025",
    #"/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/18-07-2025/",
    # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/20-07-2025_2000/",
    # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/21-07-2025_2000/",
    # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/23-07-2025_2000", # udacity
    # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/24-07-2025_2000", # udacity

    # "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/bng_recording_25-07-25_2000/25-07-2025_2000" # udacity
    ]
folder_paths = [archive_path] + additional_data_paths

# evaluate distribution
plot_steering_distribution(folder_paths, 
                           normalize=False,
                           title="Steering Distribution Original",
                           save_path="./multisim/data_dist/original")

#################### Lev ###################################################################

##### Collected Udacity - 100 tracks, Lev

data_paths = [
    "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/udacity_25-07-2025_2000", # udacity
]
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Udacity - seed2000 - tracks100 - lev",
                           save_path="./multisim/data_dist/Udacity-s2000-tracks100-lev")
##################### Collected Donkey - 100 tracks, Lev

data_paths = [
    "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/donkey_20-07-2025_2000/",
    "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/donkey_21-07-2025_2000/"
]
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Donkey - seed2000 - tracks100 - lev",
                           save_path="./multisim/data_dist/Donkey-s2000-tracks100-lev")

##################### Collected Beamng - 100 tracks, Lev

data_paths =[
    "/home/lev/Documents/testing/MultiSimulation/opensbt-multisim/recording/data/bng_recording_25-07-25_2000/25-07-2025_2000" # udacity
]

plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Beamng - seed2000 - tracks100 - lev",
                           save_path="./multisim/data_dist/Beamng-s2000-tracks100-lev")

####################### MAXIBON #########################

data_paths = [
    # maxibon - seed2000 - 25
   "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/udacity_2025-07-30_18-13-59"
]
    
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Udacity - seed2000 - tracks25 - Matteo",
                           save_path="./multisim/data_dist/Udacity-s2000-tracks25-matteo")
#######################


data_paths = [
    # maxibon - seed2000 - 25
    "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/donkey_2025-07-30_14-04-44"
]
    
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Donkey - seed2000 - tracks25 - Matteo",
                           save_path="./multisim/data_dist/Donkey-s2000-tracks25-matteo")
#######################


data_paths = [
    # maxibon - seed2000 - 25
    "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/2000/beamng_2025-07-30_14-17-01"
]
    
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Beamng - seed2000 - tracks25 - Matteo",
                           save_path="./multisim/data_dist/Beamng-s2000-tracks25-matteo")

###### Seed 3000


data_paths = [
    # maxibon - seed3000 - 25
   "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/udacity_2025-08-02_01-55-41"
]
    
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Udacity - seed3000 - tracks25 - Matteo",
                           save_path="./multisim/data_dist/Udacity-seed3000-tracks25-matteo")
#######################


data_paths = [
    # maxibon - seed3000 - 25
    "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/donkey_2025-07-31_22-47-17"
]
    
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Donkey - seed3000 - tracks25 - Matteo",
                           save_path="./multisim/data_dist/Donkey-seed3000-tracks25-matteo")
#######################


data_paths = [
    # maxibon - seed3000 - 25
    "/home/lev/Documents/testing/MultiSimulation/vit-recordings-maxi/3000/beamng_2025-07-31_22-59-29"
]
    
plot_steering_distribution(data_paths, 
                           normalize=False,
                           title="Steering Distribution Beamng - seed3000 - tracks25 -  Matteo",
                           save_path="./multisim/data_dist/Beamng-seed3000-tracks25-matteo")
