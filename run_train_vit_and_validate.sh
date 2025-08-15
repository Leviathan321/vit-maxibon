# train model and get the models dir
checkpoint_dir=""

original_dir=$(pwd)

# Define path to the other project relative to the current project
EVALUATE_VALIDITY_DIR="../opensbt-multisim/"  # adjust this path as needed

cd $EVALUATE_VALIDITY_DIR

source $EVALUATE_VALIDITY_DIR/venv/bin/activate

# TODO NEED TO SET CONFIG PATH

# evaluate model with sims
# Get current date and time with hyphen delimiters
datetime=$(date +"%d-%m-%Y")

# Run the Python script with timestamped prefix
python -m scripts.evaluate_validity \
       -s "beamng_vit" \
       -n 3 \
       -prefix "VIT_$datetime" \
       -save_folder "$checkpoint_dir"

datetime=$(date +"%d-%m-%Y")

# Run the Python script with timestamped prefix
python -m scripts.evaluate_validity \
       -s "donkey_vit" \
       -n 3 \
       -prefix "VIT_$datetime" \
       -save_folder "$checkpoint_dir"

datetime=$(date +"%d-%m-%Y")

# Run the Python script with timestamped prefix
python -m scripts.evaluate_validity \
       -s "udacity_vit" \
       -n 3 \
       -prefix "VIT_$datetime" \
       -save_folder "$checkpoint_dir"

cd "$original_dir"