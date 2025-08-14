# train model and get the models dir
checkpoint_dir=""

original_dir=$(pwd)

# Define path to the other project relative to the current project
EVALUATE_VALIDITY_DIR="../opensbt-multisim/"  # adjust this path as needed

cd $EVALUATE_VALIDITY_DIR

source $EVALUATE_VALIDITY_DIR/venv/bin/activate

# TODO NEED TO SET CONFIG PATH

# evaluate model with sims
bash $EVALUATE_VALIDITY_DIR/run_evaluate_validity_vit_ud.sh $checkpoint_dir
bash $EVALUATE_VALIDITY_DIR/run_evaluate_validity_vit_dnk.sh $checkpoint_dir
bash $EVALUATE_VALIDITY_DIR/run_evaluate_validity_vit_bng.sh $checkpoint_dir


cd "$original_dir"