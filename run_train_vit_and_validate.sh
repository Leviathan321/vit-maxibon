# train model and get the models dir
lr_rates=(0.000015 0.000013 0.00001 0.0001)
fname_suffix="tune-donkey_data-0.4_take-2"

for lr in "${lr_rates[@]}"; do
    source venv/Scripts/activate

    echo "[INFO] Starting training and retrieving checkpoint directory..."
    checkpoint_dir=$(python -m multisim.train_vit --lr_rate_tune "$lr" --suffix_folder_name "$fname_suffix" | tail -n 1)

    echo "[INFO] Training script returned checkpoint dir: $checkpoint_dir"

    export VIT_MODEL_PATH="$checkpoint_dir/vit_mixed.ckpt"
    echo "[INFO] Exported VIT_MODEL_PATH: $VIT_MODEL_PATH"

    original_dir=$(pwd)

    # Define path to the other project
    EVALUATE_VALIDITY_DIR="../opensbt-multisim/"  # adjust as needed

    cd $EVALUATE_VALIDITY_DIR

    if [[ "$OS_TYPE" == "linux" || "$OS_TYPE" == "darwin" ]]; then
        echo "[INFO] Detected Linux/macOS. Activating venv"
        source venv/bin/activate
    else
        echo "[INFO] Detected Windows. Activating venv."
        source venv/Scripts/activate
    fi

    # evaluate model with sims
    # Get current date and time with hyphen delimiters
    datetime=$(date +"%d-%m-%Y")

    # Run the Python script with timestamped prefix
    python -m scripts.evaluate_validity \
        -s "beamng_vit" \
        -n 3 \
        -prefix "VIT_$datetime" \
        -save_folder "$checkpoint_dir/validation_beamng/"

    datetime=$(date +"%d-%m-%Y")

    # Run the Python script with timestamped prefix
    python -m scripts.evaluate_validity \
        -s "donkey_vit" \
        -n 3 \
        -prefix "VIT_$datetime" \
        -save_folder "$checkpoint_dir/validation_donkey/"

    datetime=$(date +"%d-%m-%Y")

    # Run the Python script with timestamped prefix
    python -m scripts.evaluate_validity \
        -s "udacity_vit" \
        -n 3 \
        -prefix "VIT_$datetime" \
        -save_folder "$checkpoint_dir/validation_udacity/"

    # Return to original dir
    cd "$original_dir" || { echo "[ERROR] Cannot cd back to $original_dir"; exit 1; }
done