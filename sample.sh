MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 16 --timestep_respacing 200"
DATAPATH=/path/to/correct/dataset
MODELPATH=/path/to/ddpm_model.pth
CLASSIFIERPATH=/path/to/classifier.pth
DETECTORPATH=/path/to/detector.pth
OUTPUT_PATH=/path/to/saved/results
EXPNAME=name_of_experiment

# parameters of the sampling
GPU=0
S=60 
SEED=4
NUMBATCHES=40
USE_LOGITS=TRUE
CLASS_SCALES='7,11,15' # classifier scales could be different from detector scales
DETECTOR_SCALES='6,9,12' # detector scales could be different from classifier scales
LAYER=18 # layer for the perceptual loss
PERC=30 # weight of the perceptual loss
L1=0.07 # weight of the L1 loss
IMAGESIZE=256  # dataset shape

python -W ignore main_decodex.py $MODEL_FLAGS $SAMPLE_FLAGS \
  --output_path $OUTPUT_PATH --num_batches $NUMBATCHES \
  --start_step $S --dataset 'PE90_DotNoSupport' \
  --exp_name $EXPNAME --gpu $GPU \
  --model_path $MODELPATH --classifier_scales $CLASS_SCALES --detector_scales $DETECTOR_SCALES \
  --classifier_path $CLASSIFIERPATH --detector_path $DETECTORPATH --seed $SEED \
  --oracle_path $ORACLEPATH \
  --seed $SEED --data_dir $DATAPATH \
  --l1_loss $L1 --use_logits $USE_LOGITS \
  --l_perc $PERC --l_perc_layer $LAYER \
  --save_x_t True --save_z_t True \
  --use_sampling_on_x_t True \
  --save_images True --image_size $IMAGESIZE 
