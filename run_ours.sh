
# AVE
python main.py \
  --prefix reproduce \
  --dataset ave \
  --memory_size 340 \
  --memory_per_class 20 \
  --no-fixed_memory \
  --shuffle \
  --init_cls 7 \
  --increment 7 \
  --model_name avcil \
  --device 0 \
  --seed 42 \
  --project nips26 \

# Kinetics
# python main.py \
#   --prefix reproduce \
#   --dataset kinetics \
#   --memory_size 340 \
#   --memory_per_class 20 \
#   --no-fixed_memory \
#   --shuffle \
#   --init_cls 6 \
#   --increment 6 \
#   --model_name ours \
#   --device 0 \
#   --seed 42 \
#   --project nips26