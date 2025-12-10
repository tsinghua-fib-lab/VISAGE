i=6

CUDA_VISIBLE_DEVICES=$i python /data3/maruolong/VISAGE/agents/perception_agent/core/UrbanMLLM/vila/our_eval/model_vqa_loader_multiview_llm_forward.py\
      --model-path /data3/maruolong/VISAGE/data/cross_train/31/segregation_cot_40_31_data2/vila_train_all_sisv_pretrain_textenhance_att_clean_8B_new \
      --question-file  /data3/maruolong/VISAGE/data/31_cities/data2/train_data.json \
      --outputs-file /data3/maruolong/VISAGE/data/31_cities/data2/feature_vector_segregation_train.pkl \
      --image-folder  "" \
      --temperature 0.2 \
      --conv-mode llama_3