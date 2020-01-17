
### Missing file in this folder

Please download bert_model.ckpt.data-00000-of-00001 from https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip and then put this file in the folder of bert-base-chinese.

And then use the script named as 'convert_tf_checkpoint_to_pytorch.py' in the folder of OpenKG-for-Game-of-Thrones/bert_pretrain/ convert_tf_to_pytorch to generate the file called pytorch_model.bin.

The command for this generation is showed as follow:

    export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12

    python convert_tf_checkpoint_to_pytorch.py \
      --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
      --bert_config_file $BERT_BASE_DIR/bert_config.json \
      --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
