import os

SUMMARIZE_PROBLEMS = [
    #"summarize_cs_legal32k",
    "summarize_de_legal32k",
    #"summarize_en_legal32k",
    #"summarize_es_legal32k",
    #"summarize_fr_legal32k",
    #"summarize_it_legal32k",
    #"summarize_sv_legal32k",
]


def main():
    for problem in SUMMARIZE_PROBLEMS:
        successful = False
        while not successful:
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/summarize --output_dir=$TRAIN_DIR/transformer/summarize-100k/"+problem+" --schedule=train --worker_gpu=4 --train_steps=100000 --model=transformer --hparams_set=transformer_base_v3 --save_checkpoints_secs=600 --problems="+problem
            if os.system(cmd) == 0:
                successful = True

main()
