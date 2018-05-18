import os

SUMMARIZE_PROBLEMS = [
    "summarize_cs_legal32k",
    "summarize_de_legal32k",
    "summarize_en_legal32k",
    "summarize_es_legal32k",
    "summarize_fr_legal32k",
    "summarize_it_legal32k",
    "summarize_sv_legal32k",
]


def main():
    for problem in SUMMARIZE_PROBLEMS:
        os.system("mkdir -p $TRAIN_DIR/summarize/"+problem);
        successful = False
        while not successful:
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/summarize --output_dir=$TRAIN_DIR/multimodel/summarize/"+problem+" --worker_gpu=4 --train_steps=100000 --model=multi_model --hparams_set=multimodel_base --local_eval_frequency=500 --save_checkpoints_secs=180 --problems="+problem
            if os.system(cmd) == 0:
                successful = True

main()
