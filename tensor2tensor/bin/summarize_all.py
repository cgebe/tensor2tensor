import os

TRANSLATE_PROBLEMS = [
    #"summarize_cs_legal32k",
    #"summarize_de_legal32k",
    #"summarize_en_legal32k",
    #"summarize_es_legal32k",
    #"summarize_fr_legal32k",
    #"summarize_it_legal32k",
    "summarize_sv_legal32k",
]


def main():
    for problem in TRANSLATE_PROBLEMS:
        os.system("mkdir -p $TRAIN_DIR/summarize/"+problem);
        successful = False
        while not successful:
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/summarize/"+problem+" --output_dir=$TRAIN_DIR/summarize/"+problem+" --worker_gpu=4 --train_steps=50000 --model=multi_model --hparams_set=multimodel_legal --local_eval_frequency=500 --save_checkpoints_secs=180 --problems="+problem
            if os.system(cmd) == 0:
                successful = True

main()
