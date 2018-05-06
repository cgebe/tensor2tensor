import os

LABEL_PROBLEMS = [
    "multi_labeling_cs_legal32k",
    "multi_labeling_de_legal32k",
    "multi_labeling_en_legal32k",
    "multi_labeling_es_legal32k",
    "multi_labeling_fr_legal32k",
    "multi_labeling_it_legal32k",
    "multi_labeling_sv_legal32k"
]


def main():
    for problem in LABEL_PROBLEMS:
        os.system("mkdir -p $TRAIN_DIR/multi_labeling/"+problem);
        successful = False
        while not successful:
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/multi_labeling --output_dir=$TRAIN_DIR/multi_labeling/"+problem+" --worker_gpu=4 --train_steps=75000 --model=multi_model --hparams_set=multimodel_legal --local_eval_frequency=500 --save_checkpoints_secs=90 --problems="+problem
            if os.system(cmd) == 0:
                successful = True

main()
