import os

def main():
    problem = "translate_deen_legal32k"
    steps = 500000
    cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/translate/"+problem+" --output_dir=$TRAIN_DIR/translate/"+problem+" --worker_gpu=4 --train_steps="+str(steps)+" --model=multi_model --hparams_set=multimodel_legal --problems="+problem
    os.system(cmd)

main()
