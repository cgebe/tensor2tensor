import os

PROBLEMS = [
    "court_classification",
    "verdict_classification"
]

def main():
    for problem in PROBLEMS:
        decode_file = problem + ".facts-test"
        cmd = "python ./t2t-decoder --data_dir=$DATA_DIR/classification --output_dir=$TRAIN_DIR/classification/"+problem+" --model=multi_model --hparams_set=multimodel_legal --problems="+problem+" --decode_hparams='batch_size=8,beam_size=4,alpha=0.6' --decode_from_file=$DECODE_DIR/classification/"+decode_file+" --decode_to_file=$DECODE_DIR/classification/single"
        os.system(cmd)

        cmd = "python ./t2t-decoder --data_dir=$DATA_DIR/classification --output_dir=$TRAIN_DIR/classification/joint --model=multi_model --hparams_set=multimodel_legal --problems="+problem+" --decode_hparams='batch_size=8,beam_size=4,alpha=0.6' --decode_from_file=$DECODE_DIR/classification/"+decode_file+" --decode_to_file=$DECODE_DIR/classification/joint"
        os.system(cmd)

main()
