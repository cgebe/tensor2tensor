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
        lang = problem.split("_")[2]
        decode_file = "jrc_acquis." + lang + "-test.documents"
        cmd = "python ./t2t-decoder --data_dir=$DATA_DIR/multi_labeling --output_dir=$TRAIN_DIR/multi_labeling/joint --model=multi_model --hparams_set=multimodel_legal --problems="+problem+" --decode_hparams='batch_size=8,beam_size=4,alpha=0.6' --decode_from_file=$DECODE_DIR/multi_labeling/"+decode_file+" --decode_to_file=$DECODE_DIR/multi_labeling/joint"
        os.system(cmd)

main()
