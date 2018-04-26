import os

TRANSLATE_PROBLEMS = [
    "summarize_cs_legal32k",
    "summarize_de_legal32k",
    "summarize_en_legal32k",
    "summarize_es_legal32k",
    "summarize_fr_legal32k",
    "summarize_it_legal32k",
    "summarize_sv_legal32k",
]


def main():
    for problem in TRANSLATE_PROBLEMS:
        successful = False
        while not successful:
            cmd = "python ./t2t-trainer --data_dir=$DATA_DIR/summarize/joint --output_dir=$TRAIN_DIR/summarize/"+problem+" --model=multi_model --hparams_set=multimodel_legal --problems="+problem+"--decode_hparams='batch_size=4,beam_size=4,alpha=0.6' --decode_from_file=$DECODE_DIR/summarize"+" --decode_to_file=$DECODE_DIR/summarize/single"
            if os.system(cmd) == 0:
                successful = True

main()
