import os

PROBLEMS = [
    #"summarize_cs_legal32k",
    "summarize_de_legal32k",
    #"summarize_en_legal32k",
    #"summarize_es_legal32k",
    #"summarize_fr_legal32k",
    #"summarize_it_legal32k",
    #"summarize_sv_legal32k",
]


def main():
    for problem in PROBLEMS:
        lang = problem.split("_")[1]
        decode_file = "jrc_acquis." + lang + "-test.fulltexts"
        cmd = "python ./t2t-decoder --data_dir=$DATA_DIR --output_dir=$TRAIN_DIR/multimodel/joint-each --model=multi_model --hparams_set=multimodel_base --problems="+problem+" --decode_hparams='batch_size=8,beam_size=4,alpha=0.6' --decode_from_file=$DECODE_DIR/"+decode_file+" --decode_to_file=$DECODE_DIR/summarize.joint-3-de"
        os.system(cmd)

main()
