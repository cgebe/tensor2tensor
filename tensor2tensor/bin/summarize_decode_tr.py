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
        cmd = "python ./t2t-decoder --data_dir=$DATA_DIR --output_dir=$TRAIN_DIR/transformer/label/"+problem+" --model=transformer --hparams_set=transformer_base_v3 --problems="+problem+" --decode_hparams='use_last_position_only=true,batch_size=2,beam_size=4,alpha=0.6' --decode_from_file=$DECODE_DIR/"+decode_file+" --decode_to_file=$DECODE_DIR/summarize.single"
        os.system(cmd)

main()
