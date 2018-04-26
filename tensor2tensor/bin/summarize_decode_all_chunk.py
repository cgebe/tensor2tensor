import os
from itertools import izip_longest
import shutil

TRANSLATE_PROBLEMS = [
    #"summarize_cs_legal32k",
    "summarize_de_legal32k",
    #"summarize_en_legal32k",
    #"summarize_es_legal32k",
    #"summarize_fr_legal32k",
    #"summarize_it_legal32k",
    #"summarize_sv_legal32k",
]


def main():
    os.system("mkdir -p decodes");
    for problem in TRANSLATE_PROBLEMS:
        lang = problem.split("_")[1]
        decode_file = "jrc_acquis." + lang + "-test.fulltexts"
        os.system("mkdir -p chunks");

        n = 10
        with open(os.environ['DECODE_DIR'] + "/summarize/" + decode_file) as f:
            for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
                with open('chunks/chunk_{0}'.format(i * n), 'w') as fout:
                    fout.writelines(g)

        for chunk in sorted(os.listdir('chunks')):
            print(chunk)
            cmd = "python ./t2t-decoder --data_dir=$DATA_DIR/summarize/joint --output_dir=$TRAIN_DIR/summarize/"+problem+" --model=multi_model --hparams_set=multimodel_legal --problems="+problem+" --decode_hparams='batch_size=1,beam_size=4,alpha=0.6' --decode_from_file=chunks/"+chunk+" --decode_to_file=decodes/"+chunk
            os.system(cmd)

        with open(os.environ['DECODE_DIR'] + "/summarize/single-"+problem+".decodes", 'w') as outfile:
            for decodes in sorted(os.listdir('decodes')):
                with open(decodes) as infile:
                    outfile.write(infile.read())

        shutil.rmtree('chunks')
        #shutil.rmtree('decodes')

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


main()
