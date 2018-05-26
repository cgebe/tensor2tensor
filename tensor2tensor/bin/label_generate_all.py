import os


PROBLEMS = [
    #"multi_labeling_cs_legal32k",
    "multi_labeling_de_legal32k",
    #"multi_labeling_en_legal32k",
    #"multi_labeling_es_legal32k",
    #"multi_labeling_fr_legal32k",
    #"multi_labeling_it_legal32k",
    #"multi_labeling_sv_legal32k"
]


def main():
    for problem in PROBLEMS:
        if os.system("python ./t2t-datagen --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR --problem=" + problem) == 0:
            continue
        else:
            print "ERROR " + problem
            break

main()
