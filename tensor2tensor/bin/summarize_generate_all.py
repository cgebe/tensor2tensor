import os


PROBLEMS = [
    #"summarize_cs_legal32k",
    "summarize_de_legal32k",
    #"summarize_en_legal32k",
    #"summarize_es_legal32k",
    #"summarize_fr_legal32k",
    #"summarize_it_legal32k",
    #"summarize_sv_legal32k"
]

# shit doesn work
def main():
    for problem in PROBLEMS:
        if os.system("python ./t2t-datagen --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR --problem=" + problem) == 0:
            continue
        else:
            print "ERROR " + problem
            break

main()
