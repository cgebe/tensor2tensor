import os

PROBLEMS = [
    "court_classification",
    "verdict_classification"
]


def main():
    for problem in PROBLEMS:
        os.system("mkdir -p $DATA_DIR/classification/"+problem);
        if os.system("python ./t2t-datagen --data_dir=$DATA_DIR/classification --tmp_dir=$TMP_DIR --problem=" + problem) == 0:
            continue
        else:
            print "ERROR " + problem
            break

main()
