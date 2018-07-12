#!/bin/bash


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -o|--online)
    ONLINE="$2"
    shift # past argument
    shift # past value
    ;;
    -k|--kfold)
    KFOLD="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

echo ONLINE          = "${ONLINE}"
echo KFOLD          = "${KFOLD}"
if [[ ${ONLINE} == '1' ]];
then
    python face_feature_extract.py -o
    python text_classify.py -o
    python text_feature_extract.py -o
    python photo_feature_extract.py -o
    python user_feature_extract.py -o
    python feature_ensemble.py -o
    python feature_discretization.py -o
else
    if [[ ${KFOLD} == '' ]]; then
        KFOLD=1
    fi
    echo KFOLD = "${KFOLD}"
    python3 face_feature_extract.py -k ${KFOLD}
    python3 text_classify.py -k ${KFOLD}
    python3 text_feature_extract.py -k ${KFOLD}
    python3 photo_feature_extract.py -k ${KFOLD}
    python3 user_feature_extract.py -k ${KFOLD}
    python3 feature_ensemble.py -k ${KFOLD}
    python3 feature_discretization.py -k ${KFOLD}
fi
