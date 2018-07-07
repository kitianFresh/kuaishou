#!/bin/bash


POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -s|--sample)
    SAMPLE="$2"
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

echo SAMPLE          = "${SAMPLE}"
if [[ ${SAMPLE} == '1' ]];
then
    python face_feature_extract.py -s
    python text_classify.py -s
    python text_feature_extract.py -s
    python photo_feature_extract.py -s
    python user_feature_extract.py -s
    python feature_ensemble.py -s
    python feature_discretization.py -s
else
    python face_feature_extract.py
    python text_classify.py
    python text_feature_extract.py
    python photo_feature_extract.py
    python user_feature_extract.py
    python feature_ensemble.py
    python feature_discretization.py
fi
