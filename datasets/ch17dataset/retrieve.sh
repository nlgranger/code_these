#!/usr/bin/env bash


die() { echo "$@" 1>&2 ; exit 1; }


trap "echo Exited!; exit;" INT TERM


if [ "$#" -ne 3 ]; then
    die "Usage: retrieve.sh ftpuser ftppassword testarchivepassword"
fi


ftp_user="$1"
ftp_passwd="$2"
val_passwd="$3"


mkdir -p data && cd data
#if echo "259eef38ea8ac68c3ffb6da21460152348eb7178  ConGD_Phase_1.tar.gz" | sha1sum --quiet --status -c -; then
#    echo "Skipping download for ConGD_Phase_1.tar.gz"
#else
#    #aria2c --ftp-user="$ftp_user" --ftp-passwd="$ftp_passwd" ftp://cbsr.ia.ac.cn/ConGD_Phase_1.tar.gz
#    echo "259eef38ea8ac68c3ffb6da21460152348eb7178  ConGD_Phase_1.tar.gz" | sha1sum --quiet --status -c - || die "Failed to validate archive"
#fi
#if echo "1778900d01ab2f8f9c730d81fbb24513dc234b27  ConGD_Phase_2.zip" | sha1sum --quiet --status -c -; then
#    echo "Skipping download for ConGD_Phase_2.zip"
#else
#    #aria2c --ftp-user="$ftp_user" --ftp-passwd="$ftp_passwd" ftp://cbsr.ia.ac.cn/ConGD_Phase_2.zip
#    echo "1778900d01ab2f8f9c730d81fbb24513dc234b27  ConGD_Phase_2.zip" | sha1sum --quiet --status -c - || die "Failed to validate archive"
#fi

tar xf ConGD_Phase_1.tar.gz
unzip -q ConGD_Phase_2.zip
mkdir -p videos/test && cd videos/test
unzip -q -P ${val_passwd} ../../ConGD_phase_2/test.zip
cd ../
mv ConGD_phase_1/train ConGD_phase_1/valid .
cd ..
mkdir -p annotations
mv ConGD_phase_1/train.txt ConGD_phase_1/valid.txt ConGD_phase_2/test.txt ./annotations
rm ConGD_phase_1 ConGD_phase_2 -r
