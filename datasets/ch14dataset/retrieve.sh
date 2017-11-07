#!/usr/bin/env bash


die() { echo "$@" 1>&2 ; exit 1; }

trap "echo Exited!; exit;" SIGINT SIGTERM

if [ "$#" -ne 3 ]; then
    die "Usage: retrieve.sh ftpuser ftppassword testarchivepassword"
fi

username="${@: -3}"
password="${@: -2}"
testpassword="${@: -1}"


# Download archives ---------------------------------------------------------------------

urls=("http://sunai.uoc.edu/chalearnLAP/data/track3/test/ChalearnLAP2104_EvaluateTrack3.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/test/Test1e.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/test/Test2e.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/test/Test3e.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/test/Test4e.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/test/Test5e.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Train1.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Train2.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Train3.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Train4.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Train5.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Validation1.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Validation2.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/development/Validation3.zip"
      "http://sunai.uoc.edu/chalearnLAP/data/track3/test/validation.zip")

checksums=("68d1b3f7b12c696d84f57d8fd9838da852bcfb86"
           "e4bf0107fa6af27011e2cb1bc43a68617193a688"
           "8f6e8f4e17d590cdba3ee189985f4f0b2f46b118"
           "76e554ad3a7d2acb865880dd28ecbf3773d255c4"
           "fc1c3f4febf0f19aeade1200888fc15eaa1a4d55"
           "273028b2978d0aa45c8c83d50f620b002e15ef27"
           "d5ff9e784690c897fa341ef8c2f2c4986c8f7dbd"
           "024150e44a17bb2ef1a9756fde1f727e8fdb8c3b"
           "ba01e840473768098981534c732cd5f809307306"
           "4cdc16627bc6d9dca3453507ad9ac5f9ce883128"
           "9587b81471a35b002758b4d4c99f53bbd96e3e36"
           "d0ae672851d83fc5213d95c5bda895e76dab20c4"
           "7956e763f04056166f1c6dc00d8801c3618285b6"
           "cc1d7efa8b3fd30f80fc1b351791d9c5473943bc"
           "b44bcca06d5f26e4df732aae7d4a1b35a155cd6c")

archives=("ChalearnLAP2104_EvaluateTrack3.zip"
          "Test1e.zip"
          "Test2e.zip"
          "Test3e.zip"
          "Test4e.zip"
          "Test5e.zip"
          "Train1.zip"
          "Train2.zip"
          "Train3.zip"
          "Train4.zip"
          "Train5.zip"
          "Validation1.zip"
          "Validation2.zip"
          "Validation3.zip"
          "validation.zip")

for (( i=0; i<${#urls[@]}; i++ )); do
    checksum=${checksums[$i]}
    archive=${archives[$i]}
    url=${urls[$i]}

    echo "testing $archive"

    if echo "$checksum  $archive" | sha1sum -c --quiet --status -; then
        echo "$archive is valid."
    else
        echo "downloading $archive because it is missing or corrupted"
        curl -u $username:$password -O $url || die "failed to download archive"
        echo "$checksum  $archive" | sha1sum -c --quiet --status - || die "failed to validate archive"
    fi
done


# Deflate archives ----------------------------------------------------------------------

archives=("Test1e.zip"
          "Test2e.zip"
          "Test3e.zip"
          "Test4e.zip"
          "Test5e.zip"
          "Train1.zip"
          "Train2.zip"
          "Train3.zip"
          "Train4.zip"
          "Train5.zip"
          "Validation1.zip"
          "Validation2.zip"
          "Validation3.zip")

for a in "${archives[@]}"; do
    echo "processing $a"
    for f in `unzip -l $a | head -n-2 | tail -n+4 | awk -F' ' '{print $4}'`; do
        unzip -o -q -P $testpassword $a $f
        unzip -o -q $f
        rm $f
    done
done

unzip -o -q validation.zip
unzip -o -q -P $testpassword -j ChalearnLAP2104_EvaluateTrack3.zip "ChalearnLAP2104_EvaluateTrack3/input/ref/*"
