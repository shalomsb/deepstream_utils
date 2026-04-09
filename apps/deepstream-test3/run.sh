#!/bin/bash
cd "${0%/*}"

SAMPLE="file:///streams/sample_720p.h264"

echo "==============================="
echo "  DeepStream Test3 Runner"
echo "==============================="
echo "1) Single stream"
echo "2) Multi stream (2x tiled)"
echo "3) Multi stream (4x tiled)"
echo "4) Triton backend"
echo "5) Headless (no display)"
echo "6) File loop + silent"
echo "7) Custom"
echo "q) Quit"
echo "==============================="
read -p "Choose: " choice

case $choice in
    1)
        python3 deepstream_test_3.py -i $SAMPLE
        ;;
    2)
        python3 deepstream_test_3.py -i $SAMPLE $SAMPLE
        ;;
    3)
        python3 deepstream_test_3.py -i $SAMPLE $SAMPLE $SAMPLE $SAMPLE
        ;;
    4)
        python3 deepstream_test_3.py -i $SAMPLE -g nvinferserver -c config_triton_infer_primary_peoplenet.txt
        ;;
    5)
        python3 deepstream_test_3.py -i $SAMPLE --no-display
        ;;
    6)
        python3 deepstream_test_3.py -i $SAMPLE --file-loop --silent
        ;;
    7)
        read -p "Enter args: " custom_args
        python3 deepstream_test_3.py $custom_args
        ;;
    q)
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
