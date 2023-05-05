#!/bin/bash
echo "Starting Accident Avoidance Demonstrator"
source /opt/intel/openvino_2022.1.1.7030/setupvars.sh
workon openvino
python accident_avoidance_v6.py --prototxt /home/pi/MobileNetSSD_deploy.prototxt --model /home/pi/MobileNetSSD_deploy.caffemodel