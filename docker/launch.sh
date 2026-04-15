#!/bin/bash
cd "${0%/*}/.."

function usage
{
    echo "usage: ./launch.sh [-b/-d/-r]"
    echo "Choose action from:"
    echo "      -b | Build Docker image"
    echo "      -d | Develop inside Docker container"
    echo "      -r | Run DeepStream application"
}

ACTION=""

if [[ $# -ne 1 ]]; then
	usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -d | -r )
                        ACTION=$1
                        ;;
        -h )
                        usage
                        exit
                        ;;
        * )
                        usage
                        exit
                        ;;
    esac
    shift;
done

if [[ $ACTION == "" ]]; then
	usage && exit;
fi

BASE_IMAGE=nvcr.io/nvidia/deepstream:7.1-triton-multiarch
DOCKER_FILE=docker/Dockerfile
DOCKER_TAG=deepstream-dev
DOCKER_TAG_VERSION=v1.0
DOCKER_NAME=deepstream-dev

DISPLAY_ARGS=""
if [[ -n $DISPLAY ]]; then
    xhost +local:root
    DISPLAY_ARGS="-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix"
else
    echo "No display detected — running in headless mode (EGL surfaceless)"
    DISPLAY_ARGS="-e EGL_PLATFORM=surfaceless"
fi

if [[ $ACTION == '-b' ]]; then
    time \
    docker build -f $DOCKER_FILE -t $DOCKER_TAG:$DOCKER_TAG_VERSION . \
        --build-arg BASE_IMAGE=${BASE_IMAGE};
    exit;

elif [[ $ACTION == '-r' ]] || [[ $ACTION == '-d' ]] ; then
    docker run --name=${DOCKER_NAME} \
        $DISPLAY_ARGS \
        --rm --net=host --ipc=host --shm-size=4g --privileged -it \
		--runtime=nvidia \
		-v "$(pwd)":/workspace \
		-v "$(pwd)/apps":/apps \
		-v "$(pwd)/cpp_apps":/cpp_apps \
        -v "$(pwd)/models":/models \
        -v "$(pwd)/streams":/streams \
		-v "$(pwd)/triton/model_repo":/triton/model_repo \
        -v /dev:/dev \
		--entrypoint /opt/entrypoint.sh \
		$DOCKER_TAG:$DOCKER_TAG_VERSION \
		$ACTION;
    exit;

else
    echo "Invalid option from [action <-b/-r/-d>]"
    usage
    exit 1
fi
