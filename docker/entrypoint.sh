#!/usr/bin/env bash

function usage
{
    echo "usage: ./entrypoint.sh [-b/-r/-d]"
    echo "Choose action from:"
    echo "  -b | Build/setup tasks"
    echo "  -r | Run DeepStream application"
    echo "  -d | Develop using bash terminal"
    echo "  -h | Help"
}

ACTION=""

if [[ $# -ne 1 ]]; then
    usage && exit;
fi

while [[ "$1" != "" ]]; do
    case $1 in
        -b | -r | -d )    ACTION=$1    ;;
        -h )              usage && exit;;
        * )               usage && exit;;
    esac
    shift;
done

if [[ $ACTION == '-b' ]]; then
    echo "Build/setup tasks — add your setup steps here."
    /bin/bash
elif [[ $ACTION == '-r' ]]; then
    cd /workspace
    python3 main.py
elif [[ $ACTION == '-d' ]]; then
    /bin/bash
else
    usage && exit;
fi
