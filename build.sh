#!/usr/bin/env bash
source ./util.sh


info_message "Installing dependencies"
poetry install --no-root

info_message "Installing pre-commit"
pre-commit install
pre-commit install --hook-type commit-msg


success_message "Build Done!"
