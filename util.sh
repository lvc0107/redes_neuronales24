#!/usr/bin/env bash
warn_message () {
    echo -e "\033[33m$1\033[0m"
}
success_message(){
    echo -e "\033[32m$1\033[0m"
}
info_message() {
    echo -e "\033[34m$1\033[0m"
}
fail_message() {
    echo -e "\033[31m$1\033[0m"
}
