#!/usr/bin/env bash

DATA_DIR="data/github"

main() {
    cd "$DATA_DIR" || exit 1

    for folder in cuda opencl opengl vulkan; do
        echo "Cleaning $folder"
        for file in "$folder"/*; do
            # file is empty
            if [ ! -s "$file" ] || grep -q '\[\]' "$file"; then
                echo "Removing $file"
                rm "$file"
                continue
            fi

            echo "Cleaning $file"
            tmp=$(mktemp)
            echo '[' > "$tmp"
            sed -e '/\[\]/d' -e 's/\]/\],/' -e '$d' "$file" >> "$tmp"
            echo ']]' >> "$tmp"

            jq 'flatten' < "$tmp" > "$file"
        done
    done
}

main
