#!/bin/bash

# Define the base directory for datasets
DATA_DIR="./data"
mkdir -p $DATA_DIR

echo "================================================="
echo "Downloading BigTime Dataset..."
echo "================================================="
wget -c http://www.cs.cornell.edu/projects/megadepth/dataset/bigtime/BigTime_v1.tar.gz -P $DATA_DIR
# tar -xzvf $DATA_DIR/BigTime_v1.tar.gz -C $DATA_DIR

echo "================================================="
echo "Downloading VIDIT Dataset (Training set)..."
echo "================================================="
wget -c https://datasets.epfl.ch/vidit/VIDIT_train.zip -P $DATA_DIR
# unzip $DATA_DIR/VIDIT_train.zip -d $DATA_DIR

echo "================================================="
echo "Downloading RSR: Real Scene Relighting Dataset..."
echo "================================================="
echo "Note: The RSR dataset is hosted on SharePoint."
echo "Automated downloading via wget might result in an HTML page instead of the archive file."
echo "If the resulting zip file is invalid, please manually download from the following links and place them in the $DATA_DIR directory:"
echo "- RSR (256x256 resolution): https://cvcuab-my.sharepoint.com/:u:/g/personal/yixiong_cvc_uab_cat/ETWcj5yBKgJLqUZDsT9Q39QBJ8GUJYEQuzNWpV5FS2lPRg?e=jEYI5t"
echo "- RSR (Original resolution): https://cvcuab-my.sharepoint.com/:u:/g/personal/yixiong_cvc_uab_cat/ET7MLf3u27dMktC52VaMGL4Bu203mKfAcniBaPKhGktVIw?e=ZYPAjX"
echo ""
echo "Attempting to download 256x256 resolution via wget..."
wget -c "https://cvcuab-my.sharepoint.com/:u:/g/personal/yixiong_cvc_uab_cat/ETWcj5yBKgJLqUZDsT9Q39QBJ8GUJYEQuzNWpV5FS2lPRg?download=1" -O $DATA_DIR/RSR_256x256.zip

# To download the original resolution dataset via script instead, uncomment the below:
# wget -c "https://cvcuab-my.sharepoint.com/:u:/g/personal/yixiong_cvc_uab_cat/ET7MLf3u27dMktC52VaMGL4Bu203mKfAcniBaPKhGktVIw?download=1" -O $DATA_DIR/RSR_original.zip

echo "================================================="
echo "Download script finished."
echo "================================================="
