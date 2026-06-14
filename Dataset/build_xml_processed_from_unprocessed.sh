#!/usr/bin/env bash
set -euo pipefail

# Build XML-formatted training CSVs from unprocessed grid samples.
#
# Usage:
#   bash Dataset/build_xml_processed_from_unprocessed.sh
#
# Override defaults:
#   BUS_SYSTEMS="33,69,84" \
#   UNPROCESSED_DIR=Dataset/Unprocessed \
#   OUTPUT_DIR=Dataset/Processed_xml \
#   OUTPUT_NAME=train_33_69_84_nodes_open_lines_xml \
#   OUTPUT_XML_FORMAT=open_lines \
#   bash Dataset/build_xml_processed_from_unprocessed.sh
#
# For full XML (with node voltages and system loss):
#   OUTPUT_XML_FORMAT=full_xml \
#   OUTPUT_NAME=train_33_69_84_nodes_full_xml \
#   bash Dataset/build_xml_processed_from_unprocessed.sh

BUS_SYSTEMS="${BUS_SYSTEMS:-33,69,84}"
UNPROCESSED_DIR="${UNPROCESSED_DIR:-Dataset/Unprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-Dataset/Processed_xml}"
OUTPUT_NAME="${OUTPUT_NAME:-train_33_69_84_nodes_open_lines_xml}"
OUTPUT_XML_FORMAT="${OUTPUT_XML_FORMAT:-open_lines}"

mkdir -p "${OUTPUT_DIR}"

python Dataset/build_xml_processed_from_unprocessed.py \
  --bus_systems "${BUS_SYSTEMS}" \
  --unprocessed_dir "${UNPROCESSED_DIR}" \
  --output_csv "${OUTPUT_DIR}/${OUTPUT_NAME}.csv" \
  --output_xml_format "${OUTPUT_XML_FORMAT}"
