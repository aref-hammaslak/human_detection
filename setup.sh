#!/bin/sh
set -e

pip install --no-cache-dir -r requirements.txt
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0
apt-get clean
rm -rf /var/lib/apt/lists/*