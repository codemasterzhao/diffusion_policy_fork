#!/bin/bash
#SBATCH --job-name=x11-job
#SBATCH --output=result.txt
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --x11=first  # Enabling X11 forwarding

# Your commands go here
xeyes  # This is a simple X11 application for testing
