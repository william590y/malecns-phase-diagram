#!/bin/bash
#SBATCH -p thickstun
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 03:00:00
#SBATCH -o /MALECNS_ROOT/logs/%x_%j.out
# module_theory CPU job. usage: sbatch -J pd3-module_theory-<x> pd3_mt_cpu.sbatch <pd3 script basename> <args>
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python /MALECNS_ROOT/code/"$@"
