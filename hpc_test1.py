################################################################################
# HPC TEST 1
# 
# TEN SITE ANDERSON IMPURITY MODEL
################################################################################

from surrogate import *

MODEL_NAME = "AIM"
MODEL_N = 10
PARAMETER_SPACE = ()
SELECTED_PARAMETERS = ()
INIT_THETA = ()
PARTICLE_SELECTION = None

model = SurrogateModel(
    MODEL_NAME,
    SELECTED_PARAMETERS,
    MODEL_N,
)
